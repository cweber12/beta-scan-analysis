"""ViTPose++ Ground Truth scaffold job.

beta-scanner's detection-eval harness seeds its human-authored Ground Truth poses
from a *stronger, independent* model instead of MediaPipe (the detector it grades).
That model — ViTPose++ — is top-down, so it cannot find people itself; this module
runs a person detector + tracker, selects the **Climber** track from the calibration
tap, poses the Climber's box on exactly the requested frames, and writes a
``vitpose.json`` artifact into the video's bundle. See beta-scanner ``docs/adr/0019``
and this repo's ``docs/adr/0003`` for the rationale and the cross-program contract.

The heavy ML stack (torch / transformers / ultralytics) is imported lazily inside the
default backend factories, so importing this module — and unit-testing the geometry
(Climber selection, timestamp echo, artifact shape, path safety) with stub
collaborators — needs none of it. The two seams:

- ``Tracker.track(video_path)``  -> per-frame track boxes (detect + track).
- ``PoseBackend.pose(video_path, targets)`` -> keypoints for the Climber's box on
  specific frames.
"""

from __future__ import annotations

import json
import sys
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence


# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

# Frame timestamps are echoed verbatim; beta-scanner matches the seed to its
# Detection Frames within 1 ms, so we never re-derive them from the decoder clock.
ARTIFACT_NAME = "vitpose.json"
STATUS_NAME = "vitpose.status.json"
SETUP_NAME = "setup.json"
ARTIFACT_VERSION = 1

# COCO-17 keypoint order emitted by ViTPose. The first 13 are the joints beta-scanner
# scores (it must see these exact names); eyes/ears are drawn faintly as context and
# ignored by scoring — we include them because the contract permits extra points.
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# --------------------------------------------------------------------------- #
# Value types (all coordinates video-normalized to the FULL frame, [0, 1])
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Box:
    """Person box, normalized to the full frame: (x, y) is the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def contains(self, p: Point) -> bool:
        return self.x <= p.x <= self.x + self.w and self.y <= p.y <= self.y + self.h


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class FrameTracks:
    """Tracks present on one decoded frame, keyed by track id -> full-frame box."""

    timestamp: float
    boxes: dict[int, Box]


@dataclass(frozen=True)
class PoseTarget:
    """A frame the Climber must be posed on: its index in the track history + box."""

    frame_index: int
    box: Box


@dataclass(frozen=True)
class VitPoseRequest:
    video_path: str
    route_folder: str
    video_key: str
    frames: tuple[float, ...]              # requested timestamps (seconds), echoed verbatim
    climber_point: Point | None = None
    climber_crop: Box | None = None
    panning: bool = False
    # Hash of the setup.json (Climber selection) this job runs under. Stamped into the
    # artifact as the provenance anchor: downstream pairing treats a pose run and a
    # truth file as comparable only when their setupHashes match. When the request
    # omits it, the job falls back to the bundle's current setup.json (see
    # run_vitpose_job) so new artifacts always carry the field.
    setup_hash: str | None = None


# --------------------------------------------------------------------------- #
# Backend seams
# --------------------------------------------------------------------------- #

class Tracker(Protocol):
    def track(self, video_path: Path) -> list[FrameTracks]:
        """Decode the video, detect people, and track them across frames."""
        ...


class PoseBackend(Protocol):
    def pose(self, video_path: Path, targets: Sequence[PoseTarget]) -> dict[int, list[Keypoint]]:
        """Run top-down ViTPose on each target's box; key results by frame_index."""
        ...


# --------------------------------------------------------------------------- #
# Path resolution (mirrors save_detection_run's traversal guard)
# --------------------------------------------------------------------------- #

def resolve_video_path(analysis_root: Path, video_path: str) -> Path:
    """Resolve a request's video path under the analysis root, rejecting traversal.

    ``video_path`` may be repo-relative (``analysis/<route>/<key>/<file>``) or
    already rooted at the analysis dir; either way the result must stay inside it.
    """
    analysis_root = analysis_root.resolve()
    raw = Path(video_path)

    # Strip a leading "analysis/" so both "analysis/route/key/x.mp4" and
    # "route/key/x.mp4" resolve against the analysis root the same way.
    if not raw.is_absolute() and raw.parts and raw.parts[0] == analysis_root.name:
        raw = Path(*raw.parts[1:])

    candidate = raw if raw.is_absolute() else (analysis_root / raw)
    resolved = candidate.resolve()

    if analysis_root != resolved and analysis_root not in resolved.parents:
        raise ValueError("Resolved video path escapes the analysis root.")
    return resolved


def bundle_dir_for(analysis_root: Path, route_folder: str, video_key: str) -> Path:
    analysis_root = analysis_root.resolve()
    resolved = (analysis_root / route_folder / video_key).resolve()
    if analysis_root not in resolved.parents:
        raise ValueError("Resolved bundle path escapes the analysis root.")
    return resolved


# --------------------------------------------------------------------------- #
# Climber Identity — a per-frame box trajectory stitched across ByteTrack ids
# --------------------------------------------------------------------------- #
#
# ByteTrack fragments the climber into several track ids over a boulder problem
# (scale/pose changes and occlusion break the track), and no single id is both
# near the base tap and present for the whole ascent. So we don't trust one id:
# seed from the tap, then follow the climber frame-to-frame by spatial nearest-box
# association, in both directions from the seed, stitching across id switches.

# Max normalized center displacement to associate the same climber between frames,
# with a per-elapsed-frame slack so the trajectory survives short detection gaps.
_ASSOC_BASE = 0.08
_ASSOC_PER_FRAME = 0.04


def _center_dist(a: Box, b: Box) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _nearest_box(boxes: Sequence[Box], ref: Box, max_dist: float) -> Box | None:
    best: Box | None = None
    best_d = max_dist
    for box in boxes:
        d = _center_dist(box, ref)
        if d <= best_d:
            best_d, best = d, box
    return best


def _seed_climber(
    history: Sequence[FrameTracks],
    climber_point: Point | None,
    climber_crop: Box | None,
) -> tuple[int | None, Box | None]:
    """Find the (frame_index, box) to start stitching the Climber trajectory from.

    With a tap: the box that contains it (nearest such), else the nearest box center
    over the clip. Without a tap: the first box of the most prominent persistent
    track inside the crop.
    """
    if climber_point is not None:
        contains: list[tuple[float, int, Box]] = []
        nearest: tuple[float, int, Box] | None = None
        for i, frame in enumerate(history):
            for box in frame.boxes.values():
                d = ((box.cx - climber_point.x) ** 2 + (box.cy - climber_point.y) ** 2) ** 0.5
                if box.contains(climber_point):
                    contains.append((d, i, box))
                if nearest is None or d < nearest[0]:
                    nearest = (d, i, box)
        if contains:
            _, idx, box = min(contains, key=lambda t: t[0])
            return idx, box
        return (nearest[1], nearest[2]) if nearest else (None, None)

    track_id = _largest_track(history, climber_crop)
    if track_id is None:
        return None, None
    for i, frame in enumerate(history):
        if track_id in frame.boxes:
            return i, frame.boxes[track_id]
    return None, None


def build_climber_track(
    history: Sequence[FrameTracks],
    climber_point: Point | None,
    climber_crop: Box | None,
) -> dict[int, Box]:
    """Per-frame Climber box (frame_index -> box), stitched across id switches.

    Empty when no person was tracked. Frames where the climber can't be associated
    are simply absent from the map (they'll be seeded ``absent`` downstream).
    """
    if not any(f.boxes for f in history):
        return {}

    seed_idx, seed_box = _seed_climber(history, climber_point, climber_crop)
    if seed_idx is None or seed_box is None:
        return {}

    trajectory: dict[int, Box] = {seed_idx: seed_box}

    # Walk forward, then backward, from the seed; the association threshold widens
    # with the gap since the last hit so a few missed detections don't sever the track.
    for direction in (1, -1):
        prev, last = seed_box, seed_idx
        i = seed_idx + direction
        while 0 <= i < len(history):
            threshold = _ASSOC_BASE + _ASSOC_PER_FRAME * abs(i - last)
            box = _nearest_box(list(history[i].boxes.values()), prev, threshold)
            if box is not None:
                trajectory[i] = box
                prev, last = box, i
            i += direction

    return trajectory


def _largest_track(history: Sequence[FrameTracks], crop: Box | None) -> int | None:
    """The track with the greatest summed box area over the clip (inside ``crop``).

    Summed — not average — area rewards both size and persistence, so a brief
    close-up of a passerby can't outrank the climber who is smaller but on screen
    for most of the send.
    """
    areas: dict[int, float] = {}
    for frame in history:
        for track_id, box in frame.boxes.items():
            if crop is not None and not crop.contains(Point(box.cx, box.cy)):
                continue
            areas[track_id] = areas.get(track_id, 0.0) + box.area

    # If the crop filtered everyone out, fall back to the largest track anywhere.
    if not areas and crop is not None:
        for frame in history:
            for track_id, box in frame.boxes.items():
                areas[track_id] = areas.get(track_id, 0.0) + box.area

    if not areas:
        return None
    return max(areas, key=lambda tid: areas[tid])


def _nearest_frame_index(history: Sequence[FrameTracks], timestamp: float) -> int | None:
    if not history:
        return None
    best_i, best_dist = 0, float("inf")
    for i, frame in enumerate(history):
        dist = abs(frame.timestamp - timestamp)
        if dist < best_dist:
            best_dist, best_i = dist, i
    return best_i


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifact(
    request: VitPoseRequest,
    history: Sequence[FrameTracks],
    pose_backend: PoseBackend,
    video_path: Path,
) -> dict:
    """Assemble the vitpose.json payload: one echoed frame per requested timestamp.

    A requested frame gets non-empty ``keypoints`` only when the stitched Climber
    trajectory has a box on the nearest decoded frame; otherwise ``keypoints: []``
    (seeded ``absent``).
    """
    trajectory = build_climber_track(history, request.climber_point, request.climber_crop)

    # Map each requested timestamp to the nearest decoded frame that has the Climber,
    # collecting the unique (frame_index, box) targets to pose.
    ts_to_index: dict[float, int] = {}
    targets: dict[int, PoseTarget] = {}
    for ts in request.frames:
        idx = _nearest_frame_index(history, ts)
        if idx is None:
            continue
        box = trajectory.get(idx)
        if box is None:
            continue
        ts_to_index[ts] = idx
        targets.setdefault(idx, PoseTarget(frame_index=idx, box=box))

    posed: dict[int, list[Keypoint]] = (
        pose_backend.pose(video_path, list(targets.values())) if targets else {}
    )

    frames_out = []
    for ts in request.frames:
        keypoints: list[dict] = []
        idx = ts_to_index.get(ts)
        if idx is not None:
            for kp in posed.get(idx, []):
                keypoints.append({
                    "name": kp.name,
                    "x": _clamp01(float(kp.x)),
                    "y": _clamp01(float(kp.y)),
                    "score": _clamp01(float(kp.score)),
                })
        # Echo the requested timestamp verbatim — never the decoder's frame time.
        frames_out.append({"timestamp": ts, "keypoints": keypoints})

    artifact: dict = {"version": ARTIFACT_VERSION}
    # Stamp the provenance anchor when known. Omitted (not null) when absent, so
    # consumers' `artifact.get("setupHash", <fallback>)` reaches the legacy fallback.
    if request.setup_hash:
        artifact["setupHash"] = request.setup_hash
    artifact["frames"] = frames_out
    return artifact


def _read_setup_hash(bundle_dir: Path) -> str | None:
    """Return the bundle's current ``setup.json`` setupHash, or ``None`` if unreadable.

    The writer's fallback for a request that omits ``setup_hash``: stamp the setup the
    bundle is currently calibrated under. Mirrors the consumer-side fallback documented
    in the artifact contract, so a freshly written artifact never lacks the field.
    """
    setup_path = bundle_dir / SETUP_NAME
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = setup.get("setupHash")
    return value if isinstance(value, str) and value else None


def _log(message: str) -> None:
    # The job runs on a background thread that is otherwise silent; a line to stderr
    # (which uvicorn surfaces) lets a manual run see it start and finish.
    print(f"[vitpose] {message}", file=sys.stderr, flush=True)


def _write_status(bundle_dir: Path, job_id: str, status: str, *, error: str | None = None) -> None:
    payload = {
        "jobId": job_id,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if error is not None:
        payload["error"] = error
    (bundle_dir / STATUS_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_vitpose_job(
    analysis_root: Path,
    request: VitPoseRequest,
    tracker: Tracker,
    pose_backend: PoseBackend,
    job_id: str | None = None,
) -> Path:
    """Run the full job and write ``vitpose.json`` (+ a status sidecar) to the bundle.

    Returns the artifact path. On any failure the status sidecar records ``error`` so
    beta-scanner — which only polls the filesystem — can distinguish a crashed job
    from one still running instead of polling forever.
    """
    job_id = job_id or uuid.uuid4().hex
    bundle_dir = bundle_dir_for(analysis_root, request.route_folder, request.video_key)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"No bundle at route={request.route_folder!r} video_key={request.video_key!r}."
        )

    _write_status(bundle_dir, job_id, "running")
    _log(
        f"job {job_id[:8]} started: {request.route_folder}/{request.video_key} "
        f"({len(request.frames)} frames requested)"
    )
    try:
        video_path = resolve_video_path(analysis_root, request.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Stamp the setup the job ran under. Prefer the hash the request carries; fall
        # back to the bundle's current setup.json so the artifact always names a setup.
        if not request.setup_hash:
            request = replace(request, setup_hash=_read_setup_hash(bundle_dir))

        history = tracker.track(video_path)
        artifact = build_artifact(request, history, pose_backend, video_path)

        artifact_path = bundle_dir / ARTIFACT_NAME
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_status(bundle_dir, job_id, "done")
        posed = sum(1 for f in artifact["frames"] if f["keypoints"])
        _log(
            f"job {job_id[:8]} done: {posed}/{len(artifact['frames'])} frames posed "
            f"-> {artifact_path}"
        )
        return artifact_path
    except Exception as exc:  # noqa: BLE001 — surface every failure via the sidecar
        _write_status(
            bundle_dir, job_id, "error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        _log(f"job {job_id[:8]} FAILED: {type(exc).__name__}: {exc}")
        raise


# --------------------------------------------------------------------------- #
# Default (real) backends — heavy imports are lazy so this module stays importable
# and testable without torch/transformers/ultralytics installed.
# --------------------------------------------------------------------------- #

class UltralyticsTracker:
    """Person detect + ByteTrack via ultralytics YOLO (lazy-loaded)."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy

            self._model = YOLO(self._model_name)
        return self._model

    def warm(self) -> None:
        """Load the model now so the first real request doesn't pay for it."""
        self._ensure_model()

    def track(self, video_path: Path) -> list[FrameTracks]:
        import cv2  # lazy

        model = self._ensure_model()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        history: list[FrameTracks] = []
        # classes=[0] -> person only; persist=True keeps ByteTrack ids across frames.
        results = model.track(
            source=str(video_path), classes=[0], persist=True,
            tracker="bytetrack.yaml", stream=True, verbose=False,
        )
        for idx, result in enumerate(results):
            boxes: dict[int, Box] = {}
            b = getattr(result, "boxes", None)
            if b is not None and b.id is not None:
                ids = b.id.int().tolist()
                # xywhn is center-based, normalized; convert to top-left corner.
                xywhn = b.xywhn.tolist()
                for track_id, (cx, cy, w, h) in zip(ids, xywhn):
                    boxes[int(track_id)] = Box(x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h)
            history.append(FrameTracks(timestamp=idx / fps, boxes=boxes))
        return history


def _require_scipy(scipy_available: bool) -> None:
    """Fail fast, with guidance, when scipy is missing.

    The transformers ViTPose image processor warps each person box (and refines the
    output heatmaps) via ``scipy.ndimage``; with scipy absent it dies deep inside the
    library with a cryptic ``NameError: name 'inv' is not defined`` — impossible to
    act on from the job's error sidecar. scipy is declared in ``requirements.txt`` for
    exactly this; surface an actionable message if the venv hasn't been synced.
    """
    if not scipy_available:
        raise ImportError(
            "ViTPose's image processor requires scipy (it warps person boxes and "
            "refines heatmaps via scipy.ndimage). Install it: "
            "pip install -r requirements.txt"
        )


class TransformersViTPoseBackend:
    """Top-down ViTPose++ via HuggingFace transformers (lazy-loaded)."""

    # vitpose-plus-* is a mixture-of-experts model: its forward needs a dataset_index
    # to pick an expert head. 0 = the COCO expert, which emits the 17 COCO joints.
    COCO_EXPERT_INDEX = 0

    def __init__(self, model_name: str = "usyd-community/vitpose-plus-base") -> None:
        self._model_name = model_name
        self._processor = None
        self._model = None
        self._device = "cpu"

    def _ensure_model(self):
        if self._model is None:
            import torch  # lazy
            from transformers import AutoProcessor, VitPoseForPoseEstimation  # lazy
            from transformers.utils import is_scipy_available  # lazy

            _require_scipy(is_scipy_available())

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            model = VitPoseForPoseEstimation.from_pretrained(self._model_name)
            self._model = model.to(self._device).eval()
        return self._processor, self._model

    def warm(self) -> None:
        """Load processor + model now so the first real request doesn't pay for it."""
        self._ensure_model()

    def pose(self, video_path: Path, targets: Sequence[PoseTarget]) -> dict[int, list[Keypoint]]:
        import cv2  # lazy
        import torch  # lazy

        if not targets:
            return {}
        processor, model = self._ensure_model()
        by_index = {t.frame_index: t for t in targets}

        cap = cv2.VideoCapture(str(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

        out: dict[int, list[Keypoint]] = {}
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            target = by_index.get(idx)
            if target is not None:
                out[idx] = self._pose_one(processor, model, torch, frame, target.box, width, height)
            idx += 1
        cap.release()
        return out

    def _pose_one(self, processor, model, torch, frame_bgr, box: Box, width: float, height: float):
        import cv2  # lazy

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # ViTPose processor wants COCO boxes in absolute pixels: [x, y, w, h].
        px_box = [[box.x * width, box.y * height, box.w * width, box.h * height]]
        inputs = processor(rgb, boxes=[px_box], return_tensors="pt").to(self._device)
        # One person box -> batch of 1; select the COCO expert for it.
        dataset_index = torch.tensor([self.COCO_EXPERT_INDEX], device=self._device)
        with torch.no_grad():
            outputs = model(**inputs, dataset_index=dataset_index)
        results = processor.post_process_pose_estimation(outputs, boxes=[px_box])[0][0]

        keypoints: list[Keypoint] = []
        for (x_px, y_px), score, label in zip(
            results["keypoints"].tolist(),
            results["scores"].tolist(),
            results["labels"].tolist(),
        ):
            label = int(label)
            if label >= len(COCO_KEYPOINT_NAMES):
                continue
            keypoints.append(Keypoint(
                name=COCO_KEYPOINT_NAMES[label],
                x=x_px / width, y=y_px / height, score=float(score),
            ))
        return keypoints


# Cached singletons: the models are expensive to load and re-init on CUDA, so we
# keep one resident instance each and reuse it across every job (this server runs
# locally for a single user — jobs are serialized by the caller's lock).
_TRACKER: Tracker | None = None
_POSE_BACKEND: PoseBackend | None = None


def default_tracker() -> Tracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = UltralyticsTracker()
    return _TRACKER


def default_pose_backend() -> PoseBackend:
    global _POSE_BACKEND
    if _POSE_BACKEND is None:
        _POSE_BACKEND = TransformersViTPoseBackend()
    return _POSE_BACKEND


def warm_backends() -> None:
    """Pre-load both models (call at startup so the first calibration is fast)."""
    b = default_tracker()
    p = default_pose_backend()
    if hasattr(b, "warm"):
        b.warm()
    if hasattr(p, "warm"):
        p.warm()
