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
import threading
import time
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
    """Tracks present on one decoded frame, keyed by track id -> full-frame box.

    ``frame_number`` is the frame's position in the *source* video. When the tracker
    decodes with a stride it differs from the entry's index in the history list;
    ``None`` (stub histories, legacy callers) means the two coincide.
    """

    timestamp: float
    boxes: dict[int, Box]
    frame_number: int | None = None


@dataclass(frozen=True)
class PoseTarget:
    """A frame the Climber must be posed on.

    ``frame_index`` keys the pose result back into the track history;
    ``frame_number`` is the source-video frame the backend must decode (they differ
    when the tracker ran with a stride; ``None`` means they coincide).
    """

    frame_index: int
    box: Box
    frame_number: int | None = None

    @property
    def source_frame(self) -> int:
        return self.frame_number if self.frame_number is not None else self.frame_index


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
# The slack counts *source-video* frames (FrameTracks.frame_number), so a strided
# tracker history gets proportionally more room per history step.
_ASSOC_BASE = 0.08
_ASSOC_PER_FRAME = 0.04


def _source_frame_no(history: Sequence[FrameTracks], i: int) -> int:
    fn = history[i].frame_number
    return fn if fn is not None else i


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
            gap = abs(_source_frame_no(history, i) - _source_frame_no(history, last))
            threshold = _ASSOC_BASE + _ASSOC_PER_FRAME * gap
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
        targets.setdefault(
            idx, PoseTarget(frame_index=idx, box=box, frame_number=history[idx].frame_number)
        )

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


def _write_status(
    bundle_dir: Path,
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    timings: dict[str, float] | None = None,
    device: str | None = None,
) -> None:
    # Extra keys are contract-safe: beta-scanner's sidecar reader only inspects
    # `status` and `error` (app/api/dev/corpus/vitpose/route.ts).
    payload: dict = {
        "jobId": job_id,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if error is not None:
        payload["error"] = error
    if timings is not None:
        payload["timings"] = timings
    if device is not None:
        payload["device"] = device
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

        started = time.perf_counter()
        history = tracker.track(video_path)
        track_s = time.perf_counter() - started

        posed_at = time.perf_counter()
        artifact = build_artifact(request, history, pose_backend, video_path)
        pose_s = time.perf_counter() - posed_at

        artifact_path = bundle_dir / ARTIFACT_NAME
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        timings = {
            "track_s": round(track_s, 2),
            "pose_s": round(pose_s, 2),
            "total_s": round(time.perf_counter() - started, 2),
        }
        device = getattr(tracker, "device", None) or getattr(pose_backend, "device", None)
        _write_status(bundle_dir, job_id, "done", timings=timings, device=device)
        posed = sum(1 for f in artifact["frames"] if f["keypoints"])
        _log(
            f"job {job_id[:8]} done: {posed}/{len(artifact['frames'])} frames posed "
            f"in {timings['total_s']}s (track {timings['track_s']}s, "
            f"pose {timings['pose_s']}s, device {device or 'unknown'}) -> {artifact_path}"
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
    """Person detect + ByteTrack via ultralytics YOLO (lazy-loaded).

    On CUDA the video is tracked at full frame rate; on CPU — where YOLO dominates
    the job's wall time — every ``CPU_VID_STRIDE``-th frame is tracked instead, and
    each history entry carries its source ``frame_number`` so downstream timestamp
    matching, association slack, and pose-frame seeking stay exact.
    """

    CPU_VID_STRIDE = 2

    def __init__(self, model_name: str = "yolov8n.pt", vid_stride: int | None = None) -> None:
        self._model_name = model_name
        self._vid_stride = vid_stride  # explicit override; None = pick by device
        self._model = None
        self._load_lock = threading.Lock()
        self.device: str | None = None  # "cuda" / "cpu", known after the model loads

    def _ensure_model(self):
        # Locked: the boot-time warm thread and an early first job otherwise race
        # and load a second copy of the model.
        with self._load_lock:
            if self._model is None:
                import torch  # lazy
                from ultralytics import YOLO  # lazy

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = YOLO(self._model_name)
        return self._model

    def warm(self) -> None:
        """Load the model now so the first real request doesn't pay for it."""
        self._ensure_model()

    def track(self, video_path: Path) -> list[FrameTracks]:
        import cv2  # lazy

        model = self._ensure_model()
        stride = self._vid_stride or (1 if self.device == "cuda" else self.CPU_VID_STRIDE)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        history: list[FrameTracks] = []
        # classes=[0] -> person only; persist=True keeps ByteTrack ids across frames.
        # half (fp16) only on CUDA — passing the kwarg at all on CPU trips an
        # ultralytics deprecation warning.
        fp16 = {"half": True} if self.device == "cuda" else {}
        results = model.track(
            source=str(video_path), classes=[0], persist=True,
            tracker="bytetrack.yaml", stream=True, verbose=False,
            device=0 if self.device == "cuda" else "cpu",
            vid_stride=stride, **fp16,
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
            frame_no = idx * stride
            history.append(
                FrameTracks(timestamp=frame_no / fps, boxes=boxes, frame_number=frame_no)
            )
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
    """Top-down ViTPose++ via HuggingFace transformers (lazy-loaded).

    Seeks directly to each target's source frame (instead of decoding the whole
    video a second time) and runs the forwards in batches of ``BATCH_SIZE``.
    """

    # vitpose-plus-* is a mixture-of-experts model: its forward needs a dataset_index
    # to pick an expert head. 0 = the COCO expert, which emits the 17 COCO joints.
    COCO_EXPERT_INDEX = 0
    BATCH_SIZE = 16

    def __init__(self, model_name: str = "usyd-community/vitpose-plus-base") -> None:
        self._model_name = model_name
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self.device = "cpu"

    def _ensure_model(self):
        # Locked: the boot-time warm thread and an early first job otherwise race
        # and load a second copy of the model.
        with self._load_lock:
            if self._model is None:
                import torch  # lazy
                from transformers import AutoProcessor, VitPoseForPoseEstimation  # lazy
                from transformers.utils import is_scipy_available  # lazy

                _require_scipy(is_scipy_available())

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._processor = AutoProcessor.from_pretrained(self._model_name)
                model = VitPoseForPoseEstimation.from_pretrained(self._model_name)
                self._model = model.to(self.device).eval()
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

        cap = cv2.VideoCapture(str(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

        # Grab exactly the target frames, seeking in source order so every seek is
        # forward-only. A frame the decoder can't produce is simply skipped (its
        # requested timestamps stay `keypoints: []`, same as an untracked Climber).
        grabbed: list[tuple[PoseTarget, object]] = []
        for target in sorted(targets, key=lambda t: t.source_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target.source_frame)
            ok, frame = cap.read()
            if ok:
                grabbed.append((target, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        out: dict[int, list[Keypoint]] = {}
        for start in range(0, len(grabbed), self.BATCH_SIZE):
            self._pose_batch(
                processor, model, torch,
                grabbed[start:start + self.BATCH_SIZE], width, height, out,
            )
        return out

    def _pose_batch(self, processor, model, torch, batch, width: float, height: float, out):
        """Pose one chunk: N frames, one Climber box each, one forward pass."""
        images = [rgb for _, rgb in batch]
        # ViTPose processor wants COCO boxes in absolute pixels: [x, y, w, h],
        # one list of person boxes per image (always exactly one — the Climber).
        boxes = [
            [[t.box.x * width, t.box.y * height, t.box.w * width, t.box.h * height]]
            for t, _ in batch
        ]
        inputs = processor(images, boxes=boxes, return_tensors="pt").to(self.device)
        # One box per image -> one expert selection per image.
        dataset_index = torch.tensor([self.COCO_EXPERT_INDEX] * len(batch), device=self.device)
        with torch.no_grad():
            outputs = model(**inputs, dataset_index=dataset_index)
        results = processor.post_process_pose_estimation(outputs, boxes=boxes)

        for (target, _), image_results in zip(batch, results):
            person = image_results[0]
            keypoints: list[Keypoint] = []
            for (x_px, y_px), score, label in zip(
                person["keypoints"].tolist(),
                person["scores"].tolist(),
                person["labels"].tolist(),
            ):
                label = int(label)
                if label >= len(COCO_KEYPOINT_NAMES):
                    continue
                keypoints.append(Keypoint(
                    name=COCO_KEYPOINT_NAMES[label],
                    x=x_px / width, y=y_px / height, score=float(score),
                ))
            out[target.frame_index] = keypoints


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
