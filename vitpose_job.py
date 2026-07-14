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
from dataclasses import dataclass
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
# Climber selection (pure geometry — the heart of "Climber Identity")
# --------------------------------------------------------------------------- #

def select_climber_track(
    history: Sequence[FrameTracks],
    climber_point: Point | None,
    climber_crop: Box | None,
) -> int | None:
    """Pick the track id that is the Climber; None when no person was tracked.

    With a tap: the track whose box contains the tap on the **most** frames wins
    (persistence beats a fleeting spurious detection that briefly covered the tap);
    ties break toward the nearest approach. If no track ever contains the tap, the
    track whose box center comes nearest the tap over the clip. Without a tap: the
    track that is largest on average within ``climber_crop`` (spotters/passersby are
    smaller / off to the side), else the largest overall.
    """
    if not any(f.boxes for f in history):
        return None

    if climber_point is not None:
        contain_counts: dict[int, int] = {}
        nearest: dict[int, float] = {}
        for frame in history:
            for track_id, box in frame.boxes.items():
                if box.contains(climber_point):
                    contain_counts[track_id] = contain_counts.get(track_id, 0) + 1
                dist = (box.cx - climber_point.x) ** 2 + (box.cy - climber_point.y) ** 2
                if track_id not in nearest or dist < nearest[track_id]:
                    nearest[track_id] = dist
        if contain_counts:
            # Most-persistent containment; tie-break toward the nearest approach.
            return max(contain_counts, key=lambda t: (contain_counts[t], -nearest[t]))
        # No box ever contained the tap: nearest box center across the whole clip.
        return min(nearest, key=lambda t: nearest[t])

    # No tap: prefer the most prominent *persistent* track inside the climber crop.
    return _largest_track(history, climber_crop)


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

    A requested frame gets non-empty ``keypoints`` only when the Climber track has a
    box on the nearest decoded frame; otherwise ``keypoints: []`` (seeded ``absent``).
    """
    climber_id = select_climber_track(history, request.climber_point, request.climber_crop)

    # Map each requested timestamp to the nearest decoded frame that actually has the
    # Climber, collecting the unique (frame_index, box) targets to pose.
    ts_to_index: dict[float, int] = {}
    targets: dict[int, PoseTarget] = {}
    for ts in request.frames:
        idx = _nearest_frame_index(history, ts)
        if idx is None or climber_id is None:
            continue
        box = history[idx].boxes.get(climber_id)
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

    return {"version": ARTIFACT_VERSION, "frames": frames_out}


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

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            model = VitPoseForPoseEstimation.from_pretrained(self._model_name)
            self._model = model.to(self._device).eval()
        return self._processor, self._model

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


def default_tracker() -> Tracker:
    return UltralyticsTracker()


def default_pose_backend() -> PoseBackend:
    return TransformersViTPoseBackend()
