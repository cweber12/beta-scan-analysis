"""MediaPipe detection runs produced by this harness (PRD #156).

The corpus this repo analyses was **observational**: every batch varied scanner build,
calibration, truth revision and schema at once, so a metric that moved could never be
attributed. This module makes detection an *experiment* the harness controls end to end —
one factor varies, everything else is held fixed, and the configuration that produced each
run is recorded in the run.

Three properties carry that, and each exists because its absence has already cost real
work here:

1. **Every run stamps its configuration**, in the same ``diagnostics`` block that already
   carries build identity, so experiment arms group through machinery that already exists.
2. **The configuration hash covers every factor that can change the output.** This is the
   issue #149 lesson moved to the detection side: a hash that omitted model identity meant
   a model swap re-used the cached artifact and measured as "no change" from a job that
   never ran. Here the same gap would let two arms share a stamp and pool as one.
3. **Repeats are a first-class parameter.** Issue #134 found the historical corpus holds
   six genuine repeat groups — and that 27 of 33 apparent ones were a single detection pass
   re-exported. Without deliberate repeats there is no variance floor, and without a floor
   no result is checkable against noise.

Runs are written through ``save_detection_run``, the same writer the scanner posts
through, so an experimental run is — to ``evaluate``, the #15 conformance gate, the tiers,
``trends`` and the report — just a run. Nothing downstream changes.

The heavy dependency is lazy-imported and confined to the backend class at the bottom, out
of the ``analysis_pipeline`` import graph, exactly as the ViTPose scaffold is (ADR 0003).
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

# Bumped whenever this module's own behaviour changes in a way that could move a result.
# It joins the configuration hash, so a module change is as detectable as a build change —
# the property the scanner side lacked until issue #130.
MODULE_VERSION = "1"

# What marks a run as produced here rather than posted by the scanner. Same artifact
# shape, different origin, and the origin is recorded: a pooled number must never blend
# browser-produced and harness-produced runs, because whether those are the same thing is
# precisely the open parity question.
ORIGIN = "harness-mediapipe"

# MediaPipe Pose exposes three model complexities. These are the first factor swept, with
# no preprocessing and no crop, to establish the baseline every later arm is measured
# against.
DETECTION_MODES = (0, 1, 2)

# Issue #134: a batch must produce its own variance floor rather than hope one exists.
DEFAULT_REPEATS = 3

CROP_NONE = "none"
CROP_ADAPTIVE = "adaptive"
CROP_POLICIES = (CROP_NONE, CROP_ADAPTIVE)


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float          # full-frame-normalized [0, 1]
    y: float
    score: float


@dataclass(frozen=True)
class PreprocessStep:
    """One individually addressable image transform applied before detection.

    Stamped separately rather than as a lumped "preprocessed: true" flag, so a combination
    is reconstructible from the artifact and a factorial arm is distinguishable from the
    one-factor arms it is built from.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(sorted(self.params.items()))}


@dataclass(frozen=True)
class DetectionConfig:
    """Everything about *how* a run was produced that could change what it produces.

    Anything that can move the output and is not in here is a silent confound, so the
    default is to include rather than to omit.
    """

    mode: int
    preprocess: tuple[PreprocessStep, ...] = ()
    crop: str = CROP_NONE
    module_version: str = MODULE_VERSION

    def identity(self) -> dict[str, Any]:
        """The canonical, order-stable description the hash is taken over.

        ``preprocess`` keeps its declared order — transforms do not commute, so
        contrast-then-brightness is a different arm from brightness-then-contrast and must
        not collapse to the same stamp.
        """
        return {
            "mode": self.mode,
            "preprocess": [s.identity() for s in self.preprocess],
            "crop": self.crop,
            "moduleVersion": self.module_version,
            "origin": ORIGIN,
        }


def config_hash(config: DetectionConfig) -> str:
    """Stable identity of one experiment arm.

    Two runs differing in any factor must not share this, and two runs differing in none
    must share it — that is what makes an arm groupable and a batch poolable.
    """
    blob = json.dumps(config.identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DetectRequest:
    """One batch: a Bundle, a timestamp grid, one arm, and how many passes to run."""

    video_path: str
    route_folder: str
    video_key: str
    frames: tuple[float, ...]            # requested timestamps, echoed verbatim
    config: DetectionConfig
    # Pairing provenance. Evaluation only compares a run against truth whose setupHash
    # matches, so a run written without it cannot be scored.
    setup_hash: str | None = None
    repeats: int = DEFAULT_REPEATS


class Detector(Protocol):
    def detect(
        self,
        video_path: Path,
        timestamps: Sequence[float],
        config: DetectionConfig,
    ) -> dict[float, list[Keypoint]]:
        """Detect the pose on each requested timestamp. Keyed by the timestamp as given."""
        ...


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_pose_payload(
    request: DetectRequest,
    detections: dict[float, list[Keypoint]],
    pass_index: int,
    now: str | None = None,
) -> dict[str, Any]:
    """The ``data`` blob of one pose artifact, in the shape the scanner's runs use.

    Timestamps are echoed verbatim and one frame is emitted per *requested* timestamp: a
    frame the detector found nothing on carries ``keypoints: []`` rather than being
    dropped, because a thinned artifact reads downstream as "the Climber was absent"
    instead of "the detector missed".
    """

    frames = []
    for timestamp in request.frames:
        found = detections.get(timestamp) or []
        frames.append({
            "timestamp": timestamp,
            "source": "detected" if found else "missing",
            "keypoints": [
                {"name": k.name, "x": k.x, "y": k.y, "score": k.score} for k in found
            ],
        })

    return {
        "diagnostics": {
            "schemaVersion": 1,
            "recordType": "scan",
            "createdAt": now or _now_iso(),
            # Build identity's neighbours, so the existing loader picks these up with no
            # change: `appVersion` names the module, and the experiment block below is what
            # separates one arm from another.
            "appVersion": f"{ORIGIN}@{request.config.module_version}",
            "origin": ORIGIN,
            "experiment": {
                "config": request.config.identity(),
                "configHash": config_hash(request.config),
                # Which pass of the repeat set this is. Repeats must be independent
                # detection passes, and this is what makes a duplicate export detectable
                # rather than counted as evidence of stability (#134).
                "passIndex": pass_index,
                "repeats": request.repeats,
            },
        },
        "setupHash": request.setup_hash,
        "frames": frames,
    }


def build_orb_payload(request: DetectRequest) -> dict[str, Any]:
    """The explicit *not computed* ORB half of the pair.

    ``save_detection_run`` writes a pose+ORB pair and the pipeline's stem-pairing treats
    both as an invariant, but this module has no ORB cross-match to compute. An explicit
    empty artifact keeps the invariant and makes ORB metrics read as *not computed* for
    experimental runs — as against fabricating one, or borrowing a different run's ORB
    half and silently attributing it to this pose.
    """

    return {
        "diagnostics": {
            "schemaVersion": 1,
            "recordType": "orb",
            "createdAt": _now_iso(),
            "origin": ORIGIN,
        },
        "setupHash": request.setup_hash,
        "notComputed": True,
        "notComputedReason": "harness-experimental-run: no ORB cross-match is performed",
        "matches": [],
    }


def pass_requests(request: DetectRequest) -> list[DetectRequest]:
    """One request per repeat pass. Each must be an independent detection pass."""
    if request.repeats < 1:
        raise ValueError("repeats must be at least 1")
    return [replace(request, repeats=request.repeats) for _ in range(request.repeats)]


# --------------------------------------------------------------------------- #
# Backend seam — the heavy dependency lives behind this and nowhere else
# --------------------------------------------------------------------------- #

class MediaPipeDetector:
    """MediaPipe Pose over a video's requested timestamps (lazy-loaded).

    Deliberately thin: it decodes, applies the arm's preprocessing, and returns keypoints.
    Every decision about *what* to run lives in the config above, so the arm is described
    by data rather than by which code path executed.
    """

    def __init__(self, mode: int = 1) -> None:
        self._mode = mode
        self._pose = None
        self._load_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"mediapipe-pose:complexity={self._mode}"

    def _ensure_model(self):
        with self._load_lock:
            if self._pose is None:
                import mediapipe as mp  # lazy — never imported by analysis_pipeline

                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=True, model_complexity=self._mode
                )
        return self._pose
