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
of the ``analysis_pipeline`` import graph, exactly as the ViTPose scaffold is — a second
heavyweight exception, recorded in ADR 0012 rather than left to contradict ADR 0003.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from vitpose_job import bundle_dir_for, resolve_video_path
from youtube_core import generate_timestamp, save_detection_run

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

# The status sidecar, on the ``vitpose.status.json`` model (ADR 0003): a job that dies
# after its caller has been handed control is otherwise indistinguishable from one still
# working, and the caller polls forever.
STATUS_NAME = "mediapipe.status.json"
SETUP_NAME = "setup.json"

# Where each requested timestamp's frame came from. ``detected`` and ``missing`` are both
# statements about the *detector*: the frame decoded, and a pose was or was not found.
# ``undecodable`` is a statement about the *decoder*, and is kept separate on purpose —
# folding a frame the video could not produce into ``missing`` would report a container
# problem as a detection failure, which is the single easiest way to manufacture a
# detector "regression" that never happened.
FRAME_DETECTED = "detected"
FRAME_MISSING = "missing"
FRAME_UNDECODABLE = "undecodable"

# MediaPipe's three Pose Landmarker model bundles, indexed by the mode swept first. The
# mapping is 1:1, so ``mode`` alone identifies the weights *given this module version* —
# and that qualifier is the whole point: **editing this mapping is a module change and
# must bump MODULE_VERSION**, or two arms built from different weights would share a
# configuration stamp and pool as one (issue #149, moved to the detection side).
MODEL_BUNDLES = {0: "pose_landmarker_lite", 1: "pose_landmarker_full", 2: "pose_landmarker_heavy"}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{name}/float16/latest/{name}.task"
)
# Cached beside the other model downloads rather than in the repo: the bundles are 6–30 MB
# binaries, and ``analysis/`` is a data record, not a model store. Same first-run download
# the ViTPose and YOLO checkpoints already do.
MODEL_DIR_ENV = "BETA_SCAN_MEDIAPIPE_MODEL_DIR"


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
        """Detect the pose on each requested timestamp. Keyed by the timestamp as given.

        The contract carries three states in two levels, and the distinction is
        load-bearing:

        - **key present, non-empty list** — the frame decoded and a pose was found.
        - **key present, empty list** — the frame decoded and the detector found nobody.
          This is a *measurement*: it is what a detection miss looks like.
        - **key absent** — the decoder could not produce that frame at all. This is not a
          measurement, and the job records it separately rather than scoring it as a miss.

        Coordinates are normalized to the **full frame**, whatever region was actually fed
        to the model, so a cropped arm's output is comparable to an uncropped one's.
        """
        ...


# Builds a detector for one pass. A *factory* rather than an instance because repeats have
# to be independent (#134 found 27 of 33 apparent historical repeats were one pass
# re-exported): a MediaPipe graph carries state, so re-using one instance across passes
# would let pass 0 warm pass 1 and the measured floor would be an artifact of the ordering.
DetectorFactory = Callable[[DetectionConfig], Detector]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp01(value: float) -> float:
    value = float(value)
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def build_pose_payload(
    request: DetectRequest,
    detections: dict[float, list[Keypoint]],
    pass_index: int,
    now: str | None = None,
    decoded: set[float] | None = None,
) -> dict[str, Any]:
    """The ``data`` blob of one pose artifact, in the shape the scanner's runs use.

    Timestamps are echoed verbatim and one frame is emitted per *requested* timestamp: a
    frame the detector found nothing on carries ``keypoints: []`` rather than being
    dropped, because a thinned artifact reads downstream as "the Climber was absent"
    instead of "the detector missed".

    ``decoded`` names the timestamps the decoder actually produced a frame for — normally
    ``set(detections)``, since a detector reports an empty list for a decoded frame it
    found nobody on. A requested timestamp outside it is marked ``undecodable`` rather than
    ``missing``: still echoed, still empty, but not counted as the detector failing at
    something it never got to see. Omit it and every empty frame reads as ``missing``,
    which is the pre-existing behaviour.
    """

    frames = []
    for timestamp in request.frames:
        found = detections.get(timestamp) or []
        if found:
            source = FRAME_DETECTED
        elif decoded is not None and timestamp not in decoded:
            source = FRAME_UNDECODABLE
        else:
            source = FRAME_MISSING
        frames.append({
            "timestamp": timestamp,
            "source": source,
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
# Bundle inputs — the grid, the pairing anchor, and the video binary
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def read_setup_hash(bundle_dir: Path) -> str | None:
    """The Bundle's current ``setup.json`` setupHash, or ``None``.

    Evaluation scores a run against truth only when their setupHashes match, so a run
    written without one is not merely unlabelled — it is unscoreable. The job falls back
    to this whenever the request omits it, so no experimental run is ever written into
    that state by accident.
    """

    value = _read_json(bundle_dir / SETUP_NAME).get("setupHash")
    return value if isinstance(value, str) and value else None


def truth_timestamps(bundle_dir: Path) -> tuple[float, ...]:
    """The Bundle's truth frame timestamps — the grid an experimental run must sample.

    Preference order mirrors ``evaluate.load_truth``: human-reviewed ``ground-truth.json``
    when present, else the ``vitpose.json`` scaffold. Running any other grid produces a run
    whose samples land between truth frames, and evaluation's nearest-frame join then
    scores a fraction of what was computed — expensive, and silently so.
    """

    for name in ("ground-truth.json", "vitpose.json"):
        frames = _read_json(bundle_dir / name).get("frames")
        if isinstance(frames, list) and frames:
            return tuple(
                float(f.get("timestamp", 0.0)) for f in frames if isinstance(f, dict)
            )
    return ()


def resolve_bundle_video(bundle_dir: Path) -> Path | None:
    """The Bundle's video binary: whatever ``metadata.json`` recorded, else a lone file."""

    recorded = _read_json(bundle_dir / "metadata.json").get("source_video_path")
    if isinstance(recorded, str) and recorded:
        candidate = Path(recorded)
        if candidate.is_file():
            return candidate
    videos = sorted(
        p for p in bundle_dir.glob("*")
        if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")
    )
    return videos[0] if videos else None


# --------------------------------------------------------------------------- #
# Orchestration — N independent passes into the existing detection-run writer
# --------------------------------------------------------------------------- #

def _log(message: str) -> None:
    print(f"[mediapipe] {message}", file=sys.stderr, flush=True)


def write_status(bundle_dir: Path, payload: dict[str, Any]) -> Path:
    """Write the status sidecar. Extra keys are additive; ``status`` is the contract."""

    path = bundle_dir / STATUS_NAME
    body = dict(payload)
    body["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def pass_run_ts(base_ts: str, config: DetectionConfig, pass_index: int) -> str:
    """The ``run_ts`` one pass is written under — **unique per pass, by construction**.

    ``save_detection_run`` bumps a colliding *filename* stem but leaves the ``run_ts``
    inside both envelopes alone, and evaluation names its record ``<run_ts>_vs_<truth>``.
    So three passes written inside the same wall-clock second under the writer's default
    would produce three pose files and then **one** evaluation record, each overwriting the
    last — three independent passes silently collapsing into a single scored run.

    That is #134's finding wearing a different hat: apparent repeats that are really one
    measurement. The arm and the pass index go into the id so it cannot happen, and so a
    ``detections/`` listing says which arm each run belongs to without opening it.
    """

    return f"{base_ts}-{config_hash(config)[:8]}-p{pass_index}"


def _unique_run_ts(detections_dir: Path, candidate: str) -> str:
    """Belt to ``pass_run_ts``'s braces: never reuse an id already on disk.

    Only reachable when the same arm is re-run inside one second, which no real pass is
    fast enough to do — but the failure it prevents is silent, and the check is two stats.
    """

    run_ts, counter = candidate, 1
    while (detections_dir / f"{run_ts}_pose.json").exists():
        run_ts = f"{candidate}-{counter}"
        counter += 1
    return run_ts


def run_mediapipe_job(
    analysis_root: Path,
    request: DetectRequest,
    detector_factory: DetectorFactory,
    job_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run ``request.repeats`` independent detection passes over one Bundle.

    Each pass gets a **freshly built detector** and its own decode-and-infer sweep, then is
    written through ``save_detection_run`` as an ordinary pose+ORB pair. Returns one
    summary dict per written run.

    Progress and the terminal state land in the status sidecar: ``running`` while passes
    are in flight, then ``done``, or ``error`` carrying the exception type and traceback so
    a failure is diagnosable without paying for the run again. The exception is re-raised
    after it is recorded — the sidecar is for the observer, not a way to swallow failures.
    """

    job_id = job_id or uuid.uuid4().hex
    bundle_dir = bundle_dir_for(analysis_root, request.route_folder, request.video_key)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"No bundle at route={request.route_folder!r} video_key={request.video_key!r}."
        )

    # Stamp the pairing anchor before anything runs. A run written without a setupHash is
    # unscoreable, and the cost of discovering that is a whole batch.
    if not request.setup_hash:
        request = replace(request, setup_hash=read_setup_hash(bundle_dir))

    base = {
        "jobId": job_id,
        "route": request.route_folder,
        "videoKey": request.video_key,
        "moduleVersion": request.config.module_version,
        "origin": ORIGIN,
        "config": request.config.identity(),
        "configHash": config_hash(request.config),
        "setupHash": request.setup_hash,
        "repeats": request.repeats,
        "requestedFrames": len(request.frames),
    }
    written: list[dict[str, Any]] = []
    write_status(bundle_dir, {**base, "status": "running", "runs": written})
    _log(
        f"job {job_id[:8]} started: {request.route_folder}/{request.video_key} "
        f"arm {base['configHash'][:8]} ({request.repeats} passes x "
        f"{len(request.frames)} frames)"
    )

    started = time.perf_counter()
    try:
        if not request.setup_hash:
            raise ValueError(
                f"Bundle {request.route_folder}/{request.video_key} has no setupHash "
                "(setup.json missing or uncalibrated); a run written without one can "
                "never be paired with truth, so the passes would be unscoreable."
            )
        video_path = resolve_video_path(analysis_root, request.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not request.frames:
            raise ValueError("No timestamps requested; there is nothing to detect.")

        base_ts = generate_timestamp()
        detections_dir = bundle_dir / "detections"
        for pass_index, pass_request in enumerate(pass_requests(request)):
            pass_started = time.perf_counter()
            # A new detector per pass. Re-using one would make pass N a continuation of
            # pass N-1 rather than a repeat of it, and the "variance floor" so measured
            # would describe the ordering, not the detector.
            detector = detector_factory(pass_request.config)
            detections = detector.detect(
                video_path, pass_request.frames, pass_request.config
            )
            decoded = set(detections)
            pose = build_pose_payload(
                pass_request, detections, pass_index=pass_index, decoded=decoded
            )
            run_ts = _unique_run_ts(
                detections_dir, pass_run_ts(base_ts, pass_request.config, pass_index)
            )
            result = save_detection_run(
                analysis_root,
                pass_request.route_folder,
                pass_request.video_key,
                pose,
                build_orb_payload(pass_request),
                run_ts=run_ts,
            )
            detected = sum(1 for f in pose["frames"] if f["source"] == FRAME_DETECTED)
            undecodable = len(pass_request.frames) - len(decoded)
            written.append({
                "passIndex": pass_index,
                "runTs": result["run_ts"],
                "posePath": result["pose_path"],
                "orbPath": result["orb_path"],
                "framesDetected": detected,
                "framesUndecodable": undecodable,
                "seconds": round(time.perf_counter() - pass_started, 2),
            })
            # Re-written after every pass: a batch that dies on pass 3 of 5 must still say
            # which two runs reached disk, or the corpus holds runs nothing accounts for.
            write_status(bundle_dir, {**base, "status": "running", "runs": written})
            _log(
                f"job {job_id[:8]} pass {pass_index + 1}/{request.repeats}: "
                f"{detected}/{len(pass_request.frames)} frames detected"
                + (f", {undecodable} undecodable" if undecodable else "")
                + f" -> {result['run_ts']}"
            )

        write_status(bundle_dir, {
            **base,
            "status": "done",
            "runs": written,
            "timings": {"total_s": round(time.perf_counter() - started, 2)},
        })
        _log(f"job {job_id[:8]} done: {len(written)} runs written")
        return written
    except Exception as exc:  # noqa: BLE001 — every failure is surfaced via the sidecar
        write_status(bundle_dir, {
            **base,
            "status": "error",
            "runs": written,
            "errorType": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "passesWritten": len(written),
        })
        _log(f"job {job_id[:8]} FAILED after {len(written)} runs: {type(exc).__name__}: {exc}")
        raise


# --------------------------------------------------------------------------- #
# Backend seam — the heavy dependency lives behind this and nowhere else
# --------------------------------------------------------------------------- #

def model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or (Path.home() / ".cache" / "beta-scan-mediapipe"))


def ensure_model_bundle(mode: int) -> Path:
    """The ``.task`` bundle for one mode, downloaded on first use and cached.

    Same first-run download the ViTPose checkpoint and the YOLO weights already do; the
    bundles are versioned by MediaPipe at a ``latest`` URL, which is why the *module*
    version — not the URL — is what a reader can pin an arm to.
    """

    name = MODEL_BUNDLES.get(mode)
    if name is None:
        raise ValueError(f"Unknown detection mode {mode!r}; expected one of {DETECTION_MODES}.")
    target = model_dir() / f"{name}.task"
    if target.is_file() and target.stat().st_size > 0:
        return target
    from urllib.request import urlopen  # lazy — only the download path needs it

    target.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(name=name)
    _log(f"downloading {name} -> {target}")
    partial = target.with_suffix(".task.part")
    with urlopen(url, timeout=300) as response, partial.open("wb") as handle:
        handle.write(response.read())
    partial.replace(target)
    return target


class MediaPipeDetector:
    """MediaPipe Pose Landmarker over a video's requested timestamps (lazy-loaded).

    Deliberately thin: it decodes, runs the model, and returns keypoints. Every decision
    about *what* to run lives in the config, so an arm is described by data rather than by
    which code path executed — which is what lets two arms be told apart on disk.

    **It refuses a configuration it cannot honour.** Preprocessing steps and crop policies
    are part of the arm identity as of the core slice, but nothing implements them yet.
    Running them as a silent no-op would produce a run stamped "contrast, factor 1.5" whose
    pixels never saw contrast — two arms differing in their stamps and identical in their
    output, read as "preprocessing had no effect". That is a fabricated null result, and it
    is worse than a crash.
    """

    def __init__(self, mode: int = 1) -> None:
        if mode not in MODEL_BUNDLES:
            raise ValueError(f"Unknown detection mode {mode!r}; expected one of {DETECTION_MODES}.")
        self._mode = mode
        self._landmarker = None
        self._load_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"mediapipe-pose-landmarker:{MODEL_BUNDLES[self._mode]}"

    def _ensure_model(self):
        with self._load_lock:
            if self._landmarker is None:
                # Lazy — never imported by analysis_pipeline (ADR 0012).
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python import vision

                self._landmarker = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(ensure_model_bundle(self._mode))
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                    )
                )
        return self._landmarker

    def close(self) -> None:
        with self._load_lock:
            if self._landmarker is not None:
                self._landmarker.close()
                self._landmarker = None

    def _require_supported(self, config: DetectionConfig) -> None:
        if config.mode != self._mode:
            raise ValueError(
                f"Detector was built for mode {self._mode} but asked to run mode "
                f"{config.mode}; the run would carry the wrong arm's stamp."
            )
        if config.preprocess:
            names = ", ".join(step.name for step in config.preprocess)
            raise NotImplementedError(
                f"Preprocessing steps ({names}) are stamped into the arm identity but not "
                "implemented yet (PRD #156 lands them one at a time after the mode sweep). "
                "Refusing rather than running them as a no-op, which would report a "
                "measured null for a transform that never ran."
            )
        if config.crop != CROP_NONE:
            raise NotImplementedError(
                f"Crop policy {config.crop!r} is stamped into the arm identity but not "
                f"implemented yet; only {CROP_NONE!r} runs today."
            )

    def detect(
        self,
        video_path: Path,
        timestamps: Sequence[float],
        config: DetectionConfig,
    ) -> dict[float, list[Keypoint]]:
        """Decode the requested timestamps and pose each one. See ``Detector.detect``."""

        import cv2  # lazy — a core dep, but this module is never imported for its purity

        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._require_supported(config)
        landmarker = self._ensure_model()
        names = [landmark.name.lower() for landmark in vision.PoseLandmark]

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for decoding: {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0.0:
                raise RuntimeError(f"Video reports no frame rate, cannot seek: {video_path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            out: dict[float, list[Keypoint]] = {}
            # Sorted so every seek is forward-only, which is both faster and — on long-GOP
            # containers — more reliable. Keys stay the timestamps as given.
            for timestamp in sorted(set(float(t) for t in timestamps)):
                frame_no = int(round(timestamp * fps))
                if frame_count > 0 and frame_no >= frame_count:
                    # Past the last frame: genuinely undecodable, not a detector miss.
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                )
                result = landmarker.detect(image)
                poses = result.pose_landmarks or []
                # A decoded frame always gets a key — an empty list here means "the model
                # saw nobody", which is a measurement and must not look like a decode gap.
                out[timestamp] = [
                    Keypoint(
                        name=names[i] if i < len(names) else f"landmark_{i}",
                        x=_clamp01(landmark.x),
                        y=_clamp01(landmark.y),
                        score=_clamp01(landmark.visibility),
                    )
                    for i, landmark in enumerate(poses[0])
                ] if poses else []
            return out
        finally:
            capture.release()


def default_detector_factory(config: DetectionConfig) -> Detector:
    """Build a fresh MediaPipe detector for one pass (never a shared singleton)."""

    return MediaPipeDetector(mode=config.mode)


# --------------------------------------------------------------------------- #
# CLI — one Bundle, one arm, N passes
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe detection passes over one analysis Bundle."
    )
    parser.add_argument("route_folder")
    parser.add_argument("video_key")
    parser.add_argument("--analysis-root", default="analysis", type=Path)
    parser.add_argument("--mode", type=int, default=1, choices=list(DETECTION_MODES),
                        help="MediaPipe model bundle: 0 lite, 1 full, 2 heavy.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample only the first N truth timestamps (smoke runs).")
    args = parser.parse_args(argv)

    analysis_root = args.analysis_root.resolve()
    bundle_dir = bundle_dir_for(analysis_root, args.route_folder, args.video_key)
    video = resolve_bundle_video(bundle_dir)
    if video is None:
        parser.error(f"No video binary in {bundle_dir}")
    frames = truth_timestamps(bundle_dir)
    if not frames:
        parser.error(f"No truth artifact to take a timestamp grid from in {bundle_dir}")
    if args.limit is not None:
        frames = frames[:args.limit]

    runs = run_mediapipe_job(
        analysis_root,
        DetectRequest(
            video_path=str(video),
            route_folder=args.route_folder,
            video_key=args.video_key,
            frames=frames,
            config=DetectionConfig(mode=args.mode),
            repeats=args.repeats,
        ),
        default_detector_factory,
    )
    for run in runs:
        print(f"{run['runTs']}  {run['framesDetected']}/{len(frames)} detected  "
              f"{run['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
