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
import math
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

import crop_track
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

# Repeats default to **one**, and that is a reversal of the original PRD decision.
#
# #134 mandated three, because the historical corpus holds six genuine repeat groups and
# therefore no usable variance floor. But #134 measured the *scanner's* detector, where
# re-running the same video on the same build genuinely scatters (PCK 0.0055 median). This
# detector is bit-deterministic — three passes over the same frames produce byte-identical
# output, confirmed across all three modes, two videos and separate processes. Its floor is
# not small, it is exactly 0, so repeats produce provably zero information at N× the cost:
# roughly 47 of the 70.6 hours in a full three-mode sweep would have been duplicate bytes.
#
# Drift detection moves to the byte-identical canary (#168), which is strictly more
# sensitive and costs about two minutes. Repeats stay a *parameter* — a caller may still
# ask for more, and must be able to the moment the detector stops being deterministic (a
# GPU delegate, a threading change, a MediaPipe upgrade). Only the default changed.
DEFAULT_REPEATS = 1

# Frames sampled per run: ``keep = SAMPLE_COEFFICIENT · √n`` over the Bundle's truth grid.
#
# Video length spans 79× across this corpus (76 to 5,977 truth frames) while the Run is the
# unit of inference, so sampling the full grid spent 23% of all compute on 6% of the runs
# for no extra inferential weight. Measured across 55 Bundles, this rule sits at median
# 0.0017 / p90 0.0056 |ΔPCK| against the full-grid answer — a p90 essentially equal to
# #134's 0.0055 PCK noise floor, which is the stopping criterion: worst-case sampling error
# at or below noise the corpus already carries.
#
# A flat cap was rejected: its error concentrates on exactly the long videos, which are
# single continuous attempts with no repeated content to spare.
#
# **Changing this is a module change and must bump MODULE_VERSION.** The arm identity does
# not name the frame set — it does not need to, because the set is a deterministic function
# of the Bundle, which is what makes sampling error common-mode across arms and cancel in a
# delta. Change the coefficient and that stops being true: two runs on one Bundle would
# carry the same stamp over different frames. Observed for real when #169's full-grid proof
# run and the first sampled batch landed on one Bundle under one arm hash.
SAMPLE_COEFFICIENT = 12

# Experimental run ids carry this, so a ``detections/`` listing is self-describing, a
# selective wipe is a glob rather than a JSON scan, and every aggregation has a trivially
# correct segregation key.
RUN_ID_PREFIX = "exp-"

# A Bundle whose truth is this badly wrong-person is excluded from experiments (#34).
#
# Deliberately a threshold rather than "exclude all seven": ``evaluate`` already drops
# ``human-flagged-wrong`` frames from every tier's scoring (ADR 0004/0005), and dropping all
# seven Bundles would discard ~9,990 good truth frames to remove 2,113 bad ones — two of
# them are >98.7% clean, one being the corpus's largest Bundle.
#
# But not zero either, because wrong-*person* truth is the one error that does **not** cancel
# between arms. It points at a specific other human, so an arm that latches onto the same
# bystander the truth did gets *rewarded* — and identity confusion is exactly what varies
# between detection configs. Above this share the unflagged remainder is untrustworthy too.
WRONG_TRUTH_MAX_SHARE = 0.20

CROP_NONE = "none"
CROP_ADAPTIVE = "adaptive"
# Crop at the Bundle's tracked trajectory (issue #169). This is the policy that makes the
# corpus measurable at all: full-frame MediaPipe detects nothing on 24% of Bundles, where
# the scanner — which crops — reaches a median 86.9%.
CROP_TRACKED = "tracked"
CROP_POLICIES = (CROP_NONE, CROP_ADAPTIVE, CROP_TRACKED)

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
# mapping is 1:1, so ``mode`` names a bundle — but naming a bundle is not identifying
# weights, because upstream publishes at a ``latest`` URL and can republish under it. The
# bundle's sha256 is therefore pinned in ``models/mediapipe.lock.json`` and joins the arm
# identity (``DetectionConfig.model_sha``), so new weights cannot arrive under an
# unchanged stamp. Editing this *mapping* is still a module change and must bump
# MODULE_VERSION. See ADR 0012 and ``scripts/fetch_mediapipe_models.py``.
MODEL_BUNDLES = {0: "pose_landmarker_lite", 1: "pose_landmarker_full", 2: "pose_landmarker_heavy"}
# Repo-local and gitignored, with the lock file as the tracked record — the rule the repo
# already applies to video binaries, ``*.pt`` weights and ``downloads/``. Nothing is
# fetched at run time: a batch must not depend on a network round trip, and an arm must
# not depend on what upstream was serving that afternoon.
MODEL_DIR_ENV = "BETA_SCAN_MEDIAPIPE_MODEL_DIR"
MODEL_DIR_DEFAULT = Path(__file__).resolve().parent / "models" / "mediapipe"
MODEL_LOCK_PATH = Path(__file__).resolve().parent / "models" / "mediapipe.lock.json"
FETCH_HINT = "python scripts/fetch_mediapipe_models.py"


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
    # sha256 of the pinned model bundle that actually produced the run. **Derived, never
    # caller-supplied**: ``run_mediapipe_job`` reads it off the detector it is about to
    # build, so the stamp records what ran rather than what a caller claimed — the same
    # discipline ``vitpose_job.stamp_model_identity`` applies for issue #149.
    #
    # ``mode`` names a bundle, but the bundle's *contents* are what moves the output, and
    # upstream publishes at a ``latest`` URL. Without this field new weights could arrive
    # under an unchanged stamp and two arms would pool as one. ``None`` means "no model
    # identity was reported" (every stub), which keeps stub-backed hashes stable.
    model_sha: str | None = None
    # Identity of the crop trajectory this arm was cropped by (issue #169). **Derived, never
    # caller-supplied**: stamped by the job from the trajectory it actually loaded.
    #
    # ``crop`` names a *policy*; the trajectory is what decides which pixels the detector
    # saw. Two arms cropped by trajectories from different tracker settings are not
    # comparable, so without this they would share a stamp and pool — the same gap
    # ``model_sha`` closes for weights. ``None`` means no tracked crop was used.
    crop_track_hash: str | None = None

    def identity(self) -> dict[str, Any]:
        """The canonical, order-stable description the hash is taken over.

        ``preprocess`` keeps its declared order — transforms do not commute, so
        contrast-then-brightness is a different arm from brightness-then-contrast and must
        not collapse to the same stamp.
        """
        identity: dict[str, Any] = {
            "mode": self.mode,
            "preprocess": [s.identity() for s in self.preprocess],
            "crop": self.crop,
            "moduleVersion": self.module_version,
            "origin": ORIGIN,
        }
        # Omitted rather than null when absent, so a stub-backed hash is byte-identical to
        # what it was before model pinning existed and the pure-core tests stay meaningful.
        if self.model_sha is not None:
            identity["modelSha"] = self.model_sha
        if self.crop_track_hash is not None:
            identity["cropTrackHash"] = self.crop_track_hash
        return identity


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
        crop_track: Any | None = None,
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
                # How many timestamps this run actually sampled. Not part of the arm hash
                # — the frame set is a deterministic function of the Bundle — but recorded
                # so two runs of one arm on one Bundle can be *checked* to have sampled the
                # same frames rather than assumed to have.
                "frameCount": len(request.frames),
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


def sample_timestamps(frames: Sequence[float], coefficient: int = SAMPLE_COEFFICIENT
                      ) -> tuple[float, ...]:
    """The timestamps one run samples: ``coefficient · √n``, evenly spread over the grid.

    **A pure function of the grid**, deliberately. Mode batches run on different days, so a
    frame set chosen at batch time would hand the three modes three different frame sets and
    reintroduce across batches exactly the confound the design removes within one.

    **Never contiguous.** Eight 300-frame contiguous windows of one Bundle — same run, same
    truth — produced PCK from 0.104 to 0.839. That 0.735 spread is roughly 130× the noise
    floor and 15–70× any arm effect being hunted, so sampling a *stretch* would swamp the
    experiment with frame-choice noise wearing an arm's name.

    Spread by even spacing over the whole span rather than by a stride-and-truncate, which
    silently drops the tail of the video and reintroduces the same bias in miniature.
    """

    n = len(frames)
    keep = min(n, int(coefficient * math.sqrt(n))) if n else 0
    if keep <= 0:
        return ()
    if keep >= n:
        return tuple(float(t) for t in frames)
    if keep == 1:
        return (float(frames[0]),)
    return tuple(
        float(frames[round(i * (n - 1) / (keep - 1))]) for i in range(keep)
    )


def wrong_truth_share(bundle_dir: Path) -> float:
    """Fraction of a Bundle's truth frames flagged ``human-flagged-wrong`` (#34)."""

    frames = _read_json(bundle_dir / "ground-truth.json").get("frames")
    if not isinstance(frames, list) or not frames:
        return 0.0
    wrong = sum(1 for f in frames
                if isinstance(f, dict) and f.get("review") == "human-flagged-wrong")
    return wrong / len(frames)


@dataclass(frozen=True)
class BundleSelection:
    """Which Bundles a batch will run, and why the rest were left out.

    The exclusions are carried rather than silently applied: a batch that quietly skipped a
    Bundle would produce a pooled number over a population nobody can reconstruct.
    """

    included: tuple[tuple[str, str], ...] = ()
    excluded: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "included": [{"route": r, "videoKey": k} for r, k in self.included],
            "includedCount": len(self.included),
            "excluded": list(self.excluded),
            "excludedCount": len(self.excluded),
        }


def select_bundles(
    analysis_root: Path,
    only: Sequence[tuple[str, str]] | None = None,
) -> BundleSelection:
    """Every Bundle a batch can run, with each exclusion recorded and reasoned.

    ``only`` restricts to an explicit subset — the smoke-batch path — and is still filtered,
    so asking for a Bundle that cannot be run reports *why* rather than failing opaquely.
    """

    included: list[tuple[str, str]] = []
    excluded: list[dict[str, Any]] = []
    wanted = {(r, k) for r, k in only} if only is not None else None

    for bundle_dir in sorted(p for p in analysis_root.glob("*/*") if p.is_dir()):
        route, key = bundle_dir.parent.name, bundle_dir.name
        if wanted is not None and (route, key) not in wanted:
            continue

        def drop(reason: str, **extra: Any) -> None:
            excluded.append({"route": route, "videoKey": key, "reason": reason, **extra})

        if not truth_timestamps(bundle_dir):
            drop("no-truth")
            continue
        if resolve_bundle_video(bundle_dir) is None:
            drop("no-video")
            continue
        if not read_setup_hash(bundle_dir):
            drop("no-setup-hash")
            continue
        share = wrong_truth_share(bundle_dir)
        if share > WRONG_TRUTH_MAX_SHARE:
            drop("wrong-person-truth", wrongShare=round(share, 4))
            continue
        included.append((route, key))

    if wanted is not None:
        found = set(included) | {(e["route"], e["videoKey"]) for e in excluded}
        for route, key in sorted(wanted - found):
            excluded.append({"route": route, "videoKey": key, "reason": "no-bundle"})

    return BundleSelection(tuple(included), tuple(excluded))


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

    return f"{RUN_ID_PREFIX}{base_ts}-{config_hash(config)[:8]}-p{pass_index}"


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


def stamp_model_identity(
    request: DetectRequest, detector_factory: DetectorFactory
) -> DetectRequest:
    """Record which weights are about to run, read off the detector that will run them.

    The arm is a function of its model, so a model change must move the arm stamp —
    otherwise two arms pool as one and the batch measures a difference it cannot name.
    Identity comes from a *probe* detector the factory builds, never from the caller, so
    the stamp records what ran rather than what someone declared (issue #149's discipline,
    as ``vitpose_job.stamp_model_identity`` applies it).

    The probe is cheap: ``model_sha`` reads the lock file and loads no model. A detector
    with no ``model_sha`` — every stub in the test suite — reports nothing and the config
    is returned untouched, which is what keeps stub-backed hashes stable.

    Note for anyone writing a factory: this means the factory is called **once more than
    the repeat count**, and the probe never sweeps. Keep construction free of side effects
    and of model loading; ``MediaPipeDetector`` defers both to first use.
    """

    model_sha = getattr(detector_factory(request.config), "model_sha", None)
    if not model_sha:
        return request
    return replace(request, config=replace(request.config, model_sha=model_sha))


def resolve_arm(
    bundle_dir: Path, request: DetectRequest, detector_factory: DetectorFactory
) -> tuple[DetectRequest, Any | None]:
    """Resolve the arm a request will actually run under, and the trajectory it crops by.

    Three things a caller cannot supply and must not be trusted to: the pairing anchor, the
    model that will run, and the crop trajectory on disk. All three are read here, before
    any hash is taken, so the status sidecar, the run ids and every written stamp name the
    same arm — and so the arm names what ran rather than what was declared.

    Extracted so the drift canary (issue #168) resolves its arm through *this* path rather
    than a parallel one. A canary that resolved its own identity differently would be
    witnessing a slightly different job from the batch it is supposed to certify, which is
    the one thing a drift instrument may not do.
    """

    if not request.setup_hash:
        request = replace(request, setup_hash=read_setup_hash(bundle_dir))
    request = stamp_model_identity(request, detector_factory)
    track = crop_track.load_crop_track(bundle_dir) if request.config.crop == CROP_TRACKED \
        else None
    if track is not None:
        request = replace(request, config=replace(
            request.config, crop_track_hash=crop_track.track_hash(track.config)))
    return request, track


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

    # The pairing anchor, the model identity and the crop trajectory — all three resolved
    # before the arm hash is computed for anything, so the status sidecar, the run ids and
    # every written stamp name the same arm.
    request, track = resolve_arm(bundle_dir, request, detector_factory)

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
        if request.config.crop == CROP_TRACKED and track is None:
            raise FileNotFoundError(
                f"crop=tracked needs {crop_track.ARTIFACT_NAME} in "
                f"{request.route_folder}/{request.video_key}, and none is present. Build it "
                f"first — silently falling back to full frame would write runs stamped as a "
                f"cropped arm that never saw a crop."
            )

        base_ts = generate_timestamp()
        detections_dir = bundle_dir / "detections"
        for pass_index, pass_request in enumerate(pass_requests(request)):
            pass_started = time.perf_counter()
            # A new detector per pass. Re-using one would make pass N a continuation of
            # pass N-1 rather than a repeat of it, and the "variance floor" so measured
            # would describe the ordering, not the detector.
            detector = detector_factory(pass_request.config)
            detections = detector.detect(
                video_path, pass_request.frames, pass_request.config, crop_track=track
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
                # The *resolved* arm: model sha and crop trajectory are stamped inside this
                # function, so the config a caller handed in is not yet the arm that ran.
                "configHash": base["configHash"],
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
# Batch — one arm across many Bundles, single-flight, per-Bundle isolation
# --------------------------------------------------------------------------- #

BATCH_STATUS_NAME = "mediapipe-batch.status.json"

# Single-flight (PRD #156 user story 40). Two batches must never interleave writes into one
# Bundle: they would share a base timestamp, race on run ids, and produce a repeat set whose
# members came from different arms. Non-blocking on purpose — a caller that asked for a
# second batch wants to be *told no*, not silently queued behind an hour of GPU work.
_BATCH_LOCK = threading.Lock()


def batch_status_path(analysis_root: Path) -> Path:
    return analysis_root / BATCH_STATUS_NAME


def write_batch_status(analysis_root: Path, payload: dict[str, Any]) -> Path:
    path = batch_status_path(analysis_root)
    body = dict(payload)
    body["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def batch_is_running(analysis_root: Path) -> bool:
    return _BATCH_LOCK.locked()


def run_batch(
    analysis_root: Path,
    config: DetectionConfig,
    detector_factory: DetectorFactory,
    only: Sequence[tuple[str, str]] | None = None,
    repeats: int = DEFAULT_REPEATS,
    job_id: str | None = None,
    coefficient: int = SAMPLE_COEFFICIENT,
) -> dict[str, Any]:
    """Sweep one arm across many Bundles.

    Three operational properties, each because its absence costs a whole sweep:

    - **Single-flight.** Refuses to start while another batch holds the lock.
    - **Per-Bundle isolation.** A Bundle that raises is recorded with its error and skipped;
      the sweep continues and still reaches a terminal status. One bad video must not cost
      hours of work on the other eighty-five.
    - **Per-Bundle progress**, written to the batch sidecar as it goes, so a long sweep is
      observable while it runs rather than only at the end.

    Raises ``RuntimeError`` immediately if a batch is already in flight.
    """

    if not _BATCH_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "A MediaPipe batch is already running. Two batches would interleave writes "
            "into the same Bundles and produce repeat sets whose members came from "
            "different arms — refusing rather than queueing."
        )
    try:
        job_id = job_id or uuid.uuid4().hex
        selection = select_bundles(analysis_root, only)
        base = {
            "jobId": job_id,
            "status": "running",
            "origin": ORIGIN,
            "config": config.identity(),
            "configHash": config_hash(config),
            "repeats": repeats,
            "sampleCoefficient": coefficient,
            "selection": selection.as_dict(),
        }
        results: list[dict[str, Any]] = []
        write_batch_status(analysis_root, {**base, "bundles": results})
        _log(
            f"batch {job_id[:8]} started: arm {base['configHash'][:8]}, "
            f"{len(selection.included)} bundles, {repeats} repeat(s) each "
            f"({len(selection.excluded)} excluded)"
        )

        for index, (route, key) in enumerate(selection.included, start=1):
            bundle_dir = bundle_dir_for(analysis_root, route, key)
            entry: dict[str, Any] = {"route": route, "videoKey": key}
            try:
                video = resolve_bundle_video(bundle_dir)
                grid = truth_timestamps(bundle_dir)
                frames = sample_timestamps(grid, coefficient)
                entry["truthFrames"] = len(grid)
                entry["sampledFrames"] = len(frames)
                runs = run_mediapipe_job(
                    analysis_root,
                    DetectRequest(
                        video_path=str(video),
                        route_folder=route,
                        video_key=key,
                        frames=frames,
                        config=config,
                        repeats=repeats,
                    ),
                    detector_factory,
                )
                entry["status"] = "done"
                entry["runs"] = [r["runTs"] for r in runs]
                entry["framesDetected"] = [r["framesDetected"] for r in runs]
                entry["armsWritten"] = sorted({r["configHash"] for r in runs})
            except Exception as exc:  # noqa: BLE001 — one bad Bundle must not end the sweep
                entry["status"] = "error"
                entry["errorType"] = type(exc).__name__
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["traceback"] = traceback.format_exc()
                _log(f"batch {job_id[:8]} {route}/{key} FAILED: {entry['error']}")
            results.append(entry)
            write_batch_status(analysis_root, {**base, "bundles": results})
            _log(
                f"batch {job_id[:8]} [{index}/{len(selection.included)}] {route}/{key}: "
                f"{entry['status']}"
                + (f", {entry.get('sampledFrames')} frames sampled of "
                   f"{entry.get('truthFrames')}" if entry["status"] == "done" else "")
            )

        failed = [e for e in results if e["status"] == "error"]
        # The arms actually written, resolved. ``base["configHash"]`` is the *requested*
        # config, which is not yet an arm: the model sha and the crop trajectory are stamped
        # per Bundle inside the job. Reporting the request as if it were the arm would put a
        # hash in the batch record that appears on none of the runs it produced.
        arms = sorted({a for e in results for a in (e.get("armsWritten") or [])})
        final = {
            **base,
            # ``done`` even with per-Bundle failures: the *batch* completed, and the
            # failures are enumerated. ``error`` would imply nothing usable was produced.
            "status": "done",
            "requestedConfigHash": base["configHash"],
            "armsWritten": arms,
            # More than one arm in a batch means the Bundles did not share a trajectory or a
            # model — the runs are not one experimental condition and must not be pooled as
            # one. Surfaced here rather than discovered at analysis time.
            "armsMixed": len(arms) > 1,
            "bundles": results,
            "bundlesRun": len(results) - len(failed),
            "bundlesFailed": len(failed),
            "runsWritten": sum(len(e.get("runs") or []) for e in results),
        }
        if final["armsMixed"]:
            _log(f"batch {job_id[:8]} WARNING: {len(arms)} distinct arms written {arms} — "
                 "these Bundles do not share a trajectory or model and must not pool")
        write_batch_status(analysis_root, final)
        _log(
            f"batch {job_id[:8]} done: {final['runsWritten']} runs over "
            f"{final['bundlesRun']} bundles ({final['bundlesFailed']} failed)"
        )
        return final
    finally:
        _BATCH_LOCK.release()


# --------------------------------------------------------------------------- #
# Backend seam — the heavy dependency lives behind this and nowhere else
# --------------------------------------------------------------------------- #

def model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or MODEL_DIR_DEFAULT)


def pinned_model(mode: int) -> dict[str, Any]:
    """The lock entry for one mode: the sha256 and size this arm is defined against."""

    name = MODEL_BUNDLES.get(mode)
    if name is None:
        raise ValueError(f"Unknown detection mode {mode!r}; expected one of {DETECTION_MODES}.")
    entry = (_read_json(MODEL_LOCK_PATH).get("models") or {}).get(name)
    if not isinstance(entry, dict) or not entry.get("sha256"):
        raise FileNotFoundError(
            f"No pin for {name} in {MODEL_LOCK_PATH.name}. The lock file is the tracked "
            f"record of which weights an arm means; without it a run cannot be attributed "
            f"to a model. Re-pin with `{FETCH_HINT} --update`."
        )
    return entry


def resolve_model_bundle(mode: int) -> Path:
    """The local ``.task`` bundle for one mode, **verified against the lock**.

    Never downloads. A missing or altered bundle fails here, loudly, rather than at the
    other end of a sweep: a run produced by weights the lock does not describe is a run
    that cannot be attributed to an arm, which is the whole failure PRD #156 exists to
    stop happening again.
    """

    name = MODEL_BUNDLES[mode]
    entry = pinned_model(mode)
    target = model_dir() / f"{name}.task"
    if not target.is_file():
        raise FileNotFoundError(
            f"Model bundle {target} is missing. The .task binaries are gitignored; the "
            f"lock file is the tracked record. Fetch them with `{FETCH_HINT}`."
        )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"Model bundle {target} does not match the pin: expected "
            f"{entry['sha256'][:16]}, found {digest[:16]}. Refusing to run — the arm "
            f"stamp would name weights that did not produce the run. Restore with "
            f"`{FETCH_HINT}`, or adopt the change deliberately with `{FETCH_HINT} "
            f"--update` (which makes prior runs non-comparable)."
        )
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

    @property
    def model_sha(self) -> str:
        """The pinned sha256 of the weights this detector will run.

        Read from the lock file, so it costs nothing and is available *before* the model
        is loaded — which is what lets the job stamp the arm identity up front rather than
        discovering the model's identity after the first pass has already been written.
        """

        return pinned_model(self._mode)["sha256"]

    def _ensure_model(self):
        with self._load_lock:
            if self._landmarker is None:
                # Lazy — never imported by analysis_pipeline (ADR 0012).
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python import vision

                self._landmarker = vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=str(resolve_model_bundle(self._mode))
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
        if config.crop not in (CROP_NONE, CROP_TRACKED):
            raise NotImplementedError(
                f"Crop policy {config.crop!r} is stamped into the arm identity but not "
                f"implemented; {CROP_NONE!r} and {CROP_TRACKED!r} run today."
            )

    def _pose_region(self, landmarker, mp, cv2, names, frame_bgr,
                     rect: tuple[float, float, float, float] | None) -> list[Keypoint]:
        """Pose one frame, optionally restricted to ``rect``, in **full-frame** coordinates.

        The mapping back out is the load-bearing part: a cropped arm whose keypoints stayed
        in crop coordinates would be silently uncomparable with every uncropped arm, and the
        error would look like a detection quality difference rather than a units bug.
        """

        height, width = frame_bgr.shape[:2]
        x0f, y0f, x1f, y1f = rect if rect else (0.0, 0.0, 1.0, 1.0)
        x0, y0 = int(x0f * width), int(y0f * height)
        x1, y1 = int(x1f * width), int(y1f * height)
        if x1 - x0 < 32 or y1 - y0 < 32:
            return []
        sub = frame_bgr[y0:y1, x0:x1]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        poses = landmarker.detect(image).pose_landmarks or []
        if not poses:
            return []
        span_x, span_y = (x1 - x0) / width, (y1 - y0) / height
        return [
            Keypoint(
                name=names[i] if i < len(names) else f"landmark_{i}",
                x=_clamp01(x0f + landmark.x * span_x),
                y=_clamp01(y0f + landmark.y * span_y),
                score=_clamp01(landmark.visibility),
            )
            for i, landmark in enumerate(poses[0])
        ]

    def detect(
        self,
        video_path: Path,
        timestamps: Sequence[float],
        config: DetectionConfig,
        crop_track: Any | None = None,
    ) -> dict[float, list[Keypoint]]:
        """Decode the requested timestamps and pose each one. See ``Detector.detect``.

        ``crop_track`` is the Bundle's crop trajectory (issue #169), required when
        ``config.crop`` is ``tracked``. Each requested timestamp is posed inside the crop
        recorded nearest to it, which is what lets a sparse experimental grid reuse a dense
        tracking pass instead of needing frame-to-frame continuity of its own.
        """

        # Validate before importing anything heavy: a config this detector cannot honour is
        # cheap to reject, and doing it first keeps the refusal reachable — and testable —
        # on a machine with no MediaPipe installed.
        self._require_supported(config)
        if config.crop == CROP_TRACKED and crop_track is None:
            raise ValueError(
                "crop=tracked needs the Bundle's crop trajectory; running full-frame "
                "instead would silently produce a different arm from the one stamped."
            )

        import cv2  # lazy — a core dep, but this module is never imported for its purity

        import mediapipe as mp
        from mediapipe.tasks.python import vision

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
                rect = None
                if config.crop == CROP_TRACKED:
                    box = crop_track.nearest(timestamp)
                    rect = box.rect() if box is not None else None
                # A decoded frame always gets a key — an empty list here means "the model
                # saw nobody", which is a measurement and must not look like a decode gap.
                out[timestamp] = self._pose_region(
                    landmarker, mp, cv2, names, frame_bgr, rect)
            return out
        finally:
            capture.release()

    # ----------------------------------------------------------------------- #
    # Stage A — build the trajectory the runs above crop by
    # ----------------------------------------------------------------------- #

    def build_crop_track(
        self,
        video_path: Path,
        config: crop_track.CropTrackConfig,
        seed_x: float,
        seed_y: float,
        seed_t: float = 0.0,
        climb_end: float | None = None,
    ) -> crop_track.CropTrack:
        """Walk the video from the setup tap and return the crop trajectory.

        The decode loop and the MediaPipe probe live here; every decision about *how* to
        follow the climber lives in ``crop_track``, which is why that module is testable
        with a stub and this one is not.
        """

        import cv2  # lazy

        import mediapipe as mp
        from mediapipe.tasks.python import vision

        landmarker = self._ensure_model()
        names = [landmark.name.lower() for landmark in vision.PoseLandmark]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for decoding: {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0.0:
                raise RuntimeError(f"Video reports no frame rate, cannot seek: {video_path}")
            duration = (capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / fps
            end = min(duration, climb_end) if climb_end else duration

            def frames():
                t = float(seed_t)
                while t < end:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    yield round(t, 3), frame
                    t += config.step

            def probe(timestamp, frame_bgr, cx, cy, half):
                rect = (max(0.0, cx - half), max(0.0, cy - half),
                        min(1.0, cx + half), min(1.0, cy + half))
                points = self._pose_region(
                    landmarker, mp, cv2, names, frame_bgr, rect)
                if not points:
                    return None
                return crop_track.Probe(
                    cx=sum(p.x for p in points) / len(points),
                    cy=sum(p.y for p in points) / len(points),
                    appearance=torso_appearance(cv2, frame_bgr, points),
                )

            # Try each candidate crop size and keep whichever tracked best. Crop size has no
            # global optimum — measured across 12 Bundles where full-frame detects nothing,
            # 0.15 and 0.20 tie on the median and win in *opposite* directions per video, so
            # a single global size costs about ten points. Ties go to the smaller crop: a
            # tighter box means the climber fills more of what the model sees.
            best: crop_track.CropTrack | None = None
            for half in sorted(config.half_candidates):
                candidate = crop_track.track(
                    frames(), probe, config, seed_x, seed_y, half=half)
                _log(f"  half={half}: {candidate.detected}/{len(candidate.boxes)} "
                     f"({candidate.rate():.0%})")
                if best is None or candidate.rate() > best.rate():
                    best = candidate
            return best if best is not None else crop_track.CropTrack(config=config)
        finally:
            capture.release()


def torso_appearance(cv2, frame_bgr, points: Sequence[Keypoint]) -> tuple[float, ...]:
    """Clothing-colour signature of a detected pose's torso, as an HSV hue-sat histogram.

    The guard that stops a widened recovery search adopting a belayer. Deliberately the same
    measure ``vitpose_job`` uses for Climber Identity, where it was measured on the planet-x
    pair separating the reappearing climber (0.35) from base bystanders (0.76–0.84).

    Returns ``()`` when the torso region is degenerate, which reduces tracking to pure
    geometry rather than failing — a featureless probe is a supported state.
    """

    named = {p.name: p for p in points}
    corners = [named.get(n) for n in
               ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(c is None for c in corners):
        return ()
    height, width = frame_bgr.shape[:2]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    x0, x1 = int(min(xs) * width), int(max(xs) * width)
    y0, y1 = int(min(ys) * height), int(max(ys) * height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return ()
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0.0:
        return ()
    return tuple(round(float(v) / total, 5) for v in hist.flatten())


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
    parser.add_argument("--crop", default=CROP_NONE, choices=list(CROP_POLICIES))
    parser.add_argument("--track", action="store_true",
                        help="Stage A: build the crop trajectory and exit.")
    parser.add_argument("--crop-half", type=float, default=None)
    parser.add_argument("--crop-step", type=float, default=None)
    args = parser.parse_args(argv)

    analysis_root = args.analysis_root.resolve()
    bundle_dir = bundle_dir_for(analysis_root, args.route_folder, args.video_key)
    video = resolve_bundle_video(bundle_dir)
    if video is None:
        parser.error(f"No video binary in {bundle_dir}")

    if args.track:
        overrides = {k: v for k, v in
                     (("half", args.crop_half), ("step", args.crop_step)) if v is not None}
        config = replace(crop_track.CropTrackConfig(), **overrides)
        setup = _read_json(bundle_dir / SETUP_NAME)
        tap = setup.get("climberPoint") or {}
        if not isinstance(tap, dict) or tap.get("x") is None:
            parser.error(
                f"No setup tap (setup.json climberPoint) in {bundle_dir}. Tracking is seeded "
                "from calibration, never from truth — there is no valid fallback.")
        detector = MediaPipeDetector(mode=args.mode)
        track = detector.build_crop_track(
            video, config,
            seed_x=float(tap["x"]), seed_y=float(tap["y"]),
            seed_t=float(tap.get("t") or 0.0),
            climb_end=setup.get("climbEnd"),
        )
        track.setup_hash = read_setup_hash(bundle_dir)
        track.seed = {"x": tap.get("x"), "y": tap.get("y"), "t": tap.get("t"),
                      "source": "setup.json climberPoint"}
        path = crop_track.write_crop_track(bundle_dir, track)
        print(f"{track.detected}/{len(track.boxes)} tracked ({track.rate():.0%})  "
              f"arm {crop_track.track_hash(config)[:8]}  -> {path}")
        return 0

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
            config=DetectionConfig(mode=args.mode, crop=args.crop),
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
