"""Cycle integrity — the two checks that decide whether a cycle's arms are comparable.

Batches are **mode-major**: one batch per mode, run in sequence over hours or days. That
makes anything which changes between the first batch and the last *perfectly confounded
with mode* — everything mode-0 happened before the change and everything mode-2 after it,
and no amount of later analysis can separate them. Interleaving would spread drift evenly
instead; it was rejected (PRD #156 amendment 9) on one condition: that drift is **detected
exactly** rather than designed around.

This module is that detection. A **Cycle** is the comparison group — the set of batches
whose arms are meant to be read against each other — and it is opened, then closed, with
two guards spanning it:

1. **The determinism canary.** This detector is bit-deterministic: three passes over the
   same frames produce byte-identical output, confirmed across all three modes, two videos
   and separate processes. That turns drift detection from a statistical question into a
   byte comparison. One designated Bundle runs on one fixed arm at cycle open and again at
   cycle close, and the pose frames are compared byte-for-byte. Identical bytes means
   nothing moved: not the weights, not the module, not the crop trajectory, not the
   environment. Any difference at all fails the cycle.

   **The canary arm must crop.** Full-frame MediaPipe detects **0%** on the canary Bundle
   where the tracked crop reaches ~92%. A canary whose output is empty is byte-identical no
   matter which weights produced it, so an uncropped canary would sail straight through a
   model swap. This module therefore refuses to certify a cycle whose canary detected
   nothing — an empty witness is not a witness.

2. **The truth-hash snapshot.** Ground Truth is human-edited between sessions and the
   working tree is shared. If a Bundle's truth moves between the mode-0 and mode-2 batches,
   those arms pair against *different* truth and are silently incomparable. It is
   technically visible — evaluation records are named ``<run_ts>_vs_<truthHash8>.json`` —
   but nothing refuses to pool them, and nobody reads filenames when a table renders
   cleanly. So every eligible Bundle's truth identity is snapshotted at open and verified
   at close; one that moved drops out of the cycle's comparison **by name**.

Everything else that could drift is already guarded elsewhere: model weights by the sha256
in the arm identity (#165), the frame set by ``12·√n`` being a deterministic function of
the Bundle (#159), the schema by ADR 0009's v15 freeze, and the rest by the canary.

The manifest is written as a **tracked artifact** under ``analysis/cycles/``, so a
published comparison can be audited after the fact rather than taken on trust.

Pure but for the canary's detection pass, which goes through an injected detector factory —
the same seam ``mediapipe_job`` uses. The whole guard is therefore testable with stub
hashes and no MediaPipe, which is the point: a guard nobody can exercise cheaply is a guard
nobody exercises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import crop_track
import mediapipe_job as mj
from youtube_core import generate_timestamp

ARTIFACT_VERSION = 1

# Corpus-level, not per-Bundle: a cycle spans every Bundle at once, so it files beside
# ``mediapipe-batch.status.json`` rather than inside any one bundle. Tracked — the whole
# point is that a published comparison can be re-checked later by someone who was not here.
CYCLES_DIR_NAME = "cycles"

# --------------------------------------------------------------------------- #
# The canary
# --------------------------------------------------------------------------- #

# **The median Bundle, not a convenient one.** 705 truth frames — the median of the 84
# eligible Bundles — of which ``12·√n`` keeps 318, about two minutes per run. Chosen so the
# canary exercises a representative decode/crop/infer path rather than a short easy one.
CANARY_ROUTE = "planet-x"
CANARY_VIDEO_KEY = "3aUyWQp010A_20260711-185754"

# Mode 1 (``pose_landmarker_full``) on the **tracked crop**. The crop is not a detail: on
# this Bundle full-frame detects 0% and tracked crop reaches ~92%, and a canary that detects
# nothing witnesses nothing — see CANARY_MIN_DETECTION_RATE.
CANARY_MODE = 1
CANARY_CROP = mj.CROP_TRACKED

# Below this share of sampled frames the canary **refuses to certify**, because empty output
# cannot witness a model change: swap the weights under an all-empty canary and the bytes
# still match. Set well under the ~92% this Bundle/arm actually reaches and well over the 0%
# an uncropped arm produces, so it separates "the canary is blind" from "the detector moved"
# — a real drop from 92% to anything lower is caught by the byte comparison first, and this
# floor only stops a *vacuous* run being read as a clean bill of health.
CANARY_MIN_DETECTION_RATE = 0.5
# ...and an absolute floor, so a tiny Bundle cannot satisfy the rate on a handful of frames.
CANARY_MIN_DETECTED_FRAMES = 20

# How many differing frames a failure report names individually before summarising. A drift
# report is read by a human deciding what broke; 4,000 lines of hashes is not a report.
MAX_REPORTED_FRAMES = 20

# --------------------------------------------------------------------------- #
# Cycle status vocabulary
# --------------------------------------------------------------------------- #

STATUS_OPEN = "open"
STATUS_CERTIFIED = "certified"
STATUS_FAILED = "failed"
# Refused *at open*, before any batch ran: the canary could not witness, so there was never
# a cycle to certify. Recorded rather than silently discarded — a refusal is evidence.
STATUS_REFUSED = "refused"

FAILURE_CANARY_DRIFT = "canary-drift"
FAILURE_CANARY_UNWITNESSED = "canary-unwitnessed"

# Why a Bundle dropped out of a cycle's comparison. Every one of these means "the runs from
# the start of this cycle and the runs from the end were measured against different things".
REASON_TRUTH_HASH = "truth-hash-moved"
REASON_TRUTH_CONTENT = "truth-content-moved"
REASON_TRUTH_SOURCE = "truth-source-changed"
REASON_TRUTH_GONE = "truth-vanished"
REASON_SETUP_HASH = "setup-hash-moved"
REASON_CROP_CONTENT = "crop-track-moved"
REASON_CROP_CONFIG = "crop-track-config-moved"
REASON_CROP_GONE = "crop-track-vanished"
REASON_CROP_ADDED = "crop-track-added"
REASON_BUNDLE_GONE = "bundle-vanished"


class CycleIntegrityError(RuntimeError):
    """Raised where continuing would produce a comparison nobody can trust."""


def _log(message: str) -> None:
    print(f"[cycle] {message}", file=sys.stderr, flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: Any, length: int | None = None) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


# --------------------------------------------------------------------------- #
# Snapshot — what a Bundle's inputs were when the cycle opened
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BundleSnapshot:
    """One Bundle's identity at cycle open: what its runs will be measured against.

    Two truth hashes, deliberately. ``truth_hash`` is the one ``evaluate`` computes and
    names its records by (``groundTruthHash`` when the artifact declares one, else a content
    hash), so it is the hash that decides whether two records pair against the same truth.
    ``truth_content_hash`` is *always* over the content — which catches the case the
    declared hash cannot: an edit that changed the frames without re-stamping the hash. That
    edit is invisible to record naming, which makes it exactly the kind of drift that pools
    silently.
    """

    route: str
    video_key: str
    truth_source: str            # "ground-truth" | "vitpose" | ""
    truth_hash: str
    truth_content_hash: str
    truth_frames: int
    setup_hash: str
    crop_track_hash: str         # tracker config identity; "" when the Bundle has no track
    crop_track_content_hash: str
    crop_track_frames: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.route, self.video_key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "videoKey": self.video_key,
            "truthSource": self.truth_source,
            "truthHash": self.truth_hash,
            "truthContentHash": self.truth_content_hash,
            "truthFrames": self.truth_frames,
            "setupHash": self.setup_hash,
            "cropTrackHash": self.crop_track_hash,
            "cropTrackContentHash": self.crop_track_content_hash,
            "cropTrackFrames": self.crop_track_frames,
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "BundleSnapshot":
        return cls(
            route=str(doc.get("route") or ""),
            video_key=str(doc.get("videoKey") or ""),
            truth_source=str(doc.get("truthSource") or ""),
            truth_hash=str(doc.get("truthHash") or ""),
            truth_content_hash=str(doc.get("truthContentHash") or ""),
            truth_frames=int(doc.get("truthFrames") or 0),
            setup_hash=str(doc.get("setupHash") or ""),
            crop_track_hash=str(doc.get("cropTrackHash") or ""),
            crop_track_content_hash=str(doc.get("cropTrackContentHash") or ""),
            crop_track_frames=int(doc.get("cropTrackFrames") or 0),
        )


def truth_identity(bundle_dir: Path) -> tuple[str, str, str, int]:
    """``(source, truth_hash, content_hash, frame_count)`` for one Bundle.

    Mirrors ``analysis_pipeline.evaluate.load_truth``'s preference order and hash rule
    exactly — ``ground-truth.json`` over ``vitpose.json``, ``groundTruthHash`` when
    declared, else a content hash. Reimplemented rather than imported so this guard stays
    importable beside ``mediapipe_job`` without pulling the pipeline in, and asserted
    against ``evaluate``'s own value in the test suite so the two cannot drift apart. If
    they ever did, this module would be snapshotting a hash that is not the one deciding
    which records pair — a guard measuring the wrong quantity.
    """

    for name, source in (("ground-truth.json", "ground-truth"), ("vitpose.json", "vitpose")):
        path = bundle_dir / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        frames = doc.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        content = _sha(doc)
        declared = doc.get("groundTruthHash")
        return (source, str(declared) if declared else content, content, len(frames))
    return ("", "", "", 0)


def snapshot_bundle(bundle_dir: Path, route: str, video_key: str) -> BundleSnapshot:
    """Everything about a Bundle that, if it moved mid-cycle, breaks the comparison."""

    source, truth_hash, content_hash, frames = truth_identity(bundle_dir)
    track = crop_track.load_crop_track(bundle_dir)
    return BundleSnapshot(
        route=route,
        video_key=video_key,
        truth_source=source,
        truth_hash=truth_hash,
        truth_content_hash=content_hash,
        truth_frames=frames,
        setup_hash=mj.read_setup_hash(bundle_dir) or "",
        crop_track_hash=crop_track.track_hash(track.config) if track else "",
        crop_track_content_hash=crop_track.content_hash(track) if track else "",
        crop_track_frames=len(track.boxes) if track else 0,
    )


def snapshot_corpus(analysis_root: Path) -> tuple[mj.BundleSelection, list[BundleSnapshot]]:
    """Snapshot every Bundle a batch in this cycle could run.

    The population is ``mediapipe_job.select_bundles`` — the same selection a batch makes —
    so the manifest covers exactly what the cycle will measure, and the selection's own
    exclusions ride along rather than being re-derived by a second, divergent rule.
    """

    selection = mj.select_bundles(analysis_root)
    snapshots = [
        snapshot_bundle(mj.bundle_dir_for(analysis_root, route, key), route, key)
        for route, key in selection.included
    ]
    return selection, snapshots


# --------------------------------------------------------------------------- #
# Verification — which Bundles held still, and which moved
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FieldMove:
    """One field that differs between the snapshot and now."""

    field: str
    reason: str
    opened: Any
    closed: Any

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "reason": self.reason,
                "opened": self.opened, "closed": self.closed}


@dataclass(frozen=True)
class BundleVerdict:
    route: str
    video_key: str
    held: bool
    moves: tuple[FieldMove, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "videoKey": self.video_key,
            "status": "held" if self.held else "excluded",
            "reasons": sorted({m.reason for m in self.moves}),
            "moved": [m.as_dict() for m in self.moves],
        }


def verify_bundle(snapshot: BundleSnapshot, current: BundleSnapshot | None) -> BundleVerdict:
    """Compare a Bundle against its snapshot. Any movement excludes it from the cycle."""

    if current is None:
        return BundleVerdict(snapshot.route, snapshot.video_key, held=False, moves=(
            FieldMove("bundle", REASON_BUNDLE_GONE, "eligible", "absent"),))

    checks = (
        ("truthHash", REASON_TRUTH_HASH, snapshot.truth_hash, current.truth_hash),
        ("truthContentHash", REASON_TRUTH_CONTENT,
         snapshot.truth_content_hash, current.truth_content_hash),
        ("truthSource", REASON_TRUTH_SOURCE, snapshot.truth_source, current.truth_source),
        ("setupHash", REASON_SETUP_HASH, snapshot.setup_hash, current.setup_hash),
        ("cropTrackHash", REASON_CROP_CONFIG,
         snapshot.crop_track_hash, current.crop_track_hash),
        ("cropTrackContentHash", REASON_CROP_CONTENT,
         snapshot.crop_track_content_hash, current.crop_track_content_hash),
    )
    moves: list[FieldMove] = []
    for name, reason, opened, closed in checks:
        if opened == closed:
            continue
        # Name the *disappearance* rather than reporting it as a hash change: "truth
        # vanished" and "truth was revised" call for different responses from whoever reads
        # this, and collapsing them into one reason costs that distinction.
        if name.startswith("truth") and not closed:
            reason = REASON_TRUTH_GONE
        elif name.startswith("cropTrack"):
            reason = REASON_CROP_GONE if not closed else (
                REASON_CROP_ADDED if not opened else reason)
        moves.append(FieldMove(name, reason, opened, closed))

    return BundleVerdict(snapshot.route, snapshot.video_key,
                         held=not moves, moves=tuple(moves))


@dataclass
class Verification:
    held: list[BundleVerdict] = field(default_factory=list)
    excluded: list[BundleVerdict] = field(default_factory=list)
    # Bundles that became eligible *after* the cycle opened. Not excluded — they were never
    # in it — but named, because a reader comparing corpus counts would otherwise find the
    # cycle covering fewer Bundles than exist and have no way to learn why.
    added: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "heldCount": len(self.held),
            "excludedCount": len(self.excluded),
            "held": [{"route": v.route, "videoKey": v.video_key} for v in self.held],
            "excluded": [v.as_dict() for v in self.excluded],
            "added": [{"route": r, "videoKey": k} for r, k in self.added],
            "addedCount": len(self.added),
        }


def verify_corpus(analysis_root: Path, snapshots: Sequence[BundleSnapshot]) -> Verification:
    """Re-read every snapshotted Bundle and split held from excluded."""

    selection = mj.select_bundles(analysis_root)
    eligible_now = set(selection.included)
    snapshotted = {s.key for s in snapshots}

    result = Verification(added=sorted(eligible_now - snapshotted))
    for snapshot in snapshots:
        bundle_dir = mj.bundle_dir_for(analysis_root, snapshot.route, snapshot.video_key)
        current = (snapshot_bundle(bundle_dir, snapshot.route, snapshot.video_key)
                   if bundle_dir.is_dir() else None)
        verdict = verify_bundle(snapshot, current)
        (result.held if verdict.held else result.excluded).append(verdict)
    return result


# --------------------------------------------------------------------------- #
# The canary
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CanaryRun:
    """One canary pass, reduced to what a byte comparison needs.

    ``frames_sha`` is the comparison: sha256 over the canonicalised ``frames`` array of the
    pose artifact this pass *would have written* — built by ``mediapipe_job``'s own
    ``build_pose_payload``, so the bytes compared are the bytes a run records, not a
    parallel summary that could agree while the artifact differs. ``diagnostics`` is
    excluded because it carries ``createdAt``, which differs between any two passes by
    construction and would make every canary fail.

    ``frame_digests`` is per-frame, index-aligned with ``timestamps``, and exists solely so
    a failure can say *where* the two passes diverged instead of "the hashes differ".
    """

    at: str
    route: str
    video_key: str
    config: dict[str, Any]
    config_hash: str
    setup_hash: str
    timestamps: tuple[float, ...]
    frames_sha: str
    frame_digests: tuple[str, ...]
    detected: int
    missing: int
    undecodable: int
    crop_track_hash: str
    crop_track_content_hash: str
    seconds: float = 0.0

    @property
    def sampled(self) -> int:
        return len(self.timestamps)

    @property
    def detection_rate(self) -> float:
        return self.detected / self.sampled if self.sampled else 0.0

    def witnesses(self) -> bool:
        """Whether this pass can witness a change at all.

        A canary that detected nothing is byte-identical under *any* weights, so it would
        certify a model swap it never saw. That is worse than no canary, because it reads
        as a passing check.
        """

        return (self.detected >= CANARY_MIN_DETECTED_FRAMES
                and self.detection_rate >= CANARY_MIN_DETECTION_RATE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "route": self.route,
            "videoKey": self.video_key,
            "config": self.config,
            "configHash": self.config_hash,
            "setupHash": self.setup_hash,
            "timestamps": list(self.timestamps),
            "timestampsSha": _sha(list(self.timestamps), 16),
            "framesSha": self.frames_sha,
            "frameDigests": list(self.frame_digests),
            "sampledFrames": self.sampled,
            "framesDetected": self.detected,
            "framesMissing": self.missing,
            "framesUndecodable": self.undecodable,
            "detectionRate": round(self.detection_rate, 4),
            "witnesses": self.witnesses(),
            "cropTrackHash": self.crop_track_hash,
            "cropTrackContentHash": self.crop_track_content_hash,
            "seconds": round(self.seconds, 2),
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "CanaryRun":
        return cls(
            at=str(doc.get("at") or ""),
            route=str(doc.get("route") or ""),
            video_key=str(doc.get("videoKey") or ""),
            config=dict(doc.get("config") or {}),
            config_hash=str(doc.get("configHash") or ""),
            setup_hash=str(doc.get("setupHash") or ""),
            timestamps=tuple(float(t) for t in (doc.get("timestamps") or ())),
            frames_sha=str(doc.get("framesSha") or ""),
            frame_digests=tuple(str(d) for d in (doc.get("frameDigests") or ())),
            detected=int(doc.get("framesDetected") or 0),
            missing=int(doc.get("framesMissing") or 0),
            undecodable=int(doc.get("framesUndecodable") or 0),
            crop_track_hash=str(doc.get("cropTrackHash") or ""),
            crop_track_content_hash=str(doc.get("cropTrackContentHash") or ""),
            seconds=float(doc.get("seconds") or 0.0),
        )


def canary_arm(mode: int = CANARY_MODE, crop: str = CANARY_CROP) -> mj.DetectionConfig:
    return mj.DetectionConfig(mode=mode, crop=crop)


def run_canary(
    analysis_root: Path,
    detector_factory: mj.DetectorFactory,
    route: str = CANARY_ROUTE,
    video_key: str = CANARY_VIDEO_KEY,
    config: mj.DetectionConfig | None = None,
    timestamps: Sequence[float] | None = None,
) -> CanaryRun:
    """One canary pass: decode, crop, infer, and reduce to a comparable digest.

    Nothing is written to ``detections/``. A canary is an *instrument*, not a measurement —
    written as a run it would add a second copy of one Bundle under one arm to every pooled
    number, weighting that Bundle twice for data that is by construction byte-identical to
    data already there.

    ``timestamps`` is what the close pass is given: the exact grid the open pass sampled,
    read back out of the manifest. Re-deriving it from truth at close would let a truth
    revision on the canary Bundle change the *frame set*, and the byte comparison would fail
    for a reason that is not detector drift — the canary would be reporting the truth guard's
    finding, badly. Truth movement on the canary Bundle is caught by the snapshot, where it
    belongs.
    """

    config = config or canary_arm()
    bundle_dir = mj.bundle_dir_for(analysis_root, route, video_key)
    if not bundle_dir.is_dir():
        raise CycleIntegrityError(
            f"The canary Bundle {route}/{video_key} is not in this corpus. A cycle without "
            "a canary cannot detect drift between its first batch and its last."
        )
    video = mj.resolve_bundle_video(bundle_dir)
    if video is None:
        raise CycleIntegrityError(f"No video binary for the canary Bundle {bundle_dir}.")

    grid = tuple(float(t) for t in timestamps) if timestamps is not None \
        else mj.sample_timestamps(mj.truth_timestamps(bundle_dir))
    if not grid:
        raise CycleIntegrityError(
            f"The canary Bundle {route}/{video_key} has no truth grid to sample.")

    request = mj.DetectRequest(
        video_path=str(video), route_folder=route, video_key=video_key,
        frames=grid, config=config, repeats=1,
    )
    # Resolved through the batch's own path, so the canary's arm is the batch's arm — model
    # sha read off the detector, crop trajectory read off disk.
    request, track = mj.resolve_arm(bundle_dir, request, detector_factory)
    if config.crop == mj.CROP_TRACKED and track is None:
        raise CycleIntegrityError(
            f"The canary arm crops, and {route}/{video_key} has no "
            f"{crop_track.ARTIFACT_NAME}. Running it full-frame instead would produce an "
            "empty canary that passes every comparison; see CANARY_MIN_DETECTION_RATE."
        )

    started = time.perf_counter()
    detector = detector_factory(request.config)
    detections = detector.detect(video, request.frames, request.config, crop_track=track)
    elapsed = time.perf_counter() - started

    decoded = set(detections)
    payload = mj.build_pose_payload(request, detections, pass_index=0, decoded=decoded)
    frames = payload["frames"]
    counts = {mj.FRAME_DETECTED: 0, mj.FRAME_MISSING: 0, mj.FRAME_UNDECODABLE: 0}
    for frame in frames:
        counts[frame["source"]] = counts.get(frame["source"], 0) + 1

    return CanaryRun(
        at=_now_iso(),
        route=route,
        video_key=video_key,
        config=request.config.identity(),
        config_hash=mj.config_hash(request.config),
        setup_hash=request.setup_hash or "",
        timestamps=grid,
        frames_sha=_sha(frames),
        frame_digests=tuple(_sha(frame, 16) for frame in frames),
        detected=counts[mj.FRAME_DETECTED],
        missing=counts[mj.FRAME_MISSING],
        undecodable=counts[mj.FRAME_UNDECODABLE],
        crop_track_hash=crop_track.track_hash(track.config) if track else "",
        crop_track_content_hash=crop_track.content_hash(track) if track else "",
        seconds=elapsed,
    )


def compare_canary(opened: CanaryRun, closed: CanaryRun) -> dict[str, Any]:
    """Byte-compare two canary passes and name what moved.

    Scalar identity first — the arm, the weights, the module, the trajectory, the frame set
    — because those *name a cause*. Then the frames, which measure the *effect*. A report
    that only said "the bytes differ" would leave the reader to work out whether they had a
    model swap or a re-tracked crop, which is the work this is supposed to save.
    """

    fields: list[dict[str, Any]] = []

    def check(name: str, before: Any, after: Any) -> None:
        if before != after:
            fields.append({"field": name, "opened": before, "closed": after})

    check("configHash", opened.config_hash, closed.config_hash)
    for key in sorted(set(opened.config) | set(closed.config)):
        check(f"config.{key}", opened.config.get(key), closed.config.get(key))
    check("setupHash", opened.setup_hash, closed.setup_hash)
    check("cropTrackHash", opened.crop_track_hash, closed.crop_track_hash)
    check("cropTrackContentHash",
          opened.crop_track_content_hash, closed.crop_track_content_hash)
    check("timestampsSha", _sha(list(opened.timestamps), 16), _sha(list(closed.timestamps), 16))
    check("sampledFrames", opened.sampled, closed.sampled)
    check("framesDetected", opened.detected, closed.detected)
    check("framesMissing", opened.missing, closed.missing)
    check("framesUndecodable", opened.undecodable, closed.undecodable)
    check("framesSha", opened.frames_sha, closed.frames_sha)

    differing: list[dict[str, Any]] = []
    for index, (before, after) in enumerate(zip(opened.frame_digests, closed.frame_digests)):
        if before == after:
            continue
        timestamp = opened.timestamps[index] if index < len(opened.timestamps) else None
        differing.append({"index": index, "timestamp": timestamp,
                          "openedDigest": before, "closedDigest": after})

    identical = not fields and not differing
    return {
        "identical": identical,
        "fields": fields,
        "framesCompared": min(len(opened.frame_digests), len(closed.frame_digests)),
        "framesDiffering": len(differing),
        "firstDivergence": differing[0]["timestamp"] if differing else None,
        "differingFrames": differing[:MAX_REPORTED_FRAMES],
        "differingFramesTruncated": max(0, len(differing) - MAX_REPORTED_FRAMES),
    }


# --------------------------------------------------------------------------- #
# The runs a cycle covers
# --------------------------------------------------------------------------- #

# ``exp-<base_ts>-<arm8>-p<pass>``, optionally with the writer's collision suffix.
RUN_ID_PATTERN = re.compile(
    rf"^{re.escape(mj.RUN_ID_PREFIX)}(\d{{8}}-\d{{6}})-([0-9a-f]{{8}})-p(\d+)(?:-\d+)?$")


def collect_cycle_runs(analysis_root: Path, opened_ts: str, closed_ts: str) -> dict[str, Any]:
    """Which experimental runs landed inside the cycle window, by arm.

    Read off run *filenames* — ``exp-`` ids carry their timestamp and arm, which is what
    that prefix was for (#160) — so this costs a directory listing rather than parsing every
    pose artifact in the corpus.

    This is the audit trail: a published comparison naming a cycle can be checked against
    the runs the cycle actually contained, instead of against whatever is on disk when
    someone re-reads it months later.
    """

    arms: dict[str, dict[str, Any]] = {}
    bundles: set[tuple[str, str]] = set()
    total = 0
    for pose_path in analysis_root.glob("*/*/detections/*_pose.json"):
        match = RUN_ID_PATTERN.match(pose_path.name[:-len("_pose.json")])
        if not match:
            continue
        run_ts, arm, _ = match.groups()
        if not (opened_ts <= run_ts <= closed_ts):
            continue
        bundle_dir = pose_path.parent.parent
        key = (bundle_dir.parent.name, bundle_dir.name)
        bundles.add(key)
        entry = arms.setdefault(arm, {"runs": 0, "bundles": set()})
        entry["runs"] += 1
        entry["bundles"].add(key)
        total += 1

    return {
        "windowStart": opened_ts,
        "windowEnd": closed_ts,
        "runCount": total,
        "bundlesWithRuns": len(bundles),
        "arms": {
            arm: {"runs": entry["runs"], "bundles": len(entry["bundles"])}
            for arm, entry in sorted(arms.items())
        },
    }


# --------------------------------------------------------------------------- #
# Cycle artifact
# --------------------------------------------------------------------------- #

def cycles_dir(analysis_root: Path) -> Path:
    return analysis_root / CYCLES_DIR_NAME


def cycle_path(analysis_root: Path, cycle_id: str) -> Path:
    return cycles_dir(analysis_root) / f"{cycle_id}.json"


def write_cycle(analysis_root: Path, doc: dict[str, Any]) -> Path:
    path = cycle_path(analysis_root, doc["cycleId"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_cycle(analysis_root: Path, cycle_id: str) -> dict[str, Any]:
    path = cycle_path(analysis_root, cycle_id)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CycleIntegrityError(f"No readable cycle at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CycleIntegrityError(f"Cycle artifact at {path} is not an object.")
    return doc


def list_cycles(analysis_root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(cycles_dir(analysis_root).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("cycleId"):
            out.append(doc)
    return out


def open_cycle_doc(analysis_root: Path) -> dict[str, Any] | None:
    """The cycle currently open, if any. At most one may be."""

    for doc in list_cycles(analysis_root):
        if doc.get("status") == STATUS_OPEN:
            return doc
    return None


def model_locks() -> dict[str, str]:
    """Every pinned model sha, not just the canary's.

    A cycle spans all three modes, so all three locks are part of what it was measured
    under — even though only one runs the canary. Recording one and sweeping three would
    leave two of the three arms' weights unwitnessed by the manifest.
    """

    try:
        lock = json.loads(mj.MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = (lock.get("models") or {}) if isinstance(lock, dict) else {}
    return {name: str(entry.get("sha256") or "")
            for name, entry in sorted(models.items()) if isinstance(entry, dict)}


# --------------------------------------------------------------------------- #
# Open / close
# --------------------------------------------------------------------------- #

def open_cycle(
    analysis_root: Path,
    detector_factory: mj.DetectorFactory,
    cycle_id: str | None = None,
    canary_config: mj.DetectionConfig | None = None,
    canary_route: str = CANARY_ROUTE,
    canary_video_key: str = CANARY_VIDEO_KEY,
) -> dict[str, Any]:
    """Snapshot the corpus, run the opening canary, and write the manifest.

    Raises ``CycleIntegrityError`` — after writing a ``refused`` artifact — when the canary
    cannot witness. Refusing here costs two minutes; discovering it at close costs the whole
    cycle, because by then the batches have run and there is no way to establish that
    nothing drifted under them.
    """

    existing = open_cycle_doc(analysis_root)
    if existing is not None:
        raise CycleIntegrityError(
            f"Cycle {existing['cycleId']} is still open. Two overlapping cycles would both "
            "claim the same batches, and a run cannot belong to two comparisons with "
            "different manifests. Close it first."
        )

    opened_ts = generate_timestamp()
    cycle_id = cycle_id or f"cycle-{opened_ts}"
    selection, snapshots = snapshot_corpus(analysis_root)
    _log(f"{cycle_id}: snapshotted {len(snapshots)} eligible bundles "
         f"({len(selection.excluded)} not eligible)")

    canary = run_canary(analysis_root, detector_factory, route=canary_route,
                        video_key=canary_video_key, config=canary_config)
    _log(f"{cycle_id}: canary {canary.detected}/{canary.sampled} detected "
         f"({canary.detection_rate:.0%}) arm {canary.config_hash[:8]} "
         f"frames {canary.frames_sha[:16]} in {canary.seconds:.1f}s")

    doc = {
        "version": ARTIFACT_VERSION,
        "cycleId": cycle_id,
        "status": STATUS_OPEN,
        "openedAt": _now_iso(),
        "openedRunTs": opened_ts,
        "closedAt": None,
        "closedRunTs": None,
        # The identity of the *harness* this cycle ran under. Everything here is outside the
        # per-run stamp and could otherwise move between the first batch and the last with
        # nothing to say so.
        "moduleVersion": mj.MODULE_VERSION,
        "sampleCoefficient": mj.SAMPLE_COEFFICIENT,
        "modelLocks": model_locks(),
        "canary": {
            "route": canary_route,
            "videoKey": canary_video_key,
            "minDetectionRate": CANARY_MIN_DETECTION_RATE,
            "minDetectedFrames": CANARY_MIN_DETECTED_FRAMES,
            "opened": canary.as_dict(),
            "closed": None,
            "comparison": None,
        },
        "manifest": {
            "bundleCount": len(snapshots),
            "bundles": [s.as_dict() for s in snapshots],
            "selection": selection.as_dict(),
        },
        "verification": None,
        "runs": None,
        "failures": [],
        "certified": False,
    }

    if not canary.witnesses():
        doc["status"] = STATUS_REFUSED
        doc["failures"] = [FAILURE_CANARY_UNWITNESSED]
        write_cycle(analysis_root, doc)
        raise CycleIntegrityError(
            f"REFUSED to open {cycle_id}: the canary detected {canary.detected}/"
            f"{canary.sampled} frames ({canary.detection_rate:.0%}), under the "
            f"{CANARY_MIN_DETECTION_RATE:.0%} floor. Empty output is byte-identical under "
            f"any weights, so this canary would certify a model swap it never saw. Check "
            f"the arm crops ({canary.config.get('crop')!r}) and that "
            f"{canary_route}/{canary_video_key} has a crop trajectory."
        )

    path = write_cycle(analysis_root, doc)
    _log(f"{cycle_id}: open -> {path}")
    return doc


def close_cycle(
    analysis_root: Path,
    detector_factory: mj.DetectorFactory,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    """Re-run the canary, re-verify the snapshot, and certify the cycle or fail it.

    Always writes the artifact — a failed cycle is *evidence*, and the artifact is where the
    diff lives. The caller decides how loudly to die; the CLI dies loudly.
    """

    if cycle_id is None:
        current = open_cycle_doc(analysis_root)
        if current is None:
            raise CycleIntegrityError(
                f"No open cycle under {cycles_dir(analysis_root)}. Open one before the "
                "first batch: a cycle cannot be certified retroactively, because the "
                "snapshot it verifies against has to predate the batches.")
        doc = current
        cycle_id = str(doc["cycleId"])
    else:
        doc = read_cycle(analysis_root, cycle_id)

    if doc.get("status") != STATUS_OPEN:
        raise CycleIntegrityError(
            f"Cycle {cycle_id} is {doc.get('status')!r}, not open; refusing to close it "
            "again. A second close would overwrite the diff the first one recorded.")

    opened = CanaryRun.from_dict((doc.get("canary") or {}).get("opened") or {})
    snapshots = [BundleSnapshot.from_dict(b)
                 for b in ((doc.get("manifest") or {}).get("bundles") or [])]

    # The same frame set the opening pass sampled, so the comparison isolates the detector.
    closed = run_canary(
        analysis_root, detector_factory,
        route=str((doc.get("canary") or {}).get("route") or CANARY_ROUTE),
        video_key=str((doc.get("canary") or {}).get("videoKey") or CANARY_VIDEO_KEY),
        config=mj.DetectionConfig(
            mode=int(opened.config.get("mode", CANARY_MODE)),
            crop=str(opened.config.get("crop", CANARY_CROP)),
        ),
        timestamps=opened.timestamps,
    )
    comparison = compare_canary(opened, closed)
    verification = verify_corpus(analysis_root, snapshots)
    closed_ts = generate_timestamp()

    failures: list[str] = []
    if not comparison["identical"]:
        failures.append(FAILURE_CANARY_DRIFT)
    if not (opened.witnesses() and closed.witnesses()):
        failures.append(FAILURE_CANARY_UNWITNESSED)

    doc = dict(doc)
    doc["status"] = STATUS_FAILED if failures else STATUS_CERTIFIED
    doc["certified"] = not failures
    doc["failures"] = failures
    doc["closedAt"] = _now_iso()
    doc["closedRunTs"] = closed_ts
    doc["canary"] = {**(doc.get("canary") or {}),
                     "closed": closed.as_dict(), "comparison": comparison}
    doc["verification"] = verification.as_dict()
    doc["runs"] = collect_cycle_runs(
        analysis_root, str(doc.get("openedRunTs") or ""), closed_ts)
    # The population a published comparison may pool: snapshotted, still identical, and
    # nothing else. Written out explicitly so nobody has to re-derive it from the exclusions.
    doc["comparableBundles"] = [{"route": v.route, "videoKey": v.video_key}
                                for v in verification.held]

    path = write_cycle(analysis_root, doc)
    _report(doc)
    _log(f"{cycle_id}: {doc['status']} -> {path}")
    return doc


def _report(doc: dict[str, Any]) -> None:
    """Print the verdict where it cannot be missed.

    A drift finding buried at INFO next to eight hours of batch progress is a finding
    nobody acts on, so a failure gets a banner, the fields that moved, and where the frames
    first diverged.
    """

    canary = doc.get("canary") or {}
    comparison = canary.get("comparison") or {}
    verification = doc.get("verification") or {}

    if doc.get("failures"):
        _log("!" * 72)
        _log(f"CYCLE {doc['cycleId']} FAILED: {', '.join(doc['failures'])}")
        _log("The arms in this cycle are NOT comparable to each other. Do not publish a "
             "comparison over them.")
        for entry in comparison.get("fields") or []:
            _log(f"  differs: {entry['field']}: {entry['opened']!r} -> {entry['closed']!r}")
        if comparison.get("framesDiffering"):
            _log(f"  differs: {comparison['framesDiffering']} of "
                 f"{comparison['framesCompared']} canary frames, first at "
                 f"t={comparison.get('firstDivergence')}s")
            for frame in (comparison.get("differingFrames") or [])[:5]:
                _log(f"    t={frame['timestamp']}: {frame['openedDigest']} -> "
                     f"{frame['closedDigest']}")
        if FAILURE_CANARY_UNWITNESSED in (doc.get("failures") or []):
            _log("  the canary detected too little to witness anything; see "
                 "CANARY_MIN_DETECTION_RATE")
        _log("!" * 72)
    else:
        _log(f"CYCLE {doc['cycleId']} CERTIFIED: canary byte-identical over "
             f"{comparison.get('framesCompared')} frames")

    excluded = verification.get("excluded") or []
    if excluded:
        _log(f"{len(excluded)} bundle(s) dropped from this cycle's comparison "
             f"({verification.get('heldCount')} held):")
        for entry in excluded:
            _log(f"  excluded: {entry['route']}/{entry['videoKey']}: "
                 f"{', '.join(entry['reasons'])}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_manifest_summary(analysis_root: Path) -> int:
    selection, snapshots = snapshot_corpus(analysis_root)
    by_source: dict[str, int] = {}
    tracked = 0
    for snapshot in snapshots:
        by_source[snapshot.truth_source] = by_source.get(snapshot.truth_source, 0) + 1
        if snapshot.crop_track_content_hash:
            tracked += 1
    print(f"eligible bundles     {len(snapshots)}")
    print(f"not eligible         {len(selection.excluded)}")
    for source, count in sorted(by_source.items()):
        print(f"  truth={source or '(none)':<14} {count}")
    print(f"  with crop track    {tracked}")
    print(f"truth frames total   {sum(s.truth_frames for s in snapshots)}")
    reasons: dict[str, int] = {}
    for entry in selection.excluded:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"  excluded {reason:<18} {count}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open, close and inspect batch cycles (issue #168).")
    parser.add_argument("command", choices=("open", "close", "show", "manifest"))
    parser.add_argument("cycle_id", nargs="?", default=None)
    parser.add_argument("--analysis-root", default="analysis", type=Path)
    parser.add_argument("--mode", type=int, default=CANARY_MODE,
                        choices=list(mj.DETECTION_MODES))
    parser.add_argument("--crop", default=CANARY_CROP, choices=list(mj.CROP_POLICIES))
    parser.add_argument("--route", default=CANARY_ROUTE)
    parser.add_argument("--video-key", default=CANARY_VIDEO_KEY)
    args = parser.parse_args(argv)

    analysis_root = args.analysis_root.resolve()

    if args.command == "manifest":
        return _print_manifest_summary(analysis_root)

    if args.command == "show":
        cycle_id = args.cycle_id
        if cycle_id is None:
            docs = list_cycles(analysis_root)
            if not docs:
                print("no cycles")
                return 0
            for doc in docs:
                print(f"{doc['cycleId']:<28} {doc.get('status'):<10} "
                      f"{(doc.get('manifest') or {}).get('bundleCount', 0)} bundles  "
                      f"{', '.join(doc.get('failures') or []) or ''}")
            return 0
        print(json.dumps(read_cycle(analysis_root, cycle_id), indent=2))
        return 0

    try:
        if args.command == "open":
            doc = open_cycle(
                analysis_root, mj.default_detector_factory, cycle_id=args.cycle_id,
                canary_config=mj.DetectionConfig(mode=args.mode, crop=args.crop),
                canary_route=args.route, canary_video_key=args.video_key)
            print(f"{doc['cycleId']}  open  "
                  f"{(doc['manifest'])['bundleCount']} bundles  "
                  f"canary {doc['canary']['opened']['framesDetected']}/"
                  f"{doc['canary']['opened']['sampledFrames']} detected")
            return 0
        doc = close_cycle(analysis_root, mj.default_detector_factory,
                          cycle_id=args.cycle_id)
    except CycleIntegrityError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(f"{doc['cycleId']}  {doc['status']}  "
          f"{(doc['verification'])['heldCount']} comparable, "
          f"{(doc['verification'])['excludedCount']} excluded")
    # Non-zero on failure: a cycle that failed must not look like a successful command to
    # whatever ran it.
    return 0 if doc["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
