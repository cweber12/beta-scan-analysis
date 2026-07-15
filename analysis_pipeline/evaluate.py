"""Detection-vs-truth evaluation — pair scanner pose runs with the bundle truth,
compute PCK@0.5-torso per joint, and write one evaluation record per pair.

This is the first end-to-end slice of the eval path (issue #6). It walks the
``analysis/`` bundle tree, pairs every scanner pose Run with the bundle's **truth**
file (``ground-truth.json`` if present, else ``vitpose.json``), and writes an
idempotent record at ``evaluations/<run_ts>_vs_<truthHash8>.json`` inside the bundle.

Pairing is gated on ``setupHash``: a pose Run is only compared against truth authored
under the *same* calibration. Legacy truth artifacts that predate #4 do not carry
their own ``setupHash`` (ADR 0004), so the truth's *effective* setupHash falls back to
the bundle ``setup.json`` — which is exactly the setup the truth was authored against.
Mismatches (a stale Run) are reported as skipped-with-reason, never silently dropped.

Metric (v1): PCK@0.5-torso per joint over the 13 shared COCO core joints. A predicted
joint counts as correct when its distance to the truth joint is within half the
**truth** torso length (shoulder-midpoint to hip-midpoint). The denominator is the
truth's — never the scanner's — so a collapsed detection cannot shrink its own scale.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discovery import _iter_video_dirs, _load_json, _pair_stems, _unwrap

# Evaluation record schema version. Bump on any record-shape change.
SCHEMA_VERSION = 1

# The 13 shared COCO core joints (ADR 0003 / ground-truth jointSet). Every truth
# source and the scanner pose name these identically, so we join by name.
COCO_CORE_JOINTS = [
    "nose",
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
]

# PCK threshold as a fraction of truth torso length.
PCK_TORSO_FRACTION = 0.5


@dataclass
class TruthFrame:
    """One truth frame reduced to what scoring needs."""

    timestamp: float
    present: bool  # a Climber is present in this frame (scorable)
    joints: dict[str, tuple[float, float]]  # name -> (x, y), present+non-occluded only


@dataclass
class TruthDoc:
    """A bundle's truth artifact, normalised across the two on-disk shapes."""

    source: str  # "ground-truth" | "vitpose"
    setup_hash: str  # self-reported setupHash, or "" when the artifact predates #4
    truth_hash: str  # groundTruthHash, or a content hash for vitpose
    frames: list[TruthFrame]


@dataclass
class Pairing:
    """The outcome of pairing one pose Run with the bundle truth."""

    route_folder: str
    video_key: str
    run_ts: str
    truth_source: str
    status: str  # "written" | "skipped"
    reason: str = ""  # populated when skipped
    record_path: Path | None = None


@dataclass
class EvalSummary:
    """Everything the CLI needs to print a run summary."""

    pairings: list[Pairing] = field(default_factory=list)
    truthless_videos: list[str] = field(default_factory=list)  # bundles with no truth

    @property
    def written(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "written"]

    @property
    def skipped(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "skipped"]


# --------------------------------------------------------------------------- #
# Truth loading
# --------------------------------------------------------------------------- #

def _content_hash(doc: dict[str, Any]) -> str:
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _truth_from_ground_truth(doc: dict[str, Any]) -> TruthDoc:
    """``ground-truth.json`` — frames carry ``state`` + a ``joints`` dict (ADR 0004)."""

    frames: list[TruthFrame] = []
    for fr in doc.get("frames", []):
        present = fr.get("state", "present") == "present"
        joints: dict[str, tuple[float, float]] = {}
        raw = fr.get("joints", {}) or {}
        for name, j in raw.items():
            if name not in COCO_CORE_JOINTS or not isinstance(j, dict):
                continue
            if j.get("occluded"):
                continue  # can't score against a joint the human marked hidden
            x, y = j.get("x"), j.get("y")
            if x is not None and y is not None:
                joints[name] = (float(x), float(y))
        frames.append(TruthFrame(float(fr.get("timestamp", 0.0)), present, joints))
    truth_hash = doc.get("groundTruthHash") or _content_hash(doc)
    return TruthDoc("ground-truth", doc.get("setupHash") or "", truth_hash, frames)


def _truth_from_vitpose(doc: dict[str, Any]) -> TruthDoc:
    """``vitpose.json`` — frames carry a ``keypoints`` list; ``[]`` means absent."""

    frames: list[TruthFrame] = []
    for fr in doc.get("frames", []):
        kps = fr.get("keypoints", []) or []
        present = len(kps) > 0
        joints: dict[str, tuple[float, float]] = {}
        for kp in kps:
            name = kp.get("name")
            if name not in COCO_CORE_JOINTS:
                continue
            x, y = kp.get("x"), kp.get("y")
            if x is not None and y is not None:
                joints[name] = (float(x), float(y))
        frames.append(TruthFrame(float(fr.get("timestamp", 0.0)), present, joints))
    truth_hash = doc.get("groundTruthHash") or _content_hash(doc)
    return TruthDoc("vitpose", doc.get("setupHash") or "", truth_hash, frames)


def load_truth(video_dir: Path) -> TruthDoc | None:
    """Load the bundle truth, preferring ``ground-truth.json`` over ``vitpose.json``."""

    gt = video_dir / "ground-truth.json"
    if gt.exists():
        return _truth_from_ground_truth(_load_json(gt))
    vit = video_dir / "vitpose.json"
    if vit.exists():
        return _truth_from_vitpose(_load_json(vit))
    return None


# --------------------------------------------------------------------------- #
# Geometry / metric
# --------------------------------------------------------------------------- #

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def torso_length(joints: dict[str, tuple[float, float]]) -> float | None:
    """Truth torso length: shoulder-midpoint to hip-midpoint. ``None`` if undefined."""

    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(n not in joints for n in need):
        return None
    ls, rs = joints["left_shoulder"], joints["right_shoulder"]
    lh, rh = joints["left_hip"], joints["right_hip"]
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    length = _dist(shoulder_mid, hip_mid)
    return length if length > 0 else None


def _scanner_frame_interval(timestamps: list[float]) -> float:
    """Median spacing between consecutive scanner frame timestamps."""

    diffs = sorted(b - a for a, b in zip(timestamps, timestamps[1:]) if b > a)
    if not diffs:
        return 0.0
    mid = len(diffs) // 2
    return diffs[mid] if len(diffs) % 2 else (diffs[mid - 1] + diffs[mid]) / 2


def _nearest_within(sorted_ts: list[float], target: float, tol: float) -> int | None:
    """Index of the scanner frame nearest ``target`` within ``tol``, else ``None``."""

    best_i, best_d = None, None
    for i, ts in enumerate(sorted_ts):
        d = abs(ts - target)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    if best_i is not None and best_d is not None and best_d <= tol:
        return best_i
    return None


def _pose_frame_joints(frame: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Scanner keypoints reduced to ``{name: (x, y)}`` over the core joints."""

    out: dict[str, tuple[float, float]] = {}
    for kp in frame.get("keypoints", []) or []:
        name = kp.get("name")
        if name not in COCO_CORE_JOINTS:
            continue
        x, y = kp.get("x"), kp.get("y")
        if x is not None and y is not None:
            out[name] = (float(x), float(y))
    return out


def evaluate_pair(pose_frames: list[dict[str, Any]], truth: TruthDoc) -> dict[str, Any]:
    """Compute PCK@0.5-torso per joint for one pose Run against one truth doc.

    Returns the record body (counts + perJoint); provenance is stamped by the caller.
    """

    scanner_ts = sorted(float(f.get("timestamp", 0.0)) for f in pose_frames)
    by_ts: dict[float, dict[str, Any]] = {float(f.get("timestamp", 0.0)): f
                                          for f in pose_frames}
    interval = _scanner_frame_interval(scanner_ts)
    tol = interval / 2

    per_joint = {j: {"correct": 0, "total": 0} for j in COCO_CORE_JOINTS}
    n_absent = n_torso_undef = n_matched = n_scanner_missing = 0

    for tf in truth.frames:
        if not tf.present or not tf.joints:
            n_absent += 1
            continue
        torso = torso_length(tf.joints)
        if torso is None:
            n_torso_undef += 1
            continue
        idx = _nearest_within(scanner_ts, tf.timestamp, tol)
        if idx is None:
            n_scanner_missing += 1
            continue
        n_matched += 1
        scanner = _pose_frame_joints(by_ts[scanner_ts[idx]])
        threshold = PCK_TORSO_FRACTION * torso
        for name, truth_pt in tf.joints.items():
            per_joint[name]["total"] += 1
            pred = scanner.get(name)  # a thinned scanner joint == a miss
            if pred is not None and _dist(pred, truth_pt) <= threshold:
                per_joint[name]["correct"] += 1

    per_joint_out = {}
    for name, c in per_joint.items():
        total = c["total"]
        per_joint_out[name] = {
            "correct": c["correct"],
            "total": total,
            "pck": (c["correct"] / total) if total else None,
        }

    return {
        "joinToleranceSec": tol,
        "scannerFrameIntervalSec": interval,
        "counts": {
            "truthFramesTotal": len(truth.frames),
            "truthFramesAbsent": n_absent,
            "torsoUndefinedFrames": n_torso_undef,
            "matchedFrames": n_matched,
            "scannerMissingFrames": n_scanner_missing,
        },
        "perJoint": per_joint_out,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _iter_pose_runs(detections_dir: Path):
    """Yield ``(run_ts, pose_frames)`` for every pose file (no dedup — history accretes)."""

    if not detections_dir.is_dir():
        return
    for stem, kinds in _pair_stems(detections_dir).items():
        if "pose" not in kinds:
            continue
        env = _load_json(kinds["pose"])
        data = _unwrap(env)
        run_ts = env.get("run_ts", stem)
        setup_hash = data.get("setupHash", "")
        yield run_ts, setup_hash, data.get("frames", []) or []


def evaluate(analysis_root: Path) -> EvalSummary:
    """Walk the bundle tree, pair every pose Run with truth, write eval records."""

    summary = EvalSummary()

    for video_dir in _iter_video_dirs(analysis_root):
        metadata = _load_json(video_dir / "metadata.json")
        setup_path = video_dir / "setup.json"
        setup = _load_json(setup_path) if setup_path.exists() else {}
        route_folder = metadata.get("route_folder", video_dir.parent.name)
        video_key = metadata.get("video_key", video_dir.name)

        truth = load_truth(video_dir)
        if truth is None:
            summary.truthless_videos.append(f"{route_folder}/{video_key}")
            continue

        # The truth's effective setupHash: its own if it self-reports one (post-#4),
        # else the bundle setup.json it was authored against (ADR 0004).
        effective_setup_hash = truth.setup_hash or setup.get("setupHash", "")
        truth_hash8 = truth.truth_hash[:8]

        for run_ts, pose_setup_hash, pose_frames in _iter_pose_runs(video_dir / "detections"):
            if pose_setup_hash != effective_setup_hash:
                summary.pairings.append(Pairing(
                    route_folder, video_key, run_ts, truth.source, "skipped",
                    reason=(f"setupHash mismatch (run {pose_setup_hash[:8] or '∅'} "
                            f"vs truth {effective_setup_hash[:8] or '∅'})"),
                ))
                continue

            body = evaluate_pair(pose_frames, truth)
            record = {
                "schemaVersion": SCHEMA_VERSION,
                "metric": "PCK@0.5-torso",
                "routeFolder": route_folder,
                "videoKey": video_key,
                "runTs": run_ts,
                "setupHash": effective_setup_hash,
                "truthSource": truth.source,
                "truthHash": truth.truth_hash,
                "truthSetupHashSource": "truth" if truth.setup_hash else "setup.json",
                "jointSet": COCO_CORE_JOINTS,
                **body,
            }
            eval_dir = video_dir / "evaluations"
            eval_dir.mkdir(exist_ok=True)
            record_path = eval_dir / f"{run_ts}_vs_{truth_hash8}.json"
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary.pairings.append(Pairing(
                route_folder, video_key, run_ts, truth.source, "written",
                record_path=record_path))

    return summary
