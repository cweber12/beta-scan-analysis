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

Metrics (v2, issue #8), per joint and pooled: PCK@0.5-torso, median and p90
torso-normalized distance (the p90 catches intermittent tracking blowups that PCK's
threshold flattens), a per-frame presence 2x2 (truth present/absent vs scanner
detected/undetected — a hallucinated pose on a climber-absent frame is a distinct
failure mode from a misplaced wrist), and joint coverage (how often the scanner
emitted each joint at all on climber-present frames; the scanner thins low-score
joints, so a missing joint is a counted signal, not a skip). All distances are
normalized by the **truth** torso length (shoulder-midpoint to hip-midpoint) —
never the scanner's — so a collapsed detection cannot shrink its own scale.

Every record carries two tiers sharing the same pairing work (issue #11 routes the
review provenance from ADR 0004/0005 into them):

- ``auto`` frames are unchallenged ViTPose scaffold — agreement-tier evidence.
  ViTPose auto-detects absence reliably (ADR 0005), so an ``auto`` frame with no
  seeded landmarks is a trustworthy presence negative: a scanner detection there is
  a presence false positive.
- ``human-flagged-wrong`` frames carry a known-bad seed skeleton; comparing against
  it poisons the numbers, so they are excluded from every tier's scoring.
- ``human-flagged-absent`` frames come from a manual-absent button that has been
  removed (ADR 0005). They predate reliable auto-absence, may be stale (a re-seed
  can detect landmarks on a frame that was hand-flagged absent), and no new ones are
  written. They are excluded from every tier's scoring, exactly like
  ``human-flagged-wrong``.

Excluded frames stay in ``truthFramesTotal`` and surface in
``counts.agreementSkipped`` so the record's frame math reconciles;
``counts.review`` reports the per-category breakdown. Legacy ground-truth without a
``review`` field degrades gracefully: every frame is treated as ``auto``. Presence
is always ViTPose's ``state`` — never the manual flag. The accuracy tier is
structurally present but empty: no current review value is a trustworthy human
attestation (second-model verification is issue #12). Never gate on the ground-truth
``verified`` flag — under auto-accept it means "nobody objected".
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
SCHEMA_VERSION = 3

# Ground-truth review provenance vocabulary (ADR 0004 / issue #5). Any value
# outside this set — including a missing field on legacy artifacts — normalizes to
# ``auto``, so old truth degrades gracefully to agreement-tier evidence.
REVIEW_AUTO = "auto"
REVIEW_FLAGGED_WRONG = "human-flagged-wrong"
REVIEW_FLAGGED_ABSENT = "human-flagged-absent"
REVIEW_VOCAB = frozenset({REVIEW_AUTO, REVIEW_FLAGGED_WRONG, REVIEW_FLAGGED_ABSENT})

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

# Low-confidence truth gate (measure-first, "fit thresholds before prefill").
# A truth frame's visible-joint count (non-occluded core joints, i.e. ViTPose was
# confident) is a data-quality proxy: an ``occluded`` joint just means low seed
# ``score``, not geometric occlusion. v1 only *measures* the distribution (the
# ``visibleJoints`` histogram) and lists thin frames in the report worklist; it
# excludes nothing. Setting this to an int activates the gate seam in ``_score_tier``
# — excluding thin frames from PCK/normDist only (presence + coverage are counted
# first) and surfacing them in ``frames.lowVisibility``. Fit N on #15-conforming
# bundles before enabling, so it is not fit on the #34 wrong-subject truth.
MIN_VISIBLE_JOINTS: int | None = None


@dataclass
class TruthFrame:
    """One truth frame reduced to what scoring needs."""

    timestamp: float
    present: bool  # a Climber is present in this frame (scorable)
    joints: dict[str, tuple[float, float]]  # name -> (x, y), present+non-occluded only
    review: str = REVIEW_AUTO  # normalized provenance (ADR 0004)

    @property
    def flagged_wrong(self) -> bool:
        """Human marked the seed pose wrong: known-bad, excluded from scoring."""
        return self.review == REVIEW_FLAGGED_WRONG

    @property
    def flagged_absent(self) -> bool:
        """Deprecated manual absent flag (ADR 0005): untrusted, excluded from scoring."""
        return self.review == REVIEW_FLAGGED_ABSENT

    @property
    def excluded(self) -> bool:
        """Not scored in any tier — a known-bad seed or a deprecated manual absent
        flag. Excluded frames still count in ``truthFramesTotal`` and surface in
        ``counts.agreementSkipped`` so the frame math reconciles."""
        return self.flagged_wrong or self.flagged_absent

    @property
    def verified(self) -> bool:
        """Accuracy-tier eligible — a trustworthy human attestation. Nothing
        qualifies today: ADR 0005 retired manual-absent as evidence and joints are
        never hand-attested, so the accuracy tier stays empty until second-model
        verification lands (issue #12)."""
        return False


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
class Orphan:
    """An on-disk evaluation record whose run no longer pairs and whose truth hash is
    no longer current — a stale-run leftover (issue #32). ``removed`` is True only when
    ``evaluate(prune=True)`` actually deleted it; a dry run reports it without deleting."""

    route_folder: str
    video_key: str
    run_ts: str
    truth_hash8: str
    record_path: Path
    removed: bool = False


@dataclass
class EvalSummary:
    """Everything the CLI needs to print a run summary."""

    pairings: list[Pairing] = field(default_factory=list)
    truthless_videos: list[str] = field(default_factory=list)  # bundles with no truth
    orphans: list[Orphan] = field(default_factory=list)  # stale-run records (issue #32)

    @property
    def written(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "written"]

    @property
    def skipped(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "skipped"]

    @property
    def pruned(self) -> list[Orphan]:
        return [o for o in self.orphans if o.removed]


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
        review = fr.get("review")
        review = review if review in REVIEW_VOCAB else REVIEW_AUTO
        # Presence is always ViTPose's determination (ADR 0005): auto-absence is
        # reliable, and the deprecated manual absent flag never overrides ``state``.
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
        frames.append(TruthFrame(float(fr.get("timestamp", 0.0)), present, joints,
                                 review=review))
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


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile over pre-sorted values (numpy 'linear')."""

    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    rank = q * (n - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (rank - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _round6(v: float | None) -> float | None:
    """Round derived metric values so records are diff-stable across runs."""

    return None if v is None else round(v, 6)


@dataclass
class _FramePair:
    """One truth frame joined (or not) with its nearest-in-tolerance scanner frame."""

    truth: TruthFrame
    matched: bool  # a scanner frame exists within the join tolerance
    scanner: dict[str, tuple[float, float]]  # its core joints; {} when unmatched


def _score_tier(pairs: list[_FramePair]) -> dict[str, Any]:
    """Score one tier (agreement or accuracy) over its share of the frame pairs.

    Distance/PCK need a matched, torso-defined, climber-present frame; coverage
    needs only matched+present (a thinned joint counts against coverage there);
    the presence 2x2 needs only a matched frame. Unmatched frames are counted,
    never silently dropped — sparse scanner sampling is "unobserved", not
    "undetected".

    ``visibleJoints`` is a positional histogram (index i == matched-present frames
    whose truth had i non-occluded core joints; sums to ``matchedPresent``) — the
    measure-first fit input for ``MIN_VISIBLE_JOINTS``. With that gate disabled (v1)
    nothing is excluded and ``frames.lowVisibility`` stays 0.
    """

    frames = {"truthFrames": 0, "verifiedFrames": 0,
              "matchedPresent": 0, "matchedAbsent": 0,
              "unmatchedPresent": 0, "unmatchedAbsent": 0,
              "lowVisibility": 0, "torsoUndefined": 0, "scoreable": 0}
    presence = {"presentDetected": 0, "presentUndetected": 0,
                "absentDetected": 0, "absentUndetected": 0}
    cov = {j: 0 for j in COCO_CORE_JOINTS}
    pck = {j: {"correct": 0, "total": 0} for j in COCO_CORE_JOINTS}
    dists: dict[str, list[float]] = {j: [] for j in COCO_CORE_JOINTS}
    # Visible-joint histogram over matched-present frames: index i == frames whose
    # truth had i non-occluded core joints. Sums to ``matchedPresent``. The fit
    # input for MIN_VISIBLE_JOINTS.
    vis_hist = [0] * (len(COCO_CORE_JOINTS) + 1)

    for p in pairs:
        tf = p.truth
        frames["truthFrames"] += 1
        frames["verifiedFrames"] += tf.verified
        if not p.matched:
            frames["unmatchedPresent" if tf.present else "unmatchedAbsent"] += 1
            continue
        frames["matchedPresent" if tf.present else "matchedAbsent"] += 1
        detected = bool(p.scanner)
        key = ("present" if tf.present else "absent") + \
              ("Detected" if detected else "Undetected")
        presence[key] += 1
        if not tf.present:
            continue
        visible = len(tf.joints)
        vis_hist[visible] += 1
        for j in COCO_CORE_JOINTS:
            cov[j] += j in p.scanner
        # Low-confidence truth gate seam (disabled in v1: MIN_VISIBLE_JOINTS is
        # None). Presence + coverage above have already counted this frame; a thin
        # truth frame is excluded from PCK/normDist only, mirroring torsoUndefined.
        if MIN_VISIBLE_JOINTS is not None and visible < MIN_VISIBLE_JOINTS:
            frames["lowVisibility"] += 1
            continue
        torso = torso_length(tf.joints)
        if torso is None:
            frames["torsoUndefined"] += 1
            continue
        frames["scoreable"] += 1
        for name, truth_pt in tf.joints.items():
            pck[name]["total"] += 1
            pred = p.scanner.get(name)  # a thinned scanner joint == a PCK miss
            if pred is None:
                continue
            d = _dist(pred, truth_pt) / torso
            dists[name].append(d)
            if d <= PCK_TORSO_FRACTION:
                pck[name]["correct"] += 1

    cov_frames = frames["matchedPresent"]
    per_joint: dict[str, Any] = {}
    all_dists: list[float] = []
    agg_correct = agg_total = agg_emitted = 0
    for name in COCO_CORE_JOINTS:
        ds = sorted(dists[name])
        all_dists.extend(ds)
        correct, total = pck[name]["correct"], pck[name]["total"]
        agg_correct, agg_total, agg_emitted = (
            agg_correct + correct, agg_total + total, agg_emitted + cov[name])
        per_joint[name] = {
            "pck": {"correct": correct, "total": total,
                    "value": _round6(correct / total) if total else None},
            "normDist": {"n": len(ds),
                         "median": _round6(_percentile(ds, 0.5)),
                         "p90": _round6(_percentile(ds, 0.9))},
            "coverage": {"emitted": cov[name], "frames": cov_frames,
                         "rate": _round6(cov[name] / cov_frames) if cov_frames else None},
        }

    all_dists.sort()
    agg_cov_frames = cov_frames * len(COCO_CORE_JOINTS)
    return {
        "frames": frames,
        # Positional histogram: index i == matched-present frames whose truth had i
        # non-occluded core joints (len == 14, i.e. 0..13). A list, not a dict, so it
        # stays index-ordered under the record writer's key sorting.
        "visibleJoints": vis_hist,
        "presence": presence,
        "perJoint": per_joint,
        "aggregate": {
            "pck": {"correct": agg_correct, "total": agg_total,
                    "value": _round6(agg_correct / agg_total) if agg_total else None},
            "normDist": {"n": len(all_dists),
                         "median": _round6(_percentile(all_dists, 0.5)),
                         "p90": _round6(_percentile(all_dists, 0.9))},
            "coverage": {"emitted": agg_emitted, "frames": agg_cov_frames,
                         "rate": (_round6(agg_emitted / agg_cov_frames)
                                  if agg_cov_frames else None)},
        },
    }


def evaluate_pair(pose_frames: list[dict[str, Any]], truth: TruthDoc) -> dict[str, Any]:
    """Compute the full metric set for one pose Run against one truth doc.

    Returns the record body (counts + agreement/accuracy tiers); provenance is
    stamped by the caller. Both tiers share the same frame pairing. ``human-flagged-
    wrong`` (known-bad seed) and ``human-flagged-absent`` (deprecated manual flag,
    ADR 0005) frames are excluded from scoring and surface only in
    ``counts.agreementSkipped``. The accuracy tier has no trustworthy attestation
    source yet, so it is present but empty (issue #12).
    """

    scanner_ts = sorted(float(f.get("timestamp", 0.0)) for f in pose_frames)
    by_ts: dict[float, dict[str, Any]] = {float(f.get("timestamp", 0.0)): f
                                          for f in pose_frames}
    interval = _scanner_frame_interval(scanner_ts)
    tol = interval / 2

    pairs: list[_FramePair] = []
    for tf in truth.frames:
        idx = _nearest_within(scanner_ts, tf.timestamp, tol)
        if idx is None:
            pairs.append(_FramePair(tf, False, {}))
        else:
            pairs.append(_FramePair(
                tf, True, _pose_frame_joints(by_ts[scanner_ts[idx]])))

    n_present = sum(1 for p in pairs if p.truth.present)
    n_wrong = sum(1 for p in pairs if p.truth.flagged_wrong)
    n_absent_flag = sum(1 for p in pairs if p.truth.flagged_absent)
    # Both flag classes are excluded from scoring (ADR 0005); accuracy has no
    # trustworthy attestation source yet, so it stays empty.
    agreement_pairs = [p for p in pairs if not p.truth.excluded]
    accuracy_pairs = [p for p in pairs if p.truth.verified]
    return {
        "joinToleranceSec": tol,
        "scannerFrameIntervalSec": interval,
        "counts": {
            "truthFramesTotal": len(pairs),
            "truthFramesPresent": n_present,
            "truthFramesAbsent": len(pairs) - n_present,
            "truthFramesVerified": sum(1 for p in pairs if p.truth.verified),
            "review": {"auto": len(pairs) - n_wrong - n_absent_flag,
                       "flaggedWrong": n_wrong, "flaggedAbsent": n_absent_flag},
            "agreementSkipped": {"flaggedWrong": n_wrong, "flaggedAbsent": n_absent_flag},
        },
        "agreement": _score_tier(agreement_pairs),
        "accuracy": _score_tier(accuracy_pairs),
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


def _parse_record_name(name: str) -> tuple[str, str] | None:
    """Split an ``<run_ts>_vs_<truthHash8>.json`` record name into its parts.

    ``run_ts`` itself contains a hyphen (``20260719-205259``) but never ``_vs_``, so a
    right partition on the separator is unambiguous. Returns ``None`` for any name that
    doesn't fit the pattern — never touch a file we didn't write."""

    if not name.endswith(".json"):
        return None
    run_ts, sep, truth_hash8 = name[:-len(".json")].rpartition("_vs_")
    if not sep or not run_ts or not truth_hash8:
        return None
    return run_ts, truth_hash8


def _prune_orphans(eval_dir: Path, paired_run_ts: set[str], current_truth_hash8: str,
                   route_folder: str, video_key: str, prune: bool) -> list[Orphan]:
    """Find (and, when ``prune``, delete) stale-run orphan records in one bundle.

    A record is an orphan only when **both** its ``run_ts`` no longer pairs this run
    (setupHash-skipped, or the pose file is gone) **and** its ``truthHash8`` is not the
    bundle's current truth hash. A record whose run still pairs is kept even on an older
    truth hash — that is intentional truth-revision history (issue #32 out-of-scope note),
    not an orphan. A live record written this run carries the current hash and is kept."""

    orphans: list[Orphan] = []
    if not eval_dir.is_dir():
        return orphans
    for record_path in sorted(eval_dir.glob("*.json")):
        parsed = _parse_record_name(record_path.name)
        if parsed is None:
            continue
        run_ts, truth_hash8 = parsed
        if run_ts in paired_run_ts or truth_hash8 == current_truth_hash8:
            continue
        removed = False
        if prune:
            record_path.unlink()
            removed = True
        orphans.append(Orphan(route_folder, video_key, run_ts, truth_hash8,
                              record_path, removed))
    return orphans


def evaluate(analysis_root: Path, prune: bool = False) -> EvalSummary:
    """Walk the bundle tree, pair every pose Run with truth, write eval records.

    When ``prune`` is set, also delete stale-run orphan records (issue #32); with it
    unset, orphans are still reported (dry run) but nothing is deleted."""

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
        paired_run_ts: set[str] = set()

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
                "metrics": ["pck@0.5-torso", "normDistMedian", "normDistP90",
                            "presence2x2", "jointCoverage"],
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
            paired_run_ts.add(run_ts)
            summary.pairings.append(Pairing(
                route_folder, video_key, run_ts, truth.source, "written",
                record_path=record_path))

        summary.orphans.extend(_prune_orphans(
            video_dir / "evaluations", paired_run_ts, truth_hash8,
            route_folder, video_key, prune))

    return summary
