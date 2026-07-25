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

from .detector_attempts import (
    DETECTOR_ATTEMPT_STATUSES,
    condition_flags,
    parse_detector_attempts,
    rect_containment,
    rect_iou,
    region_rect,
)
from .discovery import _iter_video_dirs, _load_json, _pair_stems, _unwrap

# Evaluation record schema version. Bump on any record-shape change.
# v4 adds the per-bundle ``conformance`` block (issue #15 gate).
# v5 gives the x axis a looser r² floor (issue #16 — narrow-x-variance false positives).
# v6 adds the per-frame ``frameQuality`` block (issue #44 — auto divergence classes) and
#    the optional ``loosePaired`` flag on best-overlap fallback records. Readers fail open
#    on both (a pre-v6 record simply carries no frameQuality / loosePaired key).
# v7 adds detector-attempt-backed evidence to evaluation records (issue #73). When a
# run carries ``detectorAttempts[]``, scoring and frameQuality use that stream instead
# of dense playback ``frames[]``; frame-only runs keep the legacy fallback behavior.
# v8 splits the neutral repeated-pose diagnostic from the raw-detector stale failure
#    (issue #68): ``heldPose`` is membership in a sustained (>= frozenMinRun)
#    near-identical run, while ``frozenStale`` additionally requires the pose to be
#    direct detector output — legacy ``source == "raw"`` frames, or attempt-backed
#    evidence (detector attempts are raw MediaPipe output by construction).
# v9 scores **rejection correctness** (issue #85): every flip/quality-rejected Detector
#    Attempt's raw pose is compared against the paired truth frame, each frameQuality
#    entry gains ``rejection*`` fields, and ``frameQuality.rejectionCorrectness`` carries
#    the pooled verdict counts + over-rejection rate. Additive and fail-open — a
#    legacy (frames-only) record simply reports zero rejected attempts and a ``None``
#    rate, and pre-v9 readers ignore the new keys.
# v10 adds the per-record ``cropQuality`` block (issue #86): per matched Detector Attempt,
#    the IoU of its search regions against a truth bbox, and for each missing attempt a
#    cause class. Additive and fail-open — a legacy frames-only record carries an empty
#    block, and pre-v10 readers ignore it.
SCHEMA_VERSION = 10

# Ground-truth review provenance vocabulary (ADR 0004 / issue #5). Any value
# outside this set — including a missing field on legacy artifacts — normalizes to
# ``auto``, so old truth degrades gracefully to agreement-tier evidence.
REVIEW_AUTO = "auto"
REVIEW_FLAGGED_WRONG = "human-flagged-wrong"
REVIEW_FLAGGED_ABSENT = "human-flagged-absent"
REVIEW_VOCAB = frozenset({REVIEW_AUTO, REVIEW_FLAGGED_WRONG, REVIEW_FLAGGED_ABSENT})

# Detection-annotation taxonomy (issue #45). Human failure labels reuse the auto
# class names from issue #44; distractors are a separate vocabulary carried by the
# scanner-authored annotation ranges.
DETECTION_FAILURE_CLASSES = frozenset({
    "ok", "wrong-subject", "hallucination-fp", "flipped-rotated", "distorted",
})
DETECTION_DISTRACTORS = frozenset({
    "tree_bush", "rock_wall_shape", "crash_pad_bag", "animal", "shadow",
    "spectator", "hallucination_none", "gear", "other",
})

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

# Conformance gate (issue #15). Per axis, fit ``scanner = a·truth + b`` (OLS) over
# every matched scanner↔truth core-joint point in the bundle, then quarantine the
# bundle from *pooled* metrics when the fit is not near-identity. This catches
# per-bundle **truth mis-tracking** — the ViTPose appearance-stitch (#19) latching
# onto the wrong subject — which shows up as scattered slopes / low r² even while
# route-siblings fit clean, and which PCK alone can't distinguish from ordinary
# detector error. The #15 audit fit 26 clean bundles at a≈0.97–0.99, r²≈0.98–1.00;
# the 12 contaminated ones (#34) fall outside. Thresholds are deliberately loose so
# only genuine mis-tracking trips them. Per-record (a run×truth pairing), but the
# verdict is a truth property, so a bundle's runs agree. A near-degenerate fit
# (too few points, or a constant/zero-variance axis) can't be trusted → non-conforming.
#
# The r² floor is **asymmetric** (issue #16). A climber's horizontal spread is narrow
# relative to their vertical extent, so truth-x variance is small and x-r²
# (explained/total variance) is dragged under 0.90 by ordinary per-joint noise even
# when the x-slope sits right at identity and y fits clean — a false quarantine, not
# mis-tracking. So x uses a looser r² floor (0.75) while y keeps 0.90; the slope band
# stays symmetric on both axes and is what actually catches wrong-subject truth
# (scattered slopes / r²≈0 on *both* axes). The x-only borderline bundles fit clean-y
# at x-r² 0.79–0.87; genuine mis-tracking sits at x-r² ≤0.56 — well below 0.75.
CONFORMANCE_SLOPE_MIN = 0.85
CONFORMANCE_SLOPE_MAX = 1.15
CONFORMANCE_R2_MIN = 0.90  # y-axis floor
CONFORMANCE_R2_MIN_X = 0.75  # x-axis floor (narrow horizontal variance, issue #16)
CONFORMANCE_MIN_POINTS = 20

# Best-overlap pairing fallback (issue #44 deliverable 4). A *trusted* pairing needs a
# setupHash-matching pose Run that actually overlaps the truth timeline; a matching Run
# that samples a disjoint time span pairs to n=0 (the ``IE4T94qX55g`` case) and yields no
# usable per-frame evidence. When no setupHash-matched Run reaches this many matched,
# non-excluded present frames, ``evaluate`` falls back to the Run with the most timestamp
# overlap *regardless of setupHash*, stamps the record ``loosePaired: true``, and keeps it
# out of trusted pooling. A loose record exists only for the per-frame quality worklist +
# crops (issue #44 deliverables 1–3) — never for the trusted metrics, which stay
# setupHash-gated and conforming-only.
LOOSE_PAIR_MIN_OVERLAP = 3

# Targeted evaluate modes (issue #57). ``all`` is the default full sweep — it (re)writes
# a record for every setupHash-matched Run in every bundle, exactly as evaluate always
# did, so the default is behaviourally unchanged. ``un-analyzed`` is the incremental mode:
# a bundle whose setupHash-matched Runs already carry a current-truth record on disk is
# skipped wholesale, avoiding redundant re-scoring. The gate is deliberately per-bundle
# and coarse — a bundle with *no* setupHash-matched Run is never treated as analyzed, so it
# always reprocesses and the best-overlap loose fallback (issue #44) fires identically
# under both modes. See ``_bundle_already_analyzed``.
EVAL_MODE_ALL = "all"
EVAL_MODE_UNANALYZED = "un-analyzed"
EVAL_MODES = (EVAL_MODE_ALL, EVAL_MODE_UNANALYZED)

# Per-frame detection-quality classification (issue #44 deliverable 1). Each matched
# frame on which the scanner emitted a pose is sorted into one auto class from the
# scanner↔truth geometry, all distances normalized by the *truth* torso length (never the
# scanner's — a collapsed detection must not shrink its own scale, mirroring the PCK
# metric). Translation and shape are separated: ``centroidDist`` is the mean joint offset
# (pure displacement) and ``residual`` is the median joint offset *after removing that mean*
# (pure shape distortion), so "right shape, wrong place" (wrong-subject) is distinguished
# from "right place, wrong shape" (distorted).
#
# THRESHOLDS ARE PROVISIONAL. They are hand-set engineering estimates, not yet fit against
# the #42 manually-verified bundles (which this backend slice does not have on disk). They
# mirror the #16 ``CONFORMANCE_*`` / #23 ``SUGGESTION_THRESHOLDS`` provenance pattern and
# are echoed into every record's ``frameQuality.thresholds`` so a record captures the gate
# it was classified under. Re-fit against the #42 labels before treating the classes as
# ground truth (measure-first, as with ``MIN_VISIBLE_JOINTS``).
FQ_WRONG_SUBJECT_CENTROID = 1.0  # centroid ≥ 1 truth-torso off → locked on the wrong subject
FQ_DISTORT_RESIDUAL = 0.5        # median shape residual ≥ 0.5 torso → joints scattered
FQ_FLIP_RESIDUAL = 0.25          # a vertical flip that drops shape residual below this → flipped
FQ_FROZEN_EPS = 0.005            # max keypoint move (normalized image coords) between two
#                                  adjacent detected frames → the pair is "static"
FQ_FROZEN_MIN_RUN = 3            # a static frame is frozen/stale only inside a maximal run
#                                  of ≥ this many consecutive near-identical detected poses.
#                                  A genuine stale overlay (lost tracking, repeating the last
#                                  pose) is sustained; a single low-motion step is a paused
#                                  climber, not a freeze — that distinction is why the old
#                                  per-frame test fired on ~3/4 of frames (issue #68).

FQ_OK = "ok"
FQ_WRONG_SUBJECT = "wrong-subject"
FQ_HALLUCINATION = "hallucination-fp"
FQ_FLIPPED = "flipped-rotated"
FQ_DISTORTED = "distorted"
FQ_CLASSES = [FQ_OK, FQ_WRONG_SUBJECT, FQ_HALLUCINATION, FQ_FLIPPED, FQ_DISTORTED]

# Rejection correctness (issue #85). The scanner's flip / quality gates discard raw
# MediaPipe poses it judged wrong; the harness is the only side that can check that
# judgement, because only it holds Ground Truth. Each rejected Detector Attempt's
# ``rawKeypoints`` are scored against the paired truth frame and the rejection gets one
# verdict:
#
# - ``goodPoseRejected`` — the discarded raw pose actually agreed with truth, so the
#   gate over-rejected. This is the defect the metric exists to measure: the 2026-07-24
#   corpus rejected ~71% of truth-checkable flip rejections on plausibly-good poses.
# - ``badPoseRejected`` — the raw pose diverged from truth (or landed on a Climber-absent
#   frame, where any pose is wrong), so the gate was right to discard it.
# - ``truthUnknown`` — not checkable: the attempt carried no raw pose, or the truth frame
#   has no usable geometry (undefined torso, or no core joint shared with the raw pose).
#
# The geometry threshold is *not* new — a rejected pose counts as good exactly when
# ``_classify_detection`` (the issue #44 ``FQ_*`` constants) would have called it ``ok``,
# i.e. the scanner would have been right to keep it. The one added gate is the PCK-style
# joint-agreement floor below, which reuses ``PCK_TORSO_FRACTION`` for the per-joint
# radius and only fixes *how many* joints must agree — a majority — so that the loose
# 1.0-torso wrong-subject band can't pass a pose whose joints scatter individually.
REJECTION_STATUSES = frozenset({"flipRejected", "qualityRejected"})
REJECTION_MIN_JOINT_AGREEMENT = 0.5

REJECTION_GOOD = "goodPoseRejected"
REJECTION_BAD = "badPoseRejected"
REJECTION_UNKNOWN = "truthUnknown"
REJECTION_VERDICTS = [REJECTION_GOOD, REJECTION_BAD, REJECTION_UNKNOWN]

# Why a rejection landed on its verdict — auditability, not a separate taxonomy.
REJECTION_REASON_NO_RAW = "no-raw-pose"
REJECTION_REASON_TRUTH_ABSENT = "truth-absent"
REJECTION_REASON_UNGEOMETRIC = "truth-ungeometric"
REJECTION_REASON_AGREES = "raw-pose-agrees-truth"
REJECTION_REASON_DIVERGES = "raw-pose-diverges-truth"

# Crop quality and miss causes (issue #86). The scanner's Adaptive Crop decides *where*
# MediaPipe looks; the harness owns the truth bbox it should have looked at, so crop
# placement is only measurable backend-side. Truth bbox = the extent of the truth core
# joints, padded because 13 joints do not span a Climber's silhouette (no crown of the
# head, no hands beyond the wrists, no feet beyond the ankles). The pad is a fraction of
# the extent's *larger* side so a Climber flat against the wall — near-zero horizontal
# extent — still gets a usable box on both axes.
#
# THRESHOLDS ARE PROVISIONAL, in the ``FQ_*`` / ``CONFORMANCE_*`` tradition: hand-set
# engineering estimates echoed into every record's ``cropQuality.thresholds`` so a record
# captures the gate it was scored under. Re-fit before treating the classes as truth.
TRUTH_BBOX_PAD = 0.10          # pad each side by this fraction of the extent's larger side
CROP_CONTAINMENT_MIN = 0.5     # the searched region must hold this share of the truth bbox

# Whether a full-frame reacquire ran is decisive for miss causation, so it gets a named
# constant rather than being buried in a condition. CONTEXT.md contracts ``reacquire`` as
# a **full-frame** search, and a missing attempt reports no ``detectionRegion`` to confirm
# it from, so ``reacquireAttempted`` is what we trust. Flip this to False only if the
# scanner ever gains a non-full-frame reacquire, at which point the region must be scored
# instead of assumed.
REACQUIRE_SEARCHES_FULL_FRAME = True

# Miss causes. Ordered most-decisive first; ``_miss_cause`` returns the first that fits.
MISS_CLIMBER_ABSENT = "climber-absent"      # truth says no Climber — a correct miss
MISS_CROP_MISPLACED = "crop-misplaced"      # the only searched region excluded the Climber
MISS_ADVERSE_CONDITIONS = "adverse-conditions"  # everything searched, condition flags fired
MISS_UNEXPLAINED = "unexplained"            # everything searched, conditions clean, still lost
MISS_CAUSES = [MISS_CLIMBER_ABSENT, MISS_CROP_MISPLACED, MISS_ADVERSE_CONDITIONS,
               MISS_UNEXPLAINED]


@dataclass
class TruthFrame:
    """One truth frame reduced to what scoring needs."""

    frame_index: int | None
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
    detection_annotations: list["DetectionAnnotation"] = field(default_factory=list)


@dataclass
class DetectionAnnotation:
    """One scanner-authored annotation range over ground-truth frame indices."""

    start_frame: int
    end_frame: int
    failure_class: str
    distractor: str
    setup_hash: str


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
    loose: bool = False  # a best-overlap fallback pairing (issue #44 deliverable 4)


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
    analyzed_skipped: list[str] = field(default_factory=list)  # un-analyzed mode skips (#57)

    @property
    def written(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "written"]

    @property
    def loose(self) -> list[Pairing]:
        return [p for p in self.pairings if p.status == "written" and p.loose]

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
        frame_index = fr.get("frameIndex")
        frame_index = int(frame_index) if isinstance(frame_index, int) else None
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
        frames.append(TruthFrame(frame_index, float(fr.get("timestamp", 0.0)), present,
                                 joints, review=review))

    annotations: list[DetectionAnnotation] = []
    for ann in doc.get("detectionAnnotations", []) or []:
        if not isinstance(ann, dict):
            continue
        start = ann.get("startFrame")
        end = ann.get("endFrame")
        failure_class = ann.get("failureClass")
        distractor = ann.get("distractor")
        setup_hash = str(ann.get("setupHash") or "")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start > end:
            continue
        if failure_class not in DETECTION_FAILURE_CLASSES:
            continue
        if distractor not in DETECTION_DISTRACTORS:
            continue
        if not setup_hash:
            continue
        annotations.append(DetectionAnnotation(
            start_frame=int(start),
            end_frame=int(end),
            failure_class=str(failure_class),
            distractor=str(distractor),
            setup_hash=setup_hash,
        ))
    truth_hash = doc.get("groundTruthHash") or _content_hash(doc)
    return TruthDoc("ground-truth", doc.get("setupHash") or "", truth_hash, frames,
                    detection_annotations=annotations)


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
        frame_index = fr.get("frameIndex")
        frame_index = int(frame_index) if isinstance(frame_index, int) else None
        frames.append(TruthFrame(frame_index, float(fr.get("timestamp", 0.0)), present,
                                 joints))
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


def _detection_annotation_for_frame(
    truth: TruthDoc,
    frame_index: int | None,
    setup_hash: str,
) -> DetectionAnnotation | None:
    """Resolve the active annotation for one truth frame, if any.

    Annotations are setupHash-stamped and frame-index ranges are inclusive. A stale
    setupHash is ignored; when multiple valid ranges overlap, the last matching one
    wins so later manual refinement can override earlier annotations deterministically.
    """

    if frame_index is None or not setup_hash:
        return None
    active: DetectionAnnotation | None = None
    for ann in truth.detection_annotations:
        if ann.setup_hash != setup_hash:
            continue
        if ann.start_frame <= frame_index <= ann.end_frame:
            active = ann
    return active


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


def _centroid(joints: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Mean (x, y) over a joint dict, or ``None`` when empty."""

    if not joints:
        return None
    n = len(joints)
    return (sum(p[0] for p in joints.values()) / n,
            sum(p[1] for p in joints.values()) / n)


def _head_below_hips(joints: dict[str, tuple[float, float]]) -> bool:
    """True when the scanner's nose sits below its hip midpoint (image y grows
    downward), the tell-tale of an upside-down / flipped pose — needs only the
    scanner geometry, so it fires even before a torso-normalized comparison."""

    nose = joints.get("nose")
    lh, rh = joints.get("left_hip"), joints.get("right_hip")
    if nose is None or lh is None or rh is None:
        return False
    return nose[1] > (lh[1] + rh[1]) / 2


def _centroid_and_residual(truth: dict[str, tuple[float, float]],
                           scanner: dict[str, tuple[float, float]],
                           shared: list[str], torso: float | None
                           ) -> tuple[float | None, float | None]:
    """(centroidDist, residual) in truth-torso units over the shared joints.

    ``centroidDist`` is the magnitude of the mean scanner−truth offset (pure
    translation); ``residual`` is the median per-joint offset *after* removing that
    mean (pure shape distortion). Both ``None`` when the torso is undefined or no
    joint is shared."""

    if torso is None or not shared:
        return None, None
    dxs = [scanner[j][0] - truth[j][0] for j in shared]
    dys = [scanner[j][1] - truth[j][1] for j in shared]
    mdx, mdy = sum(dxs) / len(dxs), sum(dys) / len(dys)
    centroid_dist = math.hypot(mdx, mdy) / torso
    resids = sorted(math.hypot(dx - mdx, dy - mdy) for dx, dy in zip(dxs, dys))
    residual = (_percentile(resids, 0.5) or 0.0) / torso
    return centroid_dist, residual


def _flip_residual(truth: dict[str, tuple[float, float]],
                   scanner: dict[str, tuple[float, float]],
                   shared: list[str], torso: float | None) -> float | None:
    """Shape residual after reflecting the scanner pose vertically about its own
    centroid — small when a vertical flip would align the pose with truth."""

    if torso is None or not shared:
        return None
    scy = sum(scanner[j][1] for j in shared) / len(shared)
    flipped = {j: (scanner[j][0], 2 * scy - scanner[j][1]) for j in shared}
    _, residual = _centroid_and_residual(truth, flipped, shared, torso)
    return residual


def _classify_detection(truth: dict[str, tuple[float, float]],
                        scanner: dict[str, tuple[float, float]], torso: float | None
                        ) -> tuple[str, float | None, float | None]:
    """Classify one scanner-detected, truth-present frame → (class, centroidDist,
    residual). Order matters: flip is checked first (an upside-down pose can otherwise
    read as wrong-subject or distorted), then gross displacement, then shape scatter."""

    shared = [j for j in truth if j in scanner]
    centroid_dist, residual = _centroid_and_residual(truth, scanner, shared, torso)
    if _head_below_hips(scanner):
        return FQ_FLIPPED, centroid_dist, residual
    if torso is None or not shared:
        return FQ_OK, centroid_dist, residual  # unnormalizable — presence/coverage cover it
    flip_resid = _flip_residual(truth, scanner, shared, torso)
    if (residual is not None and residual >= FQ_DISTORT_RESIDUAL
            and flip_resid is not None and flip_resid <= FQ_FLIP_RESIDUAL):
        return FQ_FLIPPED, centroid_dist, residual
    if centroid_dist is not None and centroid_dist >= FQ_WRONG_SUBJECT_CENTROID:
        return FQ_WRONG_SUBJECT, centroid_dist, residual
    if residual is not None and residual >= FQ_DISTORT_RESIDUAL:
        return FQ_DISTORTED, centroid_dist, residual
    return FQ_OK, centroid_dist, residual


def _is_frozen(cur: dict[str, tuple[float, float]],
               prev: dict[str, tuple[float, float]] | None) -> bool:
    """True when ``cur`` is near-identical to the previous detected pose ``prev`` — every
    shared joint moved ``≤ FQ_FROZEN_EPS`` in normalized image coords. This is the
    pairwise "static" primitive; sustained-freeze detection is layered on top by
    ``_frozen_flags`` (issue #68)."""

    if not prev:
        return False
    shared = [j for j in cur if j in prev]
    if not shared:
        return False
    return max(_dist(cur[j], prev[j]) for j in shared) <= FQ_FROZEN_EPS


def _frozen_flags(poses: list[dict[str, tuple[float, float]]],
                  min_run: int = FQ_FROZEN_MIN_RUN) -> list[bool]:
    """Per-frame held-pose repeat flags over detected poses in timestamp order.

    A frame is held only when it is a non-anchor repeat inside a maximal run of
    ``≥ min_run`` consecutive near-identical poses (each adjacent pair ``_is_frozen``).
    The run anchor is the fresh pose and is not stale; source provenance later decides
    whether the held pose is a raw-detector stale failure or benign reconstruction."""

    n = len(poses)
    flags = [False] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and _is_frozen(poses[j + 1], poses[j]):
            j += 1
        if j - i + 1 >= min_run:  # frames i..j form one near-identical run
            for k in range(i + 1, j + 1):
                flags[k] = True
        i = j + 1
    return flags


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

    return _keypoint_joints(frame.get("keypoints", []) or [])


def _keypoint_joints(keypoints: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Keypoint list reduced to ``{name: (x, y)}`` over the core joints."""

    out: dict[str, tuple[float, float]] = {}
    for kp in keypoints:
        if not isinstance(kp, dict):
            continue
        name = kp.get("name")
        if name not in COCO_CORE_JOINTS:
            continue
        x, y = kp.get("x"), kp.get("y")
        if x is not None and y is not None:
            out[name] = (float(x), float(y))
    return out


def _scanner_observations(
    pose_frames: list[dict[str, Any]],
    detector_attempts: list[dict[str, Any]] | None,
) -> list[_ScannerObservation]:
    """Evaluation evidence for one Run.

    Detector Attempts are the backend-owned detector stream. When present, accepted
    attempts contribute their accepted keypoints to scoring while missing/rejected
    attempts contribute a matched detector observation with no accepted pose. Dense
    frames are used only for legacy runs where the attempt stream is absent.
    """

    if detector_attempts is not None:
        observations: list[_ScannerObservation] = []
        for attempt in detector_attempts:
            status = attempt.get("status")
            accepted = attempt.get("acceptedKeypoints") or []
            scanner = _keypoint_joints(accepted) if status == "accepted" else {}
            observations.append(_ScannerObservation(
                float(attempt.get("timestamp", 0.0)),
                scanner,
                detector_attempt=attempt,
            ))
        return observations

    return [
        _ScannerObservation(float(f.get("timestamp", 0.0)), _pose_frame_joints(f),
                            source=f.get("source"))
        for f in pose_frames
    ]


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


def _ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Ordinary least squares ``y = slope·x + intercept`` with r². Returns
    ``(slope, intercept, r2)``, or ``None`` when the fit is degenerate: fewer than
    two points, or zero variance on either axis (a vertical/constant relationship
    has no meaningful slope or r²). Hand-rolled — the math is trivial and the
    ``analysis_pipeline`` default stays numpy-free (ADR 0003 code-quality note)."""

    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    r2 = (sxy * sxy) / (sxx * syy)
    return slope, intercept, r2


def _axis_r2_min(axis: str) -> float:
    """The r² floor for an axis: looser on x (narrow horizontal variance, issue #16)."""

    return CONFORMANCE_R2_MIN_X if axis == "x" else CONFORMANCE_R2_MIN


def _axis_conforms(fit: tuple[float, float, float] | None, axis: str) -> bool:
    """One axis passes the #15 gate: a non-degenerate near-identity fit. The r² floor
    is looser on x than y (issue #16); the slope band is the same on both."""

    if fit is None:
        return False
    slope, _intercept, r2 = fit
    return (CONFORMANCE_SLOPE_MIN <= slope <= CONFORMANCE_SLOPE_MAX
            and r2 >= _axis_r2_min(axis))


def _axis_block(fit: tuple[float, float, float] | None) -> dict[str, Any]:
    if fit is None:
        return {"slope": None, "intercept": None, "r2": None}
    slope, intercept, r2 = fit
    return {"slope": _round6(slope), "intercept": _round6(intercept), "r2": _round6(r2)}


@dataclass
class _FramePair:
    """One truth frame joined (or not) with its nearest-in-tolerance scanner frame."""

    truth: TruthFrame
    matched: bool  # a scanner frame exists within the join tolerance
    scanner: dict[str, tuple[float, float]]  # its core joints; {} when unmatched
    scanner_source: str | None = None  # scanner pose ``source`` provenance, when exported
    detector_attempt: dict[str, Any] | None = None


@dataclass
class _ScannerObservation:
    """One timestamped scanner evidence item used by evaluation pairing."""

    timestamp: float
    scanner: dict[str, tuple[float, float]]
    source: str | None = None  # legacy frames[] ``source`` provenance, when exported
    detector_attempt: dict[str, Any] | None = None


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


def _conformance(pairs: list[_FramePair]) -> dict[str, Any]:
    """Per-axis identity fit of scanner onto truth over the bundle's matched joints.

    Pools every core-joint point on a matched, climber-present, non-excluded frame
    into two OLS fits (``scanner_x = a·truth_x + b`` and the y counterpart) and judges
    the bundle against the near-identity band (issue #15). This is a whole-bundle
    sanity check on the truth↔scanner coordinate relationship — a mis-tracked truth
    scatters the fit even where PCK looks plausible — not a per-joint accuracy metric.
    ``conforms`` gates the bundle out of *pooled* metrics; the per-record tiers stay
    computed either way, so a quarantined bundle is still inspectable.

    ``n`` is the point count per axis. Below ``CONFORMANCE_MIN_POINTS`` the fit is too
    thin to trust, so the bundle is non-conforming with an ``insufficient-points``
    reason rather than a spurious pass. ``reasons`` is empty exactly when ``conforms``.
    """

    tx: list[float] = []
    sx: list[float] = []
    ty: list[float] = []
    sy: list[float] = []
    for p in pairs:
        if not p.matched or not p.truth.present:
            continue
        for name, truth_pt in p.truth.joints.items():
            pred = p.scanner.get(name)
            if pred is None:
                continue
            tx.append(truth_pt[0])
            sx.append(pred[0])
            ty.append(truth_pt[1])
            sy.append(pred[1])

    n = len(tx)
    fit_x = _ols_fit(tx, sx)
    fit_y = _ols_fit(ty, sy)

    reasons: list[str] = []
    if n < CONFORMANCE_MIN_POINTS:
        reasons.append("insufficient-points")
    for axis, fit in (("x", fit_x), ("y", fit_y)):
        if not _axis_conforms(fit, axis):
            reasons.append(f"{axis}-nonconforming")
    conforms = not reasons

    return {
        "x": _axis_block(fit_x),
        "y": _axis_block(fit_y),
        "n": n,
        "conforms": conforms,
        "reasons": reasons,
        "thresholds": {
            "slopeMin": CONFORMANCE_SLOPE_MIN,
            "slopeMax": CONFORMANCE_SLOPE_MAX,
            "r2Min": CONFORMANCE_R2_MIN,  # y-axis floor
            "r2MinX": CONFORMANCE_R2_MIN_X,  # x-axis floor (issue #16)
            "minPoints": CONFORMANCE_MIN_POINTS,
        },
    }


def _attempt_status_counts(pairs: list[_FramePair]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(DETECTOR_ATTEMPT_STATUSES)}
    counts["unknown"] = 0
    for p in pairs:
        attempt = p.detector_attempt
        if not p.matched or attempt is None:
            continue
        status = attempt.get("status")
        counts[status if status in DETECTOR_ATTEMPT_STATUSES else "unknown"] += 1
    return counts


def _attempt_raw_joints(attempt: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    if not isinstance(attempt, dict):
        return {}
    return _keypoint_joints(attempt.get("rawKeypoints") or [])


def _attempt_keypoint_payload(
    attempt: dict[str, Any] | None,
    key: str,
) -> list[dict[str, Any]] | None:
    if not isinstance(attempt, dict):
        return None
    value = attempt.get(key)
    return value if isinstance(value, list) else []


def _quality_candidate(p: _FramePair) -> dict[str, tuple[float, float]]:
    if p.scanner:
        return p.scanner
    status = (p.detector_attempt or {}).get("status")
    if status in {"flipRejected", "qualityRejected"}:
        return _attempt_raw_joints(p.detector_attempt)
    return {}


def _joint_agreement(truth: dict[str, tuple[float, float]],
                     scanner: dict[str, tuple[float, float]],
                     torso: float | None) -> float | None:
    """PCK-style share of truth core joints whose scanner counterpart lands within
    ``PCK_TORSO_FRACTION`` truth-torso lengths. A thinned joint counts as a miss, exactly
    as in ``_score_tier``. ``None`` when the frame is unnormalizable or truth is empty."""

    if torso is None or not truth:
        return None
    hits = 0
    for name, truth_pt in truth.items():
        pred = scanner.get(name)
        if pred is not None and _dist(pred, truth_pt) / torso <= PCK_TORSO_FRACTION:
            hits += 1
    return hits / len(truth)


def _rejection_scoring(p: _FramePair) -> dict[str, Any] | None:
    """Score one flip/quality-rejected Detector Attempt against truth (issue #85).

    Returns ``None`` when the frame is not a rejection at all — an accepted, missing,
    unknown-status, unmatched or legacy-frames frame has nothing to second-guess. For a
    rejection, returns the verdict plus the geometry it was decided on, so a record
    carries the evidence and not just the label.

    A rejection is ``goodPoseRejected`` only when the discarded raw pose both classifies
    as ``ok`` under the issue #44 geometry *and* clears the joint-agreement floor; a
    Climber-absent frame is ``badPoseRejected`` (there is no correct pose to keep there).
    """

    attempt = p.detector_attempt
    status = attempt.get("status") if isinstance(attempt, dict) else None
    if not p.matched or status not in REJECTION_STATUSES:
        return None

    def out(verdict: str, reason: str, *, centroid: float | None = None,
            residual: float | None = None, agreement: float | None = None,
            raw_class: str | None = None) -> dict[str, Any]:
        return {
            "verdict": verdict,
            "reason": reason,
            "centroidDist": _round6(centroid),
            "residual": _round6(residual),
            "jointAgreement": _round6(agreement),
            "rawClass": raw_class,
        }

    raw = _attempt_raw_joints(attempt)
    if not raw:
        # A rejection with no raw pose is self-contradictory evidence, not a judgement
        # we can check — count it, never guess it.
        return out(REJECTION_UNKNOWN, REJECTION_REASON_NO_RAW)

    tf = p.truth
    if not tf.present:
        return out(REJECTION_BAD, REJECTION_REASON_TRUTH_ABSENT,
                   raw_class=FQ_HALLUCINATION)

    torso = torso_length(tf.joints)
    if torso is None or not any(j in raw for j in tf.joints):
        return out(REJECTION_UNKNOWN, REJECTION_REASON_UNGEOMETRIC)

    raw_class, centroid, residual = _classify_detection(tf.joints, raw, torso)
    agreement = _joint_agreement(tf.joints, raw, torso)
    good = (raw_class == FQ_OK and agreement is not None
            and agreement >= REJECTION_MIN_JOINT_AGREEMENT)
    return out(
        REJECTION_GOOD if good else REJECTION_BAD,
        REJECTION_REASON_AGREES if good else REJECTION_REASON_DIVERGES,
        centroid=centroid, residual=residual, agreement=agreement,
        raw_class=raw_class)


def _empty_verdict_counts() -> dict[str, int]:
    # ``truthAbsent`` is a *subset* of ``badPoseRejected``, tracked alongside the verdicts
    # so the truth-present-only rate below can be derived without a second pass.
    return {**{v: 0 for v in REJECTION_VERDICTS}, "truthAbsent": 0}


def _rejection_rate_block(counts: dict[str, int]) -> dict[str, Any]:
    """Verdict counts → the pooled shape shared by the record's total and per-status
    blocks.

    Two rates, because the denominator is a genuine judgement call:

    - ``overRejectionRate`` is over every *truth-checkable* rejection (good + bad).
      ``truthUnknown`` rejections have no verdict to average, so folding them into the
      denominator would dilute the rate with unmeasured frames.
    - ``overRejectionRateTruthPresent`` additionally drops the Climber-absent rejections.
      Those are correct by construction (no pose belongs on an absent frame), so they
      pull the pooled rate down without saying anything about the *gate's* geometry
      judgement. This is the number comparable to the ad-hoc 2026-07-24 corpus baseline
      (~71% of truth-checkable flip rejections), which treated Climber-absent rejections
      as unknown rather than as correct.
    """

    good, bad, absent = (counts[REJECTION_GOOD], counts[REJECTION_BAD],
                         counts["truthAbsent"])
    checkable = good + bad
    present_checkable = checkable - absent
    return {
        "rejected": good + bad + counts[REJECTION_UNKNOWN],
        "verdictCounts": {v: counts[v] for v in REJECTION_VERDICTS},
        "truthAbsent": absent,
        "truthCheckable": checkable,
        "truthPresentCheckable": present_checkable,
        "overRejectionRate": _round6(good / checkable) if checkable else None,
        "overRejectionRateTruthPresent": (
            _round6(good / present_checkable) if present_checkable else None),
    }


def _rejection_correctness(pairs: list[_FramePair]) -> dict[str, Any]:
    """Pooled rejection correctness for one Run (issue #85).

    Counts every rejected Detector Attempt in the Run, pooled and split by rejection
    status, so the flip gate and the quality gate are measurable independently (the
    corpus baseline is a *flip*-rejection rate). Legacy frames-only Runs carry no
    rejections and report zeros with a ``None`` rate — fail-open, not absent."""

    totals = _empty_verdict_counts()
    by_status = {s: _empty_verdict_counts() for s in sorted(REJECTION_STATUSES)}
    for p in pairs:
        scored = _rejection_scoring(p)
        if scored is None:
            continue
        status = (p.detector_attempt or {}).get("status")
        for counts in (totals, by_status[status]):
            counts[scored["verdict"]] += 1
            counts["truthAbsent"] += scored["reason"] == REJECTION_REASON_TRUTH_ABSENT

    return {
        **_rejection_rate_block(totals),
        "byStatus": {s: _rejection_rate_block(c) for s, c in by_status.items()},
        "thresholds": {
            "minJointAgreement": REJECTION_MIN_JOINT_AGREEMENT,
            "pckTorsoFraction": PCK_TORSO_FRACTION,
            "wrongSubjectCentroid": FQ_WRONG_SUBJECT_CENTROID,
            "distortResidual": FQ_DISTORT_RESIDUAL,
            "flipResidual": FQ_FLIP_RESIDUAL,
        },
    }


def truth_bbox(joints: dict[str, tuple[float, float]]
               ) -> tuple[float, float, float, float] | None:
    """The padded extent of one truth frame's core joints as ``(x0, y0, x1, y1)``.

    ``None`` when the frame has no joints. Not clamped to the frame: a Climber part-way
    out of shot has a box that overhangs, and clamping would silently make a crop that
    missed them look like it covered them."""

    if not joints:
        return None
    xs = [p[0] for p in joints.values()]
    ys = [p[1] for p in joints.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = max(x1 - x0, y1 - y0) * TRUTH_BBOX_PAD
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _bbox_block(bbox: tuple[float, float, float, float] | None) -> dict[str, Any] | None:
    """A truth bbox in the same ``{x, y, w, h}`` shape the scanner uses for regions, so
    a reader can compare the two without knowing which corner convention each used."""

    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return {"x": _round6(x0), "y": _round6(y0),
            "w": _round6(x1 - x0), "h": _round6(y1 - y0)}


def _miss_cause(p: _FramePair, bbox: tuple[float, float, float, float] | None,
                containment: float | None, flags_fired: bool,
                reacquire_attempted: bool) -> str:
    """Why one missing Detector Attempt found no Climber (issue #86).

    The ordering encodes what the evidence can actually support. ``crop-misplaced`` is a
    claim about *causation*, so it requires that the misplaced crop was the only place the
    scanner looked: when a full-frame reacquire also ran and still failed, the Climber was
    searched for everywhere, and the crop cannot be what lost them — however badly placed
    it was. Crop placement is still measured on every miss (``initialCropContainment``);
    it is just not allowed to masquerade as the cause.
    """

    if not p.truth.present:
        return MISS_CLIMBER_ABSENT
    searched_everywhere = reacquire_attempted and REACQUIRE_SEARCHES_FULL_FRAME
    if (not searched_everywhere and bbox is not None and containment is not None
            and containment < CROP_CONTAINMENT_MIN):
        return MISS_CROP_MISPLACED
    if flags_fired:
        return MISS_ADVERSE_CONDITIONS
    return MISS_UNEXPLAINED


def _crop_quality(pairs: list[_FramePair]) -> dict[str, Any]:
    """Crop placement and miss causes for one Run (issue #86).

    One entry per matched Detector Attempt — every status, unlike ``frameQuality``, which
    only records frames where a pose was emitted. A miss is exactly the case this block
    exists to explain, so it cannot borrow that population.

    Per attempt: the truth bbox, the IoU of ``initialSearchRegion`` and ``detectionRegion``
    against it, and the share of the bbox each region contained. IoU answers "did the crop
    frame the Climber well", containment answers "did it cover them at all" — a large but
    correctly-placed crop scores poorly on the first and perfectly on the second, and
    conflating them would read crop *size* as crop *error*. Missing attempts additionally
    carry a cause class.

    Legacy frames-only Runs have no attempts, so the block is present but empty — an
    unmeasured Run must not read as a Run with zero misplaced crops."""

    entries: list[dict[str, Any]] = []
    cause_counts = {c: 0 for c in MISS_CAUSES}
    initial_ious: list[float] = []
    detection_ious: list[float] = []
    contained = 0
    containment_scored = 0

    for p in sorted((p for p in pairs if p.matched and p.detector_attempt is not None),
                    key=lambda p: p.truth.timestamp):
        attempt = p.detector_attempt or {}
        status = attempt.get("status")
        bbox = truth_bbox(p.truth.joints) if p.truth.present else None
        initial = region_rect(attempt.get("initialSearchRegion"))
        detection = region_rect(attempt.get("detectionRegion"))
        initial_iou = rect_iou(bbox, initial)
        detection_iou = rect_iou(bbox, detection)
        containment = rect_containment(bbox, initial)
        flags = condition_flags(attempt.get("searchConditions"))
        flags_fired = any(flags.values())
        reacquire_attempted = bool(attempt.get("reacquireAttempted"))

        if initial_iou is not None:
            initial_ious.append(initial_iou)
        if detection_iou is not None:
            detection_ious.append(detection_iou)
        if containment is not None:
            containment_scored += 1
            contained += containment >= CROP_CONTAINMENT_MIN

        cause = None
        if status == "missing":
            cause = _miss_cause(p, bbox, containment, flags_fired, reacquire_attempted)
            cause_counts[cause] += 1

        entries.append({
            "t": _round6(p.truth.timestamp),
            "status": status,
            "truthPresent": p.truth.present,
            "truthBbox": _bbox_block(bbox),
            "initialSearchRegionIou": _round6(initial_iou),
            "detectionRegionIou": _round6(detection_iou),
            "initialCropContainment": _round6(containment),
            "cropContainedTruth": (None if containment is None
                                   else containment >= CROP_CONTAINMENT_MIN),
            "searchFlagsFired": flags_fired,
            "firedSearchFlags": sorted(n for n, v in flags.items() if v),
            "reacquireAttempted": reacquire_attempted,
            "missCause": cause,
        })

    def stats(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {"n": len(ordered),
                "median": _round6(_percentile(ordered, 0.5)),
                "p90": _round6(_percentile(ordered, 0.9))}

    missing = sum(cause_counts.values())
    return {
        "thresholds": {
            "truthBboxPad": TRUTH_BBOX_PAD,
            "cropContainmentMin": CROP_CONTAINMENT_MIN,
            "reacquireSearchesFullFrame": REACQUIRE_SEARCHES_FULL_FRAME,
        },
        "matchedAttempts": len(entries),
        "missingAttempts": missing,
        "missCauseCounts": cause_counts,
        "initialSearchRegionIou": stats(initial_ious),
        "detectionRegionIou": stats(detection_ious),
        "cropContainedTruth": {
            "contained": contained,
            "scored": containment_scored,
            "rate": _round6(contained / containment_scored) if containment_scored else None,
        },
        "frames": entries,
    }


def _status_driven_class(status: str | None, geometric_class: str) -> str:
    if status == "flipRejected":
        return FQ_FLIPPED
    if status == "qualityRejected":
        return FQ_DISTORTED
    return geometric_class


def _frame_quality(pairs: list[_FramePair], truth: TruthDoc,
                   setup_hash: str) -> dict[str, Any]:
    """Per-frame detection-quality classification (issue #44 deliverable 1).

    One entry per matched frame on which the scanner emitted a pose: its auto class
    (``ok`` / ``wrong-subject`` / ``hallucination-fp`` / ``flipped-rotated`` /
    ``distorted``) plus a cross-cutting ``frozenStale`` flag (near-identical keypoints
    to the previous detected frame). Ground-truth detection annotations (issue #45)
    refine the auto class when they match the active setupHash and the frame's index;
    the auto class remains in ``autoClass`` for auditability. Frames with no scanner
    detection are not detection-quality events — they are coverage/presence gaps
    counted elsewhere — so they carry no entry here. ``crop`` is a placeholder the crop
    exporter (deliverable 2) fills in for flagged frames.

    Flip/quality-rejected attempt frames additionally carry a rejection verdict (issue
    #85) scoring the discarded raw pose against truth, pooled into
    ``rejectionCorrectness``.

    Iterated in timestamp order so ``frozenStale`` compares against the true temporal
    predecessor regardless of truth-file frame order. Scored over the same non-excluded
    pairs as the agreement tier."""

    detected = sorted(
        (p for p in pairs if p.matched and _quality_candidate(p)),
        key=lambda p: p.truth.timestamp)

    # Held-pose detection is a sustained-run property (issue #68), so resolve it for the
    # whole timestamp-ordered sequence up front. ``frozenStale`` is stricter: only a held
    # pose emitted by the raw detector is a scanner stale failure.
    held_flags = _frozen_flags([p.scanner for p in detected])

    counts = {c: 0 for c in FQ_CLASSES}
    held_count = 0
    frozen_count = 0
    entries: list[dict[str, Any]] = []
    for p, held in zip(detected, held_flags):
        tf = p.truth
        attempt = p.detector_attempt
        attempt_status = attempt.get("status") if isinstance(attempt, dict) else None
        candidate = _quality_candidate(p)
        auto_cls: str
        if tf.present:
            auto_cls, centroid_dist, residual = _classify_detection(
                tf.joints, candidate, torso_length(tf.joints))
        else:
            # A pose on a climber-absent frame — the presence-2x2 ``absentDetected``
            # cell, localized to this timestamp.
            auto_cls, centroid_dist, residual = FQ_HALLUCINATION, None, None
        auto_cls = _status_driven_class(attempt_status, auto_cls)
        ann = _detection_annotation_for_frame(truth, tf.frame_index, setup_hash)
        effective_cls = ann.failure_class if ann is not None else auto_cls
        # Attempt-backed evidence is direct MediaPipe output by construction, so a
        # held pose there is a raw-detector stale failure; legacy frames must carry
        # ``source == "raw"`` to distinguish detector staleness from reconstruction.
        frozen = bool(held and (attempt is not None or p.scanner_source == "raw"))
        counts[effective_cls] += 1
        held_count += held
        frozen_count += frozen
        # Rejection correctness (issue #85) — populated only on flip/quality-rejected
        # attempts; every other entry carries the keys as None so readers can select the
        # column without branching on evidence generation.
        rejection = _rejection_scoring(p) or {}
        entry = {
            "t": _round6(tf.timestamp),
            "class": effective_cls,
            "autoClass": auto_cls,
            "failureClass": effective_cls,
            "source": p.scanner_source,
            "distractor": ann.distractor if ann is not None else None,
            "annotationSetupHash": ann.setup_hash if ann is not None else None,
            "heldPose": held,
            "frozenStale": frozen,
            "centroidDist": _round6(centroid_dist),
            "residual": _round6(residual),
            "rejectionVerdict": rejection.get("verdict"),
            "rejectionReason": rejection.get("reason"),
            "rejectionCentroidDist": rejection.get("centroidDist"),
            "rejectionResidual": rejection.get("residual"),
            "rejectionJointAgreement": rejection.get("jointAgreement"),
            "rejectionRawClass": rejection.get("rawClass"),
            "crop": None,
        }
        if isinstance(attempt, dict):
            entry.update({
                "detectorEvidence": "attempts",
                "detectorAttemptStatus": attempt_status,
                "detectorStatusKnown": bool(attempt.get("statusKnown")),
                "rawKeypoints": _attempt_keypoint_payload(attempt, "rawKeypoints"),
                "acceptedKeypoints": _attempt_keypoint_payload(attempt, "acceptedKeypoints"),
                "candidateCount": attempt.get("candidateCount"),
                "rejectedCandidateCount": attempt.get("rejectedCandidateCount"),
                "selectionMethod": attempt.get("selectionMethod"),
            })
        else:
            entry["detectorEvidence"] = "legacy-frames"
        entries.append(entry)

    return {
        "thresholds": {
            "wrongSubjectCentroid": FQ_WRONG_SUBJECT_CENTROID,
            "distortResidual": FQ_DISTORT_RESIDUAL,
            "flipResidual": FQ_FLIP_RESIDUAL,
            "frozenEps": FQ_FROZEN_EPS,
            "frozenMinRun": FQ_FROZEN_MIN_RUN,
        },
        "detectorEvidence": (
            "attempts" if any(p.detector_attempt is not None for p in pairs)
            else "legacy-frames"
        ),
        "detectorAttemptStatusCounts": _attempt_status_counts(pairs),
        # Was the scanner right to discard what it discarded (issue #85)? Scored over all
        # pairs, not just the entries above: a rejection with no raw pose yields no
        # frameQuality entry but is still a rejection that must be counted.
        "rejectionCorrectness": _rejection_correctness(pairs),
        "classCounts": counts,
        "heldPoseCount": held_count,
        "frozenStaleCount": frozen_count,
        "flaggedCount": sum(v for c, v in counts.items() if c != FQ_OK),
        "detectedFrames": len(entries),
        "frames": entries,
    }


def record_conforms(record: dict[str, Any]) -> bool:
    """Whether an on-disk record passes the #15 conformance gate. Legacy records
    (schema < 4) carry no ``conformance`` block; they predate the gate and are treated
    as conforming (fail-open) so an old corpus isn't silently emptied — regenerate to
    get a real verdict."""

    conf = record.get("conformance")
    if not isinstance(conf, dict) or "conforms" not in conf:
        return True
    return bool(conf["conforms"])


def record_trusted(record: dict[str, Any]) -> bool:
    """Whether an on-disk record may feed the *trusted* pooled metrics: it must both
    pass the #15 conformance gate and not be a best-overlap loose pairing (issue #44).
    A loose record still carries per-frame quality worth mining — pooled separately —
    but its setupHash never matched the truth, so it must stay out of the trusted pool."""

    return record_conforms(record) and not record.get("loosePaired", False)


def evaluate_pair(pose_frames: list[dict[str, Any]], truth: TruthDoc,
                  setup_hash: str = "",
                  detector_attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compute the full metric set for one pose Run against one truth doc.

    Returns the record body (counts + agreement/accuracy tiers); provenance is
    stamped by the caller. Both tiers share the same frame pairing. ``human-flagged-
    wrong`` (known-bad seed) and ``human-flagged-absent`` (deprecated manual flag,
    ADR 0005) frames are excluded from scoring and surface only in
    ``counts.agreementSkipped``. The accuracy tier has no trustworthy attestation
    source yet, so it is present but empty (issue #12).
    """

    observations = _scanner_observations(pose_frames, detector_attempts)
    scanner_ts = sorted(o.timestamp for o in observations)
    by_ts: dict[float, _ScannerObservation] = {o.timestamp: o for o in observations}
    interval = _scanner_frame_interval(scanner_ts)
    tol = interval / 2

    pairs: list[_FramePair] = []
    for tf in truth.frames:
        idx = _nearest_within(scanner_ts, tf.timestamp, tol)
        if idx is None:
            pairs.append(_FramePair(tf, False, {}))
        else:
            observation = by_ts[scanner_ts[idx]]
            pairs.append(_FramePair(
                tf, True, observation.scanner,
                scanner_source=observation.source,
                detector_attempt=observation.detector_attempt))

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
        # Whole-bundle truth↔scanner conformance (issue #15), fit over the same
        # non-excluded pairs the agreement tier scores. Gates pooled metrics.
        "conformance": _conformance(agreement_pairs),
        # Per-frame detection-quality classes (issue #44), over the same pairs.
        "frameQuality": _frame_quality(agreement_pairs, truth, setup_hash),
        # Crop placement + miss causation (issue #86). Its own block, not part of
        # frameQuality, because it scores *every* matched attempt — including the misses
        # frameQuality deliberately omits.
        "cropQuality": _crop_quality(agreement_pairs),
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
        yield run_ts, setup_hash, data.get("frames", []) or [], parse_detector_attempts(data)


def _present_overlap(pose_frames: list[dict[str, Any]], truth: TruthDoc,
                     detector_attempts: list[dict[str, Any]] | None = None) -> int:
    """How many non-excluded, present truth frames a pose Run actually overlaps.

    Mirrors the join in ``evaluate_pair`` (nearest scanner frame within half the median
    scanner interval) but counts only — the selector for the best-overlap fallback
    (issue #44 deliverable 4). Zero means the Run's samples never land near a scorable
    truth frame, so it carries no per-frame evidence no matter its setupHash."""

    scanner_ts = sorted(o.timestamp for o in _scanner_observations(pose_frames, detector_attempts))
    if not scanner_ts:
        return 0
    tol = _scanner_frame_interval(scanner_ts) / 2
    count = 0
    for tf in truth.frames:
        if tf.excluded or not tf.present:
            continue
        if _nearest_within(scanner_ts, tf.timestamp, tol) is not None:
            count += 1
    return count


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


def _bundle_already_analyzed(eval_dir: Path, matched_run_ts: list[str],
                             truth_hash8: str) -> bool:
    """Un-analyzed gate (issue #57): True when every setupHash-matched Run already carries
    a current-truth evaluation record on disk (``<run_ts>_vs_<truth_hash8>.json``).

    Coarse by design. A bundle with *no* setupHash-matched Run is never "analyzed" — it may
    still owe a best-overlap loose record (issue #44), and a fresh bundle never analyzed at
    all must not be vacuously skipped — so it always reprocesses. Because the gate keys on
    the *current* ``truth_hash8``, a truth revision (new hash) invalidates every prior
    record and reprocesses the bundle. When a bundle is skipped, its on-disk records are
    left untouched, so the loose-pair fallback outcome is identical to a full sweep."""

    if not matched_run_ts or not eval_dir.is_dir():
        return False
    return all((eval_dir / f"{run_ts}_vs_{truth_hash8}.json").exists()
               for run_ts in matched_run_ts)


def _export_crops(video_dir: Path, run_ts: str, pose_frames: list[dict[str, Any]],
                  body: dict[str, Any]) -> None:
    """Best-effort crop export for one Run's frameQuality entries (issue #44 deliverable
    2). Imported locally so the common JSON path never pulls cv2. Any failure is
    swallowed — a missing binary or decode error must not abort record writing."""

    try:
        from . import crops
        crops.export_run_crops(video_dir, run_ts, pose_frames, body["frameQuality"])
    except Exception:  # pragma: no cover - defensive; crops are non-essential
        pass


def _write_eval_record(video_dir: Path, route_folder: str, video_key: str, run_ts: str,
                       setup_hash: str, truth: TruthDoc, truth_hash8: str,
                       body: dict[str, Any], *, loose: bool = False,
                       loose_reason: str = "") -> Path:
    """Assemble and write one idempotent evaluation record; return its path.

    Shared by the trusted (setupHash-matched) path and the best-overlap loose fallback
    (issue #44 deliverable 4). A loose record stamps the pairing Run's *own* setupHash
    (not the truth's), records why it fell back, and carries ``loosePaired: true`` so
    downstream pooling can keep it out of the trusted metrics while still mining its
    per-frame quality (readers fail-open on the absent key for trusted records)."""

    record = {
        "schemaVersion": SCHEMA_VERSION,
        "metrics": ["pck@0.5-torso", "normDistMedian", "normDistP90",
                    "presence2x2", "jointCoverage"],
        "routeFolder": route_folder,
        "videoKey": video_key,
        "runTs": run_ts,
        "setupHash": setup_hash,
        "truthSource": truth.source,
        "truthHash": truth.truth_hash,
        "truthSetupHashSource": ("loose-overlap" if loose
                                 else "truth" if truth.setup_hash else "setup.json"),
        "jointSet": COCO_CORE_JOINTS,
        **body,
    }
    if loose:
        record["loosePaired"] = True
        record["loosePairReason"] = loose_reason
    eval_dir = video_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    record_path = eval_dir / f"{run_ts}_vs_{truth_hash8}.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record_path


def evaluate(analysis_root: Path, prune: bool = False,
             export_crops: bool = False, mode: str = EVAL_MODE_ALL) -> EvalSummary:
    """Walk the bundle tree, pair every pose Run with truth, write eval records.

    ``mode`` selects coverage (issue #57). ``all`` (the default) is the full sweep —
    every setupHash-matched Run in every bundle is (re)scored, exactly as before, so the
    default is behaviourally unchanged. ``un-analyzed`` skips any bundle whose matched Runs
    already carry a current-truth record (``_bundle_already_analyzed``), for incremental
    corpus work; skipped bundles are named in ``summary.analyzed_skipped``. Pruning and the
    best-overlap loose fallback are orthogonal to mode — a skipped bundle is still pruned
    (its current-truth records are hash-protected), and any bundle that *is* processed runs
    the identical loose-pair path under both modes.

    When ``prune`` is set, also delete stale-run orphan records (issue #32); with it
    unset, orphans are still reported (dry run) but nothing is deleted. When
    ``export_crops`` is set, decode the (gitignored) video binaries and write flagged-
    frame crops into each bundle's ``crops/`` dir (issue #44 deliverable 2), stamping
    the crop path into the ``frameQuality`` entries before the record is written; the
    export is best-effort and silently no-ops when cv2 or the binary is absent."""

    if mode not in EVAL_MODES:
        raise ValueError(f"unknown evaluate mode {mode!r}; expected one of {EVAL_MODES}")

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

        runs = list(_iter_pose_runs(video_dir / "detections"))
        eval_dir = video_dir / "evaluations"

        # Un-analyzed gate (issue #57): a bundle whose setupHash-matched Runs already
        # carry current-truth records is skipped wholesale — no re-scoring, records left
        # untouched — but pruning still runs. The matched Run timestamps stand in as the
        # "paired" set so truth-revision history for a still-pairing Run is retained exactly
        # as a full sweep would (current-truth records, trusted and loose, are protected by
        # the hash check regardless).
        matched_run_ts = [rt for rt, sh, _, _ in runs if sh == effective_setup_hash]
        if mode == EVAL_MODE_UNANALYZED and _bundle_already_analyzed(
                eval_dir, matched_run_ts, truth_hash8):
            summary.analyzed_skipped.append(f"{route_folder}/{video_key}")
            summary.orphans.extend(_prune_orphans(
                eval_dir, set(matched_run_ts), truth_hash8,
                route_folder, video_key, prune))
            continue

        best_trusted_overlap = 0
        for run_ts, pose_setup_hash, pose_frames, detector_attempts in runs:
            if pose_setup_hash != effective_setup_hash:
                summary.pairings.append(Pairing(
                    route_folder, video_key, run_ts, truth.source, "skipped",
                    reason=(f"setupHash mismatch (run {pose_setup_hash[:8] or '∅'} "
                            f"vs truth {effective_setup_hash[:8] or '∅'})"),
                ))
                continue

            body = evaluate_pair(pose_frames, truth, effective_setup_hash, detector_attempts)
            if export_crops:
                _export_crops(video_dir, run_ts, pose_frames, body)
            record_path = _write_eval_record(
                video_dir, route_folder, video_key, run_ts, effective_setup_hash,
                truth, truth_hash8, body)
            paired_run_ts.add(run_ts)
            best_trusted_overlap = max(
                best_trusted_overlap, body["agreement"]["frames"]["matchedPresent"])
            summary.pairings.append(Pairing(
                route_folder, video_key, run_ts, truth.source, "written",
                record_path=record_path))

        # Best-overlap loose fallback (issue #44 deliverable 4): when no trusted pairing
        # reached the overlap floor, recover per-frame evidence from the Run that
        # overlaps truth most — even one whose setupHash differs — provided it beats
        # every trusted Run's overlap. It is written loosePaired and never enters the
        # trusted pool. Recovers the IE4T94qX55g n=0 case.
        if best_trusted_overlap < LOOSE_PAIR_MIN_OVERLAP:
            best_overlap = best_trusted_overlap
            candidate: tuple[str, str, list[dict[str, Any]], list[dict[str, Any]] | None] | None = None
            for run_ts, pose_setup_hash, pose_frames, detector_attempts in runs:
                if run_ts in paired_run_ts:
                    continue
                ov = _present_overlap(pose_frames, truth, detector_attempts)
                if ov > best_overlap:
                    best_overlap, candidate = ov, (
                        run_ts, pose_setup_hash, pose_frames, detector_attempts)
            if candidate is not None and best_overlap > 0:
                run_ts, pose_setup_hash, pose_frames, detector_attempts = candidate
                body = evaluate_pair(pose_frames, truth, effective_setup_hash, detector_attempts)
                if export_crops:
                    _export_crops(video_dir, run_ts, pose_frames, body)
                reason = (
                    f"no setupHash-matched run overlapped truth "
                    f"(≥{LOOSE_PAIR_MIN_OVERLAP} present frames); paired best-overlap run "
                    f"({best_overlap} frames, run setupHash {pose_setup_hash[:8] or '∅'} "
                    f"vs truth {effective_setup_hash[:8] or '∅'})")
                record_path = _write_eval_record(
                    video_dir, route_folder, video_key, run_ts, pose_setup_hash,
                    truth, truth_hash8, body, loose=True, loose_reason=reason)
                paired_run_ts.add(run_ts)
                summary.pairings.append(Pairing(
                    route_folder, video_key, run_ts, truth.source, "written",
                    record_path=record_path, loose=True))

        summary.orphans.extend(_prune_orphans(
            eval_dir, paired_run_ts, truth_hash8,
            route_folder, video_key, prune))

    return summary
