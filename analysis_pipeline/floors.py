"""The measured uncertainty that has to travel beside a number (issue #164).

Nothing in this repo was checkable against noise until #134 asked what a metric does when
*nothing changes*. The answer turned out to depend on **who produced the run**, and this
module exists so that dependency is impossible to lose track of at render time.

Three quantities, three different producers, and they are not interchangeable:

- **Harness run-to-run scatter is exactly 0.** The MediaPipe detector this harness runs is
  bit-deterministic — repeats produce byte-identical output, which is why ``repeats``
  defaults to 1 (#159) and a byte-compared canary replaced them (#168). There is no
  harness run-to-run floor to attach, and attaching #134's 0.0055 would be a category
  error: that is the *scanner's* scatter, from a different detector in a different runtime.
- **Harness sampling error** is the real uncertainty in a harness arm's PCK. Runs score a
  ``12·√n`` sample of the Bundle's truth grid rather than the full grid, and that choice
  has a measured cost against the full-grid answer. It is **common-mode across arms** —
  same Bundle, same frames, same truth for every arm — so it largely cancels in an arm
  *delta* and matters most for the per-video *absolutes*.
- **Scanner run-to-run floors** are #134's per-metric figures, and they are labelled
  scanner-side everywhere they appear so they can never be read as a harness uncertainty.

The scanner floors are recorded here as constants rather than recomputed, for a reason
that is about to become load-bearing: the repeat set they came from is an accident of the
historical corpus (six genuine repeat groups, most apparent ones being one detection pass
re-exported), and **it does not survive the #163 corpus reset**. ``scripts/
measure_variance_floor.py`` remains the derivation and can be re-run against any corpus
that still holds the runs; this module is the measurement it produced, frozen with its
provenance and its caveats attached.
"""

from __future__ import annotations

from typing import NamedTuple

# --- who produced the number -------------------------------------------------
#
# Mirrors ``trends.ORIGIN_*``, kept as plain strings here so this module stays a leaf with
# no import back into the derivations that consume it.
SIDE_HARNESS = "harness"
SIDE_SCANNER = "scanner"


class Floor(NamedTuple):
    """One measured uncertainty, with everything needed to read it honestly."""

    metric: str
    median: float
    p90: float | None
    max: float | None
    typical_value: float | None
    n_groups: int
    side: str
    provisional: bool
    source: str

    @property
    def label(self) -> str:
        """How the number is named in prose: never a bare figure."""

        tag = f"{self.side}-side"
        if self.provisional:
            tag += ", provisional"
        return tag


# --- harness: run-to-run --------------------------------------------------- #

# Not "small". Zero. Three passes over the same frames on the same arm produce
# byte-identical pose output, confirmed across all three detection modes, two videos and
# separate processes (#159), and re-verified every Cycle by the determinism canary (#168).
#
# A harness result therefore carries no run-to-run term at all, and a report that showed
# one would be inventing scatter the detector does not have.
HARNESS_RUN_TO_RUN = 0.0

# Phrased as a clause so a renderer can lead with the number and follow with the reason,
# rather than restating "scatter is 0" twice in one sentence.
HARNESS_RUN_TO_RUN_NOTE = (
    "the detector is bit-deterministic (#159), re-verified byte-for-byte by the Cycle "
    "determinism canary (#168)"
)


# --- harness: sampling error ------------------------------------------------ #

# Frames scored per run are ``12·√n`` of the Bundle's truth grid, not the full grid. The
# cost of that, measured across 55 Bundles as |ΔPCK| against the full-grid answer.
#
# The p90 (0.0056) is essentially equal to #134's scanner PCK floor (0.0055) — not a
# coincidence but the stopping criterion the coefficient was chosen against: worst-case
# sampling error at or below noise the corpus already carries.
SAMPLING_ERROR = Floor(
    metric="agreement PCK (12·√n sampling vs full grid)",
    median=0.0017,
    p90=0.0056,
    max=None,
    typical_value=None,
    n_groups=55,
    side=SIDE_HARNESS,
    provisional=False,
    source="PRD #156 / mediapipe_job.SAMPLE_COEFFICIENT, measured over 55 Bundles",
)

# What makes the delta different from the absolute. Every arm on a Bundle scores the *same*
# frames against the *same* truth — the sample is a deterministic function of the Bundle,
# not of the arm — so the sampling error is a shared offset that largely cancels in a
# difference. It is attached to per-video absolutes and explicitly discounted in deltas.
#
# This is exactly why changing ``SAMPLE_COEFFICIENT`` must bump ``MODULE_VERSION``: it
# would let two runs on one Bundle carry the same arm stamp over *different* frames, and
# the cancellation would silently stop being true.
SAMPLING_ERROR_COMMON_MODE_NOTE = (
    "common-mode across arms — every arm on a Bundle scores the same 12·√n frames against "
    "the same truth, so it is a shared offset that largely cancels in an arm delta and "
    "matters most for the per-video absolutes"
)


# --- scanner: #134 run-to-run floors ---------------------------------------- #

# Measured by ``scripts/measure_variance_floor.py`` over the historical corpus, all builds,
# each bundle grouped with its own. A "group" is one (Bundle, build, truth revision,
# evidence generation) with two or more *independent detection passes* — duplicate exports
# of a single pass are collapsed by detection content hash, because differencing a pass
# against itself is a range of exactly zero that looks excellent and means nothing.
#
# **Six groups is not a floor**, and #134 closed on exactly that finding. The medians are
# still directional and worth printing; the p90/max columns at this n are simply the
# largest observation, so they are carried but marked provisional wherever they show.
SCANNER_FLOOR_GROUPS = 6
SCANNER_FLOOR_SOURCE = (
    "issue #134 via scripts/measure_variance_floor.py, all builds, "
    f"{SCANNER_FLOOR_GROUPS} genuine repeat groups / 12 runs"
)

SCANNER_FLOOR_CAVEAT = (
    f"PROVISIONAL — {SCANNER_FLOOR_GROUPS} repeat groups is far below the 20 a p90 needs "
    "to mean anything, so read the median as the signal and the p90/max as anecdote. The "
    "corpus could not supply a real floor: the repeats were never designed, and 27 of 33 "
    "apparent ones were one detection pass re-exported (#134)"
)


def _scanner(metric: str, median: float, p90: float, mx: float,
             typical: float) -> Floor:
    return Floor(metric=metric, median=median, p90=p90, max=mx,
                 typical_value=typical, n_groups=SCANNER_FLOOR_GROUPS,
                 side=SIDE_SCANNER, provisional=True, source=SCANNER_FLOOR_SOURCE)


# Keyed by the metric key the derivations use, so a table can look a floor up by column
# name rather than by matching prose. Values are within-Bundle *ranges* (max − min) across
# repeat runs — the scatter when nothing changed.
SCANNER_FLOORS: dict[str, Floor] = {
    "pck": _scanner("agreement PCK", 0.0055, 0.2298, 0.2298, 0.7696),
    "funnel_accepted_share": _scanner("funnel: accepted share", 0.0088, 0.0526, 0.0526, 0.9249),
    "funnel_missing_share": _scanner("funnel: missing share", 0.0124, 0.0526, 0.0526, 0.0397),
    "funnel_flip_rejected_share": _scanner(
        "funnel: flipRejected share", 0.0044, 0.0120, 0.0120, 0.0225),
    "over_rejection_rate": _scanner("over-rejection rate", 0.0167, 0.2460, 0.2460, 0.0083),
    "over_rejection_rate_truth_present": _scanner(
        "over-rejection (truth-present)", 0.1558, 0.5000, 0.5000, 0.0985),
    "crop_contained_rate": _scanner("crop containment rate", 0.0034, 0.3396, 0.3396, 0.9926),
    "crop_iou_median": _scanner("crop IoU (median)", 0.0042, 0.2742, 0.2742, 0.2790),
    "miss_no_candidates_share": _scanner(
        "miss: no-candidates share", 0.0824, 0.5000, 0.5000, 0.6295),
    "miss_identity_gated_share": _scanner(
        "miss: identity-gated share", 0.0016, 0.0233, 0.0233, 0.0050),
    "miss_climber_absent_share": _scanner(
        "miss: climber-absent share", 0.1013, 0.5000, 0.5000, 0.3495),
}

# The subset that is *funnel-derived* — what the detector did, before truth is consulted,
# plus the truth-joined rates read off the same attempt stream. These are the metrics
# #164's acceptance requires to show a floor beside them, and every one of them is
# scanner-side: the harness produces no Detector Attempt stream at all, which is a
# structural absence and must render as such rather than as a row of zeros.
FUNNEL_FLOOR_KEYS = (
    "funnel_accepted_share",
    "funnel_missing_share",
    "funnel_flip_rejected_share",
    "over_rejection_rate",
    "over_rejection_rate_truth_present",
    "crop_contained_rate",
    "crop_iou_median",
    "miss_no_candidates_share",
    "miss_identity_gated_share",
    "miss_climber_absent_share",
)

FUNNEL_ABSENT_ON_HARNESS_NOTE = (
    "The Detector Attempt funnel is a scanner-owned concept — the harness detector emits "
    "no attempt stream, so these are structurally absent on a harness arm rather than "
    "measured at zero"
)


def scanner_floor(key: str) -> Floor | None:
    """The #134 floor for one metric key, or ``None`` when it was never measured."""

    return SCANNER_FLOORS.get(key)


def below_sampling_error(delta: float | None) -> bool | None:
    """Is an arm delta smaller than the sampling error it is measured through?

    Compared against the **p90**, not the median: the question a reader is asking is
    "could this be nothing?", and the honest bar for that is the worst case over Bundles,
    not the typical one. ``None`` in, ``None`` out — an unmeasurable delta is not a small
    one.
    """

    if delta is None:
        return None
    return abs(float(delta)) < (SAMPLING_ERROR.p90 or 0.0)
