"""Establish the run-to-run variance floor from the within-build repeat set (issue #134).

No baseline in this repo has ever stated how much a metric moves when *nothing changes*.
So when flip over-rejection went 76.7% → 33.8% → 25.4%, or crop IoU 0.289 → 0.307, there
was no way to say whether the move exceeded the scatter of simply re-running the same
video on the same build. Some of the "nothing is consistent" churn may have been noise
that looked like signal.

The repeat set is an accident nobody planned: ``deaa1c0`` holds ~140 runs across ~74
bundles, most of them with two or more runs of the *same video on the same build*. That
is a repeat-measures design, and it is the only one in the corpus — a single-pass sweep
structurally cannot produce one. **It does not survive a corpus wipe**, so the floor is
measured before any reset.

For each (bundle, build) group with ≥2 runs, the within-group **range** (max − min) of
each metric is the run-to-run scatter for that bundle. The floor is reported as the
median and p90 of those ranges: half the bundles scatter by less than the median, and a
change smaller than the p90 is indistinguishable from noise on a typical bundle.

Reported beside each metric's own median value, because a range of 0.05 means very
different things on a rate near 0.5 and one near 0.02.

**The caveat this carries** (from #134, restated rather than buried): ``deaa1c0``
predates the build-identity fix (#130), so its stamp is only as trustworthy as any other
appVersion. For *variance* that matters far less than for A/B — repeat runs inside one
server lifetime are very likely the same code even when the label is unreliable — but it
is an assumption, not a fact.

Run:  python -m scripts.measure_variance_floor [analysis_root] [--build deaa1c0]
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis_pipeline.trends import _iter_eval_records, _load_pose_runs  # noqa: E402

DEFAULT_BUILD = ""   # "" = every build, each grouped with its own
MIN_RUNS_PER_GROUP = 2
# Below this many groups a p90 is just the second-largest observation, not a percentile.
# The script still prints, because a refused measurement teaches nothing, but it says
# loudly that the number is provisional (#134's acceptance asks for a floor, and this
# corpus cannot supply one).
MIN_GROUPS_FOR_A_FLOOR = 20


def _get(record: dict[str, Any], *path: str) -> Any:
    cur: Any = record
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _share(counts: Any, key: str) -> float | None:
    """One status as a share of the group's total — a count is not comparable across
    bundles of different lengths, and the funnel is read as a set of shares."""
    if not isinstance(counts, dict):
        return None
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    value = counts.get(key)
    if not total or not isinstance(value, (int, float)):
        return None
    return value / total


# (label, extractor). Every value is a rate or a share in [0, 1] except where noted, so
# the ranges below are directly comparable to a claimed improvement in the same units.
METRICS: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
    ("agreement PCK",
     lambda r: _get(r, "agreement", "aggregate", "pck", "value")),
    ("funnel: accepted share",
     lambda r: _share(_get(r, "frameQuality", "detectorAttemptStatusCounts"), "accepted")),
    ("funnel: missing share",
     lambda r: _share(_get(r, "frameQuality", "detectorAttemptStatusCounts"), "missing")),
    ("funnel: flipRejected share",
     lambda r: _share(_get(r, "frameQuality", "detectorAttemptStatusCounts"), "flipRejected")),
    ("over-rejection rate",
     lambda r: _get(r, "frameQuality", "rejectionCorrectness", "overRejectionRate")),
    ("over-rejection (truth-present)",
     lambda r: _get(r, "frameQuality", "rejectionCorrectness", "overRejectionRateTruthPresent")),
    ("crop containment rate",
     lambda r: _get(r, "cropQuality", "cropContainedTruth", "rate")),
    ("crop IoU (median)",
     lambda r: _get(r, "cropQuality", "detectionRegionIou", "median")),
    ("miss: no-candidates share",
     lambda r: _share(_get(r, "cropQuality", "missCauseCounts"), "no-candidates")),
    ("miss: identity-gated share",
     lambda r: _share(_get(r, "cropQuality", "missCauseCounts"), "identity-gated")),
    ("miss: climber-absent share",
     lambda r: _share(_get(r, "cropQuality", "missCauseCounts"), "climber-absent")),
]


def repeat_groups(analysis_root: Path, build: str) -> dict[tuple[str, ...], list[dict]]:
    """(bundle → records) for bundles with ≥2 runs of the named build.

    Build identity lives only in the pose envelope's diagnostics — evaluation records do
    not carry it — so each record is joined back to its run to be grouped.

    Grouped by ``(bundle, truthHash, detectorEvidence)``, not by bundle alone. Two runs
    of one video scored against *different truth revisions* differ for a reason that is
    not run-to-run noise, and pooling them would inflate the floor with re-seed effects —
    measuring the very confound this floor exists to rule out. The same applies to
    evidence generation: a ``legacy-frames`` run has no attempt stream at all, so its
    funnel counts are structurally zero and differencing it against an attempt-backed run
    measures the export format, not the detector.
    """
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    by_content: dict[tuple[str, ...], dict[str, dict]] = {}
    for rec in _iter_eval_records(analysis_root):
        vid = (rec.route_folder, rec.video_key)
        if vid not in cache:
            cache[vid] = _load_pose_runs(analysis_root / rec.route_folder / rec.video_key)
        run = cache[vid].get(rec.run_ts)
        if run is None or (build and not run.app_version.startswith(build)):
            continue
        truth_hash = str(rec.data.get("truthHash") or "")
        evidence = str(((rec.data.get("frameQuality") or {})
                        .get("detectorEvidence")) or "unknown")
        # A "repeat" means two independent *detection passes*. Much of this corpus
        # re-exports one pass under several run timestamps, and differencing a pass
        # against itself yields a range of exactly zero — a floor that would look
        # excellent and mean nothing. Records are keyed by the content hash of the
        # detections they were scored from, so a duplicate export collapses instead of
        # counting as evidence of stability.
        pose = analysis_root / rec.route_folder / rec.video_key / "detections" / f"{rec.run_ts}_pose.json"
        try:
            frames = (json.loads(pose.read_text(encoding="utf-8")) or {}).get("data", {}).get("frames")
        except Exception:
            continue
        content = hashlib.sha256(
            json.dumps(frames, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        key = (vid[0], vid[1], run.app_version, truth_hash, evidence)
        by_content.setdefault(key, {}).setdefault(content, rec.data)
    return {k: list(v.values()) for k, v in by_content.items()
            if len(v) >= MIN_RUNS_PER_GROUP}


def floor_for(groups: dict[tuple[str, ...], list[dict]],
              extract: Callable[[dict[str, Any]], Any]) -> dict[str, Any] | None:
    """Within-bundle range of one metric across the repeat set."""
    ranges: list[float] = []
    centres: list[float] = []
    for records in groups.values():
        values = [extract(r) for r in records]
        values = [v for v in values if isinstance(v, (int, float))]
        if len(values) < MIN_RUNS_PER_GROUP:
            continue
        ranges.append(max(values) - min(values))
        centres.append(statistics.median(values))
    if not ranges:
        return None
    ranges.sort()
    return {
        "bundles": len(ranges),
        "median_range": statistics.median(ranges),
        "p90_range": ranges[min(len(ranges) - 1, int(0.9 * len(ranges)))],
        "max_range": ranges[-1],
        "median_value": statistics.median(centres),
    }


def report(analysis_root: Path, build: str) -> None:
    groups = repeat_groups(analysis_root, build)
    runs = sum(len(v) for v in groups.values())
    scope = f"build {build!r}" if build else "all builds (each grouped with its own)"
    print(f"repeat set: {scope} — {len(groups)} groups with "
          f">={MIN_RUNS_PER_GROUP} runs, {runs} runs total")
    print("  a group is one (bundle, build, truth revision, evidence generation), and")
    print("  duplicate exports of a single detection pass are collapsed — differencing")
    print("  a pass against itself is a floor of zero that means nothing.")
    if not groups:
        print("  no repeat set for this build; nothing to measure")
        return
    print()
    print(f"  {'metric':<32}{'bundles':>8}{'median':>9}{'p90':>9}{'max':>9}   "
          f"{'typical value':>13}")
    for label, extract in METRICS:
        stats = floor_for(groups, extract)
        if stats is None:
            print(f"  {label:<32}{'—':>8}{'  not scored on this set':<28}")
            continue
        print(f"  {label:<32}{stats['bundles']:>8}"
              f"{stats['median_range']:>9.4f}{stats['p90_range']:>9.4f}"
              f"{stats['max_range']:>9.4f}   {stats['median_value']:>13.4f}")
    print()
    if len(groups) < MIN_GROUPS_FOR_A_FLOOR:
        print(f"  *** PROVISIONAL — {len(groups)} groups is below the {MIN_GROUPS_FOR_A_FLOOR} "
              f"needed for a p90 to mean anything. ***")
        print("  At this n the p90 column is simply the largest observation. Read the")
        print("  MEDIAN column as the usable signal, and treat p90/max as anecdote.")
        print("  This corpus cannot supply a floor: the repeats were never designed, and")
        print("  what looked like a repeat set is duplicate exports. A future batch must")
        print("  run each condition on each video 3+ times to produce one deliberately.")
        print()
    print("  median/p90/max are the within-bundle RANGE across repeat runs — the scatter")
    print("  when nothing changed. A batch-over-batch move below the p90 is not evidence.")


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    build = DEFAULT_BUILD
    for i, a in enumerate(argv):
        if a == "--build" and i + 1 < len(argv):
            build = argv[i + 1]
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent / "analysis"
    if not root.is_dir():
        print(f"no analysis root at {root}", file=sys.stderr)
        return 2
    report(root, build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
