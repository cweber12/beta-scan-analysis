"""Re-derive the truth-side / attested-clean split behind issue #147.

`conformance.cause` is computed from a per-axis fit of ``scanner = a*truth + b``.
A poor fit proves scanner and truth disagree; it **cannot** say which of them is
wrong. ``suspected-mistrack`` named a side anyway. This script measures how often
that side is the wrong one, by grading the cause against the only human-attested
truth-defect labels the corpus has: ``human-flagged-wrong`` frames.

The numbers it prints are cited in ADR 0010, #34, #147 and #148. Re-run it rather
than trusting those citations — it is the basis those numbers rest on, in the sense
ADR 0009 requires.

**What the attestation does and does not cover.** The review criterion was "is this
the right climber?", so a bundle with no flags is attested free of *identity* error,
not free of all truth error. A left/right laterality swap leaves the skeleton on the
correct person and passes review unseen (#148 H2). Read "attested clean" as
"no truth-side identity defect", never as "the truth is correct".

Bundle-level, deliberately: the flags mark stretches, and a bundle with any flagged
stretch had a tracker failure somewhere in it. Records are the unit of the
conformance verdict, so the cross-tab is records-by-bundle-status.

Run:  python -m scripts.measure_conformance_attribution [analysis_root]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis_pipeline.evaluate import (  # noqa: E402
    LEGACY_SUSPECTED_MISTRACK as LEGACY_DIVERGENCE,
    NONCONFORMANCE_TRAJECTORY_DIVERGENCE as DIVERGENCE,
    REVIEW_AUTO,
    REVIEW_FLAGGED_ABSENT,
    REVIEW_FLAGGED_WRONG,
)

TRUTH_SIDE = "truth-side (human-flagged)"
ATTESTED_CLEAN = "attested clean"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def review_counts(analysis_root: Path) -> tuple[dict[str, Counter], Counter]:
    """Per-bundle review breakdown of ``ground-truth.json``, plus the corpus total."""
    per_bundle: dict[str, Counter] = {}
    total: Counter = Counter()
    for truth_path in sorted(analysis_root.glob("*/*/ground-truth.json")):
        bundle = truth_path.parent
        key = f"{bundle.parent.name}/{bundle.name}"
        counts = Counter(
            (frame.get("review") or REVIEW_AUTO)
            for frame in (_load(truth_path).get("frames") or [])
        )
        per_bundle[key] = counts
        total.update(counts)
    return per_bundle, total


def _axis(reasons: set[str]) -> str:
    """Which axis of the identity fit failed — the laterality discriminator (#148 H2)."""
    x, y = "x-nonconforming" in reasons, "y-nonconforming" in reasons
    if x and y:
        return "both"
    if x:
        return "x-only"
    if y:
        return "y-only"
    return "other"


def conformance_rows(analysis_root: Path) -> tuple[list[dict], int, list[str]]:
    """Failed-gate rows, the graded-record count, and the records with no verdict.

    A record predating the #15 gate carries no ``conformance`` block at all. Such a
    record is **ungraded**, not conforming — it is excluded from the denominator and
    named, so the share never quietly rests on a population it was not measured over.
    """
    rows: list[dict] = []
    graded = 0
    ungraded: list[str] = []
    for record_path in sorted(analysis_root.glob("*/*/evaluations/*.json")):
        bundle = record_path.parent.parent
        key = f"{bundle.parent.name}/{bundle.name}"
        conformance = _load(record_path).get("conformance") or {}
        if "conforms" not in conformance:
            ungraded.append(f"{key}/{record_path.name}")
            continue
        graded += 1
        if conformance["conforms"] is not False:
            continue
        rows.append({
            "bundle": key,
            "cause": conformance.get("cause") or "unstated",
            "axis": _axis(set(conformance.get("reasons") or [])),
        })
    return rows, graded, ungraded


def _rule(width: int = 78) -> None:
    print("-" * width)


def report(analysis_root: Path) -> None:
    per_bundle, total = review_counts(analysis_root)
    rows, graded, ungraded = conformance_rows(analysis_root)

    flagged = {b: c[REVIEW_FLAGGED_WRONG] for b, c in per_bundle.items()
               if c[REVIEW_FLAGGED_WRONG]}

    def side(bundle: str) -> str:
        return TRUTH_SIDE if bundle in flagged else ATTESTED_CLEAN

    truth_frames = sum(total.values())
    print(f"corpus: {len(per_bundle)} bundles with truth, {truth_frames:,} truth frames")
    for review in (REVIEW_AUTO, REVIEW_FLAGGED_WRONG, REVIEW_FLAGGED_ABSENT):
        n = total[review]
        share = f"{100 * n / truth_frames:.1f}%" if truth_frames else "-"
        print(f"  {review:<22} {n:>7,}  {share:>6}")

    # Every bundle carrying a human wrong-flag, with how often the gate caught it.
    non_conforming = Counter(r["bundle"] for r in rows)
    record_totals: Counter = Counter()
    for record_path in analysis_root.glob("*/*/evaluations/*.json"):
        bundle = record_path.parent.parent
        record_totals[f"{bundle.parent.name}/{bundle.name}"] += 1

    _rule()
    print(f"human-attested wrong-person bundles ({len(flagged)}):")
    print(f"  {'bundle':<48} {'wrong':>7} {'of':>7}  non-conforming")
    for bundle, wrong in sorted(flagged.items(), key=lambda kv: -kv[1]):
        frames = sum(per_bundle[bundle].values())
        caught = f"{non_conforming[bundle]}/{record_totals[bundle]}"
        print(f"  {bundle[:48]:<48} {wrong:>7,} {frames:>7,}  {caught:>13}")
    print(f"  total flagged frames: {sum(flagged.values()):,} "
          f"({100 * sum(flagged.values()) / truth_frames:.1f}% of truth frames)")

    # The finding: which side does each cause actually land on?
    _rule()
    print(f"non-conforming records: {len(rows)} of {graded} graded")
    if ungraded:
        print(f"  ungraded (no conformance block, predates the #15 gate): {len(ungraded)}")
        for name in ungraded:
            print(f"    {name}")
    split: Counter = Counter()
    bundles_seen: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (row["cause"], side(row["bundle"]))
        split[key] += 1
        bundles_seen.setdefault(key, set()).add(row["bundle"])
    print(f"  {'cause':<22} {'side':<28} {'records':>8} {'bundles':>8}")
    for key in sorted(split, key=lambda k: (k[0], -split[k])):
        print(f"  {key[0]:<22} {key[1]:<28} {split[key]:>8} {len(bundles_seen[key]):>8}")

    for cause in sorted({r["cause"] for r in rows}):
        truth_side = split[(cause, TRUTH_SIDE)]
        clean = split[(cause, ATTESTED_CLEAN)]
        if not (truth_side + clean):
            continue
        print(f"  -> {cause}: {truth_side} truth-side / {clean} attested-clean "
              f"= {100 * clean / (truth_side + clean):.0f}% not truth-side identity error")

    # A left/right swap displaces joints horizontally, so an x-only failure is its
    # signature. #16 loosened the x floor blaming narrow horizontal spread; the two
    # explanations have never been distinguished (#148 H2).
    _rule()
    print(f"failing axis, {DIVERGENCE} only:")
    axes: Counter = Counter()
    for row in rows:
        if row["cause"] in (DIVERGENCE, LEGACY_DIVERGENCE):
            axes[(side(row["bundle"]), row["axis"])] += 1
    print(f"  {'side':<28} {'both':>6} {'x-only':>7} {'y-only':>7} {'other':>6}")
    for s in (ATTESTED_CLEAN, TRUTH_SIDE):
        print(f"  {s:<28} {axes[(s, 'both')]:>6} {axes[(s, 'x-only')]:>7} "
              f"{axes[(s, 'y-only')]:>7} {axes[(s, 'other')]:>6}")
    print("  x-only is consistent with laterality (#148 H2); both-axes is not.")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "analysis"
    if not root.is_dir():
        print(f"no analysis root at {root}", file=sys.stderr)
        return 2
    report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
