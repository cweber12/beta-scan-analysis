"""Fit SUGGESTION_THRESHOLDS from backfilled Video Stats + existing hand labels.

Per issue #23: after the backfill sweep, thresholds are derived by comparing the
computed continuous stats against the corpus's existing hand-labeled
``setup.json.analysisInputs`` records — simple decision-stump fits between
adjacent ordinal classes, numpy-only. Labels with too few classes fall back to
corpus distribution percentiles (annotated as such). The printed block is pasted
into ``video_stats.SUGGESTION_THRESHOLDS`` with fit date and corpus size.

Run:  python -m scripts.fit_suggestion_thresholds [analysis_root]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(d: dict | None, *path: str):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def collect(analysis_root: Path) -> list[dict]:
    """One row per bundle: hand labels + the stats that drive each suggestion."""
    rows = []
    for metadata_path in sorted(analysis_root.glob("*/*/metadata.json")):
        bundle = metadata_path.parent
        metadata = _load(metadata_path)
        setup_path = bundle / "setup.json"
        setup = _load(setup_path) if setup_path.exists() else {}
        stats_path = bundle / "video-stats.json"
        artifact = _load(stats_path) if stats_path.exists() else {}
        source = metadata.get("video_stats") or {}
        region = artifact.get("regionStats") or {}
        labels = setup.get("analysisInputs") or {}
        rows.append({
            "bundle": f"{bundle.parent.name}/{bundle.name}",
            "labels": labels,
            "shadowFraction": _get(region, "shadow", "fraction", "mean"),
            "blobLargestFraction": _get(region, "shadow", "blobs", "largestFraction"),
            "deltaE": _get(region, "climberWall", "deltaE"),
            "wallRms": _get(region, "wall", "rmsContrast"),
            "sharpness": _get(source, "sharpness", "mean"),
            "frameDiff": _get(source, "frameDiff", "mean"),
        })
    return rows


def _stump(lower: list[float], upper: list[float]) -> tuple[float, float] | None:
    """Best single cut putting ``lower`` below and ``upper`` above.

    Returns (threshold, balanced_accuracy) or None when a class is empty.
    """
    if not lower or not upper:
        return None
    lo, up = np.asarray(lower), np.asarray(upper)
    values = np.sort(np.unique(np.concatenate([lo, up])))
    if len(values) < 2:
        return None
    cuts = (values[:-1] + values[1:]) / 2.0
    best = max(cuts, key=lambda t: (lo < t).mean() + (up >= t).mean())
    acc = ((lo < best).mean() + (up >= best).mean()) / 2.0
    return round(float(best), 4), round(float(acc), 3)


def _split(rows: list[dict], label: str, stat: str, classes: dict[str, list[str]]):
    """Group a stat's values by label class; unknown/absent rows drop out."""
    out = {group: [] for group in classes}
    for row in rows:
        value = row[stat]
        lab = (row["labels"].get(label) or "unknown").strip().lower()
        if value is None:
            continue
        for group, members in classes.items():
            if lab in members:
                out[group].append(value)
    return out


# A stump that barely beats coin-flipping on the training labels is noise, not a
# fit; below this balanced accuracy the distribution fallback is more honest.
_MIN_STUMP_ACC = 0.7


def fit(rows: list[dict]) -> dict:
    thresholds: dict[str, dict] = {}
    notes: list[str] = []

    def _gated(res: tuple[float, float] | None):
        return res if res and res[1] >= _MIN_STUMP_ACC else None

    # --- shadows: none vs any, on wall shadow fraction -----------------------
    groups = _split(rows, "shadows", "shadowFraction",
                    {"none": ["none"], "some": ["low", "medium", "high"]})
    fit_none = _gated(_stump(groups["none"], groups["some"]))
    if fit_none:
        none_max, acc = fit_none
        notes.append(f"shadows.noneMaxFraction stump acc={acc} "
                     f"(n none={len(groups['none'])}, present={len(groups['some'])})")
    else:
        vals = [r["shadowFraction"] for r in rows if r["shadowFraction"] is not None]
        none_max = round(float(np.percentile(vals, 10)), 4)
        notes.append(f"shadows.noneMaxFraction: no usable 'none' stump "
                     f"(n none={len(groups['none'])} or acc < {_MIN_STUMP_ACC}); "
                     "10th-percentile fallback")
    # solid-vs-patchy is structural (one blob dominates); the legacy labels are
    # intensity-graded, so this cut is a stated prior, not a label fit.
    thresholds["shadows"] = {
        "noneMaxFraction": none_max,
        "solidMinLargestBlobFraction": 0.8,
    }

    # --- climber_contrast: low < medium < high on deltaE ----------------------
    groups = _split(rows, "climber_contrast", "deltaE",
                    {"low": ["low"], "medium": ["medium"], "high": ["high"]})
    lo_cut = _gated(_stump(groups["low"], groups["medium"] + groups["high"]))
    hi_cut = _gated(_stump(groups["low"] + groups["medium"], groups["high"]))
    if lo_cut and hi_cut and lo_cut[0] < hi_cut[0]:
        thresholds["climber_contrast"] = {"low": lo_cut[0], "high": hi_cut[0]}
        notes.append(f"climber_contrast stumps acc={lo_cut[1]}/{hi_cut[1]} "
                     f"(n={[len(groups[g]) for g in ('low', 'medium', 'high')]})")
    else:
        vals = [r["deltaE"] for r in rows if r["deltaE"] is not None]
        thresholds["climber_contrast"] = {
            "low": round(float(np.percentile(vals, 33)), 4),
            "high": round(float(np.percentile(vals, 67)), 4),
        }
        notes.append("climber_contrast: stump degenerate; tertile fallback")

    # --- wall_contrast on wall RMS contrast (labels too sparse to fit) -------
    groups = _split(rows, "wall_contrast", "wallRms",
                    {"low": ["low"], "medium": ["medium"], "high": ["high"]})
    lo_cut = _gated(_stump(groups["low"], groups["medium"] + groups["high"]))
    hi_cut = _gated(_stump(groups["low"] + groups["medium"], groups["high"]))
    if lo_cut and hi_cut and lo_cut[0] < hi_cut[0]:
        thresholds["wall_contrast"] = {"low": lo_cut[0], "high": hi_cut[0]}
        notes.append(f"wall_contrast stumps acc={lo_cut[1]}/{hi_cut[1]}")
    else:
        vals = [r["wallRms"] for r in rows if r["wallRms"] is not None]
        thresholds["wall_contrast"] = {
            "low": round(float(np.percentile(vals, 33)), 4),
            "high": round(float(np.percentile(vals, 67)), 4),
        }
        notes.append(f"wall_contrast: labels insufficient "
                     f"(n={[len(groups[g]) for g in ('low', 'medium', 'high')]}); "
                     "corpus tertiles")

    # --- motion_blur on source sharpness (all labels unknown today) ----------
    groups = _split(rows, "motion_blur", "sharpness",
                    {"low": ["low"], "medium": ["medium"], "high": ["high"]})
    # NB the mapping inverts (low sharpness = high blur): the "low" cut is the
    # sharpness below which blur reads high.
    hi_cut = _gated(_stump(groups["high"], groups["medium"] + groups["low"]))
    lo_cut = _gated(_stump(groups["high"] + groups["medium"], groups["low"]))
    if hi_cut and lo_cut and hi_cut[0] < lo_cut[0]:
        thresholds["motion_blur"] = {"low": hi_cut[0], "high": lo_cut[0]}
        notes.append(f"motion_blur stumps acc={hi_cut[1]}/{lo_cut[1]}")
    else:
        vals = [r["sharpness"] for r in rows if r["sharpness"] is not None]
        thresholds["motion_blur"] = {
            "low": round(float(np.percentile(vals, 33)), 4),
            "high": round(float(np.percentile(vals, 67)), 4),
        }
        notes.append(f"motion_blur: labels insufficient "
                     f"(n={[len(groups[g]) for g in ('low', 'medium', 'high')]}); "
                     "corpus sharpness tertiles")

    # --- camera_stability: steady vs moving on frame-diff energy -------------
    groups = _split(rows, "camera_stability", "frameDiff",
                    {"steady": ["steady"], "moving": ["moving"]})
    fit_move = _gated(_stump(groups["steady"], groups["moving"]))
    if fit_move:
        thresholds["camera_stability"] = {"movingMinFrameDiff": fit_move[0]}
        notes.append(f"camera_stability stump acc={fit_move[1]} "
                     f"(n steady={len(groups['steady'])}, moving={len(groups['moving'])})")
    else:
        vals = [r["frameDiff"] for r in rows if r["frameDiff"] is not None]
        thresholds["camera_stability"] = {
            "movingMinFrameDiff": round(float(np.percentile(vals, 80)), 4)
        }
        notes.append("camera_stability: labels insufficient; 80th-percentile fallback")

    return {"thresholds": thresholds, "notes": notes}


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0]) if argv else Path("analysis")
    rows = collect(root.resolve())
    labeled = sum(1 for r in rows if r["labels"])
    result = fit(rows)

    print(f"corpus: {len(rows)} bundles, {labeled} with analysisInputs\n")
    for note in result["notes"]:
        print(f"  - {note}")
    block = {
        "fitDate": date.today().isoformat(),
        "corpusSize": len(rows),
        "labeledBundles": labeled,
        **result["thresholds"],
    }
    print("\nSUGGESTION_THRESHOLDS = " + json.dumps(block, indent=4))


if __name__ == "__main__":
    main()
