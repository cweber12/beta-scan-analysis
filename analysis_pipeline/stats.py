"""Cluster-aware, effect-size-first statistics.

The run is the independent unit. Per-frame correlations are computed *within* each
run and summarised as a mean coefficient plus its spread across runs; nothing here
emits a p-value. Categorical hand labels get group means + Cliff's delta. Everything
is exploratory at the current corpus size.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

UNKNOWN = "unknown"

# Column groups ---------------------------------------------------------------
FRAME_PROXY_OUTCOMES = ["kp_count", "mean_score"]
FRAME_PREDICTORS = [
    "climber_sharpness",
    "climber_luma_mean",
    "climber_luma_std",
    "wall_sharpness",
    "wall_luma_mean",
    "wall_luma_std",
    "coverage",
    "velocity",
]

RUN_OUTCOMES = [
    "out_detectionRate",
    "out_flipRate",
    "out_confidence_avg",
    "out_avgKeypointCount",
    "out_goodFrames",
    "out_badStretchCount",
]


def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    """Pearson, or Spearman as Pearson-on-ranks (avoids a scipy dependency)."""

    if method == "spearman":
        a, b = a.rank(), b.rank()
    return a.corr(b, method="pearson")


def _numeric_predictor_cols(run_df: pd.DataFrame) -> list[str]:
    extra = {"motionMagnitude", "climberCoverage_avg", "climberCoverage_min", "wall_crop_area"}
    out = []
    for c in run_df.columns:
        if not (c.startswith("ref_") or c in extra):
            continue
        if "_flag_" in c:
            continue
        if pd.api.types.is_numeric_dtype(run_df[c]):
            out.append(c)
    return out


# Label pruning ---------------------------------------------------------------
def prune_labels(
    run_df: pd.DataFrame, unknown_rate_max: float = 0.5
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (kept label cols, [(dropped col, reason)]).

    Drop a hand label when it has <2 distinct non-unknown values (no contrast) or
    when more than ``unknown_rate_max`` of rows are 'unknown' (untrustworthy).
    """

    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    n = len(run_df)
    for col in [c for c in run_df.columns if c.startswith("label_")]:
        values = run_df[col].astype("string").fillna(UNKNOWN)
        unknown_rate = (values == UNKNOWN).mean() if n else 1.0
        distinct = values[values != UNKNOWN].nunique()
        if unknown_rate > unknown_rate_max:
            dropped.append((col, f"{unknown_rate:.0%} unknown"))
        elif distinct < 2:
            only = sorted(set(values[values != UNKNOWN]))
            label = only[0] if only else "all-unknown"
            dropped.append((col, f"constant ({label})"))
        else:
            kept.append(col)
    return kept, dropped


# Per-frame within-run correlations ------------------------------------------
def within_run_correlations(
    frame_df: pd.DataFrame,
    predictors: list[str] | None = None,
    outcomes: list[str] | None = None,
    methods: tuple[str, ...] = ("pearson", "spearman"),
    min_pairs: int = 4,
) -> pd.DataFrame:
    predictors = predictors or [c for c in FRAME_PREDICTORS if c in frame_df.columns]
    outcomes = outcomes or [c for c in FRAME_PROXY_OUTCOMES if c in frame_df.columns]

    rows: list[dict[str, Any]] = []
    groups = list(frame_df.groupby(["video_key", "run_ts"]))
    for predictor in predictors:
        for outcome in outcomes:
            for method in methods:
                coeffs: list[float] = []
                for _, g in groups:
                    pair = g[[predictor, outcome]].dropna()
                    if len(pair) < min_pairs:
                        continue
                    if pair[predictor].nunique() < 2 or pair[outcome].nunique() < 2:
                        continue
                    r = _corr(pair[predictor], pair[outcome], method)
                    if pd.notna(r):
                        coeffs.append(float(r))
                if not coeffs:
                    continue
                s = pd.Series(coeffs)
                rows.append(
                    {
                        "predictor": predictor,
                        "outcome": outcome,
                        "method": method,
                        "mean_r": s.mean(),
                        "std_r": s.std(ddof=0),
                        "min_r": s.min(),
                        "max_r": s.max(),
                        "n_runs": len(coeffs),
                    }
                )
    return pd.DataFrame(rows)


# Categorical effect sizes ----------------------------------------------------
def cliffs_delta(a: list[float], b: list[float]) -> float | None:
    """Cliff's delta: P(a>b) - P(a<b), in [-1, 1]. None if either side empty."""

    if not a or not b:
        return None
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(a) * len(b))


def categorical_effects(
    run_df: pd.DataFrame,
    label_cols: list[str],
    outcome_cols: list[str] | None = None,
) -> pd.DataFrame:
    outcome_cols = outcome_cols or [c for c in RUN_OUTCOMES if c in run_df.columns]
    rows: list[dict[str, Any]] = []
    for label in label_cols:
        groups = run_df.groupby(label)
        # two largest non-unknown groups drive the Cliff's delta split
        sizes = [
            (g, len(idx))
            for g, idx in groups.groups.items()
            if str(g) != UNKNOWN
        ]
        sizes.sort(key=lambda t: -t[1])
        for outcome in outcome_cols:
            group_means = {
                str(g): float(sub[outcome].mean())
                for g, sub in groups
                if pd.notna(sub[outcome].mean())
            }
            delta = None
            split = None
            if len(sizes) >= 2:
                ga, gb = sizes[0][0], sizes[1][0]
                a = run_df.loc[run_df[label] == ga, outcome].dropna().tolist()
                b = run_df.loc[run_df[label] == gb, outcome].dropna().tolist()
                delta = cliffs_delta(a, b)
                split = f"{ga} vs {gb}"
            rows.append(
                {
                    "predictor": label,
                    "outcome": outcome,
                    "group_means": group_means,
                    "cliffs_delta": delta,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ORB reference-richness correlations ----------------------------------------
def orb_correlations(run_df: pd.DataFrame) -> pd.DataFrame:
    if "orb_refKeypointCount" not in run_df.columns:
        return pd.DataFrame()
    candidate_predictors = [
        "orb_ref_wall_sharpness",
        "orb_ref_wall_mean",
        "orb_ref_wall_stdDev",
        "orb_ref_overall_sharpness",
        "wall_crop_area",
        "ref_wall_sharpness",
    ]
    rows: list[dict[str, Any]] = []
    y = run_df["orb_refKeypointCount"]
    for pred in candidate_predictors:
        if pred not in run_df.columns:
            continue
        pair = pd.concat([run_df[pred], y], axis=1).dropna()
        if len(pair) < 3 or pair[pred].nunique() < 2:
            continue
        r = pair[pred].corr(pair.iloc[:, 1], method="pearson")
        if pd.notna(r):
            rows.append(
                {"predictor": pred, "outcome": "orb_refKeypointCount", "r": float(r), "n": len(pair)}
            )
    return pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)


def run_numeric_predictor_cols(run_df: pd.DataFrame) -> list[str]:
    return _numeric_predictor_cols(run_df)


def pooled_run_correlations(
    run_df: pd.DataFrame,
    predictors: list[str] | None = None,
    outcomes: list[str] | None = None,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """Pooled Pearson across runs (descriptive at small n).

    Shaped like ``within_run_correlations`` (mean_r/min_r/max_r/n_runs) so the same
    effect-size bar chart renders it; min_r == max_r == r (no within-run spread).
    """

    predictors = predictors or _numeric_predictor_cols(run_df)
    outcomes = outcomes or [c for c in RUN_OUTCOMES if c in run_df.columns]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            pair = run_df[[predictor, outcome]].dropna()
            if len(pair) < min_pairs:
                continue
            if pair[predictor].nunique() < 2 or pair[outcome].nunique() < 2:
                continue
            r = pair[predictor].corr(pair[outcome], method="pearson")
            if pd.isna(r):
                continue
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "method": "pearson",
                    "mean_r": float(r),
                    "std_r": 0.0,
                    "min_r": float(r),
                    "max_r": float(r),
                    "n_runs": len(pair),
                }
            )
    return pd.DataFrame(rows)
