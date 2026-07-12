"""ORB all-pairs cross-match: load the scanner-produced match matrix and reduce it
to the ORB outcome the report renders.

The scanner repo runs the batch cross-match (train = a video's wall-crop ORB
features on its reference frame; query = another video's ``final_frame.png``) and
writes one ``reports/orb_match_matrix.json`` with a row per ordered ``(train,
query)`` pair. See ``docs/handoffs/scanner-data-contract.md`` and ADR 0002.

The headline ORB outcome is **route-ID separation**: same-route pairs should match
(high inlier ratio), cross-route pairs should not. Everything here is descriptive
and dependency-light (numpy/pandas only); the diagonal (train == query) is a
same-session upper-bound control and is excluded from the separation/precision
statistics but kept for the heatmap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Columns every pair row is normalised to (missing fields -> NaN/None).
PAIR_COLUMNS = [
    "trainKey", "trainRoute", "queryKey", "queryRoute", "sameRoute",
    "matches", "inliers", "inlierRatio", "homographyValid", "reprojErrorPx",
]


def load_match_matrix(path: Path) -> pd.DataFrame:
    """Load ``orb_match_matrix.json`` into a normalised pairs DataFrame.

    Returns an empty (correctly-typed) frame when the file is absent or malformed,
    so the whole pipeline still runs before the scanner has produced a matrix.
    """

    if not path or not Path(path).exists():
        return pd.DataFrame(columns=PAIR_COLUMNS)
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return pd.DataFrame(columns=PAIR_COLUMNS)

    pairs = blob.get("pairs", []) if isinstance(blob, dict) else []
    if not pairs:
        return pd.DataFrame(columns=PAIR_COLUMNS)

    df = pd.DataFrame(pairs)
    for col in PAIR_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # Derive inlierRatio when omitted but the raw counts are present.
    mask = df["inlierRatio"].isna() & df["matches"].notna()
    if mask.any():
        m = pd.to_numeric(df.loc[mask, "matches"], errors="coerce")
        i = pd.to_numeric(df.loc[mask, "inliers"], errors="coerce")
        df.loc[mask, "inlierRatio"] = (i / m).where(m > 0, 0.0)
    # sameRoute may be absent — derive from the route labels when we can.
    smask = df["sameRoute"].isna() & df["trainRoute"].notna() & df["queryRoute"].notna()
    if smask.any():
        df.loc[smask, "sameRoute"] = df.loc[smask, "trainRoute"] == df.loc[smask, "queryRoute"]
    return df[PAIR_COLUMNS]


def _offdiagonal(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the train == query diagonal (a trivial same-session control)."""

    if df.empty:
        return df
    return df[df["trainKey"] != df["queryKey"]]


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """Rank-based AUC (Mann-Whitney U / n_pos·n_neg): P(pos ranks above neg).

    0.5 = no separation, 1.0 = perfectly separable. Ties count as 0.5. No scipy.
    """

    if not pos or not neg:
        return None
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    # Average ranks over ties (1-indexed).
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # mean of the 1-indexed positions [i..j]
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def separation_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Same-route vs cross-route inlier-ratio separation (off-diagonal only)."""

    off = _offdiagonal(df)
    if off.empty:
        return {"available": False}
    ratio = pd.to_numeric(off["inlierRatio"], errors="coerce")
    same = ratio[off["sameRoute"] == True].dropna().tolist()  # noqa: E712
    cross = ratio[off["sameRoute"] == False].dropna().tolist()  # noqa: E712
    if not same or not cross:
        return {"available": False}
    same_s, cross_s = pd.Series(same), pd.Series(cross)
    return {
        "available": True,
        "n_same": len(same),
        "n_cross": len(cross),
        "same_mean": float(same_s.mean()),
        "same_median": float(same_s.median()),
        "cross_mean": float(cross_s.mean()),
        "cross_median": float(cross_s.median()),
        "separation": float(same_s.mean() - cross_s.mean()),
        "auc": _auc(same, cross),
    }


def best_threshold(df: pd.DataFrame) -> dict[str, Any]:
    """Sweep inlier-ratio thresholds for the best-F1 route-ID operating point.

    A pair is *predicted same-route* when its inlierRatio >= threshold; scored
    against the sameRoute ground truth over off-diagonal pairs.
    """

    off = _offdiagonal(df)
    if off.empty:
        return {"available": False}
    scored = off.assign(_r=pd.to_numeric(off["inlierRatio"], errors="coerce")).dropna(subset=["_r"])
    if scored.empty or scored["sameRoute"].nunique() < 2:
        return {"available": False}

    candidates = sorted(set(scored["_r"].tolist()))
    best = None
    for thr in candidates:
        pred = scored["_r"] >= thr
        truth = scored["sameRoute"] == True  # noqa: E712
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if best is None or f1 > best["f1"]:
            best = {"available": True, "threshold": float(thr), "precision": precision,
                    "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return best or {"available": False}


def ordered_matrix(df: pd.DataFrame, value: str = "inlierRatio") -> dict[str, Any]:
    """Square grid of ``value`` ordered so same-route videos block on the diagonal.

    Rows = train keys, cols = query keys, both sorted by (route, key). Returns the
    ordered keys, their routes, and a list-of-lists of values (None where a pair is
    absent) for the heatmap.
    """

    if df.empty:
        return {"available": False}
    train_route = dict(zip(df["trainKey"], df["trainRoute"]))
    query_route = dict(zip(df["queryKey"], df["queryRoute"]))
    route_of = {**query_route, **train_route}
    keys = sorted(set(df["trainKey"]) | set(df["queryKey"]),
                  key=lambda k: (str(route_of.get(k, "")), str(k)))
    lookup = {(r["trainKey"], r["queryKey"]): r.get(value) for _, r in df.iterrows()}
    grid = [[lookup.get((rk, qk)) for qk in keys] for rk in keys]
    grid = [[float(v) if isinstance(v, (int, float)) and pd.notna(v) else None for v in row]
            for row in grid]
    return {
        "available": True,
        "keys": keys,
        "routes": [str(route_of.get(k, "")) for k in keys],
        "values": grid,
        "value_name": value,
    }
