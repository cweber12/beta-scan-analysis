"""Emit features CSVs + a self-contained, theme-aware HTML correlation report.

Charts are hand-rendered inline SVG using the dataviz skill's validated palette
(diverging blue<->red for signed correlation, categorical hues per run). No
plotting dependency. Everything is framed EXPLORATORY: the run is the unit.
"""

from __future__ import annotations

import base64
import html
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:  # optional: only used to embed downscaled final-frame thumbnails
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# --- validated dataviz palette (see references/palette.md) -------------------
BLUE = (0x2A, 0x78, 0xD6)   # positive correlation pole / series-1
RED = (0xE3, 0x49, 0x48)    # negative correlation pole
GRAY_LIGHT = (0xF0, 0xEF, 0xEC)  # diverging midpoint (light surface)
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


# --- colour helpers ----------------------------------------------------------
def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore


def diverging_color(r: float) -> str:
    """Map correlation r in [-1, 1] to blue(+) / gray(0) / red(-)."""

    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "#cccccc"
    r = max(-1.0, min(1.0, r))
    if r >= 0:
        return _rgb_to_hex(_lerp(GRAY_LIGHT, BLUE, r))
    return _rgb_to_hex(_lerp(GRAY_LIGHT, RED, -r))


def seq_color(v: float | None, lo: float = 0.0, hi: float = 1.0) -> str:
    """Sequential pale->blue ramp for a value in [lo, hi] (e.g. inlier ratio)."""

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "#cccccc"
    t = (v - lo) / (hi - lo) if hi > lo else 0.0
    t = max(0.0, min(1.0, t))
    return _rgb_to_hex(_lerp(GRAY_LIGHT, BLUE, t))


def _thumb_data_uri(path: Path | str | None, max_w: int = 240) -> str | None:
    """Downscaled JPEG data-URI for a final-frame thumbnail, or None.

    Requires cv2; keeps the self-contained report from ballooning by capping the
    width and JPEG-encoding. Silently returns None when cv2 is absent or the read
    fails, so cards degrade to text-only.
    """

    if cv2 is None or not path or not Path(path).exists():
        return None
    try:
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:  # pragma: no cover - defensive
        return None


def _esc(v: Any) -> str:
    return html.escape(str(v))


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


# --- SVG chart builders ------------------------------------------------------
def svg_heatmap(corr: pd.DataFrame, title: str) -> str:
    """corr: long df with predictor, outcome, mean_r (+ optional std_r)."""

    if corr.empty:
        return f"<p class='muted'>No {_esc(title)} to show.</p>"
    predictors = list(dict.fromkeys(corr["predictor"]))
    outcomes = list(dict.fromkeys(corr["outcome"]))
    lookup = {(r["predictor"], r["outcome"]): r for _, r in corr.iterrows()}

    cw, ch = 96, 40
    left, top = 160, 70
    w = left + cw * len(outcomes) + 16
    h = top + ch * len(predictors) + 16

    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart' width='{w}' height='{h}'>"]
    for j, oc in enumerate(outcomes):
        x = left + j * cw + cw / 2
        parts.append(
            f"<text x='{x:.0f}' y='{top-12}' text-anchor='middle' class='axis'>{_esc(oc)}</text>"
        )
    for i, pr in enumerate(predictors):
        y = top + i * ch
        parts.append(
            f"<text x='{left-10}' y='{y+ch/2+4:.0f}' text-anchor='end' class='axis'>{_esc(pr)}</text>"
        )
        for j, oc in enumerate(outcomes):
            x = left + j * cw
            row = lookup.get((pr, oc))
            if row is None:
                parts.append(
                    f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' "
                    f"fill='none' stroke='var(--grid)'/>"
                )
                continue
            r = row["mean_r"]
            spread = row.get("std_r")
            fill = diverging_color(r)
            tip = f"{pr} → {oc}: mean r={_fmt(r)} (±{_fmt(spread)}, n_runs={int(row.get('n_runs', 0))})"
            parts.append(
                f"<rect x='{x+2}' y='{y+2}' width='{cw-4}' height='{ch-4}' rx='4' fill='{fill}'>"
                f"<title>{_esc(tip)}</title></rect>"
            )
            txtcol = "#0b0b0b" if abs(r) < 0.55 else "#ffffff"
            parts.append(
                f"<text x='{x+cw/2:.0f}' y='{y+ch/2+4:.0f}' text-anchor='middle' "
                f"style='fill:{txtcol}' class='cell'>{_fmt(r)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def svg_effect_bars(corr: pd.DataFrame, title: str, top_n: int = 14) -> str:
    """Horizontal bars of mean_r with within-run min..max whiskers."""

    if corr.empty:
        return f"<p class='muted'>No {_esc(title)} to show.</p>"
    d = corr.reindex(corr["mean_r"].abs().sort_values(ascending=False).index).head(top_n)

    rowh, left, right, top = 30, 260, 30, 20
    plot_w = 360
    w = left + plot_w + right
    h = top + rowh * len(d) + 30
    cx = left + plot_w / 2  # r = 0

    def xr(r: float) -> float:
        return left + plot_w * (r + 1) / 2

    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart' width='{w}' height='{h}'>"]
    # zero axis + -1/0/1 ticks
    parts.append(f"<line x1='{cx}' y1='{top}' x2='{cx}' y2='{top+rowh*len(d)}' class='grid'/>")
    for tick in (-1, 0, 1):
        x = xr(tick)
        parts.append(f"<text x='{x:.0f}' y='{top+rowh*len(d)+18:.0f}' text-anchor='middle' class='axis'>{tick}</text>")
    for i, (_, row) in enumerate(d.iterrows()):
        y = top + i * rowh + rowh / 2
        label = f"{row['predictor']} → {row['outcome']}"
        parts.append(
            f"<text x='{left-10}' y='{y+4:.0f}' text-anchor='end' class='axis'>{_esc(label)}</text>"
        )
        r = row["mean_r"]
        col = diverging_color(r)
        x0, x1 = (cx, xr(r)) if r >= 0 else (xr(r), cx)
        parts.append(
            f"<rect x='{x0:.1f}' y='{y-7:.0f}' width='{max(1,abs(x1-x0)):.1f}' height='14' rx='4' fill='{col}'>"
            f"<title>{_esc(label)}: mean r={_fmt(r)} (min {_fmt(row.get('min_r'))}, max {_fmt(row.get('max_r'))}, n_runs={int(row.get('n_runs',0))})</title></rect>"
        )
        # whisker across within-run spread
        wl, wr = xr(row.get("min_r", r)), xr(row.get("max_r", r))
        parts.append(f"<line x1='{wl:.1f}' y1='{y:.0f}' x2='{wr:.1f}' y2='{y:.0f}' class='whisker'/>")
        for wx in (wl, wr):
            parts.append(f"<line x1='{wx:.1f}' y1='{y-4:.0f}' x2='{wx:.1f}' y2='{y+4:.0f}' class='whisker'/>")
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(frame_df: pd.DataFrame, predictor: str, outcome: str) -> str:
    pair = frame_df[["video_key", predictor, outcome]].dropna()
    if len(pair) < 3:
        return ""
    runs = list(dict.fromkeys(pair["video_key"]))
    colours = {r: CATEGORICAL[i % len(CATEGORICAL)] for i, r in enumerate(runs)}

    W, H = 460, 300
    pad_l, pad_b, pad_t, pad_r = 52, 44, 16, 12
    xs, ys = pair[predictor], pair[outcome]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def px(x): return pad_l + (x - xmin) / xr * (W - pad_l - pad_r)
    def py(y): return H - pad_b - (y - ymin) / yr * (H - pad_b - pad_t)

    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' class='chart' width='{W}' height='{H}'>"]
    parts.append(f"<line x1='{pad_l}' y1='{H-pad_b}' x2='{W-pad_r}' y2='{H-pad_b}' class='grid'/>")
    parts.append(f"<line x1='{pad_l}' y1='{pad_t}' x2='{pad_l}' y2='{H-pad_b}' class='grid'/>")
    parts.append(f"<text x='{(pad_l+W-pad_r)/2:.0f}' y='{H-8}' text-anchor='middle' class='axis'>{_esc(predictor)}</text>")
    parts.append(f"<text x='14' y='{(pad_t+H-pad_b)/2:.0f}' text-anchor='middle' class='axis' transform='rotate(-90 14 {(pad_t+H-pad_b)/2:.0f})'>{_esc(outcome)}</text>")
    for _, row in pair.iterrows():
        parts.append(
            f"<circle cx='{px(row[predictor]):.1f}' cy='{py(row[outcome]):.1f}' r='3.4' "
            f"fill='{colours[row['video_key']]}' fill-opacity='0.75'><title>{_esc(row['video_key'])}</title></circle>"
        )
    parts.append("</svg>")
    legend = " ".join(
        f"<span class='chip'><i style='background:{colours[r]}'></i>{_esc(r)[:26]}</span>" for r in runs
    )
    return f"<div class='scatter'><h4>{_esc(predictor)} → {_esc(outcome)}</h4>{''.join(parts)}<div class='legend'>{legend}</div></div>"


def svg_orb_bars(orb: pd.DataFrame) -> str:
    if orb.empty:
        return "<p class='muted'>No ORB reference-richness correlations available.</p>"
    rowh, left, plot_w = 30, 240, 320
    w, h = left + plot_w + 30, 20 + rowh * len(orb) + 30
    cx = left + plot_w / 2

    def xr(r): return left + plot_w * (r + 1) / 2

    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart' width='{w}' height='{h}'>"]
    parts.append(f"<line x1='{cx}' y1='20' x2='{cx}' y2='{20+rowh*len(orb)}' class='grid'/>")
    for tick in (-1, 0, 1):
        parts.append(f"<text x='{xr(tick):.0f}' y='{20+rowh*len(orb)+18:.0f}' text-anchor='middle' class='axis'>{tick}</text>")
    for i, (_, row) in enumerate(orb.iterrows()):
        y = 20 + i * rowh + rowh / 2
        parts.append(f"<text x='{left-10}' y='{y+4:.0f}' text-anchor='end' class='axis'>{_esc(row['predictor'])}</text>")
        r = row["r"]
        col = diverging_color(r)
        x0, x1 = (cx, xr(r)) if r >= 0 else (xr(r), cx)
        parts.append(
            f"<rect x='{x0:.1f}' y='{y-7:.0f}' width='{max(1,abs(x1-x0)):.1f}' height='14' rx='4' fill='{col}'>"
            f"<title>{_esc(row['predictor'])}: r={_fmt(r)} (n={int(row['n'])})</title></rect>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --- table helpers -----------------------------------------------------------
def _df_to_table(df: pd.DataFrame, max_cols: int | None = None) -> str:
    if df.empty:
        return "<p class='muted'>(empty)</p>"
    cols = list(df.columns)[:max_cols] if max_cols else list(df.columns)
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(row[c]) if isinstance(row[c], float) else _esc(row[c])}</td>" for c in cols)
        body.append(f"<tr>{cells}</tr>")
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _dropped_table(dropped: list[tuple[str, str]]) -> str:
    if not dropped:
        return "<p class='muted'>No hand labels were pruned.</p>"
    rows = "".join(
        f"<tr><td>{_esc(c.replace('label_',''))}</td><td>{_esc(reason)}</td></tr>" for c, reason in dropped
    )
    return f"<div class='tablewrap'><table><thead><tr><th>dropped label</th><th>reason</th></tr></thead><tbody>{rows}</tbody></table></div>"


def _cat_table(cat: pd.DataFrame) -> str:
    if cat.empty:
        return "<p class='muted'>No categorical predictors survived pruning.</p>"
    rows = []
    for _, r in cat.iterrows():
        means = ", ".join(f"{k}={_fmt(v)}" for k, v in (r["group_means"] or {}).items())
        rows.append(
            f"<tr><td>{_esc(r['predictor'].replace('label_',''))}</td><td>{_esc(r['outcome'])}</td>"
            f"<td>{_esc(means)}</td><td>{_fmt(r['cliffs_delta'])}</td><td>{_esc(r['split'] or '')}</td></tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr><th>label</th><th>outcome</th>"
        "<th>group means</th><th>Cliff's δ</th><th>split</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _tier_badge(tier: str) -> str:
    label = "agreement" if tier == "agreement" else "accuracy"
    return f"<span class='flag tier'>{_esc(label)}</span>"


def _joint_ranking_table(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No evaluation-backed per-joint ranking available yet.</p>"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_tier_badge(str(r['tier']))}</td>"
            f"<td>{_esc(r['joint'])}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{_fmt(r['pck'])}</td>"
            f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
            f"<td>{_fmt(r['failure_rate'])}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>tier</th><th>joint</th><th>frame/joint n</th><th>PCK@0.5-torso</th>"
        "<th>bootstrap 95% CI</th><th>failure rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _condition_table(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No frame/joint condition trend rows available yet.</p>"
    name_map = {
        "size_frac": "climber size in frame (truth bbox height fraction)",
        "speed": "movement speed (inter-frame truth displacement)",
        "edge_dist": "edge distance (smaller = closer to frame edge)",
    }
    rows = []
    for _, r in df.sort_values(["condition", "tier", "band"]).iterrows():
        rng = f"[{_fmt(r['band_min'])}, {_fmt(r['band_max'])}]"
        rows.append(
            "<tr>"
            f"<td>{_tier_badge(str(r['tier']))}</td>"
            f"<td>{_esc(name_map.get(str(r['condition']), str(r['condition'])))}</td>"
            f"<td>band {int(r['band'])}</td>"
            f"<td>{int(r['n'])}</td>"
            f"<td>{_esc(rng)}</td>"
            f"<td>{_fmt(r['failure_rate'])}</td>"
            f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>tier</th><th>condition</th><th>quantile band</th><th>frame/joint n</th>"
        "<th>band range</th><th>failure rate</th><th>bootstrap 95% CI</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _cross_video_split_table(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No cross-video split summary available yet.</p>"
    rows = []
    for _, r in df.sort_values(["split", "metric", "tier", "value"]).iterrows():
        rows.append(
            "<tr>"
            f"<td>{_tier_badge(str(r['tier']))}</td>"
            f"<td>{_esc(r['split'])}</td>"
            f"<td>{_esc(r['value'])}</td>"
            f"<td>{_esc(r['metric'])}</td>"
            f"<td>{int(r['n_runs'])}</td>"
            f"<td>{_fmt(r['mean'])}</td>"
            f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>tier</th><th>split</th><th>value</th><th>metric</th><th>n runs</th>"
        "<th>mean</th><th>bootstrap 95% CI</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _version_overview_table(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ("<p class='muted'>No evaluation records carry a scanner appVersion "
                "yet.</p>")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td><code>{_esc(r['app_version'])}</code></td>"
            f"<td>{_esc(r['first_run_ts'])}</td>"
            f"<td>{_esc(r['last_run_ts'])}</td>"
            f"<td>{int(r['n_records'])}</td>"
            f"<td>{int(r['n_videos'])}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>appVersion</th><th>first run</th><th>last run</th>"
        "<th>eval records</th><th>videos</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _delta_cell(delta: float, lo: float, hi: float, *, lower_is_better: bool) -> str:
    """Render a delta with its CI, coloured only when the CI excludes zero."""

    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "<td>–</td><td>–</td>"
    cls = ""
    if lo > 0 or hi < 0:
        improved = (hi < 0) if lower_is_better else (lo > 0)
        cls = " class='sig-good'" if improved else " class='sig-bad'"
    sign = "+" if delta >= 0 else ""
    return (f"<td{cls}>{sign}{_fmt(delta, 3)}</td>"
            f"<td>[{_fmt(lo, 3)}, {_fmt(hi, 3)}]</td>")


def _version_delta_table(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ("<p class='muted'>No consecutive scanner versions share a truth "
                "revision on any bundle yet — deltas need evaluation records from "
                "at least two appVersions on the same video under the same truth.</p>")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td><code>{_esc(r['from_version'])}</code> → <code>{_esc(r['to_version'])}</code></td>"
            f"<td>{_tier_badge(str(r['tier']))}</td>"
            f"<td>{_esc(r['joint'])}</td>"
            f"<td>{int(r['n_from'])} / {int(r['n_to'])}</td>"
            f"<td>{_fmt(r['pck_from'], 3)} → {_fmt(r['pck_to'], 3)}</td>"
            + _delta_cell(r["pck_delta"], r["pck_ci_low"], r["pck_ci_high"],
                          lower_is_better=False)
            + f"<td>{_fmt(r['med_from'], 3)} → {_fmt(r['med_to'], 3)}</td>"
            + _delta_cell(r["med_delta"], r["med_ci_low"], r["med_ci_high"],
                          lower_is_better=True)
            + "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>version pair</th><th>tier</th><th>joint</th><th>n from / to</th>"
        "<th>PCK@0.5-torso</th><th>ΔPCK</th><th>95% CI</th>"
        "<th>median err</th><th>Δmedian</th><th>95% CI</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _shame_list_html(items: list[str], empty_text: str) -> str:
    if not items:
        return f"<p class='muted'>{_esc(empty_text)}</p>"
    rows = "".join(f"<tr><td>{_esc(v)}</td></tr>" for v in items)
    return f"<div class='tablewrap'><table><tbody>{rows}</tbody></table></div>"


# --- new sections: overview, failure cards, ORB matrix, frame timeline -------
_SOURCE_COLORS = {
    "raw": "#1baf7a", "interpolated": "#eda100", "filled": "#eb6834",
    "flipDiscarded": "#e34948", "limbExpanded": "#4a3aa7", "missing": "#c9c8c2",
}


def svg_histogram(values: list[float], lo: float = 0.0, hi: float = 1.0,
                  bins: int = 10, highlight_below: float | None = None) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "<p class='muted'>(no data)</p>"
    W, H, pad_l, pad_b, pad_t, pad_r = 420, 150, 30, 24, 10, 10
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in vals:
        b = int((min(max(v, lo), hi) - lo) / span * bins)
        counts[min(b, bins - 1)] += 1
    maxc = max(counts) or 1
    bw = (W - pad_l - pad_r) / bins
    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' class='chart' width='{W}' height='{H}'>"]
    parts.append(f"<line x1='{pad_l}' y1='{H-pad_b}' x2='{W-pad_r}' y2='{H-pad_b}' class='grid'/>")
    for i, c in enumerate(counts):
        x = pad_l + i * bw
        bh = (c / maxc) * (H - pad_b - pad_t)
        edge_hi = lo + span * (i + 1) / bins
        col = _rgb_to_hex(RED) if (highlight_below is not None and edge_hi <= highlight_below) else _rgb_to_hex(BLUE)
        parts.append(
            f"<rect x='{x+1:.1f}' y='{H-pad_b-bh:.1f}' width='{bw-2:.1f}' height='{bh:.1f}' rx='2' fill='{col}'>"
            f"<title>[{lo+span*i/bins:.2f}, {edge_hi:.2f}): {c}</title></rect>"
        )
    for frac, val in ((0.0, lo), (0.5, (lo + hi) / 2), (1.0, hi)):
        x = pad_l + frac * (W - pad_l - pad_r)
        parts.append(f"<text x='{x:.0f}' y='{H-8}' text-anchor='middle' class='axis'>{val:.2f}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _median_visible(hist: list[int]) -> str:
    """Median visible-joint count from a pre-binned histogram (index == count)."""

    total = sum(hist)
    if not total:
        return "–"
    mid, cum = total / 2, 0
    for i, c in enumerate(hist):
        cum += c
        if cum >= mid:
            return str(i)
    return str(len(hist) - 1)


def _low_confidence_html(ctx: dict[str, Any]) -> str:
    """Low-confidence truth: the visible-joint distribution (fit input for the
    measure-first gate) + a worst-first re-review worklist. Excludes nothing in
    v1 — see ``evaluate.MIN_VISIBLE_JOINTS``."""

    from .evaluate import MIN_VISIBLE_JOINTS
    from .trends import LOW_CONF_WORKLIST_TOP_K

    hist = ctx.get("visible_histogram") or []
    total = sum(int(c) for c in hist)
    if total == 0:
        return ("<p class='muted'>No matched-present truth frames measured yet "
                "(needs schema-v3 evaluation records).</p>")

    # Expand the pre-binned histogram back to values for svg_histogram — one
    # integer-wide bin per visible-count (0..13), so bins == len(hist).
    values = [i for i, c in enumerate(hist) for _ in range(int(c))]
    chart = svg_histogram(
        values, lo=0.0, hi=float(len(hist)), bins=len(hist),
        highlight_below=(None if MIN_VISIBLE_JOINTS is None else float(MIN_VISIBLE_JOINTS)))
    gate = ("no gate set — v1 measures only (fit N on #15-conforming bundles first)"
            if MIN_VISIBLE_JOINTS is None else
            f"gate active: &lt; {MIN_VISIBLE_JOINTS} visible joints excluded from PCK/normDist")

    tiles = _stat_tiles([
        (str(total), "matched-present frames measured"),
        (_median_visible(hist), "median visible joints"),
    ])

    worklist = ctx.get("low_conf_worklist")
    table = "<p class='muted'>(worklist empty)</p>"
    if isinstance(worklist, pd.DataFrame) and not worklist.empty:
        shown = worklist.head(LOW_CONF_WORKLIST_TOP_K)
        table = _df_to_table(shown)
        if len(worklist) > len(shown):
            table += (f"<p class='muted'>Showing the worst {len(shown)} of "
                      f"{len(worklist)} present truth frames — full list in "
                      "<code>eval_low_confidence_worklist.csv</code>.</p>")

    return (
        f"<p class='sub'>Distribution of visible (non-occluded) core joints over "
        f"matched-present truth frames — {gate}.</p>"
        "<div class='chartscroll'>" + chart + "</div>" + tiles
        + "<h3>Re-review worklist (fewest visible joints first)</h3>" + table)


def _stat_tiles(tiles: list[tuple[str, str]]) -> str:
    return "<div class='card'>" + "".join(
        f"<span class='stat'><b>{_esc(v)}</b>{_esc(lbl)}</span>" for v, lbl in tiles
    ) + "</div>"


def _overview_html(ctx: dict[str, Any]) -> str:
    run_df = ctx["run_df"]
    det = run_df["out_detectionRate"].dropna()
    median = det.median() if len(det) else None
    cata = run_df[run_df["out_detectionRate"] < 0.35][["video_key", "out_detectionRate"]]
    cata = cata.sort_values("out_detectionRate")
    oq = int(run_df["out_overlayQuality"].notna().sum())
    sep = ctx.get("orb_separation") or {}
    auc = sep.get("auc") if sep.get("available") else None

    tiles = [
        (_fmt(median) if median is not None else "–", "median detectionRate"),
        (str(len(cata)), "runs < 0.35"),
        (f"{oq}/{len(run_df)}", "runs w/ overlayQuality"),
        (_fmt(auc) if auc is not None else "–", "ORB route-ID AUC"),
    ]
    hist = svg_histogram(det.tolist(), 0.0, 1.0, 10, highlight_below=0.35)

    cata_rows = "".join(
        f"<tr><td>{_esc(r['video_key'])[:44]}</td><td>{_fmt(r['out_detectionRate'])}</td></tr>"
        for _, r in cata.iterrows()
    )
    cata_tbl = (
        "<div class='tablewrap'><table><thead><tr><th>catastrophic run</th>"
        f"<th>detectionRate</th></tr></thead><tbody>{cata_rows}</tbody></table></div>"
        if cata_rows else "<p class='muted'>No runs below 0.35.</p>"
    )
    # label unknown rates (worth capturing more carefully next time)
    unk = []
    n = len(run_df)
    for c in [c for c in run_df.columns if c.startswith("label_")]:
        rate = (run_df[c].astype("string").fillna("unknown") == "unknown").mean() if n else 0.0
        if rate > 0:
            unk.append((c.replace("label_", ""), rate))
    unk.sort(key=lambda t: -t[1])
    unk_txt = ", ".join(f"{name} {rate:.0%}" for name, rate in unk[:6]) or "none"

    return (
        _stat_tiles(tiles)
        + "<div class='grid2'>"
        + f"<div><h4>detectionRate distribution</h4>{hist}</div>"
        + f"<div><h4>catastrophic failures</h4>{cata_tbl}</div>"
        + "</div>"
        + f"<p class='sub'>Label <code>unknown</code> rates: {_esc(unk_txt)}.</p>"
    )


def _failure_cards_html(ctx: dict[str, Any]) -> str:
    run_df = ctx["run_df"]
    finals = ctx.get("final_frames", {})
    sort_col = "out_overlayQuality" if run_df["out_overlayQuality"].notna().any() else "out_detectionRate"
    scored = run_df.dropna(subset=[sort_col])
    if scored.empty:
        return "<p class='muted'>No scored runs to card.</p>"
    idx = scored.groupby("video_key")[sort_col].idxmin()  # worst run per video
    reps = scored.loc[idx].sort_values(sort_col).head(12)

    flag_cols = [c for c in run_df.columns if c.startswith("ref_flag_")]
    cards = []
    for _, r in reps.iterrows():
        thumb = _thumb_data_uri(finals.get(r["video_key"]))
        img = f"<img src='{thumb}' alt=''/>" if thumb else "<div class='noimg'>no thumbnail</div>"
        oq = r.get("out_overlayQuality")
        metrics = f"detRate {_fmt(r['out_detectionRate'])} · flip {_fmt(r.get('out_flipRate'))}"
        if pd.notna(oq):
            metrics = f"overlayQ {_fmt(oq)} · " + metrics
        cond = f"coverage {_fmt(r.get('climberCoverage_avg'))} · motion {_fmt(r.get('motionMagnitude'))}"
        flags = [c.replace("ref_flag_is", "") for c in flag_cols if bool(r.get(c))]
        flagchips = "".join(f"<span class='flag'>{_esc(f)}</span>" for f in flags) or \
            "<span class='muted'>no adverse flags</span>"
        cards.append(
            f"<div class='vcard'>{img}<div class='vc-body'>"
            f"<h4>{_esc(r['video_key'])[:34]}</h4>"
            f"<div class='muted vc-route'>{_esc(r['route_folder'])}</div>"
            f"<div class='vc-metrics'>{metrics}</div>"
            f"<div class='muted vc-cond'>{cond}</div>"
            f"<div class='vc-flags'>{flagchips}</div></div></div>"
        )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def svg_orb_matrix(mtx: dict[str, Any]) -> str:
    keys = mtx["keys"]
    routes = mtx["routes"]
    vals = mtx["values"]
    n = len(keys)
    cell, band = 16, 8
    left = top = band + 2
    w = left + cell * n + 8
    h = top + cell * n + 8
    uniq = list(dict.fromkeys(routes))
    route_colors = {rt: CATEGORICAL[i % len(CATEGORICAL)] for i, rt in enumerate(uniq)}

    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart' width='{w}' height='{h}'>"]
    for j, rt in enumerate(routes):
        parts.append(f"<rect x='{left+j*cell}' y='0' width='{cell}' height='{band}' fill='{route_colors[rt]}'><title>{_esc(rt)}</title></rect>")
        parts.append(f"<rect x='0' y='{top+j*cell}' width='{band}' height='{cell}' fill='{route_colors[rt]}'><title>{_esc(rt)}</title></rect>")
    for i in range(n):
        for j in range(n):
            v = vals[i][j]
            x, y = left + j * cell, top + i * cell
            fill = seq_color(v) if v is not None else "none"
            stroke = "" if v is not None else " stroke='var(--grid)'"
            tip = f"{keys[i]} → {keys[j]}: {'–' if v is None else f'{v:.2f}'}"
            parts.append(f"<rect x='{x}' y='{y}' width='{cell-1}' height='{cell-1}' fill='{fill}'{stroke}><title>{_esc(tip)}</title></rect>")
    parts.append("</svg>")
    return "".join(parts)


def _orb_matrix_html(ctx: dict[str, Any]) -> str:
    mtx = ctx.get("orb_matrix") or {"available": False}
    if not mtx.get("available"):
        return ("<p class='muted'>No ORB cross-match matrix yet. Produce "
                "<code>reports/orb_match_matrix.json</code> in the scanner repo "
                "(see <code>docs/handoffs/scanner-data-contract.md</code>) and re-run with "
                "<code>--matrix</code>.</p>")
    sep = ctx.get("orb_separation") or {}
    thr = ctx.get("orb_threshold") or {}
    tiles = ""
    if sep.get("available"):
        tiles = _stat_tiles([
            (_fmt(sep["same_mean"]), "same-route mean inlierRatio"),
            (_fmt(sep["cross_mean"]), "cross-route mean"),
            (_fmt(sep["separation"]), "separation"),
            (_fmt(sep.get("auc")), "AUC"),
        ])
    thr_txt = ""
    if thr.get("available"):
        thr_txt = (f"<p class='sub'>Best-F1 route-ID at inlierRatio ≥ {thr['threshold']:.2f}: "
                   f"precision {thr['precision']:.2f}, recall {thr['recall']:.2f}, "
                   f"F1 {thr['f1']:.2f}. Rows = train (wall crop), cols = query "
                   f"(final_frame); the coloured band marks each video's route.</p>")
    return tiles + thr_txt + "<div class='chartscroll'>" + svg_orb_matrix(mtx) + "</div>"


def svg_frame_timeline(sub: pd.DataFrame, label: str) -> str:
    rows = sub.sort_values("t")
    n = len(rows)
    if n == 0:
        return ""
    cell = max(2, min(9, int(560 / n)))
    W, H = cell * n + 2, 20
    parts = [f"<div class='tl'><span class='tl-label'>{_esc(label)[:32]}</span>",
             f"<svg viewBox='0 0 {W} {H}' role='img' class='chart' width='{W}' height='{H}'>"]
    for i, (_, r) in enumerate(rows.iterrows()):
        src = r.get("source")
        col = _SOURCE_COLORS.get(str(src), "#c9c8c2")
        parts.append(
            f"<rect x='{i*cell}' y='2' width='{max(1,cell-1)}' height='{H-4}' fill='{col}'>"
            f"<title>t={_fmt(r['t'])} · {_esc(src)}</title></rect>"
        )
    parts.append("</svg></div>")
    return "".join(parts)


def _frame_timeline_html(ctx: dict[str, Any]) -> str:
    fdf = ctx["frame_df"]
    if "source" not in fdf.columns or fdf["source"].notna().sum() == 0:
        return ("<p class='muted'>No per-frame provenance yet — needs the scanner's per-frame "
                "<code>source</code> export (Phase 2 of the data contract). Once present, each "
                "run shows raw-detect vs interpolated/filled/flip-discarded spans over time.</p>")
    # Rank runs by share of non-raw frames; show the worst few.
    order = (fdf.assign(_bad=(fdf["source"] != "raw"))
             .groupby(["video_key", "run_ts"])["_bad"].mean().sort_values(ascending=False))
    strips = []
    for (vk, rt), _ in list(order.items())[:8]:
        sub = fdf[(fdf["video_key"] == vk) & (fdf["run_ts"] == rt)]
        strips.append(svg_frame_timeline(sub, vk))
    legend = " ".join(
        f"<span class='chip'><i style='background:{c}'></i>{_esc(k)}</span>"
        for k, c in _SOURCE_COLORS.items()
    )
    return "".join(strips) + f"<div class='legend'>{legend}</div>"


# --- top-level assembly ------------------------------------------------------
_CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--baseline:#c3c2b7;--accent:#2a78d6;}
:root[data-theme=dark]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--baseline:#383835;--accent:#3987e5;}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--baseline:#383835;--accent:#3987e5;}
:root[data-theme=light]{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--baseline:#c3c2b7;--accent:#2a78d6;}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 12px;border-bottom:1px solid var(--grid);padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px}h4{font-size:13px;margin:0 0 6px;color:var(--ink2)}
.muted{color:var(--muted)}.sub{color:var(--ink2);margin:0 0 18px}
.banner{background:color-mix(in srgb,var(--accent) 14%,transparent);border:1px solid var(--accent);border-radius:10px;padding:12px 16px;margin:16px 0 8px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:16px 18px;margin:14px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.chart{max-width:100%;height:auto;display:block}
.chart .axis{fill:var(--muted);font-size:11px}.chart .cell{font-size:11px;font-weight:600}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .whisker{stroke:var(--baseline);stroke-width:1.5}
.chartscroll{overflow-x:auto}
.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
.legend{margin-top:8px;font-size:12px;color:var(--ink2)}.chip{display:inline-flex;align-items:center;margin-right:12px}
.chip i{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px}
.scatter{margin:6px 0}
.stat{display:inline-block;margin-right:26px}.stat b{font-size:22px;display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.vcard{background:var(--surface);border:1px solid var(--grid);border-radius:12px;overflow:hidden}
.vcard img{display:block;width:100%;height:auto}
.vcard .noimg{height:120px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;background:color-mix(in srgb,var(--grid) 40%,transparent)}
.vc-body{padding:10px 12px}.vc-body h4{margin:0 0 2px}.vc-route{font-size:12px;margin-bottom:6px}
.vc-metrics{font-size:12.5px;font-weight:600}.vc-cond{font-size:12px;margin:2px 0 6px}
.flag{display:inline-block;background:color-mix(in srgb,var(--accent) 16%,transparent);border:1px solid var(--accent);border-radius:6px;padding:1px 6px;font-size:11px;margin:2px 4px 0 0}
.flag.tier{text-transform:uppercase;letter-spacing:0.03em;font-weight:700}
.sig-good{color:#1baf7a;font-weight:700}.sig-bad{color:#e34948;font-weight:700}
.tl{display:flex;align-items:center;gap:10px;margin:3px 0}.tl-label{font-size:11px;color:var(--ink2);width:180px;flex:0 0 180px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
footer{margin-top:40px;color:var(--muted);font-size:12px}
"""

_THEME_JS = "<script>(function(){var m=matchMedia('(prefers-color-scheme:dark)');function s(){document.documentElement.setAttribute('data-theme',m.matches?'dark':'light')}s();m.addEventListener&&m.addEventListener('change',s)})();</script>"


def build_report_html(ctx: dict[str, Any]) -> str:
    frame_corr = ctx["frame_corr"]
    frame_corr_pearson = frame_corr[frame_corr["method"] == "pearson"] if not frame_corr.empty else frame_corr

    scatters = ""
    if not frame_corr_pearson.empty:
        top = frame_corr_pearson.reindex(
            frame_corr_pearson["mean_r"].abs().sort_values(ascending=False).index
        ).head(3)
        scatters = "".join(
            svg_scatter(ctx["frame_df"], r["predictor"], r["outcome"]) for _, r in top.iterrows()
        )

    n_runs = ctx["n_runs"]
    parts = [
        "<div class='wrap'>",
        "<h1>Beta Scanner — Detection Correlation Report</h1>",
        f"<p class='sub'>{_esc(ctx['generated_at'])} · corpus at <code>{_esc(ctx['analysis_root'])}</code></p>",
        f"<div class='banner'>EXPLORATORY — {n_runs} independent run(s). The run is the unit of "
        "inference; per-frame coefficients are summarised across runs, not pooled. Per-frame "
        "outcomes (<code>kp_count</code>, <code>mean_score</code>) are a post-processed PROXY, "
        "not raw detector output. Treat effect sizes as directional, not significant.</div>",
        "<div class='card'>"
        f"<span class='stat'><b>{n_runs}</b>distinct runs</span>"
        f"<span class='stat'><b>{ctx['n_videos']}</b>videos</span>"
        f"<span class='stat'><b>{ctx['n_collapsed']}</b>re-runs collapsed</span>"
        f"<span class='stat'><b>{ctx['n_frame_rows']}</b>per-frame samples</span></div>",

        "<h2>Corpus quality overview</h2>",
        "<p class='sub'>Where detection stands across the corpus, and which runs "
        "collapsed. Bars below 0.35 (red) are near-total detection failures.</p>",
        _overview_html(ctx),

        "<h2>Per-video failure cards</h2>",
        "<p class='sub'>Worst run per video (by overlayQuality when present, else "
        "detectionRate), worst-first, with its final frame and adverse reference-frame "
        "flags.</p>",
        _failure_cards_html(ctx),

        "<h2>Pruned hand labels</h2>",
        "<p class='sub'>Dropped for lack of contrast or too many <code>unknown</code>s "
        "(these are the labels worth capturing more carefully next time).</p>",
        _dropped_table(ctx["dropped_labels"]),

        "<h2>Per-frame image quality → pose proxy (within-run)</h2>",
        "<p class='sub'>Cell = mean Pearson r across runs; whiskers on the bars show the "
        "min–max spread of the per-run coefficients. Blue = positive, red = negative.</p>",
        "<div class='chartscroll'>", svg_heatmap(frame_corr_pearson, "per-frame correlations"), "</div>",
        "<div class='chartscroll'>", svg_effect_bars(frame_corr_pearson, "per-frame effect sizes"), "</div>",
    ]

    if scatters:
        parts += ["<h3>Strongest relationships</h3>", "<div class='grid2'>", scatters, "</div>"]

    parts += [
        "<h2>Per-frame failure timeline</h2>",
        "<p class='sub'>Per run, each sampled frame coloured by how its pose was "
        "obtained. Concentrations of non-raw frames localise where the raw detector "
        "breaks.</p>",
        _frame_timeline_html(ctx),
    ]

    parts += [
        "<h2>Evaluation trend accounting</h2>",
        "<p class='sub'>Two-tier accounting from committed evaluation records. "
        "Every value is explicitly tagged as agreement or accuracy.</p>",
        _stat_tiles([
            (str(ctx.get("eval_count", 0)), "evaluation records"),
            (str(ctx.get("verified_frames_total", 0)), "verified truth frames [accuracy]"),
            (str(ctx.get("verified_records", 0)), "records with verified truth"),
        ]),

        "<h2>Low-confidence truth (visible-joint measurement)</h2>",
        "<p class='sub'>An <code>occluded</code> truth joint means ViTPose was not "
        "confident (low seed <code>score</code>), not that it is geometrically hidden. "
        "This measures how many core joints each present frame was confident about — the "
        "fit input for a future exclusion gate — and lists the thinnest frames as a "
        "re-seed queue. It excludes nothing today (measure-first).</p>",
        _low_confidence_html(ctx),

        "<h2>Scanner version regression (appVersion run-over-run)</h2>",
        "<p class='sub'>Evaluation records grouped by the scanner commit "
        "(<code>appVersion</code> from the pose diagnostics), ordered by first-seen "
        "run timestamp. Consecutive versions are delta'd per joint over the videos "
        "both versions evaluated <em>under the same truth revision</em> — a truth "
        "change never masquerades as a scanner change. Deltas are coloured only "
        "when the bootstrap 95% CI excludes zero (green = improved, red = "
        "regressed); ΔPCK &gt; 0 and Δmedian &lt; 0 are improvements.</p>",
        _version_overview_table(ctx.get("version_overview", pd.DataFrame())),
        _version_delta_table(ctx.get("version_deltas", pd.DataFrame())),
        "<h3>Version-tracking flags</h3>",
        _shame_list_html(ctx.get("version_flags", []),
                         "No mixed-truth or unversioned records."),

        "<h2>Per-joint failure ranking (frame/joint unit)</h2>",
        "<p class='sub'>Joint ranking uses frame/joint evidence with bootstrap "
        "95% CIs (no per-video correlation coefficients).</p>",
        _joint_ranking_table(ctx.get("joint_rank", pd.DataFrame())),

        "<h2>Within-video frame-level conditions vs error</h2>",
        "<p class='sub'>Frame/joint rows are grouped into quantile bands by "
        "condition; table reports failure rates and bootstrap CIs by tier.</p>",
        _condition_table(ctx.get("condition_bands", pd.DataFrame())),

        "<h2>Cross-video descriptive splits</h2>",
        f"<p class='sub'>{_esc(ctx.get('confound_caveat', ''))}</p>",
        _cross_video_split_table(ctx.get("cross_video_splits", pd.DataFrame())),

        "<h2>Shame lists</h2>",
        "<h3>Bundles with no truth</h3>",
        _shame_list_html(ctx.get("truthless_bundles", []), "No truthless bundles."),
        "<h3>Stale setup runs</h3>",
        _shame_list_html(ctx.get("stale_runs", []), "No setup-hash stale runs."),

        "<h2>Per-run derived predictors → outcomes (pooled, n small)</h2>",
        "<p class='sub'>Pooled Pearson across runs — descriptive only at this corpus size.</p>",
        "<div class='chartscroll'>", svg_effect_bars(ctx["run_corr"], "per-run effect sizes"), "</div>",

        "<h2>Categorical labels → outcomes</h2>",
        _cat_table(ctx["cat_effects"]),

        "<h2>ORB reference feature richness</h2>",
        "<p class='sub'>Correlation of <code>refKeypointCount</code> with reference image "
        "stats and wall-crop area (per-run, descriptive). This is feature <em>supply</em>, "
        "not matchability — see the cross-match below for the real outcome.</p>",
        "<div class='chartscroll'>", svg_orb_bars(ctx["orb_corr"]), "</div>",

        "<h2>ORB cross-match (route-ID separation)</h2>",
        "<p class='sub'>Each video's wall-crop features matched against every video's "
        "final frame. Same-route pairs should match (bright), cross-route should not. Wide "
        "separation = ORB robustly identifies a wall under real condition variation (ADR "
        "0002).</p>",
        _orb_matrix_html(ctx),

        "<h2>Per-run feature table</h2>",
        _df_to_table(ctx["run_table_display"]),

        "<footer>Generated by <code>analysis_pipeline</code>. Full tables: "
        "<code>features_perrun.csv</code>, <code>features_perframe.csv</code>. "
        "Palette validated against the dataviz skill (blue↔red diverging, categorical hues).</footer>",
        "</div>",
    ]
    body = "".join(parts)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Beta Scanner Detection Correlation Report</title>"
        f"<style>{_CSS}</style></head><body>{_THEME_JS}{body}</body></html>"
    )


def write_outputs(out_dir: Path, run_df: pd.DataFrame, frame_df: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = out_dir / "features_perrun.csv"
    frame_csv = out_dir / "features_perframe.csv"
    html_path = out_dir / "report.html"
    run_df.to_csv(run_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)
    html_path.write_text(build_report_html(ctx), encoding="utf-8")
    return {"run_csv": run_csv, "frame_csv": frame_csv, "html": html_path}
