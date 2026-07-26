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


def _fmt_int(v: Any) -> str:
    """An integer cell that tolerates a missing column (older CSVs, empty frames)."""

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    try:
        return str(int(v))
    except (TypeError, ValueError):
        return "–"


def _pct(v: Any, nd: int = 1) -> str:
    """A 0..1 share as a percentage.

    The attempt funnel spans 68% down to 0.1% in one table, and ``_fmt`` renders that
    smallest share as ``0.00`` — a real bucket reading as an empty one."""

    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(f):
        return "–"
    return f"{f * 100:.{nd}f}%"


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


# What each evidence generation means to a reader of a pooled number (issue #89).
_EVIDENCE_GENERATION_BLURB = {
    "attempts": "scored from the canonical <code>detectorAttempts[]</code> stream",
    "legacy-frames": "scored from dense playback <code>frames[]</code> (pre-attempt export)",
    "unknown": "written before the evidence marker existed (pre-schema-v7)",
}


def _evidence_generation_html(summary: dict[str, Any] | None) -> str:
    """State which evidence generation a pooled section aggregates (issue #89).

    Printed on every pooled section rather than once at the top: a number read out of the
    middle of the report must carry its own provenance. A pool spanning generations says
    MIXED and enumerates the split, so blending is never something the reader has to
    notice on their own."""

    if not isinstance(summary, dict) or not summary.get("n_records"):
        return ("<p class='muted'>evidence generation: no records in this pool.</p>")
    counts = summary.get("counts") or {}
    present = summary.get("generations") or []
    mixed = bool(summary.get("mixed"))
    badge = ("<span class='flag tier'>evidence: MIXED</span>" if mixed
             else f"<span class='flag tier'>evidence: {_esc(present[0])}</span>")
    detail = ", ".join(
        f"{counts.get(g, 0)} × <code>{_esc(g)}</code> ({_EVIDENCE_GENERATION_BLURB.get(g, '')})"
        for g in present
    )
    note = (" — this section pools more than one generation; compare across batches "
            "with that in mind." if mixed else "")
    return (f"<p class='sub'>{badge} {summary.get('n_records', 0)} record(s): "
            f"{detail}.{note}</p>")


def _superseded_table(rows: list[dict[str, Any]]) -> str:
    """Records dropped from pooling because the same video+truth pairing also has an
    attempt-backed record (issue #89). Still on disk and readable — only the aggregation
    passed them over."""

    if not rows:
        return ("<p class='muted'>No superseded records — no video+truth pairing carries "
                "two evidence generations.</p>")
    head = ("<tr><th>route</th><th>video</th><th>run</th><th>truth</th>"
            "<th>generation</th><th>superseded by</th></tr>")
    body = "".join(
        f"<tr><td>{_esc(r.get('route_folder'))}</td>"
        f"<td>{_esc(r.get('video_key'))}</td>"
        f"<td>{_esc(r.get('run_ts'))}</td>"
        f"<td><code>{_esc(str(r.get('truth_hash') or '')[:8])}</code></td>"
        f"<td><code>{_esc(r.get('evidence_generation'))}</code></td>"
        f"<td>{_esc(r.get('superseded_by'))}</td></tr>"
        for r in rows
    )
    return f"<div class='tablewrap'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


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
            f"<td>{_fmt_int(r.get('n_runs'))}</td>"
            f"<td>{_esc(rng)}</td>"
            f"<td>{_fmt(r['failure_rate'])}</td>"
            f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
            f"<td>{_fmt(r.get('run_rate_median'))} / {_fmt(r.get('run_rate_p90'))}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>tier</th><th>condition</th><th>quantile band</th><th>frame/joint n</th>"
        "<th>runs</th><th>band range</th><th>failure rate</th>"
        "<th>run-unit 95% CI</th><th>per-run rate median / p90</th>"
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


def _loose_table(rows: list[dict[str, Any]]) -> str:
    """Bundles paired by the #44 best-overlap fallback (held out of trusted pooling)."""

    if not rows:
        return "<p class='muted'>No loose pairings — every scored run matched a truth setupHash.</p>"
    head = "<tr><th>route</th><th>video</th><th>run</th><th>reason</th></tr>"
    body = "".join(
        f"<tr><td>{_esc(r.get('route_folder'))}</td>"
        f"<td>{_esc(r.get('video_key'))}</td>"
        f"<td>{_esc(r.get('run_ts'))}</td>"
        f"<td>{_esc(r.get('reason'))}</td></tr>"
        for r in rows
    )
    return f"<div class='tablewrap'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


# Why each non-conformance cause is quarantined, and what to do about it (issue #88).
# Ordered mis-track first: it is the actionable group, and the only one that feeds the
# truth-repair worklist.
_NONCONFORMANCE_CAUSE_BLURB = {
    "suspected-mistrack": (
        "Ample accepted detections and the fit still misses identity — the #19 "
        "appearance-stitch signature. <strong>This is the truth-repair worklist</strong> "
        "(#21/#34): re-seed these bundles' Ground Truth."),
    "sparse-match": (
        "The detector supplied too little to fit — too few matched-present frames, or "
        "too small a share of present attempts accepted. A detector failure tripping a "
        "truth gate; re-seeding truth here would repair nothing. Take these to the "
        "attempt-funnel section instead."),
}


def _quarantine_cause_table(rows: list[dict[str, Any]]) -> str:
    head = ("<tr><th>route</th><th>video</th><th>run</th><th>reasons</th>"
            "<th>n</th><th>fit frames</th><th>accepted share</th>"
            "<th>slope x</th><th>r² x</th><th>slope y</th><th>r² y</th></tr>")
    body = "".join(
        f"<tr><td>{_esc(r.get('route_folder'))}</td>"
        f"<td>{_esc(r.get('video_key'))}</td>"
        f"<td>{_esc(r.get('run_ts'))}</td>"
        f"<td>{_esc(r.get('reasons'))}</td>"
        f"<td>{_esc(r.get('n'))}</td>"
        f"<td>{_esc(r.get('fit_frames'))}</td>"
        f"<td>{_fmt(r.get('accepted_share'))}</td>"
        f"<td>{_fmt(r.get('slope_x'))}</td><td>{_fmt(r.get('r2_x'))}</td>"
        f"<td>{_fmt(r.get('slope_y'))}</td><td>{_fmt(r.get('r2_y'))}</td></tr>"
        for r in rows
    )
    return f"<div class='tablewrap'><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"


def _quarantine_table(rows: list[dict[str, Any]]) -> str:
    """Bundles dropped from pooled metrics by the #15 conformance gate, grouped by the
    issue #88 cause so a truth problem is never read off a detector problem.

    Once anything is quarantined, every cause is named even at zero: "these all failed for
    sparse detection, none are mis-track suspects" is a result, and dropping the empty
    group would leave it to be inferred from an absence."""

    if not rows:
        return "<p class='muted'>No bundles quarantined — every record conforms.</p>"
    by_cause: dict[str, list[dict[str, Any]]] = {c: [] for c in _NONCONFORMANCE_CAUSE_BLURB}
    for r in rows:
        by_cause.setdefault(str(r.get("cause")), []).append(r)

    parts: list[str] = []
    for cause, cause_rows in by_cause.items():
        blurb = _NONCONFORMANCE_CAUSE_BLURB.get(cause, "")
        parts.append(f"<h4><code>{_esc(cause)}</code> — {len(cause_rows)} record"
                     f"{'' if len(cause_rows) == 1 else 's'}</h4>")
        if blurb:
            parts.append(f"<p class='sub'>{blurb}</p>")
        parts.append(_quarantine_cause_table(cause_rows) if cause_rows
                     else "<p class='muted'>(none this batch)</p>")
    return "".join(parts)


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


def _hallucination_split_html(ctx: dict[str, Any]) -> str:
    """The truth-presence split of ``hallucination-fp`` (issue #69).

    The class is the corpus's largest detection failure and its most actionable
    suggestion, but it conflates two scanner behaviors with different fixes — so the
    split is stated in words above the class table, not left to be read off two columns.
    Frames from pre-schema-v12 records recorded no presence and are named as unknown
    rather than folded into either side."""

    split = ctx.get("frame_quality_hallucination")
    if not isinstance(split, dict) or not split.get("total"):
        return ("<p class='sub'>No <code>hallucination-fp</code> frames pooled — "
                "nothing to split by truth presence.</p>")

    total = int(split["total"])
    absent, present = int(split["truth_absent"]), int(split["truth_present"])
    unknown = int(split["truth_unknown"])
    parts = [
        f"<strong>{absent}</strong> ({_pct(split['truth_absent_share'])}) on "
        "<em>truth-absent</em> frames — real false positives, fixed by presence gating",
        f"<strong>{present}</strong> ({_pct(split['truth_present_share'])}) on "
        "<em>truth-present</em> frames — tracking misses, fixed by tracking robustness",
    ]
    tail = ("" if not unknown else
            f" {unknown} more come from pre-schema-v12 records that never recorded "
            "presence and are excluded from those shares — re-run <code>evaluate</code> "
            "to place them.")
    return (f"<p class='sub'>Of {total} pooled <code>hallucination-fp</code> frames: "
            f"{'; '.join(parts)}.{tail}</p>")


def _frame_quality_html(ctx: dict[str, Any]) -> str:
    """Detection-quality per-frame classes (issue #44): top classes, the conditions the
    flagged-rate correlates with worst, the human distractor labels, and a worst-first
    re-review worklist. Pooled across ALL records (quarantined + loose included) — an
    independent pool from the trusted metrics."""

    from .trends import FRAME_QUALITY_WORKLIST_TOP_K

    detected = int(ctx.get("frame_quality_detected", 0))
    if detected == 0:
        return ("<p class='muted'>No per-frame quality classes yet — needs schema-v6 "
                "evaluation records (re-run <code>evaluate</code>).</p>")

    flagged = int(ctx.get("frame_quality_flagged", 0))
    held = int(ctx.get("frame_quality_held", 0))
    frozen = int(ctx.get("frame_quality_frozen", 0))
    rate = flagged / detected if detected else 0.0
    tiles = _stat_tiles([
        (str(detected), "scanner-detected frames [pooled]"),
        (str(flagged), "flagged (non-ok) frames"),
        (_fmt(rate), "flagged rate"),
        (str(held), "held-pose repeat frames"),
        (str(frozen), "raw frozen-stale frames"),
    ])

    classes = ctx.get("frame_quality_classes")
    class_tbl = "<p class='muted'>(no classes)</p>"
    if isinstance(classes, pd.DataFrame) and not classes.empty:
        rows = "".join(
            f"<tr><td>{_esc(r['class'])}</td><td>{int(r['n'])}</td>"
            f"<td>{_fmt(r['share'])}</td>"
            f"<td>{int(r.get('truth_absent', 0))}</td>"
            f"<td>{int(r.get('truth_present', 0))}</td>"
            f"<td>{int(r.get('truth_unknown', 0))}</td>"
            f"<td>{int(r['held_pose'])}</td>"
            f"<td>{int(r['frozen_stale'])}</td></tr>"
            for _, r in classes.iterrows()
        )
        class_tbl = ("<div class='tablewrap'><table><thead><tr><th>failure class</th>"
                     "<th>frames</th><th>share</th><th>truth-absent</th>"
                     "<th>truth-present</th><th>presence unknown</th>"
                     "<th>held-pose repeats</th><th>raw frozen-stale</th>"
                     f"</tr></thead><tbody>{rows}</tbody></table></div>")
    class_tbl = _hallucination_split_html(ctx) + class_tbl

    distractors = ctx.get("frame_quality_distractors")
    distractor_tbl = "<p class='muted'>(no annotated distractors yet)</p>"
    if isinstance(distractors, pd.DataFrame) and not distractors.empty:
        rows = "".join(
            f"<tr><td>{_esc(r['distractor'])}</td><td>{int(r['n'])}</td>"
            f"<td>{_fmt(r['share'])}</td><td>{int(r['held_pose'])}</td>"
            f"<td>{int(r['frozen_stale'])}</td></tr>"
            for _, r in distractors.iterrows()
        )
        distractor_tbl = ("<div class='tablewrap'><table><thead><tr><th>distractor</th>"
                          "<th>frames</th><th>share of annotated frames</th>"
                          "<th>held-pose repeats</th><th>raw frozen-stale</th></tr></thead>"
                          f"<tbody>{rows}</tbody></table></div>")

    # Worst-correlated conditions: rank Video Stats conditions by the spread of the
    # flagged-rate across their bands (max − min).
    bands = ctx.get("frame_quality_condition_bands")
    cond_tbl = ("<p class='muted'>No Video Stats condition bands yet — needs "
                "<code>video-stats.json</code> on enough pooled frames.</p>")
    if isinstance(bands, pd.DataFrame) and not bands.empty:
        spread = (bands.groupby("condition")["flagged_rate"]
                  .agg(lambda s: float(s.max() - s.min())).sort_values(ascending=False))
        order = list(spread.index)
        rows = []
        for _, r in bands.sort_values(["condition", "band"]).iterrows():
            rng = f"[{_fmt(r['band_min'])}, {_fmt(r['band_max'])}]"
            rows.append(
                "<tr>"
                f"<td>{_esc(r['condition'])}</td>"
                f"<td>band {int(r['band'])}</td>"
                f"<td>{int(r['n'])}</td>"
                f"<td>{_fmt_int(r.get('n_runs'))}</td>"
                f"<td>{_esc(rng)}</td>"
                f"<td>{_fmt(r['flagged_rate'])}</td>"
                f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
                f"<td>{_fmt(r.get('run_rate_median'))} / {_fmt(r.get('run_rate_p90'))}</td>"
                "</tr>"
            )
        worst = ", ".join(f"{c} (Δ{_fmt(spread[c])})" for c in order[:3]) or "none"
        cond_tbl = (
            f"<p class='sub'>Conditions ranked by flagged-rate spread across bands: "
            f"{_esc(worst)}. A Video Stats condition is constant within a bundle, so a "
            "band is only as strong as its <em>runs</em> column — the CI resamples runs, "
            "and the per-run median/p90 shows how much of the band's rate is one "
            "video.</p>"
            "<div class='tablewrap'><table><thead><tr><th>Video Stats condition</th>"
            "<th>tercile band</th><th>frames</th><th>runs</th><th>band range</th>"
            "<th>flagged rate</th><th>run-unit 95% CI</th>"
            "<th>per-run rate median / p90</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")

    worklist = ctx.get("frame_quality_worklist")
    wl_tbl = "<p class='muted'>No flagged or raw frozen-stale frames.</p>"
    if isinstance(worklist, pd.DataFrame) and not worklist.empty:
        shown = worklist.head(FRAME_QUALITY_WORKLIST_TOP_K)
        wl_tbl = _df_to_table(shown)
        if len(worklist) > len(shown):
            wl_tbl += (f"<p class='muted'>Showing the worst {len(shown)} of "
                       f"{len(worklist)} flagged/raw frozen-stale frames — full list in "
                       "<code>eval_frame_quality_worklist.csv</code>.</p>")

    return (
        tiles
        + "<h3>Failure class frequency</h3>" + class_tbl
        + "<h3>Distractor frequency</h3>" + distractor_tbl
        + "<h3>Flagged rate vs Video Stats conditions</h3>" + cond_tbl
        + "<h3>Re-review worklist (worst class first)</h3>" + wl_tbl)


# What each Detector Attempt status means, so the funnel reads as a funnel and not as a
# list of scanner enum values (issue #87).
_FUNNEL_STATUS_BLURB = {
    "accepted": "MediaPipe returned a pose and the scanner kept it",
    "missing": "MediaPipe returned nothing in the region(s) searched",
    "flipRejected": "a pose was found and discarded by the flip gate",
    "qualityRejected": "a pose was found and discarded by the quality gate",
    "unknown": "the scanner emitted a status outside the known vocabulary",
}

# Worst-run funnel rows to show inline; the CSV carries every run.
FUNNEL_WORKLIST_TOP_K = 20


def _attempt_funnel_html(ctx: dict[str, Any]) -> str:
    """The Detector Attempt funnel (issue #87): what the detector did, before truth.

    Every pooled share is printed beside its run-unit distribution, because the Run is the
    unit of inference: one 1500-attempt run that missed everything moves the pooled share
    as much as a dozen ordinary runs, and only the median / p90 / tail columns say which
    corpus you are looking at. No CIs here on purpose — attempts within a run are
    correlated, so a pooled-attempt interval would claim precision this design cannot
    support (#70)."""

    funnel = ctx.get("attempt_funnel_runs")
    if not isinstance(funnel, pd.DataFrame) or funnel.empty:
        return ("<p class='muted'>No Detector Attempt funnel yet — needs pose runs "
                "carrying <code>detectorAttempts[]</code> and matching evaluation "
                "records.</p>")

    totals = ctx.get("attempt_funnel") or {}
    shares = totals.get("status_shares") or {}
    tiles = _stat_tiles([
        (str(totals.get("runs", 0)), "runs with attempt evidence"),
        (str(totals.get("attempts", 0)), "Detector Attempts"),
        (_pct(shares.get("accepted")), "accepted [pooled]"),
        (_pct(shares.get("missing")), "missing [pooled]"),
        (_pct(totals.get("missing_share_run_median")), "missing [run median]"),
        (f"{totals.get('reacquire_succeeded', 0)}/{totals.get('reacquire_attempted', 0)}",
         "reacquire successes"),
        (_pct(totals.get("reacquire_success_rate")), "reacquire success [pooled]"),
        (f"{totals.get('tail_runs_missing', 0)}/{totals.get('runs', 0)}",
         "runs >50% missing"),
    ])

    status = ctx.get("attempt_funnel_status")
    status_tbl = "<p class='muted'>(no status mix)</p>"
    if isinstance(status, pd.DataFrame) and not status.empty:
        rows = "".join(
            "<tr>"
            f"<td><code>{_esc(r['status'])}</code></td>"
            f"<td>{_esc(_FUNNEL_STATUS_BLURB.get(str(r['status']), ''))}</td>"
            f"<td>{int(r['attempts'])}</td>"
            f"<td>{_pct(r['share'])}</td>"
            f"<td>{int(r['runs_with_any'])}</td>"
            f"<td>{_pct(r['run_share_median'])}</td>"
            f"<td>{_pct(r['run_share_p90'])}</td>"
            f"<td>{_pct(r['run_share_max'])}</td>"
            f"<td>{int(r['tail_runs'])}</td>"
            "</tr>"
            for _, r in status.iterrows()
        )
        status_tbl = (
            "<p class='sub'>Pooled share is over every attempt in the corpus; the run "
            "columns are the distribution of that status's share <em>within</em> a run. "
            "A gap between the pooled share and the run median means a few long runs "
            "carry the pool. <strong>Tail runs</strong> are runs where the status took "
            "more than half the attempts.</p>"
            "<div class='tablewrap'><table><thead><tr><th>status</th><th>meaning</th>"
            "<th>attempts</th><th>pooled share</th><th>runs with any</th>"
            "<th>run median</th><th>run p90</th><th>run max</th>"
            "<th>tail runs</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

    run_stats = ctx.get("attempt_funnel_run_stats")
    run_stats_tbl = ("<p class='muted'>No run-unit funnel distribution yet.</p>")
    if isinstance(run_stats, pd.DataFrame) and not run_stats.empty:
        rows = "".join(
            "<tr>"
            f"<td><code>{_esc(r['metric'])}</code></td>"
            f"<td>{_esc(r['meaning'])}</td>"
            f"<td>{int(r['n_runs'])}</td>"
            f"<td>{_fmt(r['median'], 3)}</td>"
            f"<td>{_fmt(r['p90'], 3)}</td>"
            f"<td>{_fmt(r['min'], 3)}</td>"
            f"<td>{_fmt(r['max'], 3)}</td>"
            "</tr>"
            for _, r in run_stats.iterrows()
        )
        run_stats_tbl = (
            "<p class='sub'>Reacquire success is the share of <em>attempted</em> "
            "reacquires that recovered the Climber — a run that never reacquired "
            "contributes no value rather than a zero.</p>"
            "<div class='tablewrap'><table><thead><tr><th>metric</th><th>meaning</th>"
            "<th>runs</th><th>median</th><th>p90</th><th>min</th><th>max</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

    flags = ctx.get("attempt_funnel_flags")
    flag_tbl = ("<p class='muted'>No search-condition flags on this corpus's "
                "attempts.</p>")
    if isinstance(flags, pd.DataFrame) and not flags.empty:
        rows = "".join(
            "<tr>"
            f"<td><code>{_esc(r['flag'])}</code></td>"
            f"<td><code>{_esc(r['status'])}</code></td>"
            f"<td>{int(r['flag_fired'])}/{int(r['attempts_scored'])}</td>"
            f"<td>{_pct(r['rate'])}</td>"
            f"<td>{int(r['n_runs'])}</td>"
            f"<td>{_pct(r['run_rate_median'])}</td>"
            f"<td>{_pct(r['run_rate_p90'])}</td>"
            "</tr>"
            for _, r in flags.iterrows()
        )
        flag_tbl = (
            "<p class='sub'>How often the scanner's own condition flags fired on the "
            "region it searched, split by what happened next. A flag that fires far more "
            "on <code>missing</code> than on <code>accepted</code> is a condition the "
            "detector loses the Climber in. The denominator counts only attempts whose "
            "conditions actually carry that flag, so a scanner build that never emitted "
            "it is not read as one that emitted it and found nothing. The run median is "
            "usually zero — most runs never fire a given flag — so the p90 beside it is "
            "what shows a flag a few runs fire on constantly.</p>"
            "<div class='tablewrap'><table><thead><tr><th>flag</th><th>status</th>"
            "<th>fired/scored</th><th>pooled rate</th><th>runs</th>"
            "<th>run median</th><th>run p90</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")

    worst_cols = ["route_folder", "video_key", "run_ts", "conforming", "attempt_count",
                  "attempt_status_accepted_rate", "attempt_status_missing_rate",
                  "attempt_status_flip_rejected_rate",
                  "attempt_status_quality_rejected_rate",
                  "attempt_reacquire_attempt_rate", "attempt_reacquire_success_rate"]
    worst = funnel
    if "attempt_status_missing_rate" in funnel.columns:
        worst = funnel.sort_values(
            "attempt_status_missing_rate", ascending=False, na_position="last")
    shown = worst[[c for c in worst_cols if c in worst.columns]].head(FUNNEL_WORKLIST_TOP_K)
    worst_tbl = _df_to_table(shown)
    if len(funnel) > len(shown):
        worst_tbl += (f"<p class='muted'>Showing {len(shown)} of {len(funnel)} runs; the "
                      "full funnel is in <code>eval_attempt_funnel_runs.csv</code>.</p>")

    return (
        tiles
        + "<h3>Status mix (pooled attempts vs run-unit distribution)</h3>" + status_tbl
        + "<h3>Reacquire effectiveness and per-run spread</h3>" + run_stats_tbl
        + "<h3>Search-condition flags by status</h3>" + flag_tbl
        + "<h3>Worst runs by missing share</h3>" + worst_tbl
    )


def _detection_error_attempt_html(ctx: dict[str, Any]) -> str:
    runs = ctx.get("detection_error_attempt_runs")
    if not isinstance(runs, pd.DataFrame) or runs.empty:
        return ("<p class='muted'>No run-level detector-attempt summary yet — needs "
                "schema-v6 evaluation records and matching pose runs.</p>")

    evidence = runs.get("attempt_evidence")
    unknown = int((evidence.astype("string") == "unknown").sum()) if evidence is not None else 0
    attempts = int((evidence.astype("string") == "attempts").sum()) if evidence is not None else 0

    def sum_col(name: str) -> int:
        return int(pd.to_numeric(runs.get(name, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    reacq_attempted = sum_col("attempt_reacquire_attempted_count")
    reacq_succeeded = sum_col("attempt_reacquire_succeeded_count")
    reacq_failed = sum_col("attempt_reacquire_failed_count")
    # Rejection correctness (issue #85): the over-rejection rate is over truth-checkable
    # rejections only, so the checkable/total counts sit beside it.
    rej = ctx.get("rejection_correctness") or {}
    tiles = _stat_tiles([
        (str(len(runs)), "evaluation runs"),
        (str(attempts), "with attempts"),
        (str(unknown), "unknown attempt evidence"),
        (f"{reacq_succeeded}/{reacq_attempted}", "reacquire successes"),
        (str(reacq_failed), "reacquire failures"),
        (_fmt(rej.get("over_rejection_rate")), "over-rejection rate (pooled)"),
        (_fmt(rej.get("over_rejection_rate_truth_present")),
         "over-rejection rate (Climber present)"),
        (f"{int(rej.get('truth_checkable') or 0)}/{int(rej.get('rejected_attempts') or 0)}",
         "truth-checkable rejections"),
    ])

    bands = ctx.get("detection_error_attempt_bands")
    band_tbl = ("<p class='muted'>No attempt-condition bands yet — needs at least "
                "three runs with varying attempt predictors.</p>")
    if isinstance(bands, pd.DataFrame) and not bands.empty:
        spread = (bands.groupby("predictor")["flagged_rate_mean"]
                  .agg(lambda s: float(s.max() - s.min())).sort_values(ascending=False))
        rows = []
        for _, r in bands.sort_values(["predictor", "band"]).iterrows():
            rng = f"[{_fmt(r['band_min'])}, {_fmt(r['band_max'])}]"
            rows.append(
                "<tr>"
                f"<td>{_esc(r['predictor'])}</td>"
                f"<td>band {int(r['band'])}</td>"
                f"<td>{int(r['n_runs'])}</td>"
                f"<td>{_esc(rng)}</td>"
                f"<td>{_fmt(r['flagged_rate_mean'])}</td>"
                f"<td>[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]</td>"
                "</tr>"
            )
        worst = ", ".join(f"{c} (Δ{_fmt(spread[c])})" for c in list(spread.index)[:3]) or "none"
        band_tbl = (
            f"<p class='sub'>Run-level predictors ranked by Detection Error spread: "
            f"{_esc(worst)}.</p>"
            "<div class='tablewrap'><table><thead><tr><th>attempt predictor</th>"
            "<th>band</th><th>runs</th><th>band range</th>"
            "<th>Detection Error rate</th><th>bootstrap 95% CI</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")

    # Crop placement + miss causes (issue #86). The crop-miss rate is deliberately not
    # folded into the cause table: on a corpus where full-frame reacquire always runs, no
    # miss is *caused* by the crop even though the crop may have excluded the Climber on
    # most of them.
    crop = ctx.get("crop_quality") or {}
    causes = ctx.get("crop_quality_miss_causes")
    cause_tbl = ("<p class='muted'>No missing attempts scored yet — needs schema-v10 "
                 "records with detector-attempt evidence.</p>")
    if isinstance(causes, pd.DataFrame) and not causes.empty:
        rows = []
        for _, r in causes.iterrows():
            rows.append(
                "<tr>"
                f"<td>{_esc(r['miss_cause'])}</td>"
                f"<td>{int(r['n'])}</td>"
                f"<td>{_fmt(r['share'])}</td>"
                f"<td>{int(r['crop_missed_truth'])}/{int(r['crop_containment_scored'])}</td>"
                f"<td>{_fmt(r['median_initial_crop_containment'])}</td>"
                f"<td>{int(r['flags_fired'])}</td>"
                "</tr>"
            )
        cause_tbl = (
            "<p class='sub'>Why the detector found no Climber, per matched missing "
            "attempt. <code>crop-misplaced</code> requires that the misplaced crop was "
            "the <em>only</em> place searched &mdash; when a full-frame reacquire also "
            "ran and failed, the Climber was searched for everywhere, so the crop cannot "
            "be what lost them. Crop placement is still measured on every miss in the "
            "next column.</p>"
            "<div class='tablewrap'><table><thead><tr><th>miss cause</th><th>n</th>"
            "<th>share</th><th>crop excluded Climber</th>"
            "<th>median crop containment</th><th>condition flags fired</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")

    crop_tiles = _stat_tiles([
        (_fmt(crop.get("crop_missed_truth_rate")), "attempts whose crop excluded Climber"),
        (_fmt(crop.get("median_initial_crop_containment")), "median crop containment"),
        (_fmt(crop.get("median_initial_search_region_iou")), "median crop↔truth IoU"),
        (str(int(crop.get("missing_attempts") or 0)), "missing attempts scored"),
    ])

    display_cols = [
        "route_folder", "video_key", "run_ts", "flagged_rate",
        "crop_contained_truth_rate", "miss_crop_misplaced_share",
        "miss_unexplained_share", "missing_attempts",
        "over_rejection_rate", "over_rejection_rate_truth_present",
        "flip_over_rejection_rate", "rejection_truth_checkable", "rejected_attempts",
        "attempt_evidence", "attempt_count", "attempt_reacquire_attempt_rate",
        "attempt_reacquire_success_rate", "attempt_search_luma_mean_mean",
        "attempt_search_luma_stdDev_mean", "attempt_search_sharpness_mean",
        "attempt_initial_search_region_area_mean", "attempt_detection_region_area_mean",
        "attempt_full_frame_reacquire_success_rate",
    ]
    top = runs.sort_values("flagged_rate", ascending=False, na_position="last")
    top = top[[c for c in display_cols if c in top.columns]].head(20)
    return (
        tiles
        + "<h3>Crop placement vs Ground Truth</h3>" + crop_tiles
        + "<h3>Missing-attempt causes</h3>" + cause_tbl
        + "<h3>Run-level attempt conditions vs Detection Errors</h3>" + band_tbl
        + "<h3>Worst runs with attempt evidence</h3>" + _df_to_table(top)
    )


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

    trusted_evidence = _evidence_generation_html(ctx.get("evidence_generation_trusted"))
    frames_evidence = _evidence_generation_html(ctx.get("evidence_generation_frames"))

    parts += [
        "<h2>Evaluation trend accounting</h2>",
        "<p class='sub'>Two-tier accounting from committed evaluation records. "
        "Every value is explicitly tagged as agreement or accuracy. Records superseded "
        "by a newer evidence generation for the same video+truth pairing are dropped "
        "before any of this (#89) and listed in the shame lists below.</p>",
        trusted_evidence,
        _stat_tiles([
            (str(ctx.get("eval_count", 0)), "trusted records [pooled]"),
            (str(ctx.get("quarantined_count", 0)), "quarantined records [#15 gate]"),
            (str(ctx.get("truth_repair_count", 0)),
             "of those suspected mis-tracks [#88 truth-repair]"),
            (str(ctx.get("loose_count", 0)), "loose pairings [#44 fallback]"),
            (str(ctx.get("superseded_count", 0)),
             "superseded records [#89 evidence dedup]"),
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

        "<h2>Per-frame detection quality (auto-flagged classes)</h2>",
        "<p class='sub'>Each scanner-detected frame auto-classified from the "
        "scanner↔truth geometry (<code>ok</code> / <code>wrong-subject</code> / "
        "<code>hallucination-fp</code> / <code>flipped-rotated</code> / "
        "<code>distorted</code>), plus a cross-cutting frozen-stale flag. Each class is "
        "split by whether the Climber was in the frame at all, because "
        "<code>hallucination-fp</code> otherwise conflates a real false positive "
        "(truth-absent → presence gating) with a tracking miss (truth-present → "
        "tracking robustness). Pooled across "
        "<em>all</em> records — quarantined and loose-paired included, since those hold "
        "the frames most worth fixing — an independent pool from the trusted metrics. "
        "Classes are provisional (thresholds not yet fit against verified labels).</p>",
        frames_evidence,
        _frame_quality_html(ctx),

        "<h2>Detector Attempt funnel (run unit)</h2>",
        "<p class='sub'>What the detector <em>did</em>, before Ground Truth is "
        "consulted: how the Detector Attempt stream splits across accepted / missing / "
        "flip-rejected / quality-rejected, how often full-frame reacquire ran and "
        "worked, and which search-condition flags fired under each status. The Run is "
        "the unit of inference, so every pooled share is printed beside its run-unit "
        "median, p90 and tail count — and there are deliberately no confidence "
        "intervals over pooled attempts, which are correlated within a run (#70). Runs "
        "with no Ground Truth have no evaluation record and so are not in this funnel; "
        "they are accounted for in the truthless-bundle shame list below.</p>",
        _evidence_generation_html(ctx.get("evidence_generation_funnel")),
        _attempt_funnel_html(ctx),

        "<h2>Detection Errors × Detector Attempts</h2>",
        "<p class='sub'>Detection Error rates are summarised per Run, then grouped "
        "against Detector Attempt search crops, reacquire outcomes, and search-region "
        "pixel conditions. Legacy runs stay explicit as unknown attempt evidence. "
        "<strong>Over-rejection rate</strong> is the share of truth-checkable "
        "flip/quality rejections whose discarded raw pose actually agreed with Ground "
        "Truth &mdash; the scanner's rejection gates second-guessed against truth. "
        "Rejections on Climber-absent frames are correct by construction, so the "
        "second rate drops them and judges the gates on frames where a pose was "
        "actually there to keep.</p>",
        frames_evidence,
        _detection_error_attempt_html(ctx),

        "<h2>Scanner version regression (appVersion run-over-run)</h2>",
        "<p class='sub'>Evaluation records grouped by the scanner commit "
        "(<code>appVersion</code> from the pose diagnostics), ordered by first-seen "
        "run timestamp. Consecutive versions are delta'd per joint over the videos "
        "both versions evaluated <em>under the same truth revision</em> — a truth "
        "change never masquerades as a scanner change. Deltas are coloured only "
        "when the bootstrap 95% CI excludes zero (green = improved, red = "
        "regressed); ΔPCK &gt; 0 and Δmedian &lt; 0 are improvements. Superseded legacy "
        "records are already gone (#89), so a change of evidence generation can no "
        "longer masquerade as a scanner change either.</p>",
        trusted_evidence,
        _version_overview_table(ctx.get("version_overview", pd.DataFrame())),
        _version_delta_table(ctx.get("version_deltas", pd.DataFrame())),
        "<h3>Version-tracking flags</h3>",
        _shame_list_html(ctx.get("version_flags", []),
                         "No mixed-truth or unversioned records."),

        "<h2>Per-joint failure ranking (frame/joint unit)</h2>",
        "<p class='sub'>Joint ranking uses frame/joint evidence with bootstrap "
        "95% CIs (no per-video correlation coefficients).</p>",
        trusted_evidence,
        _joint_ranking_table(ctx.get("joint_rank", pd.DataFrame())),

        "<h2>Within-video frame-level conditions vs error</h2>",
        "<p class='sub'>Frame/joint rows are grouped into quantile bands by condition; "
        "the table reports each band's pooled failure rate by tier. The rate pools "
        "frames, but the interval does <em>not</em> treat them as independent — frames "
        "within a run are correlated, so the CI is a cluster bootstrap over the band's "
        "runs (#70) and the per-run median/p90 sits beside it. Read a band as "
        "well-evidenced only when its <code>runs</code> count is large; a wide interval "
        "over many frames means few runs, not noisy frames.</p>",
        trusted_evidence,
        _condition_table(ctx.get("condition_bands", pd.DataFrame())),

        "<h2>Cross-video descriptive splits</h2>",
        f"<p class='sub'>{_esc(ctx.get('confound_caveat', ''))}</p>",
        trusted_evidence,
        _cross_video_split_table(ctx.get("cross_video_splits", pd.DataFrame())),

        "<h2>Shame lists</h2>",
        "<h3>Superseded records (#89 evidence-generation dedup)</h3>",
        "<p class='sub'>Records whose video+truth pairing also carries an attempt-backed "
        "record. The attempt-backed one is the evidence every pooled metric above drew "
        "from; these are passed over so the pairing is counted once and two generations "
        "of evidence never blend. Nothing is deleted — the records stay on disk and "
        "readable, and a pairing with no attempt-backed record keeps every record it "
        f"has. {ctx.get('superseded_count', 0)} of "
        f"{ctx.get('eval_count_on_disk', 0)} record(s) on disk, exported as "
        "<code>eval_superseded_records.csv</code>.</p>",
        _superseded_table(ctx.get("superseded_records", [])),
        "<h3>Loose-paired bundles (#44 best-overlap fallback)</h3>",
        "<p class='sub'>Bundles with no setupHash-matched run overlapping truth enough, "
        "paired instead against the run with the most timestamp overlap. Held out of "
        "every trusted pooled metric above; their per-frame quality still feeds the "
        "detection-quality worklist and crops.</p>",
        _loose_table(ctx.get("loose_bundles", [])),
        "<h3>Quarantined bundles (#15 conformance gate), by cause (#88)</h3>",
        "<p class='sub'>Bundles whose truth↔scanner fit falls outside the "
        "near-identity band (<code>scanner = a·truth + b</code>, per axis). These "
        "are excluded from every pooled metric above; their per-record tiers remain "
        "on disk for inspection. The gate verdict is unchanged — it is split here by "
        "<em>cause</em>, because a run whose detector found almost nothing fails the same "
        "gate as a run whose truth mis-tracked, and only the second is a truth problem. "
        f"Truth-repair worklist: {ctx.get('truth_repair_count', 0)} record(s), exported "
        "as <code>eval_truth_repair_worklist.csv</code>. Records written before the split "
        "carry no annotation and default to <code>suspected-mistrack</code> (their "
        "pre-#88 place) with empty evidence columns — re-run <code>evaluate</code> to "
        "classify them.</p>",
        _quarantine_table(ctx.get("quarantined_bundles", [])),
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
