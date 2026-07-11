"""Emit features CSVs + a self-contained, theme-aware HTML correlation report.

Charts are hand-rendered inline SVG using the dataviz skill's validated palette
(diverging blue<->red for signed correlation, categorical hues per run). No
plotting dependency. Everything is framed EXPLORATORY: the run is the unit.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import pandas as pd

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
        "<h2>Per-run derived predictors → outcomes (pooled, n small)</h2>",
        "<p class='sub'>Pooled Pearson across runs — descriptive only at this corpus size.</p>",
        "<div class='chartscroll'>", svg_effect_bars(ctx["run_corr"], "per-run effect sizes"), "</div>",

        "<h2>Categorical labels → outcomes</h2>",
        _cat_table(ctx["cat_effects"]),

        "<h2>ORB reference feature richness</h2>",
        "<p class='sub'>Correlation of <code>refKeypointCount</code> with reference image "
        "stats and wall-crop area (per-run, descriptive). Per-frame ORB match quality is not "
        "yet exported — see follow-ups.</p>",
        "<div class='chartscroll'>", svg_orb_bars(ctx["orb_corr"]), "</div>",

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
