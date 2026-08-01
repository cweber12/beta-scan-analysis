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

from . import cycles
from . import floors
from .evaluate import RATE_MISMATCH_MIN_RATIO

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


def _accuracy_tier_html(ctx: dict[str, Any]) -> str:
    """Name the accuracy tier's missing input wherever a tier is shown (issue #133).

    The tier is permanently empty (ADR 0010): no ground-truth ``review`` value is a
    positive human attestation, so ``truthFramesVerified`` is 0 by construction rather
    than by corpus shortfall. Nothing renders an *empty* accuracy row — the tier simply
    produces none — so a reader sees only agreement rows and has to infer why. That
    inference has been made wrong repeatedly across baselines, reading a missing input
    as a detection problem.

    Printed beside every tier-bearing section rather than once at the top, for the reason
    issue #89 established and ADR 0009 reaffirmed: a number read out of the middle of the
    report must carry its own provenance. Fails open — if verified frames ever appear, it
    reports them instead of asserting the tier is empty.
    """
    verified = int(ctx.get("verified_frames_total") or 0)
    if verified:
        records = int(ctx.get("verified_records") or 0)
        return (f"<p class='sub'><span class='flag tier'>accuracy: {verified} verified "
                f"frame(s)</span> across {records} record(s), scored against "
                f"human-attested truth.</p>")
    return (
        "<p class='sub'><span class='flag tier'>accuracy: NOT COMPUTABLE</span> "
        "0 verified truth frames, so every value in this section is "
        "<strong>agreement</strong> — scored against the unchallenged ViTPose scaffold, "
        "not against reality. The accuracy tier is permanently empty by decision "
        "(ADR 0010): no ground-truth <code>review</code> value is a positive human "
        "attestation. This is a <em>missing input</em>, not a detection failure.</p>"
    )


def _basis_banner_html(ctx: dict[str, Any]) -> str:
    """Declare the frozen basis once at the top, in addition to per-section (issue #131).

    The per-section lines are what stop two sections being compared; this is what tells a
    reader arriving at the report which baseline cycle they are looking at, without having
    to scroll to a pooled section to find out."""

    summary = ctx.get("measurement_basis_frames") or ctx.get("measurement_basis_trusted")
    if not isinstance(summary, dict):
        return ""
    frozen = summary.get("frozen_schema")
    writer = summary.get("writer_schema")
    if summary.get("cycle_broken"):
        return (f"<div class='banner'>BASIS BROKEN — the evaluation writer is at "
                f"<code>v{_esc(writer)}</code> but this baseline cycle is frozen at "
                f"<code>v{_esc(frozen)}</code>. Re-score the whole compared population "
                f"(<code>evaluate --mode all</code>) before reading any number here against "
                f"an earlier batch.</div>")
    return (f"<p class='sub'>Measurement basis: schema frozen at <code>v{_esc(frozen)}</code> "
            f"for this baseline cycle (#131). Every pooled section below states the schema "
            f"versions and build identities it rests on.</p>")


def _measurement_basis_html(summary: dict[str, Any] | None) -> str:
    """State the measurement basis a pooled section rests on (issue #131).

    Two sections resting on different schema versions or different build sets are not
    comparable, and before this the reader had no way to tell from the numbers. Printed on
    every pooled section for the same reason the evidence-generation line is: provenance
    travels with the number, not with the report.

    Three escalating conditions, loudest first — a mid-cycle schema bump (the whole
    compared population needs re-scoring), a pool spanning schema versions, and a pool
    spanning builds. Each is named rather than refused; see ``_measurement_basis``."""

    if not isinstance(summary, dict):
        return "<p class='muted'>measurement basis: no records in this pool.</p>"
    if not summary.get("n_records"):
        # An empty pool still has a Cycle, and on the arm section that is the half a reader
        # needs most: it says *why* the pool is empty rather than only that it is.
        cycle = summary.get("cycle")
        tail = ("" if not isinstance(cycle, dict) or cycle.get("posture") == cycles.POSTURE_NONE
                else f" Cycle <code>{_esc(cycle.get('cycle_id'))}</code> "
                     f"(<code>{_esc(cycle.get('status'))}</code>, rule: "
                     f"<em>{_esc(cycle.get('rule'))}</em>).")
        return f"<p class='muted'>measurement basis: no records in this pool.{tail}</p>"

    counts = summary.get("schema_counts") or {}
    versions = summary.get("schema_versions") or []
    schema_mixed = bool(summary.get("schema_mixed"))
    frozen = summary.get("frozen_schema")
    n = int(summary.get("n_records") or 0)

    badge = ("<span class='flag tier'>basis: MIXED SCHEMA</span>" if schema_mixed
             else f"<span class='flag tier'>basis: {_esc(summary.get('schema_label'))}</span>")
    detail = ", ".join(
        f"{counts.get(v, 0)} × <code>"
        f"{'schemaVersion ' + str(v) if v != 'unknown' else 'unstamped'}</code>"
        for v in versions
    )

    notes: list[str] = []
    if summary.get("cycle_broken"):
        notes.append(
            f"<strong>MID-CYCLE SCHEMA BUMP.</strong> The writer is at "
            f"<code>v{_esc(summary.get('writer_schema'))}</code> while the baseline cycle is "
            f"frozen at <code>v{_esc(frozen)}</code>. Any comparison against an earlier "
            "batch is invalid until the <em>whole</em> compared population is re-scored "
            "with <code>python -m analysis_pipeline evaluate --mode all</code> — scoring "
            "only the new batch leaves the population straddling two bases.")
    if schema_mixed:
        off = int(summary.get("off_basis") or 0)
        notes.append(
            f"This pool spans more than one schema version: {off} of {n} record(s) are "
            f"<em>not</em> on the frozen <code>v{_esc(frozen)}</code> basis. Numbers here "
            "blend bases — do not read them against a single-basis section, and re-score "
            "with <code>--mode all</code> before comparing batches.")
    if summary.get("build_mixed"):
        notes.append(
            f"Collected across {_esc(summary.get('n_builds'))} build identities: "
            f"{_esc(summary.get('build_label'))}. Comparable as a corpus-wide number, not "
            "as a statement about any one build.")
    elif summary.get("builds"):
        notes.append(f"Single build identity: <code>{_esc(summary.get('build_label'))}</code>.")
    elif not summary.get("build_set_known"):
        notes.append("Build set not established for this pool.")

    # The Cycle half of the basis (#176): the harness identity that sits outside both the
    # record stamp and the pose envelope, and could otherwise move mid-sweep unrecorded.
    cycle = summary.get("cycle")
    if isinstance(cycle, dict):
        if cycle.get("posture") == cycles.POSTURE_NONE:
            notes.append(
                "No Cycle: <code>moduleVersion</code>, <code>sampleCoefficient</code> and "
                "the model locks are <strong>not established</strong> for this pool. They "
                "sit outside both the record stamp and the pose envelope, so nothing here "
                "witnesses whether they held still while these runs were collected.")
        else:
            locks = cycle.get("model_locks") or {}
            lock_txt = ", ".join(f"<code>{_esc(name)}</code>@{_esc(str(sha)[:8])}"
                                 for name, sha in sorted(locks.items())) or "none recorded"
            notes.append(
                f"Cycle <code>{_esc(cycle.get('cycle_id'))}</code> "
                f"(<code>{_esc(cycle.get('status'))}</code>, rule: "
                f"<em>{_esc(cycle.get('rule'))}</em>): harness module "
                f"<code>{_esc(cycle.get('module_version') or '—')}</code>, sample "
                f"coefficient <code>{_esc(cycle.get('sample_coefficient') or '—')}</code>, model "
                f"locks {lock_txt}.")

    body = f"{badge} {n} record(s): {detail}."
    if notes:
        body += " " + " ".join(notes)
    cls = "warn" if (schema_mixed or summary.get("cycle_broken")) else "sub"
    return f"<p class='{cls}'>{body}</p>"


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
        code_hash = str(r.get("detector_code_hash") or "")
        rows.append(
            "<tr>"
            f"<td><code>{_esc(r['app_version'])}</code></td>"
            + (f"<td><code>{_esc(code_hash)}</code></td>" if code_hash
               else "<td class='muted'>unknown</td>")
            + f"<td>{_esc(r['first_run_ts'])}</td>"
            f"<td>{_esc(r['last_run_ts'])}</td>"
            f"<td>{int(r['n_records'])}</td>"
            f"<td>{int(r['n_videos'])}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>build</th><th>detectorCodeHash</th><th>first run</th><th>last run</th>"
        "<th>eval records</th><th>videos</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _build_conflict_table(df: pd.DataFrame) -> str:
    """One appVersion, more than one detectorCodeHash — the `c305954` signature (#130)."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return ("<p class='muted'>No appVersion stamps more than one detector build. "
                "Runs without a <code>detectorCodeHash</code> are unknown provenance, "
                "never a conflict.</p>")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td><code>{_esc(r['app_version'])}</code></td>"
            f"<td><code>{_esc(r['detector_code_hash'])}</code></td>"
            f"<td>{int(r['n_runs'])}</td>"
            f"<td>{_esc(r['first_run_ts'])}</td>"
            f"<td>{_esc(r['last_run_ts'])}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table><thead><tr>"
        "<th>appVersion</th><th>detectorCodeHash</th><th>runs</th>"
        "<th>first run</th><th>last run</th></tr></thead>"
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
#
# ``rate-mismatch`` leads the *data* group (issue #101): it is neither a truth problem
# nor a detector problem, and routing it to either worklist wastes the effort.
_NONCONFORMANCE_CAUSE_BLURB = {
    "rate-mismatch": (
        "The ViTPose scaffold sampled on a much coarser grid than the truth was "
        "exported onto, so most truth frames were never looked at and read as absent. "
        "Neither the truth nor the detector is at fault — <strong>regenerate the "
        "scaffold</strong> at the truth's sampling rate and re-export."),
    "trajectory-divergence": (
        "Ample accepted detections and the fit still diverges. <strong>This says the "
        "scanner and the truth disagree; it does not say which one is wrong.</strong> "
        "Read the <em>attribution</em> column, never this cause, to decide whether a "
        "bundle belongs on the truth-repair worklist (#21/#34) — re-seeding a bundle "
        "whose truth is sound repairs nothing. Renamed from "
        "<code>suspected-mistrack</code> in v15 (#147), which asserted the truth side: "
        "measured against the human-attested flags, 112 of its 123 firings landed on "
        "truth attested free of identity error."),
    "sparse-match": (
        "The detector supplied too little to fit — too few matched-present frames, or "
        "too small a share of present attempts accepted. A detector failure tripping a "
        "truth gate; re-seeding truth here would repair nothing. Take these to the "
        "attempt-funnel section instead."),
}


# --------------------------------------------------------------------------- #
# Conformance breakouts on failure-mode sections (issue #132)
# --------------------------------------------------------------------------- #

# Said once, above the first breakout, and referenced by the others: the #15 gate is a
# quarantine for truth-fit metrics and a covariate for failure-mode ones, because on a
# failure-mode metric the gate selects on the very failure being measured.
_CONFORMANCE_COVARIATE_NOTE = (
    "<p class='sub'>This section pools <strong>all</strong> runs. The #15 conformance "
    "gate is a quarantine for <em>truth-fit</em> metrics only (accuracy, agreement, PCK, "
    "normDist), where a bad truth fit makes the number meaningless. On a "
    "<em>failure-mode</em> metric it would select on the very failure being measured — "
    "most sharply for <code>sparse-match</code>, which <em>is</em> the detector supplying "
    "too little to fit — so conformance is reported below as a dimension instead of "
    "applied as a filter (#132). Read <strong>share of attempts</strong> against the "
    "population's own failure rate: a population holding a slice of the corpus and most "
    "of its failures is the gate's selectivity made visible, and is why a "
    "conforming-pool number drifts batch over batch as corpus composition changes.</p>")

# Non-conforming rows are a partition of the corpus at the gate; cause rows are a
# partition of *that* partition. Marked in the label rather than by indentation alone so
# the CSV and the HTML read the same way.
_CONFORMANCE_POPULATION_BLURB = {
    "all": "every run — the section's headline pool",
    "conforming": "passed the #15 gate",
    "non-conforming": "failed the #15 gate",
    "sparse-match": "of non-conforming: detector supplied too little to fit",
    "trajectory-divergence": "of non-conforming: ample detections, fit still diverges",
    "rate-mismatch": "of non-conforming: scaffold sampled coarser than the truth grid",
}


def _conformance_breakout_table(df: Any, empty: str) -> str:
    """Render a ``population``-indexed breakout (see ``trends._conformance_pools``).

    Formatting is driven off the column name — anything that is a share or a rate renders
    as a percentage, everything else as a count — so the four breakouts stay in step
    without four hand-written renderers to keep aligned."""

    if not isinstance(df, pd.DataFrame) or df.empty or "population" not in df.columns:
        return f"<p class='muted'>{empty}</p>"
    cols = [c for c in df.columns if c not in ("population", "kind")]
    head = ("<tr><th>population</th><th>meaning</th>"
            + "".join(f"<th>{_esc(c.replace('_', ' '))}</th>" for c in cols)
            + "</tr>")
    body = []
    for _, r in df.iterrows():
        pop = str(r["population"])
        cause_row = str(r.get("kind")) == "cause"
        cells = []
        for c in cols:
            v = r.get(c)
            if c.endswith("_share") or c.startswith("share_of_") or c.endswith("_rate"):
                cells.append(_pct(v))
            elif c.startswith("median_") or c.endswith("_median") or c.endswith("_p90"):
                cells.append(_fmt(v, 3))
            else:
                cells.append(_fmt_int(v))
        tr = "<tr class='muted'>" if cause_row else "<tr>"
        body.append(
            tr
            + f"<td><code>{_esc(pop)}</code></td>"
            f"<td>{_esc(_CONFORMANCE_POPULATION_BLURB.get(pop, ''))}</td>"
            + "".join(f"<td>{c}</td>" for c in cells)
            + "</tr>")
    return ("<div class='tablewrap'><table><thead>" + head
            + f"</thead><tbody>{''.join(body)}</tbody></table></div>")


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
    unconfirmed = int(split.get("truth_absent_unconfirmed") or 0)
    tail = ("" if not unknown else
            f" {unknown} more come from pre-schema-v12 records that never recorded "
            "presence and are excluded from those shares — re-run <code>evaluate</code> "
            "to place them.")
    # Issue #101: an absence the harness cannot confirm is not evidence of a false
    # positive. Say how many were held out, and why, rather than letting the shares
    # quietly rest on a population that includes scaffold gaps and tracking losses.
    if unconfirmed:
        tail += (f" A further <strong>{unconfirmed}</strong> sit on absent frames whose "
                 "absence is <em>not confirmed</em> — out of scope, never sampled, or a "
                 "tracking loss — and are held out of both shares. See the absence-reason "
                 "breakdown below before reading this split as a presence-gating result.")
    return (f"<p class='sub'>Of {total} pooled <code>hallucination-fp</code> frames: "
            f"{'; '.join(parts)}.{tail}</p>")


def _stale_truth_html(ctx: dict[str, Any]) -> str:
    """Bundles whose Ground Truth has fallen behind its scaffold (issue #101 follow-up).

    Worth its own section rather than a footnote: nothing else detects this. ``setupHash``
    tracks calibration, and re-seeding a scaffold does not change the calibration, so a
    truth authored from a superseded scaffold still reads as accepted and current on both
    sides. Every frame the new scaffold poses and the old truth calls absent becomes a
    phantom absence in the very metric this work exists to make trustworthy."""

    rows = ctx.get("stale_truth_bundles") or []
    if not rows:
        return ("<p class='muted'>No bundle's truth has fallen behind its scaffold — "
                "every authored truth broadly matches what its scaffold poses.</p>")
    head = ("<tr><th>route</th><th>video</th><th>truth present</th>"
            "<th>scaffold poses</th><th>shortfall</th><th>ratio</th></tr>")
    body = "".join(
        f"<tr><td>{_esc(r['route_folder'])}</td><td>{_esc(r['video_key'])}</td>"
        f"<td>{int(r['truth_present'])}</td><td>{int(r['scaffold_posed'])}</td>"
        f"<td>{int(r['shortfall'])}</td><td>{_pct(r['ratio'])}</td></tr>"
        for r in rows
    )
    worst = rows[0]
    return (
        f"<p class='sub'><strong>{len(rows)}</strong> bundle"
        f"{'' if len(rows) == 1 else 's'} carry Ground Truth authored from a superseded "
        "scaffold. Re-accept these in the scanner before trusting any absence-derived "
        "number from them — until then each contributes phantom absences, where the "
        "scaffold poses a Climber the truth calls missing. Worst: "
        f"<code>{_esc(worst['video_key'])}</code>, {int(worst['truth_present'])} present "
        f"against {int(worst['scaffold_posed'])} posed.</p>"
        "<p class='sub muted'>Detected by comparing present-frame counts, because "
        "nothing better exists yet: <code>setupHash</code> tracks calibration, and a "
        "re-seed does not change the calibration, so a stale truth pairs as current. "
        "The durable fix is for Ground Truth to stamp the scaffold "
        "<code>seedHash</code> it was authored from.</p>"
        "<div class='tablewrap'><table><thead>" + head +
        f"</thead><tbody>{body}</tbody></table></div>")


def _absence_reason_html(ctx: dict[str, Any]) -> str:
    """How the pooled truth-absent frames split by reason (issue #101).

    The corpus audit that motivated this found 44% of every pooled truth-absent frame
    coming from five videos where "absent" meant a scaffold that never sampled the
    frame or a tracker that lost the Climber — not a Climber who left. That population
    is the evidence base under the hallucination headline, so the breakdown belongs
    beside it rather than in a CSV."""

    table = ctx.get("frame_quality_absence_reasons")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return ("<p class='muted'>No truth-absent frames pooled — nothing to attribute. "
                "(Pre-v14 records carry no reason; re-run <code>evaluate</code>.)</p>")

    blurbs = {
        "confirmed-absent": "the Climber really is not in the frame — <strong>the only "
                            "reason that counts as an absence</strong>",
        "out-of-scope": "outside the climb window: before the climb started or after "
                        "the topout",
        "not-sampled": "the ViTPose scaffold never sampled this frame — a scaffold "
                       "artifact, fixed by regenerating truth",
        "untracked": "the scaffold's tracker lost or never acquired the Climber — a "
                     "truth-repair problem, not a scanner one",
        "unknown": "no evidence on disk to derive a reason from; never counted as an "
                   "absence",
    }
    head = ("<tr><th>reason</th><th>frames</th><th>share</th>"
            "<th>counts as absent</th><th>what it means</th></tr>")
    body = "".join(
        f"<tr><td><code>{_esc(r['reason'])}</code></td><td>{int(r['n'])}</td>"
        f"<td>{_pct(r['share'])}</td>"
        f"<td>{'yes' if r['counts_as_absent'] else 'no'}</td>"
        f"<td>{blurbs.get(r['reason'], '')}</td></tr>"
        for _, r in table.iterrows()
    )
    confirmed = table.loc[table["reason"] == "confirmed-absent", "share"]
    lead = ""
    if len(confirmed):
        lead = (f"<p class='sub'><strong>{_pct(float(confirmed.iloc[0]))}</strong> of "
                "pooled truth-absent frames are confirmed absences; the rest are held "
                "out of the presence 2×2 and the hallucination split.</p>")

    # The underlying data defect, reported whether or not it tripped the gate. A Bundle
    # can under-sample its truth grid tenfold and still fit cleanly on the frames it did
    # sample, so `rate-mismatch` (a non-conformance *cause*) stays silent on it — while
    # it fabricates absences by the thousand. The fix is the same either way.
    n_mismatch = int(ctx.get("rate_mismatch_count") or 0)
    tail = ""
    if n_mismatch:
        rows = ctx.get("rate_mismatch_records") or []
        conforming = sum(1 for r in rows if r.get("conforms"))
        tail = (f"<p class='sub'><strong>{n_mismatch}</strong> record"
                f"{'' if n_mismatch == 1 else 's'} sampled the ViTPose scaffold on a "
                "grid at least "
                f"{_fmt(RATE_MISMATCH_MIN_RATIO)}× coarser than the truth grid — the "
                "source of the <code>not-sampled</code> frames above. "
                f"{conforming} of them still <em>pass</em> the conformance gate, so the "
                "<code>rate-mismatch</code> quarantine cause never fires on them: "
                "regenerate those scaffolds at the truth's sampling rate. Full list in "
                "<code>eval_rate_mismatch_records.csv</code>.</p>")

    return (lead + "<div class='tablewrap'><table><thead>" + head
            + f"</thead><tbody>{body}</tbody></table></div>" + tail)


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
    class_tbl = (_hallucination_split_html(ctx) + class_tbl
                 + "<h3>Why the absent frames are absent (#101)</h3>"
                 + _absence_reason_html(ctx)
                 + "<h3>Truth that has fallen behind its scaffold (#101)</h3>"
                 + _stale_truth_html(ctx))

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

    conformance_tbl = _conformance_breakout_table(
        ctx.get("attempt_funnel_conformance"),
        "No conformance breakout — no run carries a conformance verdict.")

    return (
        tiles
        + "<h3>Status mix (pooled attempts vs run-unit distribution)</h3>" + status_tbl
        + "<h3>Conformance breakout (covariate, not a filter)</h3>"
        + _CONFORMANCE_COVARIATE_NOTE + conformance_tbl
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
                f"<td>{_fmt(r.get('median_best_unselected_candidate_score'))}</td>"
                "</tr>"
            )
        cause_tbl = (
            "<p class='sub'>Why the detector found no Climber, per matched missing "
            "attempt. <code>identity-gated</code> means candidates existed but the "
            "identity gate rejected every one (a scanner gating decision); "
            "<code>no-candidates</code> means MediaPipe returned nothing anywhere "
            "searched (a detector failure). Split by the scanner's "
            "<code>missReason</code>, retro-derived from <code>candidateCount</code> on "
            "older streams. <code>crop-misplaced</code> requires that the misplaced crop "
            "was the <em>only</em> place searched &mdash; when a full-frame reacquire "
            "also ran and failed, the Climber was searched for everywhere, so the crop "
            "cannot be what lost them. Crop placement is still measured on every miss "
            "in the next column.</p>"
            "<div class='tablewrap'><table><thead><tr><th>miss cause</th><th>n</th>"
            "<th>share</th><th>crop excluded Climber</th>"
            "<th>median crop containment</th><th>condition flags fired</th>"
            "<th>median best unselected candidate score</th></tr></thead>"
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
        "miss_identity_gated_share", "miss_no_candidates_share",
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

    # Issue #132: rejection correctness, crop placement and the miss-cause mix are all
    # failure-mode metrics, so each is pooled over every run and broken out by
    # conformance rather than gated on it. The covariate note is stated once, in the
    # attempt-funnel section above, and referred back to here.
    rejection_tbl = _conformance_breakout_table(
        ctx.get("rejection_conformance"),
        "No conformance breakout for rejections — no run carries a conformance verdict.")
    crop_conf_tbl = _conformance_breakout_table(
        ctx.get("crop_quality_conformance"),
        "No conformance breakout for crop placement — no scored crop attempts.")
    miss_conf_tbl = _conformance_breakout_table(
        ctx.get("crop_quality_miss_cause_conformance"),
        "No conformance breakout for miss causes — no scored missing attempts.")
    conformance_note = (
        "<p class='sub'>All three tables below pool <strong>every</strong> run and carry "
        "conformance as a dimension (#132) — see the covariate note in the attempt-funnel "
        "section above for why a failure-mode metric must not be gated on it. Cause rows "
        "(greyed) partition the non-conforming population.</p>")

    return (
        tiles
        + "<h3>Crop placement vs Ground Truth</h3>" + crop_tiles
        + "<h3>Missing-attempt causes</h3>" + cause_tbl
        + "<h3>Conformance breakout (covariate, not a filter)</h3>" + conformance_note
        + "<h4>Miss-cause mix by conformance</h4>" + miss_conf_tbl
        + "<h4>Rejection correctness by conformance</h4>" + rejection_tbl
        + "<h4>Crop placement by conformance</h4>" + crop_conf_tbl
        + "<h3>Run-level attempt conditions vs Detection Errors</h3>" + band_tbl
        + "<h3>Worst runs with attempt evidence</h3>" + _df_to_table(top)
    )


# --------------------------------------------------------------------------- #
# Arm-versus-arm reporting (issue #164)
# --------------------------------------------------------------------------- #

def _sigfmt(v: Any, nd: int = 4) -> str:
    """A floor or a delta, at the precision the measurement actually supports."""

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, (int, float)):
        return f"{float(v):.{nd}f}"
    return _esc(v)


def _signed(v: Any, nd: int = 4) -> str:
    """A delta with its sign always shown — ``+0.0123`` reads as a direction, ``0.0123``
    reads as a magnitude, and an arm comparison is about direction."""

    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{float(v):+.{nd}f}"


# The one thing a reader must not take away from this section. ADR 0010 makes the accuracy
# tier permanently empty, so no absolute here is an accuracy claim; but both arms of a
# comparison score against the *same* fixed truth, so truth error is common-mode and
# cancels in the difference. The absolute is uninterpretable, the delta is not — and that
# distinction is the whole reason arm comparison survives an empty accuracy tier.
_ARM_PCK_NOTE = (
    "<p class='sub'><strong>Agreement PCK is the primary outcome, and its absolute value "
    "is uninterpretable.</strong> Every PCK here is scored against a ViTPose scaffold that "
    "no human has attested, so the accuracy tier is permanently empty (ADR 0010, #133) and "
    "an absolute of 0.80 is <em>not</em> a claim that the detector was right 80% of the "
    "time. <strong>The arm delta is a different quantity.</strong> Both arms score against "
    "the <em>same</em> fixed truth on the <em>same</em> Bundle, so whatever the truth gets "
    "wrong it gets wrong identically for both and cancels in the difference. Read the "
    "delta columns; treat the absolutes as a scale, not a score.</p>")

_ARM_UNCERTAINTY_NOTE = (
    "<p class='sub'><strong>Which floor travels with which number.</strong> "
    f"<em>Harness run-to-run scatter is exactly {floors.HARNESS_RUN_TO_RUN:.0f}</em>, "
    f"because {_esc(floors.HARNESS_RUN_TO_RUN_NOTE)}. So no run-to-run term is attached "
    "to a harness arm, and none should be — #134's 0.0055 is the <em>scanner's</em> "
    "scatter, a different quantity from a different producer, and attaching it here would "
    "be a category error. What a harness arm carries instead is <strong>sampling "
    "error</strong>: runs score a <code>12·√n</code> sample of the Bundle's truth grid, "
    f"measured against the full grid at median <code>{floors.SAMPLING_ERROR.median:.4f}"
    f"</code> / p90 <code>{floors.SAMPLING_ERROR.p90:.4f}</code> |ΔPCK| across "
    f"{floors.SAMPLING_ERROR.n_groups} Bundles. That error is "
    f"{_esc(floors.SAMPLING_ERROR_COMMON_MODE_NOTE)} — so it is printed against the "
    "per-video absolutes below and explicitly discounted in the deltas, where whatever "
    "survives is flagged against the p90 anyway as the conservative bar. Any floor "
    "labelled <span class='flag'>scanner-side</span> is #134's figure for the scanner's "
    "detector and is never a harness uncertainty.</p>")


# --------------------------------------------------------------------------- #
# The Cycle (issue #176) — which rule the arm comparison was read under
# --------------------------------------------------------------------------- #

def _cycle_rule_html(summary: Any, swept: int = 0, pooled: int = 0) -> str:
    """State the rule applied, first thing, in the section's own words.

    #132 set the precedent that no section may be read as if it were the other, so the
    posture is declared before any table rather than inferred from whether rows are
    missing. Four postures, four different things a reader is allowed to conclude.

    ``swept`` / ``pooled`` are the Bundles the arms actually ran on and the subset that
    survived the rule — a distinct count from the Cycle's corpus-wide comparable set, and
    the one that says what the gate did *here*.
    """

    if not isinstance(summary, dict):
        return ""
    posture = summary.get("posture")
    cid = _esc(summary.get("cycle_id") or "")
    if posture == cycles.POSTURE_CERTIFIED:
        return (
            f"<p class='sub'><span class='flag tier'>Cycle {cid}: CERTIFIED</span> "
            "<strong>Rule applied: gate.</strong> The pooled arm summary and the deltas "
            "below are computed over the Bundles this Cycle certified as comparable — the "
            "ones whose truth, setup and crop trajectory were byte-identical at Cycle open "
            f"and Cycle close ({int(summary.get('comparable_count') or 0)} of "
            f"{int(summary.get('bundle_count') or 0)} corpus-wide; <strong>{pooled} of the "
            f"{swept}</strong> the arms actually ran). Those are truth-fit numbers in "
            "#132's sense, and a Bundle whose truth moved mid-Cycle yields a delta that "
            "silently contains a truth change rather than a merely noisy one. The "
            "per-Bundle table keeps <em>every</em> Bundle the arms ran on, comparability "
            "marked in its own column: the gate removes rows from the pooled lines, never "
            "from the evidence.</p>")
    if posture == cycles.POSTURE_UNCERTIFIED:
        return (
            f"<div class='banner'>CYCLE {cid} DID NOT CERTIFY "
            f"({_esc(summary.get('status') or '')}"
            + (f": {_esc(', '.join(summary.get('failures') or []))}"
               if summary.get("failures") else "")
            + ") — NO ARM COMPARISON IS PUBLISHED.</div>"
            "<p class='warn'><strong>Rule applied: refuse.</strong> The manifest carries a "
            "<code>comparableBundles</code> list even when the Cycle fails, and its own "
            "close log says <em>“The arms in this cycle are NOT comparable to each other. "
            "Do not publish a comparison over them.”</em> Keying on the presence of that "
            "list rather than on <code>certified</code> would publish exactly what the "
            "artifact forbids, so no pooled summary, no delta table and no arm ranking is "
            "rendered. The runs are still named below, as evidence of what was collected — "
            "they are not laid out as a comparison, because they are not one.</p>")
    if posture == cycles.POSTURE_IN_FLIGHT:
        return (
            f"<p class='warn'><span class='flag'>Cycle {cid}: IN FLIGHT</span> "
            "<strong>Rule applied: label, provisionally.</strong> This Cycle is open — the "
            "sweep is still running — and <code>comparableBundles</code> is written only "
            "at close, so there is nothing to gate on yet. Everything below is scoped to "
            "the runs that have landed since the Cycle opened and is <strong>provisional</strong>: "
            "it is not a certified comparison and must not be published as one. Close the "
            "Cycle (<code>python cycle_integrity.py close</code>) to learn whether the "
            "detector and the Bundles held still across it.</p>")
    return (
        "<p class='warn'><span class='flag'>NOT DRIFT-CHECKED</span> "
        "<strong>Rule applied: label, don't gate.</strong> No Cycle artifact exists under "
        "<code>analysis/cycles/</code>, so nothing establishes that the truth, the crop "
        "trajectories, the model weights or the harness module held still between the "
        "first arm below and the last. Batches are <em>mode-major</em> by design, which "
        "means any drift across a sweep is perfectly confounded with the factor being "
        "swept. The comparison is reported in full rather than withheld — there is nothing "
        "to gate against, and refusing would teach nothing — but it carries no drift "
        "guarantee, and this is the entire pre-#168 corpus.</p>")


def _cycle_verdict_html(summary: Any, bundle_rows: Any) -> str:
    """The Cycle's own verdict: the canary, the Bundles it dropped, and by what."""

    if not isinstance(summary, dict) or summary.get("posture") == cycles.POSTURE_NONE:
        return ""
    canary = summary.get("canary") or {}
    locks = summary.get("model_locks") or {}
    facts = [
        ("cycle", f"<code>{_esc(summary.get('cycle_id'))}</code>"),
        ("status", f"<code>{_esc(summary.get('status'))}</code>"
                   f" (certified: <code>{str(bool(summary.get('certified'))).lower()}</code>)"),
        ("window", f"<code>{_esc(summary.get('opened_run_ts'))}</code> → "
                   f"<code>{_esc(summary.get('closed_run_ts') or 'still open')}</code>"),
        ("bundles", f"{int(summary.get('comparable_count') or 0)} comparable / "
                    f"{int(summary.get('excluded_count') or 0)} excluded / "
                    f"{int(summary.get('newly_eligible_count') or 0)} newly eligible, of "
                    f"{int(summary.get('bundle_count') or 0)} snapshotted"),
        ("harness", f"module <code>{_esc(summary.get('module_version') or '—')}</code>, "
                    f"sample coefficient "
                    f"<code>{_esc(summary.get('sample_coefficient') or '—')}</code>, "
                    f"{len(locks)} model lock(s)"),
    ]
    runs = summary.get("runs") or {}
    if runs:
        facts.append(("cycle's own run census",
                      f"{int(runs.get('runCount') or 0)} run(s) across "
                      f"{len(runs.get('arms') or {})} arm(s) on "
                      f"{int(runs.get('bundlesWithRuns') or 0)} bundle(s)"))
    others = summary.get("other_cycles") or []
    if others:
        # Named, never merged: two Cycles are two comparison groups, and pooling them would
        # rebuild the unattributable population the Cycle exists to prevent.
        facts.append((
            "other cycles on disk",
            ", ".join(f"<code>{_esc(c.get('cycle_id'))}</code> "
                      f"({_esc(c.get('status'))})" for c in others)
            + " — not merged into this one; each is its own comparison group"))
    fact_rows = "".join(f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>" for k, v in facts)

    if canary.get("identical") is True:
        canary_html = (
            "<p class='sub'><strong>Determinism canary: byte-identical</strong> over "
            f"{int(canary.get('frames_compared') or 0)} frames of "
            f"<code>{_esc(canary.get('route'))}/{_esc(canary.get('video_key'))}</code> "
            f"at detection rate {_fmt(canary.get('opened_detection_rate'), 3)} → "
            f"{_fmt(canary.get('closed_detection_rate'), 3)}. Nothing moved between Cycle "
            "open and Cycle close: not the weights, not the module, not the crop "
            "trajectory, not the environment.</p>")
    elif canary.get("identical") is False:
        fields = "".join(
            f"<li><code>{_esc(f.get('field'))}</code>: {_esc(f.get('opened'))} → "
            f"{_esc(f.get('closed'))}</li>"
            for f in (canary.get("fields") or []))
        canary_html = (
            "<p class='warn'><strong>Determinism canary: DRIFTED.</strong> "
            f"{int(canary.get('frames_differing') or 0)} of "
            f"{int(canary.get('frames_compared') or 0)} frames differ"
            + (f", first at t={_esc(canary.get('first_divergence'))}s"
               if canary.get("first_divergence") is not None else "")
            + ". The detector that produced the first arm is not the detector that produced "
              "the last.</p>"
            + (f"<ul class='sub'>{fields}</ul>" if fields else ""))
    else:
        canary_html = (
            "<p class='sub'>Determinism canary: opened at detection rate "
            f"{_fmt(canary.get('opened_detection_rate'), 3)} on "
            f"<code>{_esc(canary.get('route'))}/{_esc(canary.get('video_key'))}</code>; "
            "the closing pass has not run, so nothing is compared yet.</p>")

    return ("<h3>Cycle verdict</h3>"
            + "<div class='tablewrap'><table><tbody>" + fact_rows + "</tbody></table></div>"
            + canary_html
            + _cycle_bundle_table(bundle_rows))


def _cycle_bundle_table(rows: Any) -> str:
    """Bundles the Cycle dropped, named with their verdicts (the #15/#88 precedent).

    ``newly-eligible`` is rendered as its own state rather than as a kind of exclusion:
    those Bundles were never snapshotted, so they did not fail anything — they were not in
    the Cycle at all — and letting them read as failures would put a Bundle on a repair
    worklist for the crime of having been created.
    """

    if not rows:
        return ("<p class='muted'>No Bundle moved between Cycle open and Cycle close, and "
                "none became eligible during it: the snapshotted set and the comparable "
                "set are the same set.</p>")
    body = "".join(
        "<tr>"
        f"<td>{_esc(r.get('route_folder'))}</td><td>{_esc(r.get('video_key'))}</td>"
        f"<td><span class='flag'>{_esc(r.get('state'))}</span></td>"
        f"<td>{_esc(r.get('reasons') or '—')}</td>"
        f"<td class='muted'>{_esc(r.get('detail') or '')}</td></tr>"
        for r in rows)
    return (
        "<p class='sub'>Bundles the Cycle names. <code>excluded</code> means an input "
        "moved between open and close — a truth re-seed, a recalibration, a rebuilt crop "
        "trajectory — so runs from the start of the Cycle and runs from the end were "
        "measured against different things. <code>newly-eligible</code> is <em>not</em> a "
        "failure: those Bundles were never snapshotted, so they were never in the Cycle "
        "rather than dropped from it.</p>"
        "<div class='tablewrap'><table><thead><tr><th>route</th><th>bundle</th>"
        "<th>state</th><th>reasons</th><th>what moved</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>")


def _cycle_window_html(rows: Any, summary: Any) -> str:
    """Harness runs the Cycle's window placed outside it — named, never silently dropped."""

    if not isinstance(summary, dict) or summary.get("posture") == cycles.POSTURE_NONE:
        return ""
    if not rows:
        return ("<p class='muted'>Every harness run on this corpus falls inside the "
                "Cycle's window, so the comparison and the corpus are the same "
                "population.</p>")
    body = "".join(
        "<tr>"
        f"<td>{_esc(r.get('route_folder'))}</td><td>{_esc(r.get('video_key'))}</td>"
        f"<td><code>{_esc(str(r.get('config_hash'))[:8])}</code></td>"
        f"<td>{_esc(r.get('arm'))}</td>"
        f"<td><code>{_esc(r.get('run_ts'))}</code></td>"
        f"<td><span class='flag'>{_esc(r.get('placement'))}</span></td></tr>"
        for r in rows)
    unplaceable = sum(1 for r in rows
                      if r.get("placement") == cycles.RUN_UNPLACEABLE)
    note = ("" if not unplaceable else
            f" {unplaceable} of them carry no <code>exp-</code> run id at all (they predate "
            "the #160 convention), so neither this join nor the Cycle's own run census can "
            "place them — that is a different statement from being outside the window, and "
            "they are marked <code>unplaceable</code> rather than excluded.")
    return (
        f"<h3>Harness runs outside the Cycle</h3>"
        f"<p class='warn'>{len(rows)} harness run(s) do not fall inside "
        f"<code>{_esc(summary.get('opened_run_ts'))}</code> → "
        f"<code>{_esc(summary.get('closed_run_ts') or 'now')}</code> and therefore do not "
        "pool into the comparison above. Nothing durable stamps a run with its Cycle — the "
        "association lives in the batch 202 response and nowhere else — so the base "
        "timestamp in the run id is the join, and a timestamp window is a weaker join than "
        f"a stamp. Every run it drops is listed rather than merely subtracted.{note}</p>"
        "<div class='tablewrap'><table><thead><tr><th>route</th><th>bundle</th>"
        "<th>arm</th><th>factors</th><th>run</th><th>placement</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>")


def _arm_evidence_table(df: Any) -> str:
    """The runs collected under a Cycle that did not certify — listed, not compared.

    Deliberately **not** the Bundle × arm matrix. A matrix is a comparison layout, and the
    failed Cycle's own artifact forbids publishing a comparison over these runs; a flat
    list by Bundle records what was collected without inviting a reader to difference two
    cells that the canary says were produced by two different detectors.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No harness runs were collected inside this Cycle.</p>"
    body = "".join(
        "<tr>"
        f"<td>{_esc(r['route_folder'])}</td><td>{_esc(r['video_key'])}</td>"
        f"<td><code>{_esc(str(r['config_hash'])[:8])}</code></td>"
        f"<td>{_esc(r['arm'])}</td><td>{int(r['runs'])}</td>"
        f"<td>{_sigfmt(r['pck'], 3)}</td></tr>"
        for _, r in df.sort_values(["route_folder", "video_key", "config_hash"]).iterrows())
    return (
        "<p class='sub'>What the Cycle collected, by Bundle. These are absolutes scored "
        "against an unattested scaffold (ADR 0010) and the Cycle did not certify that the "
        "detector producing them held still, so no two of them are a difference — read "
        "this as a record of work done, not as a result.</p>"
        "<div class='tablewrap'><table><thead><tr><th>route</th><th>bundle</th>"
        "<th>arm</th><th>factors</th><th>runs</th><th>PCK</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>")


def _arm_origin_table(df: Any) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No evaluation records on this corpus.</p>"
    rows = "".join(
        "<tr>"
        f"<td><code>{_esc(r['origin'])}</code></td>"
        f"<td>{int(r['records'])}</td><td>{int(r['bundles'])}</td>"
        f"<td>{int(r['trusted'])}</td><td>{_esc(r['pool'])}</td></tr>"
        for _, r in df.iterrows()
    )
    return ("<div class='tablewrap'><table><thead><tr><th>origin</th><th>records</th>"
            "<th>bundles</th><th>trusted</th><th>pooled into</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def _arm_overview_table(df: Any, summary: Any = None) -> str:
    """Per arm: the pooled central value, never without the spread that contradicts it."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        # Empty for two very different reasons, and reporting the wrong one would be a lie
        # about the corpus: either nothing was collected, or the Cycle gated all of it out.
        if isinstance(summary, dict) and summary.get("posture") == cycles.POSTURE_CERTIFIED:
            return ("<p class='warn'>No arm pools under this Cycle. Runs were collected, "
                    "but every Bundle they landed on was excluded from "
                    "<code>comparableBundles</code> — see the Cycle verdict above for "
                    "which inputs moved. This is a gated-empty result, not an empty "
                    "corpus.</p>")
        return ("<p class='muted'>No experimental arms — this corpus holds no "
                "harness-produced runs with evaluation records.</p>")
    rows = "".join(
        "<tr>"
        f"<td><code>{_esc(str(r['config_hash'])[:8])}</code></td>"
        f"<td>{_esc(r['arm'])}</td>"
        f"<td>{int(r['bundles'])}</td><td>{int(r['runs'])}</td>"
        f"<td>{_sigfmt(r['pck_median'], 3)}</td>"
        f"<td>{_sigfmt(r['pck_min'], 3)} – {_sigfmt(r['pck_max'], 3)}</td>"
        f"<td>{_sigfmt(r['pck_spread'], 3)}</td>"
        f"<td>{int(r['conforming_bundles'])}/{int(r['bundles'])}</td>"
        "</tr>"
        for _, r in df.iterrows()
    )
    return (
        "<p class='sub'>The median is printed with the per-Bundle range beside it because "
        "the range is usually the bigger number. Tracked-crop detection ranged 59–100% "
        "across six Bundles in this corpus, so a pooled median of 81% described none of "
        "them — a spread column wider than every delta below means the arms are not what "
        "is moving this metric.</p>"
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>factors</th>"
        "<th>bundles</th><th>runs</th><th>PCK median</th><th>PCK min – max</th>"
        "<th>spread (max−min)</th><th>conforming</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>")


def _arm_matrix_table(df: Any, reach: Any = None) -> str:
    """Bundle × arm PCK — the table that answers "which settings work for which videos".

    Every absolute carries the sampling error, once, in the caption rather than repeated in
    every cell: it is the same number for every cell by construction, and repeating it
    would suggest it varies.

    The baseline arm leads the columns, because every other column is read as a difference
    from it and hunting for the reference among alphabetised hashes is friction with no
    upside.

    **This table is the covariate half of the #176 rule.** It keeps every Bundle the arms
    ran on — including the ones the Cycle excluded from pooling — and carries comparability
    as a column instead. A Bundle that vanished from here would be an exclusion nobody
    could see, and the gate above it is only defensible while the evidence underneath
    stays visible.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No per-Bundle arm results.</p>"
    base = str((reach or {}).get("baseline_hash") or "")
    arms = list(dict.fromkeys(df["config_hash"]))
    arms.sort(key=lambda h: (0 if h == base else 1, str(h)))
    labels = {h: str(df[df["config_hash"] == h]["arm"].iloc[0]) for h in arms}
    badge = "<span class='flag'>baseline</span>"
    head = "".join(
        f"<th title='{_esc(labels[h])}'><code>{_esc(str(h)[:8])}</code>"
        + (" " + badge if h == base else "") + "</th>"
        for h in arms)
    has_cycle = "cycle_state" in df.columns
    cycle_head = "<th>cycle</th>" if has_cycle else ""
    body = []
    for (route, key), g in df.groupby(["route_folder", "video_key"], dropna=False):
        by_arm = {r["config_hash"]: r for _, r in g.iterrows()}
        cells = []
        for h in arms:
            r = by_arm.get(h)
            if r is None:
                cells.append("<td class='muted'>not run</td>")
                continue
            mark = "" if r["conforms"] else " <span class='flag'>#15</span>"
            cells.append(f"<td>{_sigfmt(r['pck'], 3)}{mark}</td>")
        cycle_cell = ""
        if has_cycle:
            state = str(g["cycle_state"].iloc[0])
            detail = str(g["cycle_detail"].iloc[0] or "")
            pooled = bool(g["cycle_comparable"].iloc[0])
            cls = "" if pooled else " class='muted'"
            cycle_cell = (f"<td{cls} title='{_esc(detail)}'>"
                          f"<span class='flag'>{_esc(state)}</span></td>")
        body.append(f"<tr><td>{_esc(route)}</td><td>{_esc(key)}</td>"
                    f"{cycle_cell}{''.join(cells)}</tr>")
    cycle_note = ("" if not has_cycle else
                  " The <code>cycle</code> column is the comparability covariate: a row "
                  "marked anything but <code>comparable</code> is still shown here in full "
                  "but does <em>not</em> contribute to the pooled summary or the deltas.")
    return (
        "<p class='sub'>Agreement PCK per Bundle per arm. <strong>±"
        f"{floors.SAMPLING_ERROR.p90:.4f}</strong> sampling error (p90) applies to every "
        "absolute in this table identically — it is stated once because it is the same "
        "number in every cell by construction, and a per-cell ± would imply it varies. "
        "<code>not run</code> is a gap in the sweep, not a zero. <span class='flag'>#15</span> "
        "marks a Bundle whose truth fit failed the conformance gate on that arm: its "
        "absolute is not comparable to a conforming one, though the arm delta on it still "
        f"is, since both arms face the same broken fit.{cycle_note}</p>"
        "<div class='tablewrap'><table><thead><tr><th>route</th><th>bundle</th>"
        f"{cycle_head}{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def _arm_delta_table(summary: Any, per_video: Any, cycle: Any = None) -> str:
    """Arm versus baseline, pooled over shared Bundles and then per Bundle underneath."""

    if not isinstance(summary, pd.DataFrame) or summary.empty:
        gated = (isinstance(cycle, dict)
                 and cycle.get("posture") == cycles.POSTURE_CERTIFIED)
        note = (" Note that this Cycle <strong>gates</strong> the delta population to its "
                "comparable Bundles, so a pair that shares a Bundle only outside "
                "<code>comparableBundles</code> produces no row here by design — the "
                "Bundle × arm table above still shows both."
                if gated else "")
        return ("<p class='muted'><strong>No arm comparison is possible on this corpus.</strong> "
                "A delta needs two arms that ran the <em>same</em> Bundle; every arm here "
                "either stands alone or shares no Bundle with another. Nothing is reported "
                "rather than a difference of pooled means over different videos, which "
                f"would measure which videos each arm happened to run on.{note}</p>")
    rows = []
    for _, r in summary.iterrows():
        flag = ("<span class='flag'>below sampling error</span>"
                if bool(r["all_below_sampling_error"]) else "")
        rows.append(
            "<tr>"
            f"<td><code>{_esc(str(r['config_hash'])[:8])}</code></td>"
            f"<td>{_esc(r['arm'])}</td>"
            f"<td><code>{_esc(str(r['baseline_hash'])[:8])}</code></td>"
            f"<td>{int(r['shared_bundles'])}</td>"
            f"<td>{_signed(r['delta_median'])}</td>"
            f"<td>{_signed(r['delta_min'])} – {_signed(r['delta_max'])}</td>"
            f"<td>{int(r['bundles_improved'])}↑ {int(r['bundles_regressed'])}↓</td>"
            f"<td>{int(r['bundles_below_sampling_error'])}/{int(r['shared_bundles'])} {flag}</td>"
            "</tr>")
    pooled = (
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>factors</th>"
        "<th>vs baseline</th><th>shared bundles</th><th>ΔPCK median</th>"
        "<th>ΔPCK range</th><th>direction</th><th>below sampling error</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")

    detail = ""
    if isinstance(per_video, pd.DataFrame) and not per_video.empty:
        drows = "".join(
            "<tr>"
            f"<td><code>{_esc(str(r['config_hash'])[:8])}</code></td>"
            f"<td>{_esc(r['route_folder'])}</td><td>{_esc(r['video_key'])}</td>"
            f"<td>{_sigfmt(r['baseline_pck'], 3)}</td><td>{_sigfmt(r['pck'], 3)}</td>"
            f"<td>{_signed(r['delta_pck'])}</td>"
            f"<td>{'yes — indistinguishable from noise' if r['below_sampling_error'] else 'no'}</td>"
            "</tr>"
            for _, r in per_video.iterrows())
        detail = (
            "<h4>Per Bundle</h4>"
            "<div class='tablewrap'><table><thead><tr><th>arm</th><th>route</th>"
            "<th>bundle</th><th>baseline PCK</th><th>arm PCK</th><th>ΔPCK</th>"
            "<th>below sampling error</th></tr></thead>"
            f"<tbody>{drows}</tbody></table></div>")

    return (
        "<p class='sub'><strong>Paired on shared Bundles, not pooled means.</strong> Only "
        "Bundles that <em>both</em> arms ran contribute a delta; a pair with no shared "
        "Bundle produces no row at all. Bundles differ from one another far more than arms "
        "differ on one Bundle, so a difference of pooled means over non-identical Bundle "
        "sets would measure the sweep's coverage rather than the condition. "
        "<strong>Sampling error is discounted here</strong> — both arms scored the same "
        "frames of the same Bundle against the same truth, so it is a shared offset that "
        "largely cancels — and what survives is still flagged against its p90 "
        f"(<code>{floors.SAMPLING_ERROR.p90:.4f}</code>) as the conservative bar for "
        "\"could this be nothing?\". A median near zero with a wide range is not a null "
        "result: it is an arm that helps some videos and hurts others, which is precisely "
        "what the per-Bundle rows below are for.</p>"
        + pooled + detail)


def _arm_floor_table(arm_bundles: Any) -> str:
    """Every funnel-derived metric with its measured noise floor beside it.

    The floors are #134's and every one of them is scanner-side, which the table says in a
    column rather than in a footnote. A harness arm's cells are *structurally* empty — the
    harness emits no Detector Attempt stream — and that renders as "not produced by this
    origin" rather than 0.000, because a detector that was never asked to reject anything
    has not rejected nothing.
    """

    harness_present = (isinstance(arm_bundles, pd.DataFrame)
                       and not arm_bundles.empty)
    rows = []
    for key in floors.FUNNEL_FLOOR_KEYS:
        f = floors.scanner_floor(key)
        if f is None:
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_esc(key)}</code></td><td>{_esc(f.metric)}</td>"
            "<td class='muted'>not produced by this origin</td>"
            f"<td>{_sigfmt(f.median)}</td><td>{_sigfmt(f.p90)}</td>"
            f"<td>{_sigfmt(f.typical_value)}</td>"
            f"<td><span class='flag'>{_esc(f.label)}</span></td>"
            "</tr>")
    pck = floors.scanner_floor("pck")
    pck_row = (
        "<tr>"
        "<td><code>pck</code></td><td>agreement PCK</td>"
        "<td>measured — see the tables above</td>"
        f"<td>{_sigfmt(floors.SAMPLING_ERROR.median)}</td>"
        f"<td>{_sigfmt(floors.SAMPLING_ERROR.p90)}</td>"
        f"<td>{_sigfmt(pck.typical_value) if pck else '—'}</td>"
        "<td><span class='flag'>harness-side sampling error</span></td>"
        "</tr>")
    note = ("" if harness_present else
            "<p class='muted'>No harness arm on this corpus, so the value column is empty "
            "for a second reason as well.</p>")
    return (
        "<p class='sub'>"
        f"{_esc(floors.FUNNEL_ABSENT_ON_HARNESS_NOTE)}. The floors beside "
        "them are #134's measurement of the <em>scanner's</em> detector, labelled as such "
        "in every row so they can never be read as a harness uncertainty. "
        f"<strong>{_esc(floors.SCANNER_FLOOR_CAVEAT)}.</strong> The PCK row is the "
        "exception: it is the one metric the harness does produce, and the floor beside it "
        "is the harness sampling error, not #134's scanner figure "
        f"(<code>{pck.median if pck else 0:.4f}</code>, which the p90 of the sampling rule "
        "was chosen to sit at).</p>"
        "<div class='tablewrap'><table><thead><tr><th>metric key</th><th>metric</th>"
        "<th>harness arms</th><th>floor (median)</th><th>floor (p90)</th>"
        "<th>typical value</th><th>whose floor</th></tr></thead>"
        f"<tbody>{pck_row}{''.join(rows)}</tbody></table></div>" + note)


def _arm_reach_html(reach: Any) -> str:
    """How far the comparison reaches — the caveats that are properties of the sweep.

    Rendered as a warning band rather than a footnote because both findings invalidate a
    reading rather than qualifying it: a delta smaller than the between-Bundle spread does
    not generalise, and a delta from one Bundle is an anecdote. Neither is visible in any
    cell of the tables they sit above.
    """

    if not isinstance(reach, dict):
        return ""
    parts: list[str] = []
    spread = reach.get("baseline_spread")
    biggest = reach.get("max_abs_delta")
    under = int(reach.get("deltas_under_spread") or 0)
    n_arms = int(reach.get("delta_arms") or 0)
    if spread is not None and biggest is not None and n_arms:
        if under == n_arms:
            verdict = (f"<strong>All {n_arms} arm effect(s) measured are smaller than "
                       f"that</strong> — the largest is only <code>{biggest:.4f}</code>.")
        else:
            verdict = (f"<strong>{under} of the {n_arms} arm effect(s) measured are "
                       f"smaller than that</strong>; only "
                       f"<em>{_esc(reach.get('max_abs_delta_arm', ''))}</em>, at "
                       f"<code>{biggest:.4f}</code>, exceeds it.")
        parts.append(
            "<strong>Which video, not which arm, is the bigger term.</strong> The "
            f"baseline arm (<em>{_esc(reach.get('baseline_arm', ''))}</em>) ranges "
            f"<code>{spread:.4f}</code> PCK across the "
            f"{int(reach.get('baseline_bundles') or 0)} Bundle(s) it ran, on its own. "
            + verdict + " The paired deltas below stay valid, because each compares two "
            "arms on one Bundle and the Bundle cancels; what does not follow is that an "
            "arm ranking measured on these videos would hold on others.")
    single = int(reach.get("single_bundle_arms") or 0)
    if single:
        parts.append(
            f"{single} arm(s) rest on a <strong>single shared Bundle</strong>. A "
            "one-Bundle delta is reported rather than withheld — refusing to report it "
            "teaches nothing — but it is an anecdote with a decimal point, not a result "
            "that generalises.")
    uncomparable = reach.get("uncomparable_arms") or []
    if uncomparable:
        listed = ", ".join(
            f"<code>{_esc(str(a['config_hash'])[:8])}</code> ({_esc(a['arm'])}, "
            f"{int(a['bundles'])} bundle(s))" for a in uncomparable)
        parts.append(
            f"{len(uncomparable)} arm(s) share <strong>no Bundle</strong> with the "
            f"baseline and therefore produce no delta at all: {listed}. That is a gap in "
            "the sweep, not a null result — those factors are <em>unmeasured</em> here, "
            "and the difference between the two matters.")
    if not parts:
        return ""
    return "".join(f"<p class='warn'>{p}</p>" for p in parts)


def _arm_repeat_flag_html(flags: Any) -> str:
    """Groups whose repeats are not repeats — a check only a zero floor makes possible."""

    if not flags:
        return ("<p class='muted'>No arm/Bundle group disagrees with itself. On a "
                "bit-deterministic detector that is the expected result, and any other "
                "would mean the runs are not the same measurement.</p>")
    rows = "".join(
        "<tr>"
        f"<td><code>{_esc(str(f['config_hash'])[:8])}</code></td>"
        f"<td>{_esc(f['route_folder'])}</td><td>{_esc(f['video_key'])}</td>"
        f"<td>{int(f['runs'])}</td><td>{_sigfmt(f['pck_range'])}</td>"
        f"<td>{_esc(f['cause'])}</td></tr>"
        for f in flags)
    return (
        "<p class='warn'>These arm/Bundle groups hold several runs that <em>disagree</em>. "
        "The harness detector is bit-deterministic, so a nonzero range between runs of one "
        "arm on one Bundle is not scatter — it is evidence that the runs are not the same "
        "measurement, and averaging them would manufacture a variance floor out of a "
        "bookkeeping collision.</p>"
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>route</th><th>bundle</th>"
        "<th>runs</th><th>PCK range</th><th>what that means</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>")


def _arm_frame_set_html(summary: Any) -> str:
    """Runs whose stamped frame count is not the one their Bundle prescribes (issue #178).

    The generalising half of the repeat check above. That one needs a sampled run and a
    full-grid run to *collide* on one (arm, Bundle) before it can fire; this one compares
    each run against its Bundle's ``12·√n`` and therefore catches a whole sweep of runs that
    agree with each other and with nothing else in the corpus.
    """

    if not isinstance(summary, dict) or not summary.get("runs"):
        return ("<p class='muted'>No harness runs in scope to check a frame set on.</p>")

    basis = (f"<p class='sub'>Checked against <code>{int(summary['coefficient'])}·√n</code>"
             f" of each Bundle's truth grid, taken from {_esc(summary['coefficient_source'])}"
             f". {int(summary['runs'])} run(s) in scope: "
             f"{int(summary['matches'])} on the rule, {int(summary['flagged'])} off it, "
             f"{int(summary['unstamped'])} with no <code>frameCount</code> stamped, "
             f"{int(summary['no_truth_grid'])} on a Bundle with no readable truth grid. "
             "The last two are <em>unknown</em>, not mismatched — the stamps cannot say, "
             "which is a different statement from saying the run is wrong.</p>")

    if not summary.get("flagged"):
        return (basis + "<p class='muted'>Every stamped run scored exactly the frame set "
                "its Bundle prescribes. That is what makes the arm deltas above readable: "
                "the arm identity does not name the frame set, so two arms are only "
                "comparable while both took the sample the Bundle determines.</p>")

    arms = "".join(
        "<tr>"
        f"<td><code>{_esc(str(a['config_hash'])[:8])}</code></td>"
        f"<td>{_esc(a['arm'])}</td>"
        f"<td>{int(a['flagged_runs'])}/{int(a['runs'])}</td>"
        f"<td>{'every run in this arm' if a['whole_arm'] else 'some runs in this arm'}</td>"
        "</tr>"
        for a in summary.get("arms_affected") or ())
    runs = "".join(
        "<tr>"
        f"<td><code>{_esc(str(r['config_hash'])[:8])}</code></td>"
        f"<td>{_esc(r['route_folder'])}</td><td>{_esc(r['video_key'])}</td>"
        f"<td><code>{_esc(r['run_ts'])}</code></td>"
        f"<td>{_fmt_int(r['frame_count'])}</td><td>{_fmt_int(r['expected_frames'])}</td>"
        f"<td>{_esc(r['status'])}</td><td>{_esc(r['detail'])}</td></tr>"
        for r in summary.get("flagged_runs") or ())
    return (
        basis
        + "<p class='warn'>These runs scored a frame set their Bundle does not prescribe. "
        "The arm identity deliberately does not name the frame set — that is what makes "
        "sampling error common-mode across arms and cancel in a delta — so an off-rule run "
        "carries a stamp indistinguishable from an on-rule one. Any delta computed across "
        "the mismatch is partly a frame-set artifact rather than a condition effect. An arm "
        "whose runs are <em>all</em> off the rule is the dangerous shape: internally "
        "consistent, mutually comparable, and comparable to nothing else.</p>"
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>factors</th>"
        "<th>off-rule runs</th><th>reach</th></tr></thead>"
        f"<tbody>{arms}</tbody></table></div>"
        "<div class='tablewrap'><table><thead><tr><th>arm</th><th>route</th>"
        "<th>bundle</th><th>run</th><th>scored</th><th>prescribed</th><th>status</th>"
        "<th>what that means</th></tr></thead>"
        f"<tbody>{runs}</tbody></table></div>")


def _arm_video_stats_html(df: Any) -> str:
    """The condition Predictors behind each Bundle in the sweep."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "<p class='muted'>No Bundles in the sweep to join conditions to.</p>"
    cols = [c for c in df.columns if c not in {"route_folder", "video_key", "arms"}]
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        f"<tr><td>{_esc(r['route_folder'])}</td><td>{_esc(r['video_key'])}</td>"
        f"<td>{int(r['arms'])}</td>"
        + "".join(f"<td>{_fmt(r[c], 3) if isinstance(r[c], float) else _esc(r[c])}</td>"
                  for c in cols)
        + "</tr>"
        for _, r in df.iterrows())
    return (
        "<p class='sub'>Without these, an arm result is a property of the specific videos "
        "in the sweep. With them a finding can be stated as a <em>condition</em> — "
        "\"contrast preprocessing helps on low-contrast walls\" generalises beyond the "
        "sample; \"contrast preprocessing helps on planet-x\" does not. "
        "<code>luma_mean</code> / <code>rms_contrast</code> / "
        "<code>sharpness_mean</code> / <code>frame_diff_mean</code> are phase-1 source "
        "stats stamped at import and never stale; the <code>wall_*</code> / "
        "<code>climber_wall_*</code> / <code>shadow_*</code> columns are phase-2 region "
        "stats and go stale exactly as Ground Truth does when recalibration mints a new "
        "<code>setupHash</code>, which <code>vs_stale</code> reports. At this sweep size "
        "the join is descriptive: it says what conditions the arms were tested under, not "
        "yet which condition explains a delta.</p>"
        "<div class='tablewrap'><table><thead><tr><th>route</th><th>bundle</th>"
        f"<th>arms</th>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def _experiment_arms_html(ctx: dict[str, Any]) -> str:
    """The whole arm-comparison section (issues #164 and #176).

    The Cycle leads. Which rule the comparison was read under decides what every table
    below it means, so it is declared before the first number rather than footnoted after
    the last — and on a Cycle that did not certify, the comparison tables are not rendered
    at all.
    """

    arm_count = int(ctx.get("experiment_arm_count") or 0)
    summary = ctx.get("cycle_summary") or {}
    posture = summary.get("posture") or cycles.POSTURE_NONE
    scoped_runs = int(ctx.get("arm_scope_run_count") or 0)
    total_runs = int(ctx.get("experiment_run_count") or 0)
    scoped_arms = int(ctx.get("arm_scope_arm_count") or 0)
    swept = int(ctx.get("arm_swept_bundles") or 0)
    pooled = int(ctx.get("arm_pooled_bundles") or 0)
    # Both numbers whenever they differ, because "6 arms" over a table showing 5 is the
    # kind of quiet disagreement this whole issue is about.
    run_tile = (str(total_runs) if scoped_runs == total_runs
                else f"{scoped_runs}/{total_runs}")
    arm_tile = (str(arm_count) if scoped_arms == arm_count
                else f"{scoped_arms}/{arm_count}")
    tiles = _stat_tiles([
        (str(posture), "cycle posture"),
        (arm_tile, "experiment arms [in cycle / on disk]"),
        (run_tile, "harness runs [in cycle / on disk]"),
        (str(ctx.get("arm_bundle_count", 0)), "arm × bundle results"),
        (str(ctx.get("arm_pooled_bundle_count", 0)), "pooled after the cycle rule"),
        (f"{floors.SAMPLING_ERROR.p90:.4f}", "sampling error p90 [harness]"),
        (str(len(ctx.get("arm_repeat_flags") or [])), "non-repeat collisions"),
        (str(int((ctx.get("arm_frame_set_summary") or {}).get("flagged") or 0)),
         "runs off the 12·√n rule"),
    ])
    if not arm_count:
        return (tiles + _cycle_rule_html(summary, swept, pooled)
                + "<p class='muted'>No harness-produced runs carry an experiment "
                "stamp on this corpus, so there is nothing to compare. Every section "
                "elsewhere in this report is scanner-origin.</p>")

    head = (tiles
            + _cycle_rule_html(summary, swept, pooled)
            + _measurement_basis_html(ctx.get("measurement_basis_arms"))
            + _cycle_verdict_html(summary, ctx.get("cycle_bundles"))
            + _cycle_window_html(ctx.get("arm_runs_outside_cycle"), summary)
            + _ARM_PCK_NOTE
            + _ARM_UNCERTAINTY_NOTE
            + "<h3>Origin populations (never blended)</h3>"
            + "<p class='sub'>A harness run and a scanner run are two producers, not two "
            "generations of one evidence stream. Whether a browser-WASM run and a Python "
            "run agree is the open parity question (#162); pooling them before it is "
            "answered would assume the answer. Every pooled section elsewhere in this "
            "report draws on the scanner population only.</p>"
            + _arm_origin_table(ctx.get("origin_populations")))

    # A Cycle that failed or refused publishes no comparison at all — not a captioned one.
    # The runs are still recorded, because an expensive sweep that produced no publishable
    # result is a finding in its own right and deleting it from the report loses it.
    if posture == cycles.POSTURE_UNCERTIFIED:
        return (head
                + "<h3>Runs collected under this Cycle (not a comparison)</h3>"
                + _arm_evidence_table(ctx.get("arm_bundles"))
                + "<h3>Frame-set integrity</h3>"
                + _arm_frame_set_html(ctx.get("arm_frame_set_summary"))
                + "<h3>Repeat integrity</h3>"
                + _arm_repeat_flag_html(ctx.get("arm_repeat_flags"))
                + "<h3>Video Stats conditions for the swept Bundles</h3>"
                + _arm_video_stats_html(ctx.get("arm_video_stats")))

    return (
        head
        + "<h3>Arms</h3>" + _arm_overview_table(ctx.get("arm_overview"), summary)
        + "<h3>Per-Bundle results (Bundle × arm)</h3>"
        + _arm_matrix_table(ctx.get("arm_bundles"), ctx.get("arm_reach"))
        + "<h3>Arm versus baseline</h3>"
        + _arm_reach_html(ctx.get("arm_reach"))
        + _arm_delta_table(ctx.get("arm_delta_summary"), ctx.get("arm_deltas"), summary)
        + "<h3>Frame-set integrity</h3>"
        + _arm_frame_set_html(ctx.get("arm_frame_set_summary"))
        + "<h3>Repeat integrity</h3>"
        + _arm_repeat_flag_html(ctx.get("arm_repeat_flags"))
        + "<h3>Funnel-derived metrics and their floors</h3>"
        + _arm_floor_table(ctx.get("arm_bundles"))
        + "<h3>Video Stats conditions for the swept Bundles</h3>"
        + _arm_video_stats_html(ctx.get("arm_video_stats")))


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
/* A basis problem (#131): a pool blending schema versions, or a mid-cycle bump. Reads
   like .sub but cannot be skimmed past, because the numbers beside it are not comparable. */
.warn{color:var(--ink2);margin:0 0 18px;background:color-mix(in srgb,#e34948 10%,transparent);border-left:3px solid #e34948;border-radius:4px;padding:8px 12px}
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
        _basis_banner_html(ctx),
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
    # Issue #131: the basis line sits immediately under the evidence line on every pooled
    # section, so schema + build set travel with the number the same way generation does.
    trusted_basis = _measurement_basis_html(ctx.get("measurement_basis_trusted"))
    frames_basis = _measurement_basis_html(ctx.get("measurement_basis_frames"))

    # Issue #132: every truth-fit section says out loud that it quarantines, and how much
    # it is quarantining. These metrics are measured *against* Ground Truth, so a run whose
    # truth does not fit yields a meaningless number rather than a bad one — the gate stays.
    # Stated per section rather than once at the top, because the failure-mode sections
    # below deliberately do the opposite and the reader must not carry one rule into both.
    n_quarantined = int(ctx.get("quarantined_count") or 0)
    n_trusted = int(ctx.get("eval_count") or 0)
    n_loose = int(ctx.get("loose_count") or 0)
    quarantine_note = (
        "<p class='sub'><strong>Truth-fit metric — quarantined pool.</strong> Measured "
        f"over the {n_trusted} trusted record(s) only: the {n_quarantined} record(s) that "
        f"failed the #15 conformance gate and the {n_loose} #44 loose pairing(s) are held "
        "out, because these numbers are scored <em>against</em> Ground Truth and a truth "
        "fit that misses identity makes them meaningless rather than merely bad. This is "
        "the opposite of the failure-mode sections (attempt funnel, miss causes, "
        "rejection, crop), which pool every run and carry conformance as a covariate "
        "(#132). Quarantined records are listed by cause under Shame lists.</p>")

    parts += [
        # Placed ahead of the pooled accounting on purpose (#164): a reader has to know
        # this report holds two populations before reading a section that pools one of
        # them, or "the corpus" silently means whichever they read first.
        "<h2>Experiment arms (harness-produced, arm versus arm)</h2>",
        "<p class='sub'>Detection as an experiment rather than an observation (PRD #156): "
        "one factor varies, everything else is held fixed, and the configuration that "
        "produced each Run is stamped in the Run. Arms group by <code>configHash</code>, "
        "which covers every factor that can move the output — mode, each preprocessing "
        "step and its parameters, crop policy, crop trajectory, model weights, module "
        "version — so two Runs differing in anything cannot share a stamp. Results are "
        "per-Bundle first and pooled only afterwards, because the corpus this PRD exists "
        "to escape is exactly what pooled-first reporting produces.</p>",
        _experiment_arms_html(ctx),

        "<h2>Evaluation trend accounting</h2>",
        "<p class='sub'>Two-tier accounting from committed evaluation records. "
        "Every value is explicitly tagged as agreement or accuracy — and the accuracy "
        "tier is permanently empty, which each section states rather than leaving to "
        "inference (ADR 0010, #133). Records superseded "
        "by a newer evidence generation for the same video+truth pairing are dropped "
        "before any of this (#89) and listed in the shame lists below.</p>",
        # The #15 gate has two roles and they point opposite ways (#132). Say which is
        # which here, once, so no section below has to be read as if it were the other.
        "<p class='sub'><strong>How the #15 conformance gate is applied.</strong> "
        "<em>Truth-fit</em> metrics — accuracy, agreement, PCK, normDist, and everything "
        "derived from them (version regression, joint ranking, condition bands, "
        "cross-video splits) — <strong>quarantine</strong> non-conforming records: they "
        "are scored against Ground Truth, so a truth fit that misses identity makes them "
        "meaningless. <em>Failure-mode</em> metrics — the attempt funnel, miss causes, "
        "rejection correctness and crop placement — <strong>pool every run</strong> and "
        "report conformance as a covariate instead, because gating them would select on "
        "the very failure being measured. The tile counts below are the trusted "
        "(quarantined) pool; each failure-mode section carries its own conformance "
        "breakout.</p>",
        trusted_evidence,
        trusted_basis,
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
        _accuracy_tier_html(ctx),

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
        frames_basis,
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
        _measurement_basis_html(ctx.get("measurement_basis_funnel")),
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
        frames_basis,
        _detection_error_attempt_html(ctx),

        "<h2>Scanner version regression (build identity run-over-run)</h2>",
        "<p class='sub'>Evaluation records grouped by <strong>build identity</strong> — "
        "the pair <code>(appVersion, detectorCodeHash)</code> from the pose diagnostics, "
        "not the commit stamp alone (#130). <code>appVersion</code> is resolved once at "
        "dev-server start, so a hot reload moves the code without moving the stamp; "
        "<code>detectorCodeHash</code> is derived per run from the detector source that "
        "actually executed. A build that hot-reloaded mid-batch therefore splits into "
        "its real behavioural groups instead of averaging them, and two commits sharing "
        "one hash — a commit that did not touch detection — stay pooled instead of "
        "reading as a version boundary. Runs with no hash group on the appVersion alone "
        "and are labelled plainly; the field is fail-open by contract. Builds are "
        "ordered by first-seen "
        "both versions evaluated <em>under the same truth revision</em> — a truth "
        "change never masquerades as a scanner change. Deltas are coloured only "
        "when the bootstrap 95% CI excludes zero (green = improved, red = "
        "regressed); ΔPCK &gt; 0 and Δmedian &lt; 0 are improvements. Superseded legacy "
        "records are already gone (#89), so a change of evidence generation can no "
        "longer masquerade as a scanner change either.</p>",
        quarantine_note,
        trusted_evidence,
        _version_overview_table(ctx.get("version_overview", pd.DataFrame())),
        _version_delta_table(ctx.get("version_deltas", pd.DataFrame())),
        _accuracy_tier_html(ctx),
        "<h3>Build-identity conflicts</h3>",
        "<p class='sub'>One <code>appVersion</code> covering runs that executed "
        "different detector code — the signature of a hot reload that left the stamp "
        "frozen, which is how the 07-25/26 batch came to be stamped "
        "<code>c305954</code> while behaviourally running a later build. Derived over "
        "<em>every</em> pose run on disk, not just the ones an evaluation record "
        "scored: a hot reload during an unscored batch contaminates just as much, and "
        "is most worth knowing about before the scoring pass rather than after.</p>",
        _build_conflict_table(ctx.get("build_conflicts", pd.DataFrame())),
        "<h3>Version-tracking flags</h3>",
        _shame_list_html(ctx.get("version_flags", []),
                         "No mixed-truth or unversioned records."),

        "<h2>Per-joint failure ranking (frame/joint unit)</h2>",
        "<p class='sub'>Joint ranking uses frame/joint evidence with bootstrap "
        "95% CIs (no per-video correlation coefficients).</p>",
        quarantine_note,
        trusted_evidence,
        _joint_ranking_table(ctx.get("joint_rank", pd.DataFrame())),
        _accuracy_tier_html(ctx),

        "<h2>Within-video frame-level conditions vs error</h2>",
        "<p class='sub'>Frame/joint rows are grouped into quantile bands by condition; "
        "the table reports each band's pooled failure rate by tier. The rate pools "
        "frames, but the interval does <em>not</em> treat them as independent — frames "
        "within a run are correlated, so the CI is a cluster bootstrap over the band's "
        "runs (#70) and the per-run median/p90 sits beside it. Read a band as "
        "well-evidenced only when its <code>runs</code> count is large; a wide interval "
        "over many frames means few runs, not noisy frames.</p>",
        quarantine_note,
        trusted_evidence,
        _condition_table(ctx.get("condition_bands", pd.DataFrame())),
        _accuracy_tier_html(ctx),

        "<h2>Cross-video descriptive splits</h2>",
        f"<p class='sub'>{_esc(ctx.get('confound_caveat', ''))}</p>",
        quarantine_note,
        trusted_evidence,
        _cross_video_split_table(ctx.get("cross_video_splits", pd.DataFrame())),
        _accuracy_tier_html(ctx),

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
        "carry no annotation and default to <code>trajectory-divergence</code> (their "
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
