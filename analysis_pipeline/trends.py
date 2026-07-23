"""Evaluation-record trend summaries for the analysis report.

The analysis command reads committed evaluation records under each bundle's
``evaluations/`` folder and derives trend sections for issue #9:

- per-joint failure ranking with bootstrap CIs (frame/joint unit),
- within-video condition trends (size, speed, edge proximity) vs joint error,
- cross-video descriptive splits (resolution, panning, source type) with CIs,
- coverage/shame accounting (truthless bundles, stale setup runs),
- scanner appVersion run-over-run regression tracking (issue #10): consecutive
  versions delta'd per joint over a truth-hash-matched video pool.

This module never writes evaluation records and never calls the evaluate
subcommand; it only consumes existing artifacts in the bundle tree.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .discovery import _iter_video_dirs, _load_json, _pair_stems, _unwrap
from .evaluate import (
    COCO_CORE_JOINTS,
    _dist,
    _iter_pose_runs,
    _nearest_within,
    _pose_frame_joints,
    _scanner_frame_interval,
    load_truth,
    record_conforms,
    record_trusted,
    torso_length,
)

N_BOOT = 300
BOOT_SEED = 42


@dataclass
class EvalRecord:
    path: Path
    route_folder: str
    video_key: str
    run_ts: str
    truth_hash: str
    data: dict[str, Any]


def _pct_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not samples:
        return (math.nan, math.nan)
    s = sorted(samples)
    lo_i = max(0, int((alpha / 2) * (len(s) - 1)))
    hi_i = min(len(s) - 1, int((1 - alpha / 2) * (len(s) - 1)))
    return (s[lo_i], s[hi_i])


def _bootstrap_rate(values: list[int], n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    if not values:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(values)
    mean = sum(values) / n
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    lo, hi = _pct_ci(draws)
    return (mean, lo, hi)


def _iter_eval_records(analysis_root: Path) -> list[EvalRecord]:
    latest_by_run: dict[tuple[str, str, str], EvalRecord] = {}
    for video_dir in _iter_video_dirs(analysis_root):
        eval_dir = video_dir / "evaluations"
        if not eval_dir.is_dir():
            continue
        for path in sorted(eval_dir.glob("*.json")):
            try:
                data = _load_json(path)
            except Exception:
                continue
            route = str(data.get("routeFolder") or video_dir.parent.name)
            key = str(data.get("videoKey") or video_dir.name)
            run_ts = str(data.get("runTs") or "")
            if not run_ts:
                continue
            rec = EvalRecord(
                path=path,
                route_folder=route,
                video_key=key,
                run_ts=run_ts,
                truth_hash=str(data.get("truthHash") or ""),
                data=data,
            )
            dedup = (route, key, run_ts)
            cur = latest_by_run.get(dedup)
            if cur is None or path.stat().st_mtime > cur.path.stat().st_mtime:
                latest_by_run[dedup] = rec
    return sorted(latest_by_run.values(), key=lambda r: (r.route_folder, r.video_key, r.run_ts))


def _load_pose_runs(video_dir: Path) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    """Map ``run_ts -> (scanner appVersion, pose frames)`` for one bundle.

    The appVersion (a scanner commit hash) lives only in the pose envelope's
    diagnostics — evaluation records don't carry it — so version tracking
    resolves it from the detection files at trend time.
    """

    out: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    detections_dir = video_dir / "detections"
    if not detections_dir.is_dir():
        return out
    for stem, kinds in _pair_stems(detections_dir).items():
        if "pose" not in kinds:
            continue
        try:
            env = _load_json(kinds["pose"])
        except Exception:
            continue
        data = _unwrap(env)
        run_ts = str(env.get("run_ts", stem))
        app_version = str((data.get("diagnostics") or {}).get("appVersion") or "")
        out[run_ts] = (app_version, data.get("frames", []) or [])
    return out


def _bundle_meta(video_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(video_dir / "metadata.json")
    setup_path = video_dir / "setup.json"
    setup = _load_json(setup_path) if setup_path.exists() else {}
    return metadata, setup


def _resolution_bucket(metadata: dict[str, Any]) -> str:
    src = metadata.get("source_video", {}) if isinstance(metadata, dict) else {}
    h = src.get("height")
    if isinstance(h, (int, float)) and h > 0:
        return f"{int(h)}p"
    return "unknown"


def _frame_bbox_metrics(joints: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    if not joints:
        return None
    xs = [v[0] for v in joints.values()]
    ys = [v[1] for v in joints.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox_h = max(0.0, ymax - ymin)
    edge_dist = max(0.0, min(xmin, 1 - xmax, ymin, 1 - ymax))
    return bbox_h, edge_dist


def _build_frame_joint_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: dict[tuple[str, str], dict[str, tuple[str, list[dict[str, Any]]]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in recs:
        video_dir = analysis_root / rec.route_folder / rec.video_key
        if not video_dir.exists():
            continue
        truth = load_truth(video_dir)
        if truth is None:
            continue
        if rec.truth_hash and truth.truth_hash and rec.truth_hash != truth.truth_hash:
            # Keep trend analysis anchored to the same truth revision as the record.
            continue

        pose_runs = pose_cache.get((rec.route_folder, rec.video_key), {})
        app_version, pose_frames = pose_runs.get(rec.run_ts, ("", None))
        if pose_frames is None:
            continue

        metadata, setup = _bundle_meta(video_dir)
        source_type = str(metadata.get("source_type") or "unknown")
        resolution = _resolution_bucket(metadata)
        panning = setup.get("panning")
        panning_label = "panning" if panning is True else "static" if panning is False else "unknown"

        scanner_ts = sorted(float(f.get("timestamp", 0.0)) for f in pose_frames)
        if not scanner_ts:
            continue
        by_ts = {float(f.get("timestamp", 0.0)): f for f in pose_frames}
        interval = _scanner_frame_interval(scanner_ts)
        tol = interval / 2

        scored_frames: list[dict[str, Any]] = []
        for tf in truth.frames:
            if tf.excluded:
                continue  # known-bad seed or deprecated manual flag (ADR 0005)
            if not tf.present:
                continue
            torso = torso_length(tf.joints)
            if torso is None:
                continue
            bm = _frame_bbox_metrics(tf.joints)
            if bm is None:
                continue
            idx = _nearest_within(scanner_ts, tf.timestamp, tol)
            scanner = _pose_frame_joints(by_ts[scanner_ts[idx]]) if idx is not None else {}
            cx = sum(j[0] for j in tf.joints.values()) / len(tf.joints)
            cy = sum(j[1] for j in tf.joints.values()) / len(tf.joints)
            scored_frames.append({
                "timestamp": tf.timestamp,
                "verified": bool(tf.verified),
                "torso": torso,
                "bbox_h": bm[0],
                "edge_dist": bm[1],
                "cx": cx,
                "cy": cy,
                "truth_joints": tf.joints,
                "scanner": scanner,
            })

        scored_frames.sort(key=lambda r: r["timestamp"])
        prev_center: tuple[float, float] | None = None
        for sf in scored_frames:
            center = (sf["cx"], sf["cy"])
            speed = None
            if prev_center is not None:
                speed = _dist(center, prev_center)
            prev_center = center

            for joint in COCO_CORE_JOINTS:
                truth_pt = sf["truth_joints"].get(joint)
                if truth_pt is None:
                    continue
                pred = sf["scanner"].get(joint)
                norm_dist = None
                correct = 0
                if pred is not None:
                    norm_dist = _dist(pred, truth_pt) / sf["torso"]
                    correct = 1 if norm_dist <= 0.5 else 0
                base = {
                    "route_folder": rec.route_folder,
                    "video_key": rec.video_key,
                    "run_ts": rec.run_ts,
                    "app_version": app_version,
                    "truth_hash": truth.truth_hash,
                    "source_type": source_type,
                    "resolution": resolution,
                    "panning": panning_label,
                    "joint": joint,
                    "correct": correct,
                    "failure": 1 - correct,
                    "norm_dist": norm_dist,
                    "size_frac": sf["bbox_h"],
                    "speed": speed,
                    "edge_dist": sf["edge_dist"],
                }
                rows.append({**base, "tier": "agreement"})
                if sf["verified"]:
                    rows.append({**base, "tier": "accuracy"})

    return pd.DataFrame(rows)


def _joint_ranking(frame_joint_df: pd.DataFrame) -> pd.DataFrame:
    if frame_joint_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (tier, joint), g in frame_joint_df.groupby(["tier", "joint"]):
        vals = g["correct"].astype(int).tolist()
        boot = _bootstrap_rate(vals)
        if boot is None:
            continue
        rows.append({
            "tier": tier,
            "joint": joint,
            "n": len(vals),
            "pck": boot[0],
            "ci_low": boot[1],
            "ci_high": boot[2],
            "failure_rate": 1 - boot[0],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["tier", "pck", "joint"], ascending=[True, True, True])


def _condition_bands(frame_joint_df: pd.DataFrame, col: str, bins: int = 3) -> pd.DataFrame:
    if frame_joint_df.empty or col not in frame_joint_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tier, tg in frame_joint_df.groupby("tier"):
        d = tg[[col, "failure"]].dropna()
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            vals = bg["failure"].astype(int).tolist()
            boot = _bootstrap_rate(vals)
            if boot is None:
                continue
            rows.append({
                "tier": tier,
                "condition": col,
                "band": int(band) + 1,
                "n": len(vals),
                "failure_rate": boot[0],
                "ci_low": boot[1],
                "ci_high": boot[2],
                "band_min": float(bg[col].min()),
                "band_max": float(bg[col].max()),
            })
    return pd.DataFrame(rows)


def _cross_video_splits(recs: list[EvalRecord], analysis_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in recs:
        video_dir = analysis_root / rec.route_folder / rec.video_key
        if not video_dir.exists():
            continue
        metadata, setup = _bundle_meta(video_dir)
        row_base = {
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "source_type": str(metadata.get("source_type") or "unknown"),
            "resolution": _resolution_bucket(metadata),
            "panning": "panning" if setup.get("panning") is True else "static" if setup.get("panning") is False else "unknown",
        }
        for tier in ("agreement", "accuracy"):
            agg = ((rec.data.get(tier) or {}).get("aggregate") or {})
            pck = ((agg.get("pck") or {}).get("value"))
            cov = ((agg.get("coverage") or {}).get("rate"))
            if pck is None and cov is None:
                continue
            rows.append({
                **row_base,
                "tier": tier,
                "pck": pck,
                "coverage": cov,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    out_rows: list[dict[str, Any]] = []
    split_cols = ["resolution", "panning", "source_type"]
    for split_col in split_cols:
        for (tier, split_val), g in df.groupby(["tier", split_col]):
            for metric in ("pck", "coverage"):
                vals = [float(v) for v in g[metric].dropna().tolist()]
                if len(vals) < 2:
                    continue
                boot = _bootstrap_rate([1 if v >= 0.5 else 0 for v in vals])
                rng = random.Random(BOOT_SEED)
                draws = []
                n = len(vals)
                for _ in range(N_BOOT):
                    s = [vals[rng.randrange(n)] for _ in range(n)]
                    draws.append(sum(s) / n)
                lo, hi = _pct_ci(draws)
                out_rows.append({
                    "tier": tier,
                    "split": split_col,
                    "value": str(split_val),
                    "metric": metric,
                    "n_runs": n,
                    "mean": sum(vals) / n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "share_ge_0_5": boot[0] if boot is not None else None,
                })
    return pd.DataFrame(out_rows)


def _bootstrap_rate_delta(a: list[int], b: list[int],
                          n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Delta of means ``b - a`` for 0/1 outcomes with a percentile bootstrap CI.

    Resampling n iid 0/1 values and taking the mean is Binomial(n, p̂)/n, so the
    bootstrap draws come straight from the binomial (vectorised, deterministic).
    """

    rng = np.random.default_rng(BOOT_SEED)
    na, nb = len(a), len(b)
    pa, pb = sum(a) / na, sum(b) / nb
    draws = rng.binomial(nb, pb, n_boot) / nb - rng.binomial(na, pa, n_boot) / na
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return (pb - pa, float(lo), float(hi))


def _bootstrap_median_delta(a: list[float], b: list[float],
                            n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Delta of medians ``b - a`` with a percentile bootstrap CI."""

    rng = np.random.default_rng(BOOT_SEED)

    def boot_medians(vals: list[float]) -> np.ndarray:
        v = np.asarray(vals, dtype=float)
        n = len(v)
        out = np.empty(n_boot)
        batch = max(1, 20_000_000 // n)  # cap the index matrix at ~20M cells
        i = 0
        while i < n_boot:
            j = min(n_boot, i + batch)
            out[i:j] = np.median(v[rng.integers(0, n, size=(j - i, n))], axis=1)
            i = j
        return out

    draws = boot_medians(b) - boot_medians(a)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    delta = float(np.median(np.asarray(b)) - np.median(np.asarray(a)))
    return (delta, float(lo), float(hi))


_ALL_JOINTS = "(all joints)"


def _version_regression(
    recs: list[EvalRecord],
    frame_joint_df: pd.DataFrame,
    app_versions: dict[tuple[str, str, str], str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Group eval records by scanner appVersion and delta consecutive versions.

    Versions are ordered by first-seen run timestamp. For each consecutive pair
    the comparison pool is restricted to ``(video, truthHash)`` combos with
    records on *both* sides — a truth revision must never masquerade as a
    scanner change — and per-joint PCK / median-error deltas carry bootstrap
    CIs so noise at small n reads as noise. Videos where both versions ran but
    never under the same truth are flagged as mixed-truth and excluded.
    """

    flags: list[str] = []
    by_version: dict[str, list[EvalRecord]] = {}
    unknown = 0
    for rec in recs:
        av = app_versions.get((rec.route_folder, rec.video_key, rec.run_ts), "")
        if not av:
            unknown += 1
            continue
        by_version.setdefault(av, []).append(rec)
    if unknown:
        flags.append(
            f"{unknown} evaluation record(s) without a scanner appVersion "
            "(pose diagnostics missing) excluded from version tracking")

    ordered = sorted(by_version, key=lambda v: min(r.run_ts for r in by_version[v]))
    overview = pd.DataFrame([{
        "app_version": v,
        "first_run_ts": min(r.run_ts for r in by_version[v]),
        "last_run_ts": max(r.run_ts for r in by_version[v]),
        "n_records": len(by_version[v]),
        "n_videos": len({(r.route_folder, r.video_key) for r in by_version[v]}),
    } for v in ordered])

    if frame_joint_df.empty:
        pool_key = pd.Series(dtype=object)
    else:
        pool_key = pd.Series(
            list(zip(frame_joint_df["route_folder"], frame_joint_df["video_key"],
                     frame_joint_df["truth_hash"])),
            index=frame_joint_df.index)

    delta_rows: list[dict[str, Any]] = []
    for va, vb in zip(ordered, ordered[1:]):
        truths: list[dict[tuple[str, str], set[str]]] = []
        for version in (va, vb):
            per_video: dict[tuple[str, str], set[str]] = {}
            for r in by_version[version]:
                if r.truth_hash:
                    per_video.setdefault((r.route_folder, r.video_key), set()).add(r.truth_hash)
            truths.append(per_video)
        truths_a, truths_b = truths

        comparable: set[tuple[str, str, str]] = set()
        for vid in sorted(set(truths_a) & set(truths_b)):
            shared = truths_a[vid] & truths_b[vid]
            if shared:
                comparable.update((vid[0], vid[1], th) for th in shared)
            else:
                flags.append(
                    f"{va} → {vb}: {vid[0]}/{vid[1]} has runs from both versions "
                    "but never under the same truth revision — excluded (mixed truth)")
        if not comparable:
            flags.append(f"{va} → {vb}: no videos with both versions under a "
                         "shared truth revision — no deltas computed")
            continue
        if frame_joint_df.empty:
            continue

        n_videos = len({(r, k) for r, k, _ in comparable})
        in_pool = pool_key.isin(comparable)
        sub_a = frame_joint_df[(frame_joint_df["app_version"] == va) & in_pool]
        sub_b = frame_joint_df[(frame_joint_df["app_version"] == vb) & in_pool]
        for tier in ("agreement", "accuracy"):
            ta = sub_a[sub_a["tier"] == tier]
            tb = sub_b[sub_b["tier"] == tier]
            if ta.empty or tb.empty:
                continue
            for joint in [_ALL_JOINTS, *COCO_CORE_JOINTS]:
                ja = ta if joint == _ALL_JOINTS else ta[ta["joint"] == joint]
                jb = tb if joint == _ALL_JOINTS else tb[tb["joint"] == joint]
                a_correct = ja["correct"].astype(int).tolist()
                b_correct = jb["correct"].astype(int).tolist()
                if not a_correct or not b_correct:
                    continue
                pck_delta, pck_lo, pck_hi = _bootstrap_rate_delta(a_correct, b_correct)
                a_dist = ja["norm_dist"].dropna().tolist()
                b_dist = jb["norm_dist"].dropna().tolist()
                if a_dist and b_dist:
                    med_a = float(np.median(a_dist))
                    med_b = float(np.median(b_dist))
                    med_delta, med_lo, med_hi = _bootstrap_median_delta(a_dist, b_dist)
                else:
                    med_a = med_b = med_delta = med_lo = med_hi = math.nan
                delta_rows.append({
                    "from_version": va,
                    "to_version": vb,
                    "tier": tier,
                    "joint": joint,
                    "n_videos": n_videos,
                    "n_from": len(a_correct),
                    "n_to": len(b_correct),
                    "pck_from": sum(a_correct) / len(a_correct),
                    "pck_to": sum(b_correct) / len(b_correct),
                    "pck_delta": pck_delta,
                    "pck_ci_low": pck_lo,
                    "pck_ci_high": pck_hi,
                    "med_from": med_a,
                    "med_to": med_b,
                    "med_delta": med_delta,
                    "med_ci_low": med_lo,
                    "med_ci_high": med_hi,
                })

    return overview, pd.DataFrame(delta_rows), flags


def _shame_lists(analysis_root: Path) -> tuple[list[str], list[str]]:
    no_truth: list[str] = []
    stale_runs: list[str] = []
    for video_dir in _iter_video_dirs(analysis_root):
        metadata = _load_json(video_dir / "metadata.json")
        route = str(metadata.get("route_folder") or video_dir.parent.name)
        key = str(metadata.get("video_key") or video_dir.name)
        truth = load_truth(video_dir)
        if truth is None:
            no_truth.append(f"{route}/{key}")
            continue
        setup = _load_json(video_dir / "setup.json") if (video_dir / "setup.json").exists() else {}
        effective_setup_hash = truth.setup_hash or setup.get("setupHash", "")
        for run_ts, pose_setup_hash, _ in _iter_pose_runs(video_dir / "detections"):
            if pose_setup_hash != effective_setup_hash:
                stale_runs.append(
                    f"{route}/{key} {run_ts} (run {pose_setup_hash[:8] or '∅'} vs truth {effective_setup_hash[:8] or '∅'})"
                )
    return no_truth, stale_runs


# Worklist rows to surface in the report (the truth re-review queue is long; the
# CSV keeps the full list, the HTML shows the worst K).
LOW_CONF_WORKLIST_TOP_K = 40


def _visible_histogram(recs: list[EvalRecord]) -> list[int]:
    """Corpus visible-joint histogram, index ``i`` == matched-present frames whose
    truth carried ``i`` non-occluded core joints, pooled across records from each
    agreement tier's ``visibleJoints``. This is the measure-first fit input for
    ``evaluate.MIN_VISIBLE_JOINTS`` — the exact population the gate would act on
    (matched-present frames). Records predating schema v3 simply contribute nothing.
    """

    hist = [0] * (len(COCO_CORE_JOINTS) + 1)
    for rec in recs:
        vj = (rec.data.get("agreement") or {}).get("visibleJoints") or []
        if not isinstance(vj, list):
            continue  # pre-v3 records carried no positional histogram
        for i, v in enumerate(vj):
            if 0 <= i < len(hist):
                hist[i] += int(v)
    return hist


def _low_confidence_worklist(analysis_root: Path) -> pd.DataFrame:
    """Present truth frames ranked by fewest visible (non-occluded) core joints —
    the re-seed / re-review queue for low-confidence truth.

    Truth-side and per-bundle (independent of scanner runs), so a bundle's frames
    are listed once regardless of how many pose runs it has. Excluded frames
    (flagged-wrong / deprecated manual-absent) are skipped. A frame's occluded
    joints are the core joints ``load_truth`` dropped as occluded (ADR 0004),
    i.e. the ones ViTPose was not confident about.
    """

    rows: list[dict[str, Any]] = []
    for video_dir in _iter_video_dirs(analysis_root):
        truth = load_truth(video_dir)
        if truth is None:
            continue
        metadata = _load_json(video_dir / "metadata.json")
        route = str(metadata.get("route_folder") or video_dir.parent.name)
        key = str(metadata.get("video_key") or video_dir.name)
        for tf in truth.frames:
            if tf.excluded or not tf.present:
                continue
            occluded = [j for j in COCO_CORE_JOINTS if j not in tf.joints]
            rows.append({
                "route_folder": route,
                "video_key": key,
                "timestamp": tf.timestamp,
                "visible": len(tf.joints),
                "occluded_joints": ", ".join(occluded),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["visible", "route_folder", "video_key", "timestamp"],
        ascending=True,
    ).reset_index(drop=True)


# Per-frame quality worklist rows to surface in the HTML (the CSV keeps the full list).
FRAME_QUALITY_WORKLIST_TOP_K = 40

# The auto classes that count as a detection-quality *failure* (issue #44 deliverable 1);
# ``ok`` is the only non-failure. ``frozen-stale`` is a cross-cutting flag, not a class.
_FQ_FLAGGED = frozenset({"wrong-subject", "hallucination-fp", "flipped-rotated", "distorted"})

# Worst-first severity order for the worklist.
_FQ_SEVERITY = {"hallucination-fp": 0, "wrong-subject": 1, "flipped-rotated": 2,
                "distorted": 3, "ok": 4}

# A small set of numeric Video Stats conditions (issue #23) to band the per-frame class
# rate against (issue #44 deliverable 3). Nested key paths into ``video-stats.json``.
_VS_CONDITION_PATHS = {
    "wall_luma_mean": ("regionStats", "wall", "luma", "mean"),
    "wall_rms_contrast": ("regionStats", "wall", "rmsContrast"),
    "climber_wall_deltaE": ("regionStats", "climberWall", "deltaE"),
    "shadow_fraction": ("regionStats", "shadow", "fraction", "mean"),
}


def _video_stats_conditions(video_dir: Path) -> dict[str, float]:
    """Numeric Video Stats condition values for one bundle (issue #23 → #44), or {}."""

    path = video_dir / "video-stats.json"
    if not path.exists():
        return {}
    try:
        doc = _load_json(path)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for name, keys in _VS_CONDITION_PATHS.items():
        cur: Any = doc
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, (int, float)):
            out[name] = float(cur)
    return out


def _frame_quality_rows(analysis_root: Path, recs: list[EvalRecord]) -> pd.DataFrame:
    """Pool every record's ``frameQuality`` frames into one long table (issue #44).

    Pooled across **all** records — including #15-quarantined and #44-loose ones —
    because the frames most worth fixing live in exactly those bundles; the trusted
    metric pool (conforming, setupHash-matched only) is an independent pool. Each row
    carries the bundle's Video Stats conditions so the class rate can be banded against
    them. Records predating schema v6 carry no ``frameQuality`` and contribute nothing."""

    rows: list[dict[str, Any]] = []
    vs_cache: dict[tuple[str, str], dict[str, float]] = {}
    for rec in recs:
        fq = rec.data.get("frameQuality")
        if not isinstance(fq, dict):
            continue
        vid = (rec.route_folder, rec.video_key)
        if vid not in vs_cache:
            vs_cache[vid] = _video_stats_conditions(
                analysis_root / rec.route_folder / rec.video_key)
        conds = vs_cache[vid]
        loose = bool(rec.data.get("loosePaired"))
        conforming = record_conforms(rec.data)
        for e in fq.get("frames") or []:
            cls = str(e.get("class") or "ok")
            rows.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": e.get("t"),
                "class": cls,
                "auto_class": e.get("autoClass"),
                "failure_class": e.get("failureClass"),
                "distractor": e.get("distractor"),
                "annotation_setup_hash": e.get("annotationSetupHash"),
                "flagged": int(cls in _FQ_FLAGGED),
                "frozen_stale": int(bool(e.get("frozenStale"))),
                "centroid_dist": e.get("centroidDist"),
                "residual": e.get("residual"),
                "crop": e.get("crop"),
                "loose": loose,
                "conforming": conforming,
                **{f"vs_{k}": v for k, v in conds.items()},
            })
    return pd.DataFrame(rows)


def _frame_quality_classes(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Failure-class frequency table over the pooled per-frame quality rows."""

    if fq_df.empty:
        return pd.DataFrame()
    total = len(fq_df)
    rows: list[dict[str, Any]] = []
    for cls, g in fq_df.groupby("class"):
        rows.append({
            "class": str(cls),
            "n": int(len(g)),
            "share": len(g) / total,
            "frozen_stale": int(g["frozen_stale"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "class"], ascending=[False, True]).reset_index(drop=True)


def _frame_quality_distractors(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Human distractor frequency table over annotated per-frame quality rows."""

    if fq_df.empty or "distractor" not in fq_df.columns:
        return pd.DataFrame()
    sub = fq_df[fq_df["distractor"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    total = len(sub)
    rows: list[dict[str, Any]] = []
    for distractor, g in sub.groupby("distractor"):
        rows.append({
            "distractor": str(distractor),
            "n": int(len(g)),
            "share": len(g) / total,
            "frozen_stale": int(g["frozen_stale"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "distractor"], ascending=[False, True]).reset_index(drop=True)


def _frame_quality_worklist(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Flagged + frozen frames, worst-first — the per-frame re-review / crop queue."""

    if fq_df.empty:
        return pd.DataFrame()
    sub = fq_df[(fq_df["flagged"] == 1) | (fq_df["frozen_stale"] == 1)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_sev"] = sub["class"].map(lambda c: _FQ_SEVERITY.get(c, 4))
    sub = sub.sort_values(
        ["_sev", "centroid_dist"], ascending=[True, False], na_position="last")
    cols = ["route_folder", "video_key", "run_ts", "t", "class", "frozen_stale",
            "centroid_dist", "residual", "crop"]
    return sub[cols].reset_index(drop=True)


def _frame_quality_condition_bands(fq_df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """Flagged-frame rate per Video Stats condition tercile (issue #44 deliverable 3).

    Reuses the condition-band machinery (``pd.qcut`` + ``_bootstrap_rate``) from the
    within-video trends, but the outcome is the auto ``flagged`` flag and the predictor
    is a per-bundle Video Stats condition rather than a per-frame geometric one."""

    if fq_df.empty:
        return pd.DataFrame()
    cond_cols = [c for c in fq_df.columns if c.startswith("vs_")]
    rows: list[dict[str, Any]] = []
    for col in cond_cols:
        d = fq_df[[col, "flagged"]].dropna()
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            vals = bg["flagged"].astype(int).tolist()
            boot = _bootstrap_rate(vals)
            if boot is None:
                continue
            rows.append({
                "condition": col[len("vs_"):],
                "band": int(band) + 1,
                "n": len(vals),
                "flagged_rate": boot[0],
                "ci_low": boot[1],
                "ci_high": boot[2],
                "band_min": float(bg[col].min()),
                "band_max": float(bg[col].max()),
            })
    return pd.DataFrame(rows)


def _quarantined_rows(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Non-conforming records (issue #15 gate), flattened for the report's shame
    accounting: which bundle/run tripped the gate, why, and the offending fit."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if record_conforms(rec.data):
            continue
        conf = rec.data.get("conformance") or {}
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "reasons": ", ".join(conf.get("reasons") or []),
            "n": conf.get("n"),
            "slope_x": (conf.get("x") or {}).get("slope"),
            "r2_x": (conf.get("x") or {}).get("r2"),
            "slope_y": (conf.get("y") or {}).get("slope"),
            "r2_y": (conf.get("y") or {}).get("r2"),
        })
    return sorted(rows, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"]))


def _loose_rows(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Best-overlap loose pairings (issue #44), flattened for the report's shame
    accounting: which bundle/run fell back, and why. Held out of the trusted pool but
    kept for the per-frame quality worklist + crops."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if not rec.data.get("loosePaired"):
            continue
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "reason": str(rec.data.get("loosePairReason") or ""),
        })
    return sorted(rows, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"]))


def build_trend_context(analysis_root: Path) -> dict[str, Any]:
    all_recs = _iter_eval_records(analysis_root)
    # Issue #15 gate: quarantine non-conforming bundles (truth mis-tracking) from
    # every *pooled* derivation below. Issue #44: best-overlap loose pairings are
    # likewise held out of the trusted pool (their setupHash never matched the truth).
    # Both classes stay on disk and inspectable; only the aggregation drops them, and
    # the report accounts for each by name.
    quarantined = _quarantined_rows(all_recs)
    loose_records = _loose_rows(all_recs)
    recs = [r for r in all_recs if record_trusted(r.data)]
    pose_cache: dict[tuple[str, str], dict[str, tuple[str, list[dict[str, Any]]]]] = {}
    for rec in recs:
        vid = (rec.route_folder, rec.video_key)
        if vid not in pose_cache:
            pose_cache[vid] = _load_pose_runs(analysis_root / rec.route_folder / rec.video_key)
    app_versions = {
        (route, key, run_ts): av
        for (route, key), runs in pose_cache.items()
        for run_ts, (av, _) in runs.items()
    }
    frame_joint_df = _build_frame_joint_rows(analysis_root, recs, pose_cache)
    joint_rank = _joint_ranking(frame_joint_df)
    version_overview, version_deltas, version_flags = _version_regression(
        recs, frame_joint_df, app_versions)
    cond_df = pd.concat(
        [
            _condition_bands(frame_joint_df, "size_frac"),
            _condition_bands(frame_joint_df, "speed"),
            _condition_bands(frame_joint_df, "edge_dist"),
        ],
        ignore_index=True,
    ) if not frame_joint_df.empty else pd.DataFrame()
    split_df = _cross_video_splits(recs, analysis_root)
    no_truth, stale_runs = _shame_lists(analysis_root)
    visible_hist = _visible_histogram(recs)
    low_conf_worklist = _low_confidence_worklist(analysis_root)

    # Per-frame detection quality (issue #44): pooled across ALL records — quarantined
    # and loose included — because those bundles hold the frames most worth fixing. This
    # is an independent pool from the trusted metrics above (conforming-only).
    fq_df = _frame_quality_rows(analysis_root, all_recs)
    fq_classes = _frame_quality_classes(fq_df)
    fq_distractors = _frame_quality_distractors(fq_df)
    fq_worklist = _frame_quality_worklist(fq_df)
    fq_condition_bands = _frame_quality_condition_bands(fq_df)

    verified_total = 0
    verified_records = 0
    for rec in recs:
        counts = rec.data.get("counts") or {}
        vf = int(counts.get("truthFramesVerified") or 0)
        verified_total += vf
        if vf > 0:
            verified_records += 1

    return {
        "eval_records": recs,
        "eval_count": len(recs),
        "eval_count_total": len(all_recs),
        "quarantined_bundles": quarantined,
        "quarantined_count": len(quarantined),
        "loose_bundles": loose_records,
        "loose_count": len(loose_records),
        "frame_joint_df": frame_joint_df,
        "joint_rank": joint_rank,
        "condition_bands": cond_df,
        "cross_video_splits": split_df,
        "version_overview": version_overview,
        "version_deltas": version_deltas,
        "version_flags": version_flags,
        "truthless_bundles": no_truth,
        "stale_runs": stale_runs,
        "visible_histogram": visible_hist,
        "low_conf_worklist": low_conf_worklist,
        "frame_quality_classes": fq_classes,
        "frame_quality_distractors": fq_distractors,
        "frame_quality_worklist": fq_worklist,
        "frame_quality_condition_bands": fq_condition_bands,
        "frame_quality_detected": int(len(fq_df)),
        "frame_quality_flagged": int(fq_df["flagged"].sum()) if not fq_df.empty else 0,
        "frame_quality_frozen": int(fq_df["frozen_stale"].sum()) if not fq_df.empty else 0,
        "verified_frames_total": verified_total,
        "verified_records": verified_records,
        "confound_caveat": (
            "Cross-video splits are descriptive only: route and videographer are "
            "confounded with source/resolution/panning in this corpus."
        ),
    }


def write_trend_tables(out_dir: Path, ctx: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    quarantined = ctx.get("quarantined_bundles") or []
    quarantined_df = pd.DataFrame(quarantined) if quarantined else pd.DataFrame()
    tables = {
        "eval_joint_ranking.csv": ctx.get("joint_rank"),
        "eval_condition_bands.csv": ctx.get("condition_bands"),
        "eval_cross_video_splits.csv": ctx.get("cross_video_splits"),
        "eval_version_overview.csv": ctx.get("version_overview"),
        "eval_version_deltas.csv": ctx.get("version_deltas"),
        "eval_low_confidence_worklist.csv": ctx.get("low_conf_worklist"),
        "eval_quarantined_bundles.csv": quarantined_df,
        "eval_frame_quality_classes.csv": ctx.get("frame_quality_classes"),
        "eval_frame_quality_distractors.csv": ctx.get("frame_quality_distractors"),
        "eval_frame_quality_worklist.csv": ctx.get("frame_quality_worklist"),
        "eval_frame_quality_condition_bands.csv": ctx.get("frame_quality_condition_bands"),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            p = out_dir / name
            table.to_csv(p, index=False)
            outputs[name] = p
    return outputs
