"""Evaluation-record trend summaries for the analysis report.

The analysis command reads committed evaluation records under each bundle's
``evaluations/`` folder and derives trend sections for issue #9:

- per-joint failure ranking with bootstrap CIs (frame/joint unit),
- within-video condition trends (size, speed, edge proximity) vs joint error,
- cross-video descriptive splits (resolution, panning, source type) with CIs,
- coverage/shame accounting (truthless bundles, stale setup runs).

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

import pandas as pd

from .discovery import _iter_video_dirs, _load_json
from .evaluate import (
    COCO_CORE_JOINTS,
    _dist,
    _iter_pose_runs,
    _nearest_within,
    _pose_frame_joints,
    _scanner_frame_interval,
    load_truth,
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


def _build_frame_joint_rows(analysis_root: Path, recs: list[EvalRecord]) -> pd.DataFrame:
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

        pose_runs = {run_ts: frames for run_ts, _, frames in _iter_pose_runs(video_dir / "detections")}
        pose_frames = pose_runs.get(rec.run_ts)
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


def build_trend_context(analysis_root: Path) -> dict[str, Any]:
    recs = _iter_eval_records(analysis_root)
    frame_joint_df = _build_frame_joint_rows(analysis_root, recs)
    joint_rank = _joint_ranking(frame_joint_df)
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
        "frame_joint_df": frame_joint_df,
        "joint_rank": joint_rank,
        "condition_bands": cond_df,
        "cross_video_splits": split_df,
        "truthless_bundles": no_truth,
        "stale_runs": stale_runs,
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
    tables = {
        "eval_joint_ranking.csv": ctx.get("joint_rank"),
        "eval_condition_bands.csv": ctx.get("condition_bands"),
        "eval_cross_video_splits.csv": ctx.get("cross_video_splits"),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            p = out_dir / name
            table.to_csv(p, index=False)
            outputs[name] = p
    return outputs
