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


def build_trend_context(analysis_root: Path) -> dict[str, Any]:
    recs = _iter_eval_records(analysis_root)
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
        "version_overview": version_overview,
        "version_deltas": version_deltas,
        "version_flags": version_flags,
        "truthless_bundles": no_truth,
        "stale_runs": stale_runs,
        "visible_histogram": visible_hist,
        "low_conf_worklist": low_conf_worklist,
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
        "eval_version_overview.csv": ctx.get("version_overview"),
        "eval_version_deltas.csv": ctx.get("version_deltas"),
        "eval_low_confidence_worklist.csv": ctx.get("low_conf_worklist"),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            p = out_dir / name
            table.to_csv(p, index=False)
            outputs[name] = p
    return outputs
