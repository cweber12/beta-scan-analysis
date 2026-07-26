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

from .detector_attempts import parse_detector_attempts
from .discovery import _iter_video_dirs, _load_json, _pair_stems, _unwrap
from .detector_attempts import (
    DETECTOR_ATTEMPT_STATUS_ORDER,
    DETECTOR_ATTEMPT_STATUS_UNKNOWN,
    DETECTOR_ATTEMPT_STATUSES,
    _slug,
    condition_flags as _condition_flags,
    region_metric as _region_metric,
)
from .runs import _detector_attempt_summary
from .evaluate import (
    ABSENCE_CONFIRMED,
    RATE_MISMATCH_MIN_RATIO,
    ABSENCE_REASONS,
    ABSENCE_UNKNOWN,
    COCO_CORE_JOINTS,
    EVIDENCE_ATTEMPTS,
    EVIDENCE_GENERATIONS,
    MISS_CAUSES,
    NONCONFORMANCE_CAUSES,
    NONCONFORMANCE_SUSPECTED_MISTRACK,
    _dist,
    _iter_pose_runs,
    _nearest_within,
    _pose_frame_joints,
    _scanner_frame_interval,
    load_truth,
    record_conforms,
    record_evidence_generation,
    record_nonconformance_cause,
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


def _p90(values: pd.Series) -> float | None:
    return float(np.quantile(values.to_numpy(dtype=float), 0.9)) if len(values) else None


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


# The three columns that identify a Run. The Run is the unit of inference (CONTEXT.md),
# so anything that puts a CI on a per-frame outcome resamples these, not the frames.
_RUN_KEY_COLS = ("route_folder", "video_key", "run_ts")


def _cluster_bootstrap_rate(sums: list[float], counts: list[int],
                            n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    """Percentile bootstrap of a pooled rate that resamples **runs**, not frames (#70).

    ``sums[i]`` / ``counts[i]`` are run *i*'s outcome total and row count. Each draw
    takes ``len(counts)`` runs with replacement and recomputes the pooled rate over the
    drawn runs, so the interval's width tracks how many independent runs the estimate
    rests on rather than how many correlated frames they happened to contribute. The
    point estimate is untouched — it is still the pooled rate over every frame.
    """
    total = sum(counts)
    if not counts or total == 0:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(counts)
    pooled = sum(sums) / total
    draws: list[float] = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += sums[i]
            c += counts[i]
        if c:
            draws.append(s / c)
    if not draws:
        return None
    lo, hi = _pct_ci(draws)
    return (pooled, lo, hi)


def _run_unit_rate(df: pd.DataFrame, outcome: str) -> dict[str, Any] | None:
    """Pooled ``outcome`` rate with a run-unit CI and the per-run dispersion beside it.

    Frames inside a run are massively pseudo-replicated — a band can hold 100k frames
    drawn from a few dozen runs — so an iid bootstrap over those frames reports a CI far
    tighter than the design supports and makes marginal condition effects read as
    significant (#70). The rate stays pooled (that is what the corpus shows); the
    interval comes from the cluster bootstrap, and ``run_rate_median`` / ``run_rate_p90``
    expose the spread across runs that a single pooled number hides.

    Returns ``None`` when the frame carries no run identity — better to drop the band
    than to publish a frame-pooled interval that looks like a run-unit one.
    """
    key_cols = [c for c in _RUN_KEY_COLS if c in df.columns]
    if not key_cols or df.empty:
        return None
    grouped = df.groupby(key_cols, dropna=False)[outcome]
    sums = [float(v) for v in grouped.sum()]
    counts = [int(v) for v in grouped.size()]
    boot = _cluster_bootstrap_rate(sums, counts)
    if boot is None:
        return None
    run_rates = pd.Series([s / c for s, c in zip(sums, counts) if c], dtype=float)
    return {
        "n": int(sum(counts)),
        "n_runs": len(counts),
        "rate": boot[0],
        "ci_low": boot[1],
        "ci_high": boot[2],
        "run_rate_median": float(run_rates.median()),
        "run_rate_p90": _p90(run_rates),
    }


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


def _dedup_evidence_generations(
    recs: list[EvalRecord],
) -> tuple[list[EvalRecord], list[dict[str, Any]]]:
    """Keep one **evidence generation** per video+truth pairing (issue #89).

    A video re-scanned after the scanner started exporting ``detectorAttempts[]`` carries
    two records for the same ``(route, video, truthHash)`` pairing: the attempt-backed one
    and the legacy-frames one it superseded. Pooling both counts that pairing twice, and
    blends two generations of evidence into one number — the legacy record's frame-derived
    quality answers a question the attempt stream answers directly, and its appVersion
    differs, so a generation change would read as a scanner change.

    So: when a pairing has any attempt-backed record, only its attempt-backed records
    pool. Everything else in that pairing is *superseded* — returned as rows for the
    report's accounting, never deleted. Records stay on disk and readable exactly as
    written; only the aggregation drops them.

    A pairing with no attempt-backed record is untouched, so a legacy-only corpus
    aggregates exactly as it did before this gate existed. Truth revision is part of the
    pairing key on purpose: an attempt-backed run under a *different* truth supersedes
    nothing, because the two records were never measuring the same thing.
    """

    by_pairing: dict[tuple[str, str, str], list[EvalRecord]] = {}
    for rec in recs:
        by_pairing.setdefault((rec.route_folder, rec.video_key, rec.truth_hash), []).append(rec)

    kept: list[EvalRecord] = []
    superseded: list[dict[str, Any]] = []
    for group in by_pairing.values():
        attempt_backed = [
            r for r in group
            if record_evidence_generation(r.data) == EVIDENCE_ATTEMPTS
        ]
        if not attempt_backed:
            kept.extend(group)
            continue
        kept.extend(attempt_backed)
        superseded_by = ", ".join(sorted(r.run_ts for r in attempt_backed))
        for rec in group:
            if record_evidence_generation(rec.data) == EVIDENCE_ATTEMPTS:
                continue
            superseded.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "truth_hash": rec.truth_hash,
                "evidence_generation": record_evidence_generation(rec.data),
                "superseded_by": superseded_by,
            })

    return (
        sorted(kept, key=lambda r: (r.route_folder, r.video_key, r.run_ts)),
        sorted(superseded, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"])),
    )


def _evidence_generation_summary(recs: list[EvalRecord], pool: str) -> dict[str, Any]:
    """What evidence generation(s) one pooled set of records is made of (issue #89).

    Every pooled section reports this, so a mixed pool is never something a reader has to
    infer. Dedup removes the *superseded* mixture (same pairing, two generations); a pool
    can still legitimately span generations across different videos — a corpus mid-
    migration — and that is exactly the case worth naming rather than silently averaging.
    """

    counts = {g: 0 for g in EVIDENCE_GENERATIONS}
    for rec in recs:
        counts[record_evidence_generation(rec.data)] += 1
    present = [g for g in EVIDENCE_GENERATIONS if counts[g]]
    return {
        "pool": pool,
        "n_records": len(recs),
        "counts": counts,
        "generations": present,
        "mixed": len(present) > 1,
        "label": " + ".join(present) if present else "none",
    }


# One bundle's pose runs: ``run_ts -> (appVersion, pose frames, detector attempts|None)``.
PoseRun = tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]
# ...and the corpus-wide cache of them, keyed ``(route_folder, video_key)``. Four
# derivations below need the same pose files; one cache threaded through them all is what
# keeps a trend build from re-reading every detection file once per derivation.
PoseRunCache = dict[tuple[str, str], dict[str, PoseRun]]

# A record whose run has no pose file at all: no appVersion, no frames, and — the part
# that matters — *unknown* rather than empty detector attempts.
_NO_POSE_RUN: PoseRun = ("", [], None)


def _pose_run(
    analysis_root: Path,
    rec: EvalRecord,
    cache: PoseRunCache,
) -> PoseRun | None:
    """One record's pose run, loading (and caching) its bundle on first use.

    ``None`` means the bundle has no pose file for that ``run_ts`` — distinct from a run
    whose pose file holds zero frames, which callers must be able to tell apart."""

    vid = (rec.route_folder, rec.video_key)
    if vid not in cache:
        cache[vid] = _load_pose_runs(analysis_root / rec.route_folder / rec.video_key)
    return cache[vid].get(rec.run_ts)


def _load_pose_runs(video_dir: Path) -> dict[str, PoseRun]:
    """Map ``run_ts -> (scanner appVersion, pose frames)`` for one bundle.

    The appVersion (a scanner commit hash) lives only in the pose envelope's
    diagnostics — evaluation records don't carry it — so version tracking
    resolves it from the detection files at trend time.
    """

    out: dict[str, PoseRun] = {}
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
        attempts = parse_detector_attempts(data)
        out[run_ts] = (app_version, data.get("frames", []) or [], attempts)
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
    pose_cache: PoseRunCache,
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

        pose_run = _pose_run(analysis_root, rec, pose_cache)
        if pose_run is None:
            continue  # no pose file for this run_ts — nothing to score against truth
        app_version, pose_frames, _ = pose_run

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
    """Failure rate per quantile band of a geometric condition, CI'd at the run unit.

    A frame/joint row is not an independent observation — one run contributes thousands
    of them — so the band's interval comes from ``_run_unit_rate``'s cluster bootstrap
    and the per-run median/p90 travel with it (#70)."""

    if frame_joint_df.empty or col not in frame_joint_df.columns:
        return pd.DataFrame()
    key_cols = [c for c in _RUN_KEY_COLS if c in frame_joint_df.columns]
    rows: list[dict[str, Any]] = []
    for tier, tg in frame_joint_df.groupby("tier"):
        d = tg[[*key_cols, col, "failure"]].dropna(subset=[col, "failure"])
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            stats = _run_unit_rate(bg.assign(failure=bg["failure"].astype(int)), "failure")
            if stats is None:
                continue
            rows.append({
                "tier": tier,
                "condition": col,
                "band": int(band) + 1,
                "n": stats["n"],
                "n_runs": stats["n_runs"],
                "failure_rate": stats["rate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "run_rate_median": stats["run_rate_median"],
                "run_rate_p90": stats["run_rate_p90"],
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
        for run_ts, pose_setup_hash, _, _ in _iter_pose_runs(video_dir / "detections"):
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

_ATTEMPT_CONDITION_KEYS = {
    "mean": "luma_mean",
    "stdDev": "luma_stdDev",
    "sharpness": "sharpness",
}
_ATTEMPT_REGION_METRICS = ("area", "cx", "cy", "edge_distance")


def _attempts_by_timestamp(
    attempts: list[dict[str, Any]] | None,
) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for attempt in attempts or []:
        out[round(float(attempt.get("timestamp", 0.0)), 1)] = attempt
    return out


def _attempt_frame_context(
    attempt: dict[str, Any] | None,
    evidence: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "detector_attempt_evidence": evidence,
        "detector_attempt_status": None,
        "reacquire_attempted": None,
        "reacquire_succeeded": None,
        "reacquire_failed": None,
        "candidate_count": None,
        "rejected_candidate_count": None,
        "selection_method": None,
    }
    for prefix in ("initial_search_region", "detection_region"):
        for metric in _ATTEMPT_REGION_METRICS:
            out[f"{prefix}_{metric}"] = None
    for prefix in ("search", "reacquire"):
        for suffix in _ATTEMPT_CONDITION_KEYS.values():
            out[f"{prefix}_{suffix}"] = None

    if not isinstance(attempt, dict):
        return out

    attempted = bool(attempt.get("reacquireAttempted"))
    reacquired = bool(attempt.get("reacquired"))
    out.update({
        "detector_attempt_status": attempt.get("status"),
        "reacquire_attempted": attempted,
        "reacquire_succeeded": attempted and reacquired,
        "reacquire_failed": attempted and not reacquired,
        "candidate_count": attempt.get("candidateCount"),
        "rejected_candidate_count": attempt.get("rejectedCandidateCount"),
        "selection_method": attempt.get("selectionMethod"),
    })
    for prefix, source_key in (
        ("initial_search_region", "initialSearchRegion"),
        ("detection_region", "detectionRegion"),
    ):
        for metric in _ATTEMPT_REGION_METRICS:
            out[f"{prefix}_{metric}"] = _region_metric(attempt.get(source_key), metric)
    for prefix, source_key in (
        ("search", "searchConditions"),
        ("reacquire", "reacquireConditions"),
    ):
        conditions = attempt.get(source_key)
        for src_key, suffix in _ATTEMPT_CONDITION_KEYS.items():
            value = conditions.get(src_key) if isinstance(conditions, dict) else None
            out[f"{prefix}_{suffix}"] = float(value) if isinstance(value, (int, float)) else None
    for name, value in _condition_flags(attempt.get("searchConditions")).items():
        out[f"search_flag_{name}"] = value
    return out


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


def _frame_quality_rows(analysis_root: Path, recs: list[EvalRecord],
                        pose_cache: PoseRunCache | None = None) -> pd.DataFrame:
    """Pool every record's ``frameQuality`` frames into one long table (issue #44).

    Pooled across **all** records — including #15-quarantined and #44-loose ones —
    because the frames most worth fixing live in exactly those bundles; the trusted
    metric pool (conforming, setupHash-matched only) is an independent pool. Each row
    carries the bundle's Video Stats conditions so the class rate can be banded against
    them. Records predating schema v6 carry no ``frameQuality`` and contribute nothing."""

    rows: list[dict[str, Any]] = []
    vs_cache: dict[tuple[str, str], dict[str, float]] = {}
    pose_cache = {} if pose_cache is None else pose_cache
    for rec in recs:
        fq = rec.data.get("frameQuality")
        if not isinstance(fq, dict):
            continue
        vid = (rec.route_folder, rec.video_key)
        if vid not in vs_cache:
            vs_cache[vid] = _video_stats_conditions(
                analysis_root / rec.route_folder / rec.video_key)
        conds = vs_cache[vid]
        _, _, attempts = _pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN
        attempt_index = _attempts_by_timestamp(attempts)
        attempt_evidence = "unknown" if attempts is None else "attempts"
        loose = bool(rec.data.get("loosePaired"))
        conforming = record_conforms(rec.data)
        for e in fq.get("frames") or []:
            cls = str(e.get("class") or "ok")
            t = e.get("t")
            attempt = (
                attempt_index.get(round(float(t), 1))
                if isinstance(t, (int, float))
                else None
            )
            rows.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": t,
                "class": cls,
                "auto_class": e.get("autoClass"),
                "failure_class": e.get("failureClass"),
                # Tri-state on purpose (issue #69): True / False / None, where None is a
                # pre-schema-v12 frame that never recorded presence. Never coerce the
                # missing case to False — that would count unknown frames as real
                # false positives.
                "truth_present": e.get("truthPresent"),
                # Why an absent frame is absent (issue #101). ``None`` on a present
                # frame; a pre-v14 record carries no field and reads as ``unknown``
                # downstream — never as a confirmed absence.
                "absence_reason": e.get("absenceReason"),
                "source": e.get("source"),
                "distractor": e.get("distractor"),
                "annotation_setup_hash": e.get("annotationSetupHash"),
                "flagged": int(cls in _FQ_FLAGGED),
                "held_pose": int(bool(e.get("heldPose"))),
                "frozen_stale": int(bool(e.get("frozenStale"))),
                "centroid_dist": e.get("centroidDist"),
                "residual": e.get("residual"),
                # Rejection correctness (issue #85), read from the record rather than
                # re-derived. None on non-rejection frames and pre-v9 records.
                "rejection_verdict": e.get("rejectionVerdict"),
                "rejection_reason": e.get("rejectionReason"),
                "rejection_centroid_dist": e.get("rejectionCentroidDist"),
                "rejection_joint_agreement": e.get("rejectionJointAgreement"),
                "rejection_raw_class": e.get("rejectionRawClass"),
                "crop": e.get("crop"),
                "loose": loose,
                "conforming": conforming,
                **_attempt_frame_context(attempt, attempt_evidence),
                **{f"vs_{k}": v for k, v in conds.items()},
            })
    return pd.DataFrame(rows)


def _truth_presence_counts(g: pd.DataFrame) -> dict[str, Any]:
    """Split one class's pooled frames by truth presence (issue #69, #101).

    Counts plus the two shares *within the class*, taken over the frames whose presence
    is actually known: a pre-schema-v12 record carries no ``truthPresent``, and folding
    those into the denominator would report a split the records never measured. When
    nothing is known the shares are ``None``, not 0.0.

    From v14 an absence additionally has to be **confirmed** to count (issue #101). An
    absence that is out of scope, never sampled or a tracking loss is reported as
    ``truth_absent_unconfirmed`` and kept out of both the numerator and the
    denominator — those frames are the 44% of the pooled absent population that made
    the old truth-absent share unsafe to act on."""

    col = g["truth_present"] if "truth_present" in g.columns else pd.Series(dtype=object)
    reasons = (g["absence_reason"] if "absence_reason" in g.columns
               else pd.Series([None] * len(g), index=g.index, dtype=object))
    known = col.notna()
    vals = col[known].astype(bool)
    present = int(vals.sum())
    absent_idx = vals[~vals].index
    confirmed_mask = reasons.reindex(absent_idx) == ABSENCE_CONFIRMED
    absent = int(confirmed_mask.sum())
    unconfirmed = int(len(absent_idx) - absent)
    n_known = present + absent
    return {
        "truth_present": present,
        "truth_absent": absent,
        "truth_absent_unconfirmed": unconfirmed,
        "truth_unknown": int(len(g) - present - len(absent_idx)),
        "truth_present_share": present / n_known if n_known else None,
        "truth_absent_share": absent / n_known if n_known else None,
    }


def _absence_reason_counts(fq_df: pd.DataFrame) -> pd.DataFrame:
    """How the pooled truth-*absent* frames split by reason (issue #101).

    The table that says how much of "the Climber was not there" actually means that.
    Every reason is keyed even at zero, so "no scaffold gaps this batch" is a readable
    result rather than something inferred from an absent row."""

    if fq_df.empty or "truth_present" not in fq_df.columns:
        return pd.DataFrame()
    absent = fq_df[fq_df["truth_present"] == False]  # noqa: E712 — object column
    if absent.empty:
        return pd.DataFrame()
    reasons = (absent["absence_reason"] if "absence_reason" in absent.columns
               else pd.Series([None] * len(absent), index=absent.index, dtype=object))
    reasons = reasons.fillna(ABSENCE_UNKNOWN)
    total = len(absent)
    rows = [{
        "reason": reason,
        "n": int((reasons == reason).sum()),
        "share": float((reasons == reason).sum()) / total,
        "counts_as_absent": reason == ABSENCE_CONFIRMED,
    } for reason in ABSENCE_REASONS]
    return pd.DataFrame(rows)


def _frame_quality_classes(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Failure-class frequency table over the pooled per-frame quality rows.

    Each class is additionally split by truth presence (issue #69). The split matters
    most for ``hallucination-fp``, where truth-absent is a real false positive
    (presence gating) and truth-present is a tracking miss (tracking robustness), but
    it is carried for every class because the axis is per-frame, not per-class."""

    if fq_df.empty:
        return pd.DataFrame()
    total = len(fq_df)
    rows: list[dict[str, Any]] = []
    for cls, g in fq_df.groupby("class"):
        rows.append({
            "class": str(cls),
            "n": int(len(g)),
            "share": len(g) / total,
            "held_pose": int(g["held_pose"].sum()),
            "frozen_stale": int(g["frozen_stale"].sum()),
            **_truth_presence_counts(g),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "class"], ascending=[False, True]).reset_index(drop=True)


def _hallucination_split_totals(fq_df: pd.DataFrame) -> dict[str, Any]:
    """Pooled truth-presence split of the ``hallucination-fp`` frames (issue #69).

    The headline the class table's extra columns are there to support: of every frame
    pooled as a hallucination, how many were emitted where no Climber was (a real false
    positive, fixed by presence gating) versus where one was (a tracking miss)."""

    empty = fq_df.empty or "class" not in fq_df.columns
    sub = pd.DataFrame() if empty else fq_df[fq_df["class"] == "hallucination-fp"]
    if sub.empty:
        return {"total": 0, "truth_present": 0, "truth_absent": 0,
                "truth_absent_unconfirmed": 0, "truth_unknown": 0,
                "truth_present_share": None, "truth_absent_share": None}
    return {"total": int(len(sub)), **_truth_presence_counts(sub)}


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
            "held_pose": int(g["held_pose"].sum()),
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
    cols = ["route_folder", "video_key", "run_ts", "t", "class", "truth_present",
            "source", "held_pose", "frozen_stale", "centroid_dist", "residual",
            "rejection_verdict", "rejection_centroid_dist", "rejection_joint_agreement",
            "detector_attempt_evidence", "detector_attempt_status",
            "reacquire_attempted", "reacquire_succeeded", "reacquire_failed",
            "search_luma_mean", "search_luma_stdDev", "search_sharpness",
            "initial_search_region_area", "detection_region_area", "crop"]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def _frame_quality_condition_bands(fq_df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """Flagged-frame rate per Video Stats condition tercile (issue #44 deliverable 3).

    Reuses the condition-band machinery (``pd.qcut`` + ``_run_unit_rate``) from the
    within-video trends, but the outcome is the auto ``flagged`` flag and the predictor
    is a per-bundle Video Stats condition rather than a per-frame geometric one.

    A Video Stats condition is constant *within* a bundle, so a band here is really a
    handful of videos' worth of frames — the pseudo-replication is even starker than in
    the geometric bands, and the CI is likewise a run-unit cluster bootstrap (#70)."""

    if fq_df.empty:
        return pd.DataFrame()
    cond_cols = [c for c in fq_df.columns if c.startswith("vs_")]
    key_cols = [c for c in _RUN_KEY_COLS if c in fq_df.columns]
    rows: list[dict[str, Any]] = []
    for col in cond_cols:
        d = fq_df[[*key_cols, col, "flagged"]].dropna(subset=[col, "flagged"])
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            stats = _run_unit_rate(bg.assign(flagged=bg["flagged"].astype(int)), "flagged")
            if stats is None:
                continue
            rows.append({
                "condition": col[len("vs_"):],
                "band": int(band) + 1,
                "n": stats["n"],
                "n_runs": stats["n_runs"],
                "flagged_rate": stats["rate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "run_rate_median": stats["run_rate_median"],
                "run_rate_p90": stats["run_rate_p90"],
                "band_min": float(bg[col].min()),
                "band_max": float(bg[col].max()),
            })
    return pd.DataFrame(rows)


def _bootstrap_mean(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(vals)
    mean = sum(vals) / n
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    lo, hi = _pct_ci(draws)
    return mean, lo, hi


_ATTEMPT_ERROR_PREDICTORS = [
    "attempt_search_luma_mean_mean",
    "attempt_search_luma_stdDev_mean",
    "attempt_search_sharpness_mean",
    "attempt_initial_search_region_area_mean",
    "attempt_initial_search_region_edge_distance_mean",
    "attempt_detection_region_area_mean",
    "attempt_detection_region_edge_distance_mean",
    "attempt_reacquire_attempt_rate",
    "attempt_reacquire_success_rate",
    "attempt_full_frame_reacquire_success_rate",
]


def _crop_quality_rows(recs: list[EvalRecord]) -> pd.DataFrame:
    """Pool every record's ``cropQuality`` attempts into one long table (issue #86).

    Pooled across **all** records — quarantined and loose included — for the same reason
    the per-frame quality pool is: the runs whose crops wander are exactly the ones worth
    inspecting. Records predating schema v10 carry no ``cropQuality`` and contribute
    nothing."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        cq = rec.data.get("cropQuality")
        if not isinstance(cq, dict):
            continue
        loose = bool(rec.data.get("loosePaired"))
        conforming = record_conforms(rec.data)
        for e in cq.get("frames") or []:
            bbox = e.get("truthBbox") or {}
            rows.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": e.get("t"),
                "status": e.get("status"),
                "truth_present": e.get("truthPresent"),
                "miss_cause": e.get("missCause"),
                "miss_reason": e.get("missReason"),
                "best_unselected_candidate_score": e.get("bestUnselectedCandidateScore"),
                "initial_search_region_iou": e.get("initialSearchRegionIou"),
                "detection_region_iou": e.get("detectionRegionIou"),
                "initial_crop_containment": e.get("initialCropContainment"),
                "crop_contained_truth": e.get("cropContainedTruth"),
                "search_flags_fired": e.get("searchFlagsFired"),
                "fired_search_flags": ", ".join(e.get("firedSearchFlags") or []),
                "reacquire_attempted": e.get("reacquireAttempted"),
                "truth_bbox_area": (
                    bbox.get("w") * bbox.get("h")
                    if isinstance(bbox.get("w"), (int, float))
                    and isinstance(bbox.get("h"), (int, float)) else None),
                "loose": loose,
                "conforming": conforming,
            })
    return pd.DataFrame(rows)


def _miss_cause_table(crop_df: pd.DataFrame) -> pd.DataFrame:
    """Miss-cause frequency over the pooled attempts, with the crop-placement evidence
    beside each cause.

    ``crop_missed_truth`` is carried per cause on purpose: on a corpus where full-frame
    reacquire always runs, no miss is *caused* by the crop, yet the crop can still have
    excluded the Climber on most of them. Showing both stops the reader inferring either
    fact from the other."""

    if crop_df.empty or "miss_cause" not in crop_df.columns:
        return pd.DataFrame()
    sub = crop_df[crop_df["miss_cause"].notna()]
    if sub.empty:
        return pd.DataFrame()
    total = len(sub)
    rows: list[dict[str, Any]] = []
    for cause, g in sub.groupby("miss_cause"):
        contained = g["crop_contained_truth"]
        scored = int(contained.notna().sum())
        best = g.get("best_unselected_candidate_score")
        rows.append({
            "miss_cause": str(cause),
            "n": int(len(g)),
            "share": len(g) / total,
            "crop_missed_truth": int((contained == False).sum()),  # noqa: E712
            "crop_containment_scored": scored,
            "median_initial_crop_containment": (
                float(g["initial_crop_containment"].median())
                if g["initial_crop_containment"].notna().any() else None),
            "flags_fired": int((g["search_flags_fired"] == True).sum()),  # noqa: E712
            # The gate-tuning number (scanner issues 03-04): on identity-gated misses,
            # how confident the best candidate the gate rejected was.
            "median_best_unselected_candidate_score": (
                float(best.median()) if best is not None and best.notna().any() else None),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "miss_cause"], ascending=[False, True]).reset_index(drop=True)


def _crop_run_columns(cq: Any) -> dict[str, Any]:
    """Per-run crop-quality columns read off a record's ``cropQuality`` (issue #86).

    Pre-v10 and legacy frames-only records carry no block, so counts are zero and every
    rate/median is ``None`` — an unmeasured Run must not read as a Run with perfect
    crops."""

    cq = cq if isinstance(cq, dict) else {}
    causes = cq.get("missCauseCounts") if isinstance(cq.get("missCauseCounts"), dict) else {}
    contained = cq.get("cropContainedTruth") if isinstance(
        cq.get("cropContainedTruth"), dict) else {}
    initial = cq.get("initialSearchRegionIou") if isinstance(
        cq.get("initialSearchRegionIou"), dict) else {}
    missing = int(cq.get("missingAttempts") or 0)
    out: dict[str, Any] = {
        "crop_matched_attempts": int(cq.get("matchedAttempts") or 0),
        "missing_attempts": missing,
        "crop_contained_truth_rate": contained.get("rate"),
        "initial_search_region_iou_median": initial.get("median"),
    }
    for cause in MISS_CAUSES:
        count = int(causes.get(cause) or 0)
        out[f"miss_{_slug(cause)}_count"] = count
        out[f"miss_{_slug(cause)}_share"] = (count / missing) if missing else None
    return out


def _rejection_run_columns(fq: dict[str, Any]) -> dict[str, Any]:
    """Per-run rejection-correctness columns read off a record's ``frameQuality``
    (issue #85). Pre-v9 and legacy frames-only records carry no
    ``rejectionCorrectness`` block, so every count is zero and the rates are ``None`` —
    an unmeasured run must not read as a zero over-rejection rate.

    ``over_rejection_rate`` is the pooled rate across both rejection gates;
    ``flip_over_rejection_rate`` isolates the flip gate, which is the one the corpus
    baseline and the scanner-side flip-gate work are about. The ``*_truth_present``
    variants drop Climber-absent rejections from the denominator — see
    ``evaluate._rejection_rate_block`` for why both denominators are reported."""

    rc = fq.get("rejectionCorrectness")
    rc = rc if isinstance(rc, dict) else {}
    counts = rc.get("verdictCounts") if isinstance(rc.get("verdictCounts"), dict) else {}
    flip = rc.get("byStatus", {}).get("flipRejected") if isinstance(
        rc.get("byStatus"), dict) else None
    flip = flip if isinstance(flip, dict) else {}
    return {
        "rejected_attempts": int(rc.get("rejected") or 0),
        "good_pose_rejected": int(counts.get("goodPoseRejected") or 0),
        "bad_pose_rejected": int(counts.get("badPoseRejected") or 0),
        "rejection_truth_absent": int(rc.get("truthAbsent") or 0),
        "rejection_truth_unknown": int(counts.get("truthUnknown") or 0),
        "rejection_truth_checkable": int(rc.get("truthCheckable") or 0),
        "rejection_truth_present_checkable": int(rc.get("truthPresentCheckable") or 0),
        "over_rejection_rate": rc.get("overRejectionRate"),
        "over_rejection_rate_truth_present": rc.get("overRejectionRateTruthPresent"),
        "flip_rejected_attempts": int(flip.get("rejected") or 0),
        "flip_rejection_truth_checkable": int(flip.get("truthCheckable") or 0),
        "flip_over_rejection_rate": flip.get("overRejectionRate"),
        "flip_over_rejection_rate_truth_present": flip.get(
            "overRejectionRateTruthPresent"),
    }


def _detection_error_attempt_run_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: PoseRunCache | None = None,
) -> pd.DataFrame:
    """One row per evaluation record, joining Detection Errors to attempt summaries.

    The outcome is the record's frameQuality flagged rate; predictors are aggregated
    over that Run's Detector Attempts. This preserves the Run as the independent unit.
    Rejection correctness (issue #85) rides along as per-run columns so flip-gate
    changes are comparable batch-over-batch at the Run unit.
    """

    rows: list[dict[str, Any]] = []
    pose_cache = {} if pose_cache is None else pose_cache
    for rec in recs:
        fq = rec.data.get("frameQuality")
        if not isinstance(fq, dict):
            continue
        frames = fq.get("frames") or []
        detected = len(frames)
        flagged = sum(
            1 for e in frames
            if str((e or {}).get("class") or "ok") in _FQ_FLAGGED
        )
        _, _, attempts = _pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "loose": bool(rec.data.get("loosePaired")),
            "conforming": record_conforms(rec.data),
            "detected_frames": detected,
            "flagged_frames": flagged,
            "flagged_rate": flagged / detected if detected else None,
            "frozen_stale_frames": int(fq.get("frozenStaleCount") or 0),
            **_rejection_run_columns(fq),
            **_crop_run_columns(rec.data.get("cropQuality")),
            **_detector_attempt_summary(attempts),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["route_folder", "video_key", "run_ts"]).reset_index(drop=True)


def _detection_error_attempt_bands(run_df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """Group Detection Error rates against attempt evidence at the Run unit."""

    if run_df.empty or "flagged_rate" not in run_df.columns:
        return pd.DataFrame()
    predictors = [
        c for c in _ATTEMPT_ERROR_PREDICTORS
        if c in run_df.columns and pd.api.types.is_numeric_dtype(run_df[c])
    ]
    predictors += [
        c for c in run_df.columns
        if c.startswith("attempt_search_flag_") and c.endswith("_rate")
        and pd.api.types.is_numeric_dtype(run_df[c])
    ]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        d = run_df[[predictor, "flagged_rate"]].dropna()
        if len(d) < max(3, bins):
            continue
        if d[predictor].nunique() < 2:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[predictor], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            vals = bg["flagged_rate"].astype(float).tolist()
            boot = _bootstrap_mean(vals)
            if boot is None:
                continue
            rows.append({
                "predictor": predictor,
                "band": int(band) + 1,
                "n_runs": len(vals),
                "flagged_rate_mean": boot[0],
                "ci_low": boot[1],
                "ci_high": boot[2],
                "band_min": float(bg[predictor].min()),
                "band_max": float(bg[predictor].max()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Detector Attempt funnel (issue #87)
#
# What the detector *did*, before Ground Truth is consulted: how the attempt stream
# splits across accepted / missing / flipRejected / qualityRejected, how often reacquire
# ran and worked, and which search conditions were flagged under each status.
#
# The Run is the unit of inference (CONTEXT.md), so every pooled share here is reported
# beside its run-unit distribution — median, p90, and the count of runs the status
# dominates. A corpus where one very long run misses everything and a corpus where every
# run misses a quarter of its frames produce the same pooled missing share; only the
# run-unit columns tell them apart. Deliberately no pooled-frame CIs: attempts within a
# run are correlated, so a CI over pooled attempts would claim precision the design
# cannot support (#70).
# --------------------------------------------------------------------------- #

# A run is a *tail* run for a status when that status takes more than this share of its
# attempts. 0.5 is the corpus baseline's "runs > 50% missing" line — a run that misses
# most of what it looked at is a different failure from one that misses some of it.
ATTEMPT_FUNNEL_TAIL_SHARE = 0.5

# Run-unit distributions worth reporting outside the status mix (the per-status shares
# already carry their own median/p90 in the status table).
_FUNNEL_RUN_METRICS = [
    ("attempt_count", "attempts per run"),
    ("attempt_reacquire_attempt_rate", "reacquire attempted / attempts"),
    ("attempt_reacquire_success_rate", "reacquire succeeded / reacquires attempted"),
    ("attempt_full_frame_reacquire_success_rate",
     "full-frame reacquire succeeded / attempts"),
]


def _status_columns(status: str) -> tuple[str, str]:
    slug = _slug(status)
    return f"attempt_status_{slug}_count", f"attempt_status_{slug}_rate"


def _attempt_funnel_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run funnel rows — the attempt-backed subset of the Detection Error run table.

    Derived from that table rather than re-walking the attempt streams, so the funnel and
    the Detection Error section can never disagree about a run's status mix: there is one
    ``_detector_attempt_summary`` per run and both read it. Legacy runs are dropped by the
    same rule that makes them legacy — no attempt stream, so no funnel to report."""

    if run_df.empty or "attempt_evidence" not in run_df.columns:
        return pd.DataFrame()
    sub = run_df[run_df["attempt_evidence"].astype("string") == EVIDENCE_ATTEMPTS]
    if sub.empty:
        return pd.DataFrame()
    cols = ["route_folder", "video_key", "run_ts", "conforming", "loose", "attempt_count"]
    for status in DETECTOR_ATTEMPT_STATUS_ORDER:
        cols.extend(_status_columns(status))
    cols += [
        "attempt_reacquire_attempted_count", "attempt_reacquire_succeeded_count",
        "attempt_reacquire_failed_count", "attempt_reacquire_attempt_rate",
        "attempt_reacquire_success_rate",
        "attempt_full_frame_reacquire_success_count",
        "attempt_full_frame_reacquire_success_rate",
    ]
    cols += sorted(c for c in sub.columns if c.startswith("attempt_search_flag_"))
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def _attempt_funnel_status_table(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """The status mix, pooled over attempts *and* distributed over runs.

    Every status is a row even at zero: "nothing was quality-rejected this batch" is a
    result, and leaving the row out would leave it to be inferred from an absence."""

    if funnel_df.empty:
        return pd.DataFrame()
    counts_total = pd.to_numeric(funnel_df["attempt_count"], errors="coerce").fillna(0)
    total = int(counts_total.sum())
    rows: list[dict[str, Any]] = []
    for status in DETECTOR_ATTEMPT_STATUS_ORDER:
        count_col, rate_col = _status_columns(status)
        if count_col not in funnel_df.columns:
            continue
        counts = pd.to_numeric(funnel_df[count_col], errors="coerce").fillna(0)
        shares = pd.to_numeric(funnel_df.get(rate_col), errors="coerce").dropna()
        n = int(counts.sum())
        rows.append({
            "status": status,
            "attempts": n,
            "share": (n / total) if total else None,
            "runs_with_any": int((counts > 0).sum()),
            "run_share_median": float(shares.median()) if len(shares) else None,
            "run_share_p90": _p90(shares),
            "run_share_max": float(shares.max()) if len(shares) else None,
            "tail_runs": int((shares > ATTEMPT_FUNNEL_TAIL_SHARE).sum()),
        })
    return pd.DataFrame(rows)


def _attempt_funnel_run_stats(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """Run-unit distribution of the funnel measures that are not per-status shares."""

    if funnel_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for metric, label in _FUNNEL_RUN_METRICS:
        if metric not in funnel_df.columns:
            continue
        vals = pd.to_numeric(funnel_df[metric], errors="coerce").dropna()
        if vals.empty:
            continue  # e.g. no run ever attempted a reacquire — an unmeasured metric
        rows.append({
            "metric": metric,
            "meaning": label,
            "n_runs": int(len(vals)),
            "median": float(vals.median()),
            "p90": _p90(vals),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    return pd.DataFrame(rows)


def _attempt_funnel_flag_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: PoseRunCache | None = None,
) -> pd.DataFrame:
    """Condition-flag rate per attempt status — the one funnel table the per-run summary
    cannot supply, because it needs each attempt's flags *and* its status together.

    The denominator is attempts of that status **whose conditions carry the flag**: a
    scanner build that never emitted ``underexposed`` must not read as one that emitted it
    and found nothing. Pooled rates come with the per-run distribution beside them, since
    flags cluster hard within a run (one dark video floods the pool). The p90 is there
    because the median is usually zero — most runs never fire a given flag, so the median
    alone would report "no signal" for a flag a handful of runs fire on constantly."""

    pose_cache = {} if pose_cache is None else pose_cache
    pooled: dict[tuple[str, str], list[int]] = {}          # (flag, status) -> [scored, fired]
    per_run: dict[tuple[str, str], list[float]] = {}       # (flag, status) -> run rates
    for rec in recs:
        if record_evidence_generation(rec.data) != EVIDENCE_ATTEMPTS:
            continue
        _, _, attempts = _pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN
        run_tally: dict[tuple[str, str], list[int]] = {}
        for attempt in attempts or []:
            raw_status = attempt.get("status")
            status = (raw_status if raw_status in DETECTOR_ATTEMPT_STATUSES
                      else DETECTOR_ATTEMPT_STATUS_UNKNOWN)
            for flag, fired in _condition_flags(attempt.get("searchConditions")).items():
                for tally in (pooled, run_tally):
                    slot = tally.setdefault((flag, status), [0, 0])
                    slot[0] += 1
                    slot[1] += int(bool(fired))
        for key, (scored, fired) in run_tally.items():
            if scored:
                per_run.setdefault(key, []).append(fired / scored)

    order = {status: i for i, status in enumerate(DETECTOR_ATTEMPT_STATUS_ORDER)}
    rows: list[dict[str, Any]] = []
    for (flag, status), (scored, fired) in pooled.items():
        run_rates = per_run.get((flag, status), [])
        rows.append({
            "flag": flag,
            "status": status,
            "attempts_scored": scored,
            "flag_fired": fired,
            "rate": (fired / scored) if scored else None,
            "n_runs": len(run_rates),
            "run_rate_median": (float(np.median(run_rates)) if run_rates else None),
            "run_rate_p90": (float(np.quantile(run_rates, 0.9)) if run_rates else None),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_order"] = df["status"].map(lambda s: order.get(s, len(order)))
    return df.sort_values(["flag", "_order"]).drop(columns="_order").reset_index(drop=True)


def _attempt_funnel_totals(funnel_df: pd.DataFrame,
                           status_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide funnel headline, summed off the per-run rows so the tiles and the CSV
    can never disagree."""

    if funnel_df.empty:
        return {"runs": 0, "attempts": 0, "status_shares": {},
                "reacquire_attempted": 0, "reacquire_succeeded": 0,
                "reacquire_success_rate": None, "reacquire_success_rate_run_median": None,
                "missing_share_run_median": None, "tail_runs_missing": 0}

    def total(name: str) -> int:
        return int(pd.to_numeric(funnel_df.get(name), errors="coerce").fillna(0).sum())

    by_status = status_df.set_index("status") if not status_df.empty else pd.DataFrame()
    attempted = total("attempt_reacquire_attempted_count")
    succeeded = total("attempt_reacquire_succeeded_count")
    run_success = pd.to_numeric(
        funnel_df.get("attempt_reacquire_success_rate"), errors="coerce").dropna()
    return {
        "runs": int(len(funnel_df)),
        "attempts": total("attempt_count"),
        "status_shares": ({s: by_status.loc[s, "share"] for s in by_status.index}
                          if not by_status.empty else {}),
        "reacquire_attempted": attempted,
        "reacquire_succeeded": succeeded,
        "reacquire_success_rate": (succeeded / attempted) if attempted else None,
        "reacquire_success_rate_run_median": (
            float(run_success.median()) if len(run_success) else None),
        "missing_share_run_median": (
            by_status.loc["missing", "run_share_median"]
            if "missing" in getattr(by_status, "index", []) else None),
        "tail_runs_missing": (
            int(by_status.loc["missing", "tail_runs"])
            if "missing" in getattr(by_status, "index", []) else 0),
    }


def _rejection_totals(run_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide rejection-correctness headline (issue #85), summed over the per-run
    rows so the counts and the CSV can never disagree.

    Counts pool, but the *rate* is reported three ways on purpose. Two are denominator
    choices carried up from the record (``over_rejection_rate`` over every truth-checkable
    rejection, ``..._truth_present`` dropping the Climber-absent ones — see
    ``evaluate._rejection_rate_block``). The third, ``over_rejection_rate_run_mean``,
    averages the per-run rates: the Run is the unit of inference, so a corpus where one
    long run dominates the pooled frames is visibly different from one where every run
    over-rejects."""

    def col(name: str) -> pd.Series:
        return pd.to_numeric(run_df.get(name, pd.Series(dtype=float)), errors="coerce")

    def total(name: str) -> int:
        return 0 if run_df.empty else int(col(name).fillna(0).sum())

    rates = pd.Series(dtype=float) if run_df.empty else col("over_rejection_rate").dropna()
    runs_with = int(len(rates))
    good, bad, absent = (total("good_pose_rejected"), total("bad_pose_rejected"),
                         total("rejection_truth_absent"))
    checkable = good + bad
    present_checkable = checkable - absent
    return {
        "rejected_attempts": total("rejected_attempts"),
        "good_pose_rejected": good,
        "bad_pose_rejected": bad,
        "truth_absent": absent,
        "truth_unknown": total("rejection_truth_unknown"),
        "truth_checkable": checkable,
        "truth_present_checkable": present_checkable,
        "over_rejection_rate": (good / checkable) if checkable else None,
        "over_rejection_rate_truth_present": (
            (good / present_checkable) if present_checkable else None),
        "over_rejection_rate_run_mean": float(rates.mean()) if runs_with else None,
        "runs_with_checkable_rejections": runs_with,
    }


def _crop_totals(crop_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide crop-placement headline (issue #86).

    ``crop_missed_truth_rate`` is over truth-present attempts with a scorable crop, and is
    reported independently of the miss-cause mix: it is the crop-placement defect, not a
    causal claim about misses."""

    if crop_df.empty:
        return {"matched_attempts": 0, "missing_attempts": 0,
                "crop_containment_scored": 0, "crop_missed_truth": 0,
                "crop_missed_truth_rate": None,
                "median_initial_crop_containment": None,
                "median_initial_search_region_iou": None,
                "miss_cause_counts": {c: 0 for c in MISS_CAUSES}}

    contained = crop_df["crop_contained_truth"]
    scored = int(contained.notna().sum())
    missed = int((contained == False).sum())  # noqa: E712
    causes = crop_df["miss_cause"].dropna()
    return {
        "matched_attempts": int(len(crop_df)),
        "missing_attempts": int(len(causes)),
        "crop_containment_scored": scored,
        "crop_missed_truth": missed,
        "crop_missed_truth_rate": (missed / scored) if scored else None,
        "median_initial_crop_containment": (
            float(crop_df["initial_crop_containment"].median())
            if crop_df["initial_crop_containment"].notna().any() else None),
        "median_initial_search_region_iou": (
            float(crop_df["initial_search_region_iou"].median())
            if crop_df["initial_search_region_iou"].notna().any() else None),
        "miss_cause_counts": {
            c: int((causes == c).sum()) for c in MISS_CAUSES},
    }


def _quarantined_rows(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Non-conforming records (issue #15 gate), flattened for the report's shame
    accounting: which bundle/run tripped the gate, why, and the offending fit.

    Each row carries the issue #88 ``cause`` and the evidence behind it, so the section
    can be read cause-first: a sparse-match row is a detector problem that happens to trip
    a truth gate, and only a suspected-mistrack row is a truth problem."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if record_conforms(rec.data):
            continue
        conf = rec.data.get("conformance") or {}
        evidence = conf.get("causeEvidence") or {}
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "cause": record_nonconformance_cause(rec.data),
            "reasons": ", ".join(conf.get("reasons") or []),
            "n": conf.get("n"),
            "fit_frames": evidence.get("fitFrames"),
            "accepted_share": evidence.get("acceptedShare"),
            "slope_x": (conf.get("x") or {}).get("slope"),
            "r2_x": (conf.get("x") or {}).get("r2"),
            "slope_y": (conf.get("y") or {}).get("slope"),
            "r2_y": (conf.get("y") or {}).get("r2"),
        })
    return sorted(rows, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"]))


def _quarantine_cause_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Non-conforming records per cause. Every cause is keyed even at zero, so a report
    reading "0 suspected mis-tracks" is distinguishable from a report that never split."""

    counts = {c: 0 for c in NONCONFORMANCE_CAUSES}
    for row in rows:
        cause = row.get("cause")
        if cause in counts:
            counts[cause] += 1
    return counts


def _rate_mismatch_records(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Records whose scaffold sampled coarser than the truth grid (issue #101).

    Reported independently of the conformance gate on purpose. ``rate-mismatch`` is a
    *non-conformance cause*, so it only speaks when a record also fails — and a Bundle
    can under-sample its truth grid tenfold while still fitting cleanly on the frames it
    did sample. Those Bundles fabricate absences by the thousand, and absence provenance
    now keeps that out of the numbers; this list is what stops the underlying data defect
    from staying invisible, because the fix (regenerate the scaffold) is the same either
    way."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        conf = rec.data.get("conformance") or {}
        evidence = conf.get("causeEvidence") or {}
        ratio = evidence.get("samplingRatio")
        if not isinstance(ratio, (int, float)) or ratio < RATE_MISMATCH_MIN_RATIO:
            continue
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "scaffold_step_sec": evidence.get("scaffoldStepSec"),
            "truth_step_sec": evidence.get("truthStepSec"),
            "sampling_ratio": ratio,
            "conforms": bool(conf.get("conforms")),
        })
    return sorted(rows, key=lambda r: (-r["sampling_ratio"], r["route_folder"],
                                       r["video_key"], r["run_ts"]))


def _truth_repair_worklist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of quarantined records worth re-seeding truth for (issues #21/#34).

    Scoped to ``suspected-mistrack`` by issue #88: a sparse-match record fails the gate
    because the detector found almost nothing, and re-seeding its Ground Truth would burn
    review effort on a bundle whose truth may be fine."""

    return [r for r in rows if r.get("cause") == NONCONFORMANCE_SUSPECTED_MISTRACK]


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
    # Issue #89: dedup evidence generations *before* anything pools or is accounted for.
    # A superseded legacy record is not a quarantined record and not a loose pairing — it
    # is the same pairing measured twice — so it must not appear in either shame list.
    on_disk = _iter_eval_records(analysis_root)
    all_recs, superseded = _dedup_evidence_generations(on_disk)
    # Issue #15 gate: quarantine non-conforming bundles (truth mis-tracking) from
    # every *pooled* derivation below. Issue #44: best-overlap loose pairings are
    # likewise held out of the trusted pool (their setupHash never matched the truth).
    # Both classes stay on disk and inspectable; only the aggregation drops them, and
    # the report accounts for each by name.
    quarantined = _quarantined_rows(all_recs)
    quarantine_causes = _quarantine_cause_counts(quarantined)
    truth_repair = _truth_repair_worklist(quarantined)
    loose_records = _loose_rows(all_recs)
    recs = [r for r in all_recs if record_trusted(r.data)]
    evidence_trusted = _evidence_generation_summary(recs, "trusted pooled metrics")
    evidence_frames = _evidence_generation_summary(
        all_recs, "per-frame / attempt pools (all records)")
    # One pose-file read per bundle for the whole trend build: the frame/joint rows, the
    # per-frame quality pool, the Detection Error run table and the attempt funnel all
    # draw from this cache.
    pose_cache: PoseRunCache = {}
    for rec in all_recs:
        _pose_run(analysis_root, rec, pose_cache)
    app_versions = {
        (route, key, run_ts): av
        for (route, key), runs in pose_cache.items()
        for run_ts, (av, _, _) in runs.items()
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
    fq_df = _frame_quality_rows(analysis_root, all_recs, pose_cache)
    fq_classes = _frame_quality_classes(fq_df)
    fq_hallucination = _hallucination_split_totals(fq_df)
    fq_absence_reasons = _absence_reason_counts(fq_df)
    rate_mismatches = _rate_mismatch_records(all_recs)
    fq_distractors = _frame_quality_distractors(fq_df)
    fq_worklist = _frame_quality_worklist(fq_df)
    fq_condition_bands = _frame_quality_condition_bands(fq_df)
    fq_attempt_runs = _detection_error_attempt_run_rows(analysis_root, all_recs, pose_cache)
    fq_attempt_bands = _detection_error_attempt_bands(fq_attempt_runs)
    rejection_totals = _rejection_totals(fq_attempt_runs)

    # Detector Attempt funnel (issue #87): scanner behavior before truth is consulted,
    # over the attempt-backed records only — a legacy run has no attempt stream to funnel.
    # Quarantined and loose records stay in: a run that fails the #15 gate is often
    # exactly the run whose funnel collapsed, and dropping it would hide the failure the
    # section exists to show.
    funnel_recs = [r for r in all_recs
                   if record_evidence_generation(r.data) == EVIDENCE_ATTEMPTS]
    funnel_runs = _attempt_funnel_runs(fq_attempt_runs)
    funnel_status = _attempt_funnel_status_table(funnel_runs)
    funnel_run_stats = _attempt_funnel_run_stats(funnel_runs)
    funnel_flags = _attempt_funnel_flag_rows(analysis_root, funnel_recs, pose_cache)
    funnel_totals = _attempt_funnel_totals(funnel_runs, funnel_status)
    evidence_funnel = _evidence_generation_summary(funnel_recs, "attempt funnel")

    # Crop placement + miss causes (issue #86), pooled over the same all-records set.
    crop_df = _crop_quality_rows(all_recs)
    miss_causes = _miss_cause_table(crop_df)
    crop_totals = _crop_totals(crop_df)

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
        "eval_count_on_disk": len(on_disk),
        "superseded_records": superseded,
        "superseded_count": len(superseded),
        "evidence_generation_trusted": evidence_trusted,
        "evidence_generation_frames": evidence_frames,
        "quarantined_bundles": quarantined,
        "quarantined_count": len(quarantined),
        "quarantine_cause_counts": quarantine_causes,
        "truth_repair_worklist": truth_repair,
        "truth_repair_count": len(truth_repair),
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
        "frame_quality_hallucination": fq_hallucination,
        "frame_quality_absence_reasons": fq_absence_reasons,
        "rate_mismatch_records": rate_mismatches,
        "rate_mismatch_count": len(rate_mismatches),
        "frame_quality_distractors": fq_distractors,
        "frame_quality_worklist": fq_worklist,
        "frame_quality_condition_bands": fq_condition_bands,
        "detection_error_attempt_runs": fq_attempt_runs,
        "detection_error_attempt_bands": fq_attempt_bands,
        "rejection_correctness": rejection_totals,
        "attempt_funnel_runs": funnel_runs,
        "attempt_funnel_status": funnel_status,
        "attempt_funnel_run_stats": funnel_run_stats,
        "attempt_funnel_flags": funnel_flags,
        "attempt_funnel": funnel_totals,
        "evidence_generation_funnel": evidence_funnel,
        "crop_quality_attempts": crop_df,
        "crop_quality_miss_causes": miss_causes,
        "crop_quality": crop_totals,
        "frame_quality_detected": int(len(fq_df)),
        "frame_quality_flagged": int(fq_df["flagged"].sum()) if not fq_df.empty else 0,
        "frame_quality_held": int(fq_df["held_pose"].sum()) if not fq_df.empty else 0,
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
    truth_repair = ctx.get("truth_repair_worklist") or []
    truth_repair_df = pd.DataFrame(truth_repair) if truth_repair else pd.DataFrame()
    superseded = ctx.get("superseded_records") or []
    superseded_df = pd.DataFrame(superseded) if superseded else pd.DataFrame()
    rate_mismatch = ctx.get("rate_mismatch_records") or []
    rate_mismatch_df = pd.DataFrame(rate_mismatch) if rate_mismatch else pd.DataFrame()
    tables = {
        "eval_joint_ranking.csv": ctx.get("joint_rank"),
        "eval_condition_bands.csv": ctx.get("condition_bands"),
        "eval_cross_video_splits.csv": ctx.get("cross_video_splits"),
        "eval_version_overview.csv": ctx.get("version_overview"),
        "eval_version_deltas.csv": ctx.get("version_deltas"),
        "eval_low_confidence_worklist.csv": ctx.get("low_conf_worklist"),
        "eval_quarantined_bundles.csv": quarantined_df,
        "eval_truth_repair_worklist.csv": truth_repair_df,
        "eval_superseded_records.csv": superseded_df,
        "eval_frame_quality_classes.csv": ctx.get("frame_quality_classes"),
        "eval_frame_quality_absence_reasons.csv": ctx.get("frame_quality_absence_reasons"),
        "eval_rate_mismatch_records.csv": rate_mismatch_df,
        "eval_frame_quality_distractors.csv": ctx.get("frame_quality_distractors"),
        "eval_frame_quality_worklist.csv": ctx.get("frame_quality_worklist"),
        "eval_frame_quality_condition_bands.csv": ctx.get("frame_quality_condition_bands"),
        "eval_detection_error_attempt_runs.csv": ctx.get("detection_error_attempt_runs"),
        "eval_detection_error_attempt_bands.csv": ctx.get("detection_error_attempt_bands"),
        "eval_attempt_funnel_status.csv": ctx.get("attempt_funnel_status"),
        "eval_attempt_funnel_runs.csv": ctx.get("attempt_funnel_runs"),
        "eval_attempt_funnel_run_stats.csv": ctx.get("attempt_funnel_run_stats"),
        "eval_attempt_funnel_flags.csv": ctx.get("attempt_funnel_flags"),
        "eval_crop_quality_attempts.csv": ctx.get("crop_quality_attempts"),
        "eval_crop_quality_miss_causes.csv": ctx.get("crop_quality_miss_causes"),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            p = out_dir / name
            table.to_csv(p, index=False)
            outputs[name] = p
    return outputs
