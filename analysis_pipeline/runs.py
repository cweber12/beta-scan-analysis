"""Flatten each RunRecord into one per-run row (predictors + outcomes)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .detector_attempts import (
    ATTEMPT_STAT_KEYS as _ATTEMPT_STAT_KEYS,
    DETECTOR_ATTEMPT_STATUSES,
    condition_flags as _condition_flags,
    is_full_frame as _is_full_frame,
    region_metric as _region_metric,
    _num,
    _slug,
)
from .discovery import RunRecord

# Hand labels carried in setup.json -> analysisInputs. Prefixed ``label_``.
LABEL_KEYS = [
    "route_orientation",
    "camera_angle",
    "shadows",
    "climber_contrast",
    "wall_contrast",
    "motion_blur",
    "occlusion",
    "camera_stability",
]

# Region stat blocks present on both the pose input.referenceFrame and the orb
# referenceFrameMeta objects.
_REGIONS = ("overall", "climber", "wall")
_STATS = ("mean", "stdDev", "sharpness")

# Video Stats predictor columns (issue #23). Phase-1 source stats live in
# metadata.json["video_stats"]; phase-2 region stats in video-stats.json.
# Each entry: column suffix -> path inside the respective block.
_SOURCE_STAT_PATHS: dict[str, tuple[str, ...]] = {
    "lumaMean": ("luma", "mean"),
    "lumaStd": ("luma", "std"),
    "lumaP5": ("luma", "p5"),
    "lumaP95": ("luma", "p95"),
    "clippedHighlightFraction": ("clippedHighlightFraction",),
    "crushedShadowFraction": ("crushedShadowFraction",),
    "rmsContrast": ("rmsContrast",),
    "sharpnessMean": ("sharpness", "mean"),
    "sharpnessMin": ("sharpness", "min"),
    "frameDiffMean": ("frameDiff", "mean"),
    "frameDiffMax": ("frameDiff", "max"),
    "exposureDriftSlope": ("exposureDrift", "slopePerMinute"),
    "exposureDriftRange": ("exposureDrift", "range"),
    "colorCastROverG": ("colorCast", "rOverG"),
    "colorCastBOverG": ("colorCast", "bOverG"),
    "bitsPerPixel": ("bitsPerPixel",),
}

_REGION_STAT_PATHS: dict[str, tuple[str, ...]] = {
    "wallLumaMean": ("wall", "luma", "mean"),
    "wallRmsContrast": ("wall", "rmsContrast"),
    "wallEdgeDensity": ("wall", "texture", "edgeDensity"),
    "wallLaplacianVar": ("wall", "texture", "laplacianVar"),
    "wallHueConcentration": ("wall", "hue", "concentration"),
    "wallSaturationMean": ("wall", "saturation", "mean"),
    "climberWallDeltaE": ("climberWall", "deltaE"),
    "climberWallLumaSep": ("climberWall", "lumaSeparation"),
    "shadowFractionMean": ("shadow", "fraction", "mean"),
    "shadowFractionStd": ("shadow", "fraction", "std"),
    "shadowInOutLumaRatio": ("shadow", "inOutLumaRatio"),
    "shadowBlobCount": ("shadow", "blobs", "count"),
    "shadowBlobLargestFraction": ("shadow", "blobs", "largestFraction"),
    "shadowDriftRange": ("shadow", "drift", "range"),
}

_REGION_GEOM_KEYS = ("area", "cx", "cy", "edge_distance")


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _reference_stats(ref: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for region in _REGIONS:
        for stat in _STATS:
            out[f"{prefix}_{region}_{stat}"] = _get(ref, region, stat)
    flags = ref.get("flags", {}) if isinstance(ref, dict) else {}
    for flag, val in flags.items():
        out[f"{prefix}_flag_{flag}"] = bool(val)
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _detector_attempt_summary(attempts: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Run-unit summaries of Detector Attempt evidence.

    Missing attempts stay unknown (``None`` metrics) so legacy runs are not confused
    with zero failures or zero reacquires.
    """

    out: dict[str, Any] = {
        "attempt_evidence": "unknown" if attempts is None else "attempts",
        "attempt_count": None,
        "attempt_reacquire_attempted_count": None,
        "attempt_reacquire_succeeded_count": None,
        "attempt_reacquire_failed_count": None,
        "attempt_reacquire_attempt_rate": None,
        "attempt_reacquire_success_rate": None,
        "attempt_full_frame_reacquire_success_count": None,
        "attempt_full_frame_reacquire_success_rate": None,
    }
    for status in sorted(DETECTOR_ATTEMPT_STATUSES):
        out[f"attempt_status_{_slug(status)}_count"] = None
        out[f"attempt_status_{_slug(status)}_rate"] = None
    out["attempt_status_unknown_count"] = None
    out["attempt_status_unknown_rate"] = None
    for region_prefix in ("initial_search_region", "detection_region"):
        for metric in _REGION_GEOM_KEYS:
            out[f"attempt_{region_prefix}_{metric}_mean"] = None
            out[f"attempt_{region_prefix}_{metric}_min"] = None
            out[f"attempt_{region_prefix}_{metric}_max"] = None
    for prefix in ("search", "reacquire"):
        for suffix in _ATTEMPT_STAT_KEYS.values():
            out[f"attempt_{prefix}_{suffix}_mean"] = None

    if attempts is None:
        return out

    n = len(attempts)
    out["attempt_count"] = n
    if n == 0:
        return out

    status_counts = {status: 0 for status in sorted(DETECTOR_ATTEMPT_STATUSES)}
    status_counts["unknown"] = 0
    reacquire_attempted = 0
    reacquire_succeeded = 0
    reacquire_failed = 0
    full_frame_reacquire_success = 0
    flag_counts: dict[str, int] = {}
    flag_names: set[str] = set()

    for attempt in attempts:
        status = attempt.get("status")
        status_counts[status if status in DETECTOR_ATTEMPT_STATUSES else "unknown"] += 1
        did_reacquire = bool(attempt.get("reacquireAttempted"))
        reacquired = bool(attempt.get("reacquired"))
        if did_reacquire:
            reacquire_attempted += 1
            if reacquired:
                reacquire_succeeded += 1
                if _is_full_frame(attempt.get("detectionRegion")):
                    full_frame_reacquire_success += 1
            else:
                reacquire_failed += 1
        for name, value in _condition_flags(attempt.get("searchConditions")).items():
            flag_names.add(name)
            if value:
                flag_counts[name] = flag_counts.get(name, 0) + 1

    for status, count in status_counts.items():
        key = _slug(status)
        out[f"attempt_status_{key}_count"] = count
        out[f"attempt_status_{key}_rate"] = count / n

    out["attempt_reacquire_attempted_count"] = reacquire_attempted
    out["attempt_reacquire_succeeded_count"] = reacquire_succeeded
    out["attempt_reacquire_failed_count"] = reacquire_failed
    out["attempt_reacquire_attempt_rate"] = reacquire_attempted / n
    out["attempt_reacquire_success_rate"] = (
        reacquire_succeeded / reacquire_attempted if reacquire_attempted else None
    )
    out["attempt_full_frame_reacquire_success_count"] = full_frame_reacquire_success
    out["attempt_full_frame_reacquire_success_rate"] = full_frame_reacquire_success / n

    for region_prefix, source_key in (
        ("initial_search_region", "initialSearchRegion"),
        ("detection_region", "detectionRegion"),
    ):
        for metric in _REGION_GEOM_KEYS:
            vals = [
                v for v in (_region_metric(a.get(source_key), metric) for a in attempts)
                if v is not None
            ]
            if vals:
                out[f"attempt_{region_prefix}_{metric}_mean"] = _mean(vals)
                out[f"attempt_{region_prefix}_{metric}_min"] = min(vals)
                out[f"attempt_{region_prefix}_{metric}_max"] = max(vals)

    for prefix, source_key in (
        ("search", "searchConditions"),
        ("reacquire", "reacquireConditions"),
    ):
        for src_key, suffix in _ATTEMPT_STAT_KEYS.items():
            vals = [
                float(v)
                for a in attempts
                if isinstance(a.get(source_key), dict)
                for v in [a[source_key].get(src_key)]
                if isinstance(v, (int, float))
            ]
            if vals:
                out[f"attempt_{prefix}_{suffix}_mean"] = _mean(vals)

    for name in sorted(flag_names):
        count = flag_counts.get(name, 0)
        out[f"attempt_search_flag_{name}_rate"] = count / n
        out[f"attempt_search_flag_{name}_count"] = count

    return out


def build_run_table(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rec in records:
        diag = rec.pose.get("diagnostics", {})
        inp = diag.get("input", {})
        result_pose = _get(diag, "result", "pose", default={}) or {}
        ref = inp.get("referenceFrame", {})

        # Condition labels now live in setup.json.analysisInputs (written by the
        # scanner at calibration). Older bundles are backfilled by the one-off
        # migration, so this is the single source of truth.
        labels = rec.setup.get("analysisInputs", {}) or {}
        row: dict[str, Any] = {
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "config_hash": rec.config_hash,
        }
        # --- predictors: hand labels ---
        for key in LABEL_KEYS:
            row[f"label_{key}"] = labels.get(key, "unknown")

        # --- predictors: derived reference-frame stats ---
        row.update(_reference_stats(ref, "ref"))
        row["motionMagnitude"] = inp.get("motionMagnitude")
        row["climberCoverage_avg"] = _get(inp, "climberFrameCoverage", "avg")
        row["climberCoverage_min"] = _get(inp, "climberFrameCoverage", "min")

        # --- predictors: Video Stats (issue #23) ---
        source_stats = rec.metadata.get("video_stats") or {}
        for suffix, path in _SOURCE_STAT_PATHS.items():
            row[f"src_{suffix}"] = _get(source_stats, *path)

        region_stats = rec.video_stats.get("regionStats") or {}
        for suffix, path in _REGION_STAT_PATHS.items():
            row[f"vs_{suffix}"] = _get(region_stats, *path)
        row["vs_panningFlagged"] = (
            region_stats.get("panningFlagged") if region_stats else None
        )
        # Staleness: region stats describe the crops of the setup they were
        # computed under; a run under a different setupHash must be visible as
        # stale rather than silently wrong. None = no region stats at all.
        if region_stats:
            row["vs_stale"] = (rec.video_stats.get("setupHash") or "") != (
                rec.setup_hash or ""
            )
        else:
            row["vs_stale"] = None
        row["vs_cameraAngle"] = _get(rec.video_stats, "cameraAngle", "estimate")

        # --- predictors: Detector Attempt run-unit summaries (issue #74) ---
        row.update(_detector_attempt_summary(rec.detector_attempts))

        # --- outcomes: pose (per-run aggregates) ---
        sampled = result_pose.get("sampledFrames")
        flipped = result_pose.get("flippedFrames")
        row["out_detectionRate"] = result_pose.get("detectionRate")
        row["out_sampledFrames"] = sampled
        row["out_detectedFrames"] = result_pose.get("detectedFrames")
        row["out_flippedFrames"] = flipped
        row["out_flipRate"] = (
            flipped / sampled if sampled and flipped is not None else None
        )
        row["out_goodFrames"] = result_pose.get("goodFrames")
        row["out_keptFrames"] = result_pose.get("keptFrames")
        row["out_confidence_avg"] = _get(result_pose, "confidence", "avg")
        row["out_confidence_min"] = _get(result_pose, "confidence", "min")
        row["out_avgKeypointCount"] = result_pose.get("avgKeypointCount")
        row["out_limbExpandedFrames"] = result_pose.get("limbExpandedFrames")
        row["out_gapsRefined"] = _get(result_pose, "refinement", "gapsRefined")

        # --- primary pose outcome: the scanner's end-to-end verdict (ADR 0001) ---
        # Currently null across the corpus; populated once the scanner ships it.
        row["out_overlayQuality"] = _get(diag, "result", "overlayQuality")
        badstretches = _get(diag, "result", "badStretches", default=[]) or []
        row["out_badStretchCount"] = len(badstretches)
        row["out_badStretchSeconds"] = sum(
            max(0.0, float(s.get("endSec", 0.0)) - float(s.get("startSec", 0.0)))
            for s in badstretches
            if isinstance(s, dict)
        )

        # --- outcomes: orb (reference feature richness only) ---
        orb_meta = rec.orb.get("referenceFrameMeta", {})
        orb_summary = rec.orb.get("summary", {})
        row["orb_refKeypointCount"] = orb_meta.get(
            "refKeypointCount", orb_summary.get("refKeypointCount")
        )
        # orb reference region stats (may echo the pose ones, kept for ORB section)
        row.update(_reference_stats(orb_meta, "orb_ref"))

        # crop geometry (a plausible driver of ORB feature count)
        wall = rec.setup.get("wallCrop", {})
        row["wall_crop_area"] = (
            (wall.get("w") or 0) * (wall.get("h") or 0) if wall else None
        )

        rows.append(row)

    return pd.DataFrame(rows)
