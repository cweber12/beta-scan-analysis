"""Flatten each RunRecord into one per-run row (predictors + outcomes)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .discovery import RunRecord

# Hand labels carried in metadata.json -> analysis_inputs. Prefixed ``label_``.
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


def build_run_table(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rec in records:
        diag = rec.pose.get("diagnostics", {})
        inp = diag.get("input", {})
        result_pose = _get(diag, "result", "pose", default={}) or {}
        ref = inp.get("referenceFrame", {})

        labels = rec.metadata.get("analysis_inputs", {})
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
        badstretches = _get(diag, "result", "badStretches", default=[]) or []
        row["out_badStretchCount"] = len(badstretches)

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
