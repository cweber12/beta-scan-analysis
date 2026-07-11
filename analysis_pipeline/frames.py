"""Per-frame table: decode the video at sampled timestamps and join image-quality
predictors to the post-processed keypoint PROXY outcome.

Per-frame detector provenance is not exported (every frame is post-processed and
filled), so ``kp_count`` and ``mean_score`` are an explicit quality *proxy*, not the
detector's raw output. See the report banner and the plan's cross-repo follow-ups.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .discovery import RunRecord

try:  # optional at import time so discovery/stats work without cv2 installed
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only when cv2 is absent
    cv2 = None
    np = None

# torso keypoints used for the centroid-velocity predictor
_TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _sample_interval_sec(config: dict[str, Any]) -> float:
    frame_step = config.get("frameStep", 1) or 1
    frame_interval_ms = config.get("frameIntervalMs", 100) or 100
    return frame_step * frame_interval_ms / 1000.0


def _frames_by_timestamp(pose: dict[str, Any]) -> dict[float, list[dict[str, Any]]]:
    """Index exported frames by timestamp rounded to the 0.1s grid."""

    out: dict[float, list[dict[str, Any]]] = {}
    for fr in pose.get("frames", []) or []:
        ts = round(float(fr.get("timestamp", 0.0)), 1)
        out[ts] = fr.get("keypoints", []) or []
    return out


def _proxy_and_kinematics(keypoints: list[dict[str, Any]]) -> dict[str, Any]:
    if not keypoints:
        return {"kp_count": 0, "mean_score": None, "coverage": None, "_cx": None, "_cy": None}
    scores = [kp.get("score", 0.0) for kp in keypoints]
    xs = [kp.get("x") for kp in keypoints if kp.get("x") is not None]
    ys = [kp.get("y") for kp in keypoints if kp.get("y") is not None]
    coverage = None
    if xs and ys:
        coverage = (max(xs) - min(xs)) * (max(ys) - min(ys))

    by_name = {kp.get("name"): kp for kp in keypoints}
    torso = [by_name[n] for n in _TORSO if n in by_name]
    src = torso if torso else keypoints
    cx = sum(kp.get("x", 0.0) for kp in src) / len(src)
    cy = sum(kp.get("y", 0.0) for kp in src) / len(src)

    return {
        "kp_count": len(keypoints),
        "mean_score": sum(scores) / len(scores),
        "coverage": coverage,
        "_cx": cx,
        "_cy": cy,
    }


def _crop_stats(gray, box: dict[str, Any], h: int, w: int, prefix: str) -> dict[str, Any]:
    """Laplacian-variance sharpness + luma mean/std for a normalized crop box."""

    keys = [f"{prefix}_sharpness", f"{prefix}_luma_mean", f"{prefix}_luma_std"]
    if not box:
        return {k: None for k in keys}
    x0 = max(0, min(w - 1, int(round(box.get("x", 0) * w))))
    y0 = max(0, min(h - 1, int(round(box.get("y", 0) * h))))
    x1 = max(x0 + 1, min(w, int(round((box.get("x", 0) + box.get("w", 0)) * w))))
    y1 = max(y0 + 1, min(h, int(round((box.get("y", 0) + box.get("h", 0)) * h))))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return {k: None for k in keys}
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    return {
        keys[0]: sharpness,
        keys[1]: float(crop.mean()),
        keys[2]: float(crop.std()),
    }


def build_frame_table(records: list[RunRecord], decode: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for rec in records:
        config = rec.config
        interval = _sample_interval_sec(config)
        duration = (
            rec.pose.get("diagnostics", {}).get("input", {}).get("video", {}).get("durationSec")
            or 0.0
        )
        frame_index = _frames_by_timestamp(rec.pose)

        cap = None
        vh = vw = 0
        can_decode = decode and cv2 is not None and rec.video_path is not None
        if can_decode:
            cap = cv2.VideoCapture(str(rec.video_path))
            if not cap.isOpened():
                cap = None
                can_decode = False
            else:
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        climber_box = rec.setup.get("climberCrop", {})
        wall_box = rec.setup.get("wallCrop", {})

        prev_c: tuple[float, float] | None = None
        n_samples = int(math.floor(duration / interval)) + 1 if interval > 0 else 0

        for k in range(n_samples):
            t = round(k * interval, 1)
            keypoints = frame_index.get(t, [])
            kin = _proxy_and_kinematics(keypoints)

            velocity = None
            cx, cy = kin.pop("_cx"), kin.pop("_cy")
            if cx is not None and prev_c is not None:
                velocity = math.hypot(cx - prev_c[0], cy - prev_c[1])
            if cx is not None:
                prev_c = (cx, cy)

            row: dict[str, Any] = {
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": t,
                "velocity": velocity,
                **kin,
            }

            if can_decode:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    row.update(_crop_stats(gray, climber_box, vh, vw, "climber"))
                    row.update(_crop_stats(gray, wall_box, vh, vw, "wall"))

            rows.append(row)

        if cap is not None:
            cap.release()

    return pd.DataFrame(rows)
