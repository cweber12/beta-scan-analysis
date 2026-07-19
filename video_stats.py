"""Video Stats — automatic image-statistic Predictors for analysis bundles (issue #23).

Two-phase extraction:

- **Phase 1 (source stats)** — whole-frame stats sampled across the video, computed
  at download/import and stored as the ``video_stats`` block of the bundle's
  ``metadata.json`` (immutable source facts, never stale).
- **Phase 2 (region stats)** — crop-aware stats (wall color/texture, climber↔wall
  contrast, shadow structure), computed when the scanner POSTs its freshly drawn
  calibration crops. Stored in the per-bundle ``video-stats.json`` artifact stamped
  with the ``setupHash`` it was computed under — the same staleness pattern as
  Ground Truth (ADR 0004).

The stats functions are pure frames-in / stats-out (numpy + OpenCV only — this
module stays outside the ViTPose heavy-dependency quarantine, ADR 0003). The video
decode adapter around them is deliberately thin and untested.

From the continuous stats, corpus-fit thresholds (``SUGGESTION_THRESHOLDS``) derive
suggested hand labels the scanner prefills for the user to *verify*; the
human-verified layer in ``setup.json.analysisInputs`` stays authoritative.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


VIDEO_STATS_NAME = "video-stats.json"
VIDEO_STATS_VERSION = 1

# Frame sampling: uniform ~1 frame/sec capped at 30 frames, one decode pass.
SAMPLE_HZ = 1.0
MAX_SAMPLES = 30

# Exposure clipping bounds on 8-bit luma.
_HIGHLIGHT_CLIP = 250
_SHADOW_CRUSH = 5

# A wall pixel is "in shadow" when darker than this fraction of the wall's median
# luma for that sample. Relative (not absolute) so overall exposure doesn't move it.
_SHADOW_LUMA_FRACTION = 0.70

# Shadow blobs smaller than this fraction of the wall area are noise, not structure.
_MIN_BLOB_FRACTION = 0.005

# Cap on pixels fed to k-means / histogram passes so cost stays flat per video.
_MAX_REGION_PIXELS = 20_000

_DOMINANT_COLOR_K = 3


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _round(v: float, nd: int = 4) -> float:
    return round(float(v), nd)


def _gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": _round(arr.mean()),
        "std": _round(arr.std()),
        "min": _round(arr.min()),
        "max": _round(arr.max()),
    }


def _crop_region(frame: np.ndarray, crop: dict[str, float]) -> np.ndarray:
    """Slice a normalized {x, y, w, h} crop out of a frame, clamped to >= 1 px."""
    h, w = frame.shape[:2]
    x0 = min(max(int(round(crop["x"] * w)), 0), w - 1)
    y0 = min(max(int(round(crop["y"] * h)), 0), h - 1)
    x1 = min(max(int(round((crop["x"] + crop["w"]) * w)), x0 + 1), w)
    y1 = min(max(int(round((crop["y"] + crop["h"]) * h)), y0 + 1), h)
    return frame[y0:y1, x0:x1]


def _subsample_pixels(region_bgr: np.ndarray, cap: int = _MAX_REGION_PIXELS) -> np.ndarray:
    """Region -> (N, 3) float32 pixel rows, decimated to at most ``cap`` rows."""
    pixels = region_bgr.reshape(-1, 3)
    if len(pixels) > cap:
        step = int(np.ceil(len(pixels) / cap))
        pixels = pixels[::step]
    return pixels.astype(np.float32)


def _bgr_rows_to_lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """(N, 3) uint8-range BGR rows -> (N, 3) real CIELAB (L 0..100, a/b signed)."""
    as_image = pixels_bgr.reshape(-1, 1, 3).astype(np.uint8)
    lab8 = cv2.cvtColor(as_image, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    lab8[:, 0] *= 100.0 / 255.0
    lab8[:, 1] -= 128.0
    lab8[:, 2] -= 128.0
    return lab8


def _mean_lab(region_bgr: np.ndarray) -> np.ndarray:
    pixels = _subsample_pixels(region_bgr)
    if len(pixels) == 0:
        return np.zeros(3)
    return _bgr_rows_to_lab(pixels).mean(axis=0)


# --------------------------------------------------------------------------- #
# Phase 1 — whole-frame source stats
# --------------------------------------------------------------------------- #

def compute_source_stats(
    frames_bgr: Sequence[np.ndarray], timestamps: Sequence[float]
) -> dict[str, Any]:
    """Whole-frame stats over uniformly sampled frames. Pure: frames in, stats out."""
    if not frames_bgr:
        raise ValueError("compute_source_stats needs at least one frame.")
    if len(frames_bgr) != len(timestamps):
        raise ValueError("frames and timestamps must pair 1:1.")

    luma_means: list[float] = []
    luma_all = []  # decimated pixels pooled across frames for percentiles
    clipped: list[float] = []
    crushed: list[float] = []
    rms: list[float] = []
    sharpness: list[float] = []
    diffs: list[float] = []
    bgr_means = np.zeros(3, dtype=np.float64)

    prev_small: np.ndarray | None = None
    for frame in frames_bgr:
        gray = _gray(frame)
        luma_means.append(float(gray.mean()))
        flat = gray.reshape(-1)
        if len(flat) > _MAX_REGION_PIXELS:
            flat = flat[:: int(np.ceil(len(flat) / _MAX_REGION_PIXELS))]
        luma_all.append(flat)
        clipped.append(float((gray >= _HIGHLIGHT_CLIP).mean()))
        crushed.append(float((gray <= _SHADOW_CRUSH).mean()))
        rms.append(float(gray.std()) / 255.0)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if frame.ndim == 3:
            bgr_means += frame.reshape(-1, 3).mean(axis=0)

        # Frame-diff energy on a small fixed-width grayscale so shake registers as
        # structure change, not compression noise; normalized to 0..1.
        scale = 160.0 / max(gray.shape[1], 1)
        small = cv2.resize(gray, (160, max(int(round(gray.shape[0] * scale)), 1)))
        if prev_small is not None and prev_small.shape == small.shape:
            diffs.append(
                float(np.abs(small.astype(np.int16) - prev_small.astype(np.int16)).mean())
                / 255.0
            )
        prev_small = small

    pooled = np.concatenate(luma_all)
    p5, p50, p95 = (float(np.percentile(pooled, q)) for q in (5, 50, 95))

    means_arr = np.asarray(luma_means)
    span_min = (timestamps[-1] - timestamps[0]) / 60.0 if len(timestamps) > 1 else 0.0
    slope = 0.0
    if span_min > 0:
        slope = float(np.polyfit(np.asarray(timestamps) / 60.0, means_arr, 1)[0])

    bgr_means /= max(sum(1 for f in frames_bgr if f.ndim == 3), 1)
    g = max(bgr_means[1], 1e-6)

    return {
        "sampledFrames": len(frames_bgr),
        "sampleSpanSeconds": _round(timestamps[-1] - timestamps[0], 2),
        "luma": {
            "mean": _round(means_arr.mean()),
            "std": _round(float(pooled.std())),
            "p5": _round(p5),
            "p50": _round(p50),
            "p95": _round(p95),
        },
        "clippedHighlightFraction": _round(float(np.mean(clipped)), 5),
        "crushedShadowFraction": _round(float(np.mean(crushed)), 5),
        "rmsContrast": _round(float(np.mean(rms))),
        "sharpness": _aggregate(sharpness),
        "frameDiff": _aggregate(diffs) if diffs else None,
        "exposureDrift": {
            "slopePerMinute": _round(slope),
            "range": _round(float(means_arr.max() - means_arr.min())),
        },
        "colorCast": {
            "rOverG": _round(bgr_means[2] / g),
            "bOverG": _round(bgr_means[0] / g),
        },
    }


def bits_per_pixel(
    filesize: int | None,
    width: int | None,
    height: int | None,
    fps: float | None,
    duration_seconds: float | None,
) -> float | None:
    """Compression-quality proxy from already-recorded source facts."""
    if not all((filesize, width, height, fps, duration_seconds)):
        return None
    pixels = float(width) * float(height) * float(fps) * float(duration_seconds)
    if pixels <= 0:
        return None
    return _round(float(filesize) * 8.0 / pixels, 5)


# --------------------------------------------------------------------------- #
# Phase 2 — region stats (calibration crops)
# --------------------------------------------------------------------------- #

def _dominant_colors(pixels_bgr: np.ndarray, k: int = _DOMINANT_COLOR_K) -> list[dict]:
    """k-means in Lab over wall pixels -> [{fraction, lab, rgbHex}] by weight."""
    if len(pixels_bgr) < k:
        return []
    lab = _bgr_rows_to_lab(pixels_bgr).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        lab, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.reshape(-1), minlength=k).astype(np.float64)
    out = []
    for i in np.argsort(-counts):
        if counts[i] == 0:
            continue
        L, a, b = (float(v) for v in centers[i])
        lab8 = np.array([[[L * 255.0 / 100.0, a + 128.0, b + 128.0]]], dtype=np.uint8)
        bgr = cv2.cvtColor(lab8, cv2.COLOR_LAB2BGR).reshape(3)
        out.append({
            "fraction": _round(counts[i] / counts.sum()),
            "lab": [_round(L, 2), _round(a, 2), _round(b, 2)],
            "rgbHex": f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}",
        })
    return out


def _hue_saturation(pixels_bgr: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3).astype(np.float64)
    # OpenCV hue is 0..179 (degrees / 2); circular mean keeps red (~0/360) sane.
    ang = np.deg2rad(hsv[:, 0] * 2.0)
    mean_deg = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))) % 360.0
    resultant = float(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()))
    return {
        "hue": {
            "meanDeg": _round(mean_deg, 1),
            # 0 = hues uniformly scattered, 1 = single hue.
            "concentration": _round(resultant),
        },
        "saturation": {
            "mean": _round(hsv[:, 1].mean()),
            "std": _round(hsv[:, 1].std()),
        },
    }


def _shadow_sample(gray_wall: np.ndarray) -> dict[str, float]:
    """Shadow measurements for one wall sample: fraction + in/out luma & contrast."""
    med = float(np.median(gray_wall))
    mask = gray_wall < _SHADOW_LUMA_FRACTION * med
    frac = float(mask.mean())
    inside = gray_wall[mask]
    outside = gray_wall[~mask]
    return {
        "fraction": frac,
        "inLuma": float(inside.mean()) if inside.size else 0.0,
        "outLuma": float(outside.mean()) if outside.size else float(gray_wall.mean()),
        "inContrast": float(inside.std()) / 255.0 if inside.size else 0.0,
        "outContrast": float(outside.std()) / 255.0 if outside.size else 0.0,
    }


def _shadow_blobs(gray_wall: np.ndarray) -> dict[str, Any]:
    """Connected-component structure of the shadow mask on one representative frame."""
    med = float(np.median(gray_wall))
    mask = (gray_wall < _SHADOW_LUMA_FRACTION * med).astype(np.uint8)
    total = mask.size
    n_all, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [
        int(stats_cc[i, cv2.CC_STAT_AREA])
        for i in range(1, n_all)
        if stats_cc[i, cv2.CC_STAT_AREA] >= _MIN_BLOB_FRACTION * total
    ]
    shadow_total = sum(areas)
    return {
        "count": len(areas),
        "largestFraction": _round(max(areas) / shadow_total) if shadow_total else 0.0,
        "meanAreaFraction": _round(np.mean(areas) / total) if areas else 0.0,
    }


def compute_region_stats(
    frames_bgr: Sequence[np.ndarray],
    timestamps: Sequence[float],
    wall_crop: dict[str, float],
    climber_crop: dict[str, float] | None = None,
    climber_point_t: float | None = None,
    panning: bool = False,
) -> dict[str, Any]:
    """Crop-aware stats over sampled frames. Pure: frames + geometry in, stats out.

    Wall aggregates span all samples (valid when ``panning`` is false; still
    recorded but flagged when true, since the reference-frame crop then drifts off
    the wall). Climber↔wall contrast is computed at the sample nearest the frame
    the climber crop was drawn against (``climber_point_t``).
    """
    if not frames_bgr:
        raise ValueError("compute_region_stats needs at least one frame.")
    if len(frames_bgr) != len(timestamps):
        raise ValueError("frames and timestamps must pair 1:1.")

    wall_luma_mean: list[float] = []
    wall_rms: list[float] = []
    edge_density: list[float] = []
    laplacian_var: list[float] = []
    shadow_samples: list[dict[str, float]] = []
    color_pixels: list[np.ndarray] = []

    for frame in frames_bgr:
        wall = _crop_region(frame, wall_crop)
        gray = _gray(wall)
        wall_luma_mean.append(float(gray.mean()))
        wall_rms.append(float(gray.std()) / 255.0)
        edges = cv2.Canny(gray, 50, 150)
        edge_density.append(float((edges > 0).mean()))
        laplacian_var.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        shadow_samples.append(_shadow_sample(gray))
        if wall.ndim == 3:
            color_pixels.append(_subsample_pixels(wall, _MAX_REGION_PIXELS // max(len(frames_bgr), 1)))

    pooled_pixels = (
        np.concatenate(color_pixels) if color_pixels else np.zeros((0, 3), np.float32)
    )

    fractions = [s["fraction"] for s in shadow_samples]
    # Blob structure from the sample with the median shadow fraction — one
    # representative frame, not a mix of different shadow states.
    rep_idx = int(np.argsort(fractions)[len(fractions) // 2])
    rep_gray = _gray(_crop_region(frames_bgr[rep_idx], wall_crop))

    in_luma = float(np.mean([s["inLuma"] for s in shadow_samples]))
    out_luma = float(np.mean([s["outLuma"] for s in shadow_samples]))

    shadow_block: dict[str, Any] = {
        "fraction": _aggregate(fractions),
        "inOutLumaRatio": _round(in_luma / out_luma) if out_luma > 0 else None,
        "inShadowContrast": _round(float(np.mean([s["inContrast"] for s in shadow_samples]))),
        "outShadowContrast": _round(float(np.mean([s["outContrast"] for s in shadow_samples]))),
        "blobs": _shadow_blobs(rep_gray),
        "drift": {
            "range": _round(max(fractions) - min(fractions)),
            "firstToLast": _round(fractions[-1] - fractions[0]),
        },
    }

    climber_wall: dict[str, Any] | None = None
    if climber_crop is not None:
        t = climber_point_t if climber_point_t is not None else timestamps[0]
        idx = int(np.argmin([abs(ts - t) for ts in timestamps]))
        frame = frames_bgr[idx]
        climber_region = _crop_region(frame, climber_crop)
        wall_region = _crop_region(frame, wall_crop)
        climber_lab = _mean_lab(climber_region)
        # Surrounding wall = wall crop minus the climber box; masking the exact
        # overlap is overkill for a mean — the climber box is a small fraction of
        # the wall, so the whole wall crop is a stable stand-in for "surround".
        wall_lab = _mean_lab(wall_region)
        delta = climber_lab - wall_lab
        climber_wall = {
            "deltaE": _round(float(np.linalg.norm(delta)), 2),
            "lumaSeparation": _round(abs(delta[0]) * 255.0 / 100.0, 2),
            "frameTimestamp": _round(timestamps[idx], 3),
        }

    wall_block: dict[str, Any] = {
        "luma": {
            "mean": _round(float(np.mean(wall_luma_mean))),
            "std": _round(float(np.std(wall_luma_mean))),
        },
        "rmsContrast": _round(float(np.mean(wall_rms))),
        "texture": {
            "edgeDensity": _round(float(np.mean(edge_density))),
            "laplacianVar": _round(float(np.mean(laplacian_var)), 2),
        },
        "dominantColors": _dominant_colors(pooled_pixels) if len(pooled_pixels) else [],
    }
    if len(pooled_pixels):
        wall_block.update(_hue_saturation(pooled_pixels))

    return {
        "sampledFrames": len(frames_bgr),
        "panning": bool(panning),
        # Wall aggregates assume a static wall crop; under panning they mix
        # off-crop content, so they are recorded but flagged.
        "panningFlagged": bool(panning),
        "wall": wall_block,
        "climberWall": climber_wall,
        "shadow": shadow_block,
    }


# --------------------------------------------------------------------------- #
# Suggested hand labels (suggest + verify)
# --------------------------------------------------------------------------- #

# Corpus-fit constants — fit by scripts/fit_suggestion_thresholds.py against the
# backfilled stats + existing hand labels. None until the first fit lands;
# suggestions do not ship before thresholds are fit (issue #23).
SUGGESTION_THRESHOLDS: dict[str, Any] | None = None


def _get(d: dict | None, *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def suggest_labels(
    source_stats: dict[str, Any] | None,
    region_stats: dict[str, Any] | None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Derive suggested categorical labels from continuous stats via thresholds.

    Returns only the labels whose driving stat and thresholds are both available;
    an empty dict when thresholds have not been fit. Values use the established
    ``analysisInputs`` vocabularies.
    """
    thresholds = thresholds if thresholds is not None else SUGGESTION_THRESHOLDS
    if not thresholds:
        return {}

    out: dict[str, str] = {}

    def _band(value: float | None, cfg: dict | None, low_high: tuple[str, str, str]) -> str | None:
        """Two-cut ordinal banding: value < lo -> first, > hi -> last, else middle."""
        if value is None or not cfg:
            return None
        lo, hi = cfg.get("low"), cfg.get("high")
        if lo is None or hi is None:
            return None
        return low_high[0] if value < lo else low_high[2] if value > hi else low_high[1]

    # shadows: none / solid / patchy from shadow fraction + blob structure.
    cfg = thresholds.get("shadows")
    frac = _get(region_stats, "shadow", "fraction", "mean")
    if cfg and frac is not None:
        if frac < cfg.get("noneMaxFraction", 0.0):
            out["shadows"] = "none"
        else:
            largest = _get(region_stats, "shadow", "blobs", "largestFraction") or 0.0
            out["shadows"] = (
                "solid" if largest >= cfg.get("solidMinLargestBlobFraction", 1.1) else "patchy"
            )

    val = _band(
        _get(region_stats, "climberWall", "deltaE"),
        thresholds.get("climber_contrast"),
        ("low", "medium", "high"),
    )
    if val:
        out["climber_contrast"] = val

    val = _band(
        _get(region_stats, "wall", "rmsContrast"),
        thresholds.get("wall_contrast"),
        ("low", "medium", "high"),
    )
    if val:
        out["wall_contrast"] = val

    # motion blur: LOW sharpness = HIGH blur, so the band inverts.
    val = _band(
        _get(source_stats, "sharpness", "mean"),
        thresholds.get("motion_blur"),
        ("high", "medium", "low"),
    )
    if val:
        out["motion_blur"] = val

    cfg = thresholds.get("camera_stability")
    diff = _get(source_stats, "frameDiff", "mean")
    if cfg and diff is not None and cfg.get("movingMinFrameDiff") is not None:
        out["camera_stability"] = (
            "moving" if diff >= cfg["movingMinFrameDiff"] else "steady"
        )

    return out


# --------------------------------------------------------------------------- #
# Camera viewing angle from ViTPose keypoint geometry
# --------------------------------------------------------------------------- #

_ANGLE_MIN_SCORE = 0.5
_ANGLE_MIN_FRAMES = 3
# Median shoulder/hip width ratio bands. A camera above the climber foreshortens
# and shrinks the (farther) hips -> ratio grows; below, shoulders shrink -> falls.
_ANGLE_HIGH_MIN_RATIO = 1.9
_ANGLE_LOW_MAX_RATIO = 0.95


def estimate_camera_angle(frames: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Viewing-angle estimate from ViTPose scaffold frames (keypoints in, angle out).

    Uses torso foreshortening: per usable frame, the shoulder-width / hip-width
    ratio and torso-length / shoulder-width aspect; medians across frames classify
    into the ``camera_angle`` hand-label vocabulary (level / high / low). Returns
    None when fewer than ``_ANGLE_MIN_FRAMES`` frames carry a confident torso.
    """
    ratios: list[float] = []
    aspects: list[float] = []

    for frame in frames:
        kps = {
            kp["name"]: kp
            for kp in frame.get("keypoints", [])
            if kp.get("score", 0.0) >= _ANGLE_MIN_SCORE
        }
        needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if any(name not in kps for name in needed):
            continue
        ls, rs, lh, rh = (np.array([kps[n]["x"], kps[n]["y"]]) for n in needed)
        sw = float(np.linalg.norm(ls - rs))
        hw = float(np.linalg.norm(lh - rh))
        tl = float(np.linalg.norm((ls + rs) / 2.0 - (lh + rh) / 2.0))
        if sw <= 1e-6 or hw <= 1e-6:
            continue
        ratios.append(sw / hw)
        aspects.append(tl / sw)

    if len(ratios) < _ANGLE_MIN_FRAMES:
        return None

    ratio = float(np.median(ratios))
    aspect = float(np.median(aspects))
    estimate = (
        "high" if ratio >= _ANGLE_HIGH_MIN_RATIO
        else "low" if ratio <= _ANGLE_LOW_MAX_RATIO
        else "level"
    )
    return {
        "estimate": estimate,
        "shoulderHipRatio": _round(ratio),
        "torsoAspect": _round(aspect),
        "framesUsed": len(ratios),
    }


# --------------------------------------------------------------------------- #
# Decode adapter (thin, untested) + artifact writers
# --------------------------------------------------------------------------- #

def sample_video_frames(
    video_path: Path, sample_hz: float = SAMPLE_HZ, max_samples: int = MAX_SAMPLES
) -> tuple[list[np.ndarray], list[float]]:
    """Uniformly sample frames in one decode pass -> (frames_bgr, timestamps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            # Unreliable header: fall back to decoding everything, then decimate.
            frames, timestamps = [], []
            t = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                timestamps.append(t)
                t += 1.0 / 30.0
            step = max(len(frames) // max_samples, 1)
            return frames[::step][:max_samples], timestamps[::step][:max_samples]

        duration = total / fps
        n = min(max(int(duration * sample_hz), 1), max_samples, total)
        wanted = {int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)}
        frames, timestamps = [], []
        index = 0
        while index <= max(wanted):
            if not cap.grab():
                break
            if index in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame)
                    timestamps.append(index / fps)
            index += 1
        if not frames:
            raise RuntimeError(f"Decoded no frames from: {video_path}")
        return frames, timestamps
    finally:
        cap.release()


def build_source_stats_block(
    video_path: Path,
    source_video: dict[str, Any] | None = None,
    frames: Sequence[np.ndarray] | None = None,
    timestamps: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the full phase-1 block stored in metadata.json.

    Decodes the video unless the caller passes already-sampled frames (the
    endpoint reuses its phase-2 decode).
    """
    if frames is None or timestamps is None:
        frames, timestamps = sample_video_frames(video_path)
    block = compute_source_stats(frames, timestamps)
    sv = source_video or {}
    block["bitsPerPixel"] = bits_per_pixel(
        sv.get("filesize"), sv.get("width"), sv.get("height"),
        sv.get("fps"), sv.get("duration_seconds"),
    )
    block["version"] = VIDEO_STATS_VERSION
    block["computedAt"] = datetime.now().isoformat(timespec="seconds")
    return block


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def write_source_stats(bundle_dir: Path, block: dict[str, Any]) -> Path:
    """Merge the phase-1 block into the bundle's metadata.json (``video_stats``)."""
    metadata_path = bundle_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    metadata["video_stats"] = block
    _dump_json(metadata_path, metadata)
    return metadata_path


def write_region_stats(
    bundle_dir: Path,
    region_stats: dict[str, Any],
    suggestions: dict[str, str],
    setup_hash: str | None,
    source: str = "endpoint",
) -> Path:
    """Write/overwrite the phase-2 artifact, preserving any cameraAngle block.

    The top-level ``setupHash`` is the provenance anchor for ``regionStats`` —
    recalibration re-POSTs and overwrites, exactly like Ground Truth (ADR 0004).
    """
    path = bundle_dir / VIDEO_STATS_NAME
    doc: dict[str, Any] = {}
    if path.exists():
        try:
            doc = _load_json(path)
        except (OSError, ValueError):
            doc = {}
    doc["version"] = VIDEO_STATS_VERSION
    if setup_hash:
        doc["setupHash"] = setup_hash
    else:
        doc.pop("setupHash", None)
    doc["computedAt"] = datetime.now().isoformat(timespec="seconds")
    doc["source"] = source
    doc["regionStats"] = region_stats
    doc["suggestions"] = suggestions
    _dump_json(path, doc)
    return path


def write_camera_angle(bundle_dir: Path, camera_angle: dict[str, Any]) -> Path:
    """Merge the ViTPose-derived viewing-angle block into video-stats.json.

    Creates a minimal artifact when phase 2 hasn't run yet. The block carries its
    own ``setupHash``/``source`` provenance, independent of the region stats'.
    """
    path = bundle_dir / VIDEO_STATS_NAME
    doc: dict[str, Any] = {"version": VIDEO_STATS_VERSION}
    if path.exists():
        try:
            doc = _load_json(path)
        except (OSError, ValueError):
            pass
    doc.setdefault("version", VIDEO_STATS_VERSION)
    doc["cameraAngle"] = camera_angle
    _dump_json(path, doc)
    return path
