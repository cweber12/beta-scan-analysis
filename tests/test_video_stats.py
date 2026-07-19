"""Tests for the Video Stats extraction core (issue #23).

The stats functions are pure frames-in / stats-out, so tests feed synthetic
numpy frames and assert stat values and suggestion *directions* — never internal
intermediates. The video decode adapter stays a thin untested wrapper.

Runnable with pytest, or standalone: ``python -m tests.test_video_stats``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

import video_stats as vs


FULL = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def _flat(luma: int, w: int = 160, h: int = 120) -> np.ndarray:
    return np.full((h, w, 3), luma, dtype=np.uint8)


def _textured(seed: int = 0, w: int = 160, h: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Phase 1 — source stats
# --------------------------------------------------------------------------- #

def test_source_stats_flat_frame_basics():
    frames = [_flat(128)] * 3
    out = vs.compute_source_stats(frames, [0.0, 1.0, 2.0])
    assert abs(out["luma"]["mean"] - 128.0) < 1.5  # BGR->gray rounding
    assert out["clippedHighlightFraction"] == 0.0
    assert out["crushedShadowFraction"] == 0.0
    assert out["rmsContrast"] == 0.0
    assert out["frameDiff"]["mean"] == 0.0
    assert out["sampledFrames"] == 3


def test_source_stats_clipping_fractions():
    frame = _flat(128)
    frame[:, :40] = 255   # 25% blown
    frame[:, 40:56] = 0   # 10% crushed
    out = vs.compute_source_stats([frame], [0.0])
    assert abs(out["clippedHighlightFraction"] - 0.25) < 0.01
    assert abs(out["crushedShadowFraction"] - 0.10) < 0.01


def test_source_stats_sharpness_orders_blur():
    sharp = _textured(1)
    import cv2
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    out_sharp = vs.compute_source_stats([sharp], [0.0])
    out_blur = vs.compute_source_stats([blurred], [0.0])
    assert out_sharp["sharpness"]["mean"] > out_blur["sharpness"]["mean"] * 10


def test_source_stats_jitter_raises_frame_diff():
    base = _textured(2)
    static = [base, base, base]
    jittered = [base, np.roll(base, 4, axis=1), np.roll(base, -4, axis=0)]
    ts = [0.0, 1.0, 2.0]
    still = vs.compute_source_stats(static, ts)["frameDiff"]["mean"]
    shaky = vs.compute_source_stats(jittered, ts)["frameDiff"]["mean"]
    assert still == 0.0
    assert shaky > 0.05


def test_source_stats_exposure_drift_slope():
    frames = [_flat(v) for v in (100, 120, 140)]
    out = vs.compute_source_stats(frames, [0.0, 30.0, 60.0])
    # 40 luma over one minute -> slope ~40/min, range ~40.
    assert 30.0 < out["exposureDrift"]["slopePerMinute"] < 50.0
    assert abs(out["exposureDrift"]["range"] - 40.0) < 3.0


def test_bits_per_pixel():
    # 1 MB, 100x100 px, 10 fps, 10 s -> 8e6 bits / 1e6 px = 8.0
    assert vs.bits_per_pixel(1_000_000, 100, 100, 10, 10) == 8.0
    assert vs.bits_per_pixel(None, 100, 100, 10, 10) is None
    assert vs.bits_per_pixel(1, 0, 100, 10, 10) is None


# --------------------------------------------------------------------------- #
# Phase 2 — region stats
# --------------------------------------------------------------------------- #

def test_shadow_fraction_and_solid_blob():
    wall = _flat(200)
    wall[40:80, 40:120] = 90  # one solid dark blob: 40x80 of 120x160 = 1/6
    out = vs.compute_region_stats([wall], [0.0], FULL)
    frac = out["shadow"]["fraction"]["mean"]
    assert abs(frac - (40 * 80) / (120 * 160)) < 0.02
    assert out["shadow"]["blobs"]["count"] == 1
    assert out["shadow"]["blobs"]["largestFraction"] == 1.0
    assert out["shadow"]["inOutLumaRatio"] < 0.5


def test_shadow_blob_count_distinguishes_patchy():
    wall = _flat(200)
    for x0 in (10, 60, 110):  # three separated dapple blobs
        wall[20:50, x0:x0 + 30] = 90
    out = vs.compute_region_stats([wall], [0.0], FULL)
    assert out["shadow"]["blobs"]["count"] == 3
    assert out["shadow"]["blobs"]["largestFraction"] < 0.5


def test_no_shadow_on_uniform_wall():
    out = vs.compute_region_stats([_flat(180)], [0.0], FULL)
    assert out["shadow"]["fraction"]["mean"] < 0.01
    assert out["shadow"]["blobs"]["count"] == 0


def test_climber_wall_delta_e_direction():
    crop = {"x": 0.25, "y": 0.25, "w": 0.25, "h": 0.25}

    def scene(climber_bgr):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:] = (60, 140, 60)  # greenish wall
        region = vs._crop_region(frame, crop)
        region[:] = climber_bgr
        return frame

    contrasting = scene((40, 40, 220))   # red climber on green wall
    camouflaged = scene((70, 150, 70))   # near-wall green
    hi = vs.compute_region_stats([contrasting], [0.0], FULL, climber_crop=crop)
    lo = vs.compute_region_stats([camouflaged], [0.0], FULL, climber_crop=crop)
    assert hi["climberWall"]["deltaE"] > lo["climberWall"]["deltaE"] * 3
    assert lo["climberWall"]["deltaE"] < 10.0


def test_wall_texture_orders_feature_richness():
    rich = _textured(3)
    poor = _flat(150)
    out_rich = vs.compute_region_stats([rich], [0.0], FULL)
    out_poor = vs.compute_region_stats([poor], [0.0], FULL)
    assert out_rich["wall"]["texture"]["edgeDensity"] > out_poor["wall"]["texture"]["edgeDensity"]
    assert out_rich["wall"]["texture"]["laplacianVar"] > out_poor["wall"]["texture"]["laplacianVar"]
    assert out_poor["wall"]["dominantColors"][0]["fraction"] > 0.5


def test_shadow_drift_across_samples():
    still = _flat(200)
    shadowed = _flat(200)
    shadowed[:, :80] = 90  # shadow arrives, covering half the wall
    out = vs.compute_region_stats([still, shadowed], [0.0, 1.0], FULL)
    assert out["shadow"]["drift"]["firstToLast"] > 0.4
    assert out["shadow"]["drift"]["range"] > 0.4


def test_panning_is_flagged_not_dropped():
    out = vs.compute_region_stats([_flat(150)], [0.0], FULL, panning=True)
    assert out["panningFlagged"] is True
    assert out["wall"]["luma"]["mean"] > 0  # still recorded


# --------------------------------------------------------------------------- #
# Suggestions (directions, with explicit thresholds)
# --------------------------------------------------------------------------- #

_THRESHOLDS = {
    "shadows": {"noneMaxFraction": 0.02, "solidMinLargestBlobFraction": 0.75},
    "climber_contrast": {"low": 15.0, "high": 40.0},
    "wall_contrast": {"low": 0.08, "high": 0.2},
    "motion_blur": {"low": 50.0, "high": 300.0},
    "camera_stability": {"movingMinFrameDiff": 0.03},
}


def test_suggestions_empty_without_fitted_thresholds():
    assert vs.suggest_labels({}, {}, thresholds=None) == {}


def test_suggestion_directions():
    wall_solid = _flat(200)
    wall_solid[30:90, 30:130] = 90
    region_solid = vs.compute_region_stats([wall_solid], [0.0], FULL)
    assert vs.suggest_labels(None, region_solid, _THRESHOLDS)["shadows"] == "solid"

    wall_patchy = _flat(200)
    for x0 in (10, 60, 110):
        wall_patchy[20:50, x0:x0 + 30] = 90
    region_patchy = vs.compute_region_stats([wall_patchy], [0.0], FULL)
    assert vs.suggest_labels(None, region_patchy, _THRESHOLDS)["shadows"] == "patchy"

    region_none = vs.compute_region_stats([_flat(180)], [0.0], FULL)
    assert vs.suggest_labels(None, region_none, _THRESHOLDS)["shadows"] == "none"

    base = _textured(4)
    shaky = vs.compute_source_stats([base, np.roll(base, 6, axis=1)], [0.0, 1.0])
    steady = vs.compute_source_stats([base, base], [0.0, 1.0])
    assert vs.suggest_labels(shaky, None, _THRESHOLDS)["camera_stability"] == "moving"
    assert vs.suggest_labels(steady, None, _THRESHOLDS)["camera_stability"] == "steady"
    # sharp/textured video -> low blur (inverted band)
    assert vs.suggest_labels(steady, None, _THRESHOLDS)["motion_blur"] == "low"


# --------------------------------------------------------------------------- #
# Camera viewing angle from keypoints
# --------------------------------------------------------------------------- #

def _pose_frame(sw: float, hw: float, score: float = 0.9) -> dict:
    cx = 0.5
    return {"timestamp": 0.0, "keypoints": [
        {"name": "left_shoulder", "x": cx - sw / 2, "y": 0.4, "score": score},
        {"name": "right_shoulder", "x": cx + sw / 2, "y": 0.4, "score": score},
        {"name": "left_hip", "x": cx - hw / 2, "y": 0.7, "score": score},
        {"name": "right_hip", "x": cx + hw / 2, "y": 0.7, "score": score},
    ]}


def test_camera_angle_level_high_low():
    level = [_pose_frame(0.20, 0.16)] * 4      # ratio 1.25
    high = [_pose_frame(0.20, 0.08)] * 4       # ratio 2.5 (hips foreshortened)
    low = [_pose_frame(0.08, 0.16)] * 4        # ratio 0.5 (shoulders foreshortened)
    assert vs.estimate_camera_angle(level)["estimate"] == "level"
    assert vs.estimate_camera_angle(high)["estimate"] == "high"
    assert vs.estimate_camera_angle(low)["estimate"] == "low"


def test_camera_angle_needs_confident_torso_frames():
    # Too few usable frames -> None (low scores are filtered out).
    assert vs.estimate_camera_angle([_pose_frame(0.2, 0.16, score=0.1)] * 5) is None
    assert vs.estimate_camera_angle([_pose_frame(0.2, 0.16)] * 2) is None
    assert vs.estimate_camera_angle([{"timestamp": 0.0, "keypoints": []}] * 5) is None


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #

def test_region_write_then_camera_angle_merge():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        vs.write_region_stats(bundle, {"wall": {}}, {"shadows": "none"},
                              setup_hash="sh1", source="endpoint")
        vs.write_camera_angle(bundle, {"estimate": "level", "source": "vitpose",
                                       "setupHash": "sh1"})
        doc = json.loads((bundle / vs.VIDEO_STATS_NAME).read_text(encoding="utf-8"))
        assert doc["setupHash"] == "sh1"
        assert doc["regionStats"] == {"wall": {}}
        assert doc["suggestions"] == {"shadows": "none"}
        assert doc["cameraAngle"]["estimate"] == "level"

        # Recalibration re-POST overwrites region stats but keeps the angle block.
        vs.write_region_stats(bundle, {"wall": {"new": 1}}, {}, setup_hash="sh2")
        doc = json.loads((bundle / vs.VIDEO_STATS_NAME).read_text(encoding="utf-8"))
        assert doc["setupHash"] == "sh2"
        assert doc["regionStats"] == {"wall": {"new": 1}}
        assert doc["cameraAngle"]["estimate"] == "level"


def test_camera_angle_creates_minimal_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        vs.write_camera_angle(bundle, {"estimate": "high", "source": "vitpose"})
        doc = json.loads((bundle / vs.VIDEO_STATS_NAME).read_text(encoding="utf-8"))
        assert doc["version"] == vs.VIDEO_STATS_VERSION
        assert doc["cameraAngle"]["estimate"] == "high"
        assert "regionStats" not in doc


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all video-stats tests passed")


if __name__ == "__main__":
    _run_all()
