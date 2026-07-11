"""Smoke tests: discovery + dedup + label pruning + stats, on a synthetic bundle.

No cv2 decode needed (build_frame_table is called with decode=False and stub frames).
Runnable with pytest, or standalone: ``python -m analysis_pipeline.tests.test_smoke``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from analysis_pipeline import stats
from analysis_pipeline.discovery import discover_runs
from analysis_pipeline.frames import build_frame_table
from analysis_pipeline.runs import build_run_table


def _write_run(video_dir: Path, stem: str, *, video_hash: str, setup_hash: str,
               config: dict, labels: dict, det_rate: float, written_at: str) -> None:
    det = video_dir / "detections"
    det.mkdir(parents=True, exist_ok=True)
    frames = [
        {"timestamp": round(i * 1.0, 1),
         "keypoints": [{"name": "nose", "x": 0.5, "y": 0.5, "score": 0.9},
                       {"name": "left_shoulder", "x": 0.4, "y": 0.6, "score": 0.8}]}
        for i in range(4)
    ]
    pose = {
        "video_key": video_dir.name, "route_folder": video_dir.parent.name,
        "run_ts": stem, "written_at": written_at, "type": "pose",
        "data": {
            "setupHash": setup_hash,
            "diagnostics": {
                "videoHash": video_hash, "config": config,
                "input": {"video": {"durationSec": 3.0},
                          "referenceFrame": {"wall": {"sharpness": 100.0, "mean": 80.0, "stdDev": 20.0}},
                          "motionMagnitude": 0.03,
                          "climberFrameCoverage": {"avg": 0.05, "min": 0.01}},
                "result": {"pose": {"sampledFrames": 4, "detectedFrames": int(det_rate * 4),
                                    "detectionRate": det_rate, "flippedFrames": 0,
                                    "goodFrames": 3, "confidence": {"avg": 0.88, "min": 0.7},
                                    "avgKeypointCount": 20.0}, "badStretches": []},
            },
            "frames": frames,
        },
    }
    orb = {"video_key": video_dir.name, "run_ts": stem, "type": "orb",
           "data": {"referenceFrameMeta": {"refKeypointCount": 2000,
                                           "wall": {"sharpness": 100.0}}, "summary": {}}}
    (det / f"{stem}_pose.json").write_text(json.dumps(pose), encoding="utf-8")
    (det / f"{stem}_orb.json").write_text(json.dumps(orb), encoding="utf-8")
    md = {"route_folder": video_dir.parent.name, "video_key": video_dir.name,
          "analysis_inputs": labels}
    (video_dir / "metadata.json").write_text(json.dumps(md), encoding="utf-8")
    (video_dir / "setup.json").write_text(
        json.dumps({"climberCrop": {"x": 0.4, "y": 0.5, "w": 0.1, "h": 0.4},
                    "wallCrop": {"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.6},
                    "setupHash": setup_hash}), encoding="utf-8")


def _build_corpus(root: Path) -> None:
    cfg = {"frameStep": 10, "frameIntervalMs": 100}
    base_labels = {"route_orientation": "head-on", "camera_angle": "level",
                   "shadows": "high", "climber_contrast": "low", "wall_contrast": "medium",
                   "motion_blur": "low", "occlusion": "unknown", "camera_stability": "steady"}
    # video A: THREE identical re-runs (must collapse to one)
    a = root / "routeA" / "vidA"
    for i, ts in enumerate(("20260101-000001", "20260101-000002", "20260101-000003")):
        _write_run(a, ts, video_hash="ha", setup_hash="sa", config=cfg,
                   labels={**base_labels, "route_orientation": "left"},
                   det_rate=0.66, written_at=f"2026-01-01T00:0{i}:00")
    # video B, C: distinct
    _write_run(root / "routeB" / "vidB", "20260102-000001", video_hash="hb", setup_hash="sb",
               config=cfg, labels={**base_labels, "route_orientation": "head-on"},
               det_rate=1.0, written_at="2026-01-02T00:00:00")
    _write_run(root / "routeC" / "vidC", "20260103-000001", video_hash="hc", setup_hash="sc",
               config=cfg, labels={**base_labels, "route_orientation": "right",
                                   "camera_stability": "moving"},
               det_rate=0.9, written_at="2026-01-03T00:00:00")


def test_discovery_dedup_prune_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_corpus(root)

        records = discover_runs(root)
        # 3 identical vidA runs collapse to 1; vidB, vidC distinct -> 3 total
        assert len(records) == 3, f"expected 3 deduped runs, got {len(records)}"
        assert sorted(r.video_key for r in records) == ["vidA", "vidB", "vidC"]

        run_df = build_run_table(records)
        assert len(run_df) == 3
        assert run_df["orb_refKeypointCount"].notna().all()

        kept, dropped = stats.prune_labels(run_df)
        dropped_names = {c for c, _ in dropped}
        # camera_angle constant, occlusion 100% unknown -> dropped
        assert "label_camera_angle" in dropped_names
        assert "label_occlusion" in dropped_names
        # route_orientation varies (left/head-on/right) -> kept
        assert "label_route_orientation" in kept

        frame_df = build_frame_table(records, decode=False)
        # 4 samples per run (duration 3.0s, 1.0s interval -> t=0,1,2,3)
        assert len(frame_df) == 3 * 4
        assert frame_df["kp_count"].eq(2).all()

        corr = stats.within_run_correlations(frame_df)
        assert set(["predictor", "outcome", "mean_r", "n_runs"]).issubset(corr.columns) or corr.empty
        # velocity/coverage are constant here -> may be empty; ensure it runs without error
        assert isinstance(corr, pd.DataFrame)


def test_cliffs_delta_bounds():
    assert stats.cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    assert stats.cliffs_delta([1, 2], [3, 4, 5]) == -1.0
    assert stats.cliffs_delta([2, 2], [2, 2]) == 0.0
    assert stats.cliffs_delta([], [1]) is None


def _run_all():
    fns = [test_discovery_dedup_prune_and_stats, test_cliffs_delta_bounds]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all smoke tests passed")


if __name__ == "__main__":
    _run_all()
