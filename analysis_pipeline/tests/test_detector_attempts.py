from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from analysis_pipeline.discovery import discover_runs
from analysis_pipeline.frames import build_frame_table


def _kp(name: str = "nose", x: float = 0.5, y: float = 0.5) -> dict:
    return {"name": name, "x": x, "y": y, "score": 0.9}


def _write_bundle(root: Path, *, detector_attempts: list[dict] | None = None) -> Path:
    video_dir = root / "routeA" / "vidA"
    det = video_dir / "detections"
    det.mkdir(parents=True)
    cfg = {"frameStep": 1, "frameIntervalMs": 100}

    frames = [
        {
            "timestamp": round(i * 0.1, 1),
            "source": "raw",
            "keypoints": [_kp(x=0.1 + i)],
        }
        for i in range(4)
    ]
    data: dict = {
        "setupHash": "setup-a",
        "diagnostics": {
            "videoHash": "video-a",
            "config": cfg,
            "input": {"video": {"durationSec": 0.31}},
            "result": {"pose": {"sampledFrames": 4}},
        },
        "frames": frames,
    }
    if detector_attempts is not None:
        data["detectorAttempts"] = detector_attempts

    pose = {
        "video_key": "vidA",
        "route_folder": "routeA",
        "run_ts": "20260101-000001",
        "written_at": "2026-01-01T00:00:00",
        "type": "pose",
        "data": data,
    }
    orb = {
        "video_key": "vidA",
        "run_ts": "20260101-000001",
        "type": "orb",
        "data": {"summary": {}},
    }
    (det / "20260101-000001_pose.json").write_text(json.dumps(pose), encoding="utf-8")
    (det / "20260101-000001_orb.json").write_text(json.dumps(orb), encoding="utf-8")
    (video_dir / "metadata.json").write_text(
        json.dumps({"route_folder": "routeA", "video_key": "vidA"}), encoding="utf-8"
    )
    (video_dir / "setup.json").write_text(json.dumps({"setupHash": "setup-a"}), encoding="utf-8")
    return video_dir


def test_detector_attempts_are_loaded_and_preferred_over_frames():
    full_frame = {"x": 0, "y": 0, "w": 1, "h": 1}
    crop = {"x": 0.12, "y": 0.23, "w": 0.34, "h": 0.45}
    raw = [_kp("nose", 0.41, 0.52)]
    accepted = [_kp("nose", 0.43, 0.54)]
    attempts = [
        {
            "timestamp": 0.0,
            "status": "accepted",
            "initialSearchRegion": crop,
            "detectionRegion": crop,
            "reacquireAttempted": False,
            "reacquired": False,
            "rawKeypoints": raw,
            "acceptedKeypoints": accepted,
            "searchConditions": {"mean": 70, "stdDev": 12, "sharpness": 99},
            "reacquireConditions": None,
            "candidateCount": 2,
            "rejectedCandidateCount": 1,
            "selectionMethod": "tracked",
        },
        {
            "timestamp": 0.1,
            "status": "missing",
            "initialSearchRegion": crop,
            "detectionRegion": None,
            "reacquireAttempted": True,
            "reacquired": False,
            "rawKeypoints": [],
            "acceptedKeypoints": [],
        },
        {
            "timestamp": 0.2,
            "status": "flipRejected",
            "initialSearchRegion": crop,
            "detectionRegion": full_frame,
            "reacquireAttempted": True,
            "reacquired": False,
            "rawKeypoints": raw,
            "acceptedKeypoints": [],
        },
        {
            "timestamp": 0.3,
            "status": "qualityRejected",
            "initialSearchRegion": None,
            "detectionRegion": full_frame,
            "reacquireAttempted": False,
            "reacquired": False,
            "rawKeypoints": raw,
            "acceptedKeypoints": [],
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        _write_bundle(Path(tmp), detector_attempts=attempts)
        records = discover_runs(Path(tmp))

        assert len(records) == 1
        rec = records[0]
        assert rec.detector_attempt_evidence == "attempts"
        assert rec.detector_attempts is not None
        assert [a["status"] for a in rec.detector_attempts] == [
            "accepted",
            "missing",
            "flipRejected",
            "qualityRejected",
        ]
        assert rec.detector_attempts[0]["rawKeypoints"] == raw
        assert rec.detector_attempts[0]["acceptedKeypoints"] == accepted
        assert (
            rec.detector_attempts[0]["rawKeypoints"]
            is not rec.detector_attempts[0]["acceptedKeypoints"]
        )
        assert rec.detector_attempts[2]["detectionRegion"] == full_frame
        assert rec.detector_attempts[3]["initialSearchRegion"] is None

        frame_df = build_frame_table(records, decode=False).sort_values("t")
        assert list(frame_df["detector_attempt_status"]) == [
            "accepted",
            "missing",
            "flipRejected",
            "qualityRejected",
        ]
        assert list(frame_df["source"]) == [
            "raw",
            "missing",
            "flipRejected",
            "qualityRejected",
        ]
        assert list(frame_df["raw_detected"]) == [1.0, 0.0, 0.0, 0.0]
        assert frame_df.iloc[0]["raw_keypoints"] == raw
        assert frame_df.iloc[0]["accepted_keypoints"] == accepted
        assert frame_df.iloc[2]["detection_region"] == full_frame
        assert frame_df.iloc[0]["initial_search_region"] == crop
        assert frame_df.iloc[0]["climber_luma_mean"] == 70.0
        assert frame_df.iloc[0]["candidate_count"] == 2


def test_legacy_frame_only_runs_have_unknown_detector_attempt_evidence():
    with tempfile.TemporaryDirectory() as tmp:
        _write_bundle(Path(tmp), detector_attempts=None)
        records = discover_runs(Path(tmp))

        rec = records[0]
        assert rec.detector_attempts is None
        assert rec.detector_attempt_evidence == "unknown"

        frame_df = build_frame_table(records, decode=False)
        assert frame_df["detector_attempt_evidence"].eq("unknown").all()
        assert frame_df["detector_attempt_status"].isna().all()
        assert frame_df["source"].eq("raw").all()
        assert frame_df["raw_detected"].sum() == 4
        assert isinstance(frame_df, pd.DataFrame)
