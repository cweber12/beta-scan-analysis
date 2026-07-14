"""Tests for the ViTPose++ Ground Truth scaffold job.

Exercises the pure geometry (Climber selection, timestamp echo, coord clamping,
artifact shape, path safety, status sidecar) with stub Tracker/PoseBackend
collaborators — no torch/transformers/ultralytics required.

Runnable with pytest, or standalone: ``python test_vitpose_job.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import vitpose_job as vj
from vitpose_job import (
    Box,
    FrameTracks,
    Keypoint,
    Point,
    PoseTarget,
    VitPoseRequest,
    build_artifact,
    resolve_video_path,
    run_vitpose_job,
    select_climber_track,
)


# --------------------------------------------------------------------------- #
# Stub collaborators
# --------------------------------------------------------------------------- #

class StubTracker:
    def __init__(self, history):
        self.history = history

    def track(self, video_path):
        return self.history


class StubPoseBackend:
    """Poses each target's box with a single 'nose' kp at the box center."""

    def __init__(self):
        self.seen_targets = []

    def pose(self, video_path, targets):
        self.seen_targets = list(targets)
        out = {}
        for t in targets:
            out[t.frame_index] = [
                Keypoint("nose", t.box.cx, t.box.cy, 0.9),
                # Deliberately out-of-range to prove clamping.
                Keypoint("left_wrist", 1.4, -0.2, 1.7),
            ]
        return out


def _history_two_people():
    # Track 1 = climber (upper region), track 2 = spotter (lower-left, smaller).
    return [
        FrameTracks(0.0, {1: Box(0.45, 0.30, 0.12, 0.30), 2: Box(0.05, 0.70, 0.06, 0.15)}),
        FrameTracks(0.5, {1: Box(0.46, 0.25, 0.12, 0.30), 2: Box(0.05, 0.70, 0.06, 0.15)}),
        FrameTracks(1.0, {1: Box(0.47, 0.20, 0.12, 0.30)}),  # spotter left the frame
    ]


# --------------------------------------------------------------------------- #
# Climber selection
# --------------------------------------------------------------------------- #

def test_select_climber_by_containment():
    history = _history_two_people()
    tap = Point(0.50, 0.40)  # inside track 1's box on frame 0
    assert select_climber_track(history, tap, None) == 1


def test_select_climber_nearest_when_no_containment():
    history = _history_two_people()
    tap = Point(0.05, 0.95)  # inside nobody; nearest the spotter (track 2, lower-left)
    assert select_climber_track(history, tap, None) == 2


def test_select_climber_no_tap_largest_in_crop():
    history = _history_two_people()
    crop = Box(0.3, 0.1, 0.5, 0.6)  # contains the climber's center, not the spotter's
    assert select_climber_track(history, None, crop) == 1


def test_select_climber_none_when_no_tracks():
    history = [FrameTracks(0.0, {}), FrameTracks(0.5, {})]
    assert select_climber_track(history, Point(0.5, 0.5), None) is None


# --------------------------------------------------------------------------- #
# Artifact assembly
# --------------------------------------------------------------------------- #

def _request(frames, tap=Point(0.50, 0.40)):
    return VitPoseRequest(
        video_path="analysis/r/k/k.mp4", route_folder="r", video_key="k",
        frames=tuple(frames), climber_point=tap,
    )


def test_artifact_echoes_timestamps_verbatim_and_in_order():
    history = _history_two_people()
    backend = StubPoseBackend()
    # Odd values that must survive untouched; not equal to any decoder frame time.
    requested = [0.4667, 0.0, 0.9333]
    art = build_artifact(_request(requested), history, backend, Path("x.mp4"))

    assert art["version"] == 1
    assert [f["timestamp"] for f in art["frames"]] == requested


def test_artifact_clamps_coords_to_unit_range():
    history = _history_two_people()
    art = build_artifact(_request([0.0]), history, StubPoseBackend(), Path("x.mp4"))
    kps = {kp["name"]: kp for kp in art["frames"][0]["keypoints"]}
    assert 0.0 <= kps["nose"]["x"] <= 1.0 and 0.0 <= kps["nose"]["y"] <= 1.0
    # left_wrist was posed at (1.4, -0.2, 1.7) -> clamped.
    assert kps["left_wrist"] == {"name": "left_wrist", "x": 1.0, "y": 0.0, "score": 1.0}


def test_artifact_empty_keypoints_when_climber_untracked():
    history = _history_two_people()  # climber (track 1) present on every frame here
    # Force selection to the spotter, who is ABSENT on the 1.0s frame.
    req = VitPoseRequest(
        video_path="a", route_folder="r", video_key="k",
        frames=(0.0, 1.0), climber_point=Point(0.06, 0.75),  # taps the spotter
    )
    art = build_artifact(req, history, StubPoseBackend(), Path("x.mp4"))
    by_ts = {f["timestamp"]: f["keypoints"] for f in art["frames"]}
    assert by_ts[0.0]  # spotter tracked at 0.0 -> posed
    assert by_ts[1.0] == []  # spotter gone at 1.0 -> absent


def test_artifact_all_empty_when_no_climber():
    history = [FrameTracks(0.0, {}), FrameTracks(0.5, {})]
    backend = StubPoseBackend()
    art = build_artifact(_request([0.0, 0.5]), history, backend, Path("x.mp4"))
    assert all(f["keypoints"] == [] for f in art["frames"])
    assert backend.seen_targets == []  # backend never invoked with no targets


def test_two_timestamps_nearest_same_frame_are_both_posed():
    history = _history_two_people()  # decoded times 0.0, 0.5, 1.0
    backend = StubPoseBackend()
    # 0.48 and 0.52 are both nearest the 0.5s frame (index 1): both echoed, one pose.
    art = build_artifact(_request([0.48, 0.52]), history, backend, Path("x.mp4"))
    assert [f["timestamp"] for f in art["frames"]] == [0.48, 0.52]
    assert all(f["keypoints"] for f in art["frames"])
    assert len(backend.seen_targets) == 1  # the shared frame is posed only once


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #

def test_resolve_video_path_accepts_relative_and_analysis_prefixed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vid = root / "r" / "k" / "k.mp4"
        vid.parent.mkdir(parents=True)
        vid.write_bytes(b"x")
        assert resolve_video_path(root, "analysis/r/k/k.mp4") == vid.resolve()
        assert resolve_video_path(root, "r/k/k.mp4") == vid.resolve()


def test_resolve_video_path_rejects_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        root.mkdir()
        try:
            resolve_video_path(root, "../../etc/passwd")
        except ValueError:
            return
        raise AssertionError("expected traversal to be rejected")


# --------------------------------------------------------------------------- #
# End-to-end job (stub backends)
# --------------------------------------------------------------------------- #

def _make_bundle(root: Path):
    vid = root / "r" / "k" / "k.mp4"
    vid.parent.mkdir(parents=True)
    vid.write_bytes(b"x" * 200_000)
    return vid


def test_run_job_writes_artifact_and_done_status():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)
        req = _request([0.0, 1.0])
        path = run_vitpose_job(root, req, StubTracker(_history_two_people()), StubPoseBackend())

        assert path.name == "vitpose.json"
        art = json.loads(path.read_text())
        assert [f["timestamp"] for f in art["frames"]] == [0.0, 1.0]

        status = json.loads((path.parent / "vitpose.status.json").read_text())
        assert status["status"] == "done"
        # The scored-run invariant: no detections/ pose|orb file was written.
        assert not (path.parent / "detections").exists()


def test_run_job_missing_bundle_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        root.mkdir()
        try:
            run_vitpose_job(root, _request([0.0]), StubTracker([]), StubPoseBackend())
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for missing bundle")


def test_run_job_records_error_status_on_failure():
    class BoomTracker:
        def track(self, video_path):
            raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)
        try:
            run_vitpose_job(root, _request([0.0]), BoomTracker(), StubPoseBackend())
        except RuntimeError:
            pass
        status = json.loads((root / "r" / "k" / "vitpose.status.json").read_text())
        assert status["status"] == "error"
        assert "boom" in status["error"]


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
