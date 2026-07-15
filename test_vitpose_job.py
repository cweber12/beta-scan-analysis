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
    _require_scipy,
    build_artifact,
    build_climber_track,
    resolve_video_path,
    run_vitpose_job,
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
# Climber Identity — trajectory stitching
# --------------------------------------------------------------------------- #

def test_climber_track_stitches_across_bytetrack_id_switch():
    # THE regression for the real bug: ByteTrack fragments the climber into id 1
    # (at the base) then id 9 (as they ascend); a spotter (id 2) sits still at the
    # base. Seeding from the base tap must follow the climber UP across the id
    # switch, covering every frame — and never latch onto the stationary spotter.
    def climber(y):  # ascends: box top-left y decreases over time
        return Box(0.45, y, 0.10, 0.15)
    spot = Box(0.05, 0.80, 0.08, 0.15)   # cx ~0.09, far from the climber (cx 0.50)
    history = [
        FrameTracks(0.0, {1: climber(0.80), 2: spot}),   # climber at base = id 1
        FrameTracks(0.1, {1: climber(0.72), 2: spot}),
        FrameTracks(0.2, {9: climber(0.64), 2: spot}),   # id switch 1 -> 9
        FrameTracks(0.3, {9: climber(0.56), 2: spot}),
        FrameTracks(0.4, {9: climber(0.48), 2: spot}),
    ]
    tap = Point(0.50, 0.85)  # on the climber at the base (inside id 1 on frame 0)
    traj = build_climber_track(history, tap, None)
    assert set(traj.keys()) == {0, 1, 2, 3, 4}          # every frame covered
    assert all(b.cx > 0.3 for b in traj.values())       # the climber, not the spotter


def test_climber_track_seeds_nearest_when_no_containment():
    a0 = Box(0.40, 0.40, 0.10, 0.10)
    a1 = Box(0.42, 0.42, 0.10, 0.10)
    history = [FrameTracks(0.0, {3: a0}), FrameTracks(0.1, {3: a1})]
    tap = Point(0.90, 0.90)  # inside nobody; seed on the nearest box, then follow it
    assert set(build_climber_track(history, tap, None).keys()) == {0, 1}


def test_climber_track_no_tap_seeds_persistent_over_brief_closeup():
    # A huge one-frame close-up (id 8) must not be seeded over the persistent
    # climber (id 5); the trajectory should follow the climber across the clip.
    big_closeup = Box(0.1, 0.1, 0.7, 0.7)     # area 0.49, single frame
    climber = Box(0.45, 0.40, 0.12, 0.30)     # cx 0.51, every frame
    history = [FrameTracks(0.0, {8: big_closeup, 5: climber})]
    history += [FrameTracks(0.1 * i, {5: climber}) for i in range(1, 30)]
    traj = build_climber_track(history, None, None)
    assert len(traj) == 30
    assert all(abs(b.cx - 0.51) < 1e-9 for b in traj.values())


def test_climber_track_empty_when_no_tracks():
    history = [FrameTracks(0.0, {}), FrameTracks(0.5, {})]
    assert build_climber_track(history, Point(0.5, 0.5), None) == {}


def test_climber_track_association_slack_scales_with_source_frame_gap():
    # A strided tracker history: consecutive entries are 2 source frames apart, and
    # the climber moves 0.13 per entry — beyond the unscaled consecutive threshold
    # (0.08 + 0.04 = 0.12) but within the frame-gap-scaled one (0.08 + 0.04*2 = 0.16).
    # Stitching must use the source-frame gap, not the history-index gap.
    history = [
        FrameTracks(i * 2 / 24.0, {1: Box(0.10 + 0.13 * i, 0.40, 0.10, 0.15)}, frame_number=i * 2)
        for i in range(5)
    ]
    tap = Point(0.15, 0.45)  # on the climber in the first entry
    traj = build_climber_track(history, tap, None)
    assert set(traj.keys()) == {0, 1, 2, 3, 4}


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


def test_artifact_stamps_setup_hash_from_request():
    # The provenance anchor: the stamped setupHash equals the request's setup_hash.
    history = _history_two_people()
    req = VitPoseRequest(
        video_path="analysis/r/k/k.mp4", route_folder="r", video_key="k",
        frames=(0.0,), climber_point=Point(0.50, 0.40), setup_hash="abc123",
    )
    art = build_artifact(req, history, StubPoseBackend(), Path("x.mp4"))
    assert art["setupHash"] == "abc123"


def test_artifact_omits_setup_hash_when_request_has_none():
    # No setup hash known -> the key is omitted (not null), so consumers'
    # artifact.get("setupHash", <fallback>) reaches the legacy fallback.
    art = build_artifact(_request([0.0]), _history_two_people(), StubPoseBackend(), Path("x.mp4"))
    assert "setupHash" not in art


def test_targets_carry_source_frame_numbers_from_strided_history():
    # A strided history: entry i is source frame i*2. The PoseTarget handed to the
    # backend must carry that source frame number (it's what the backend seeks),
    # keyed by the history index the artifact assembly uses.
    history = [
        FrameTracks(i * 2 / 24.0, {1: Box(0.45, 0.40, 0.10, 0.15)}, frame_number=i * 2)
        for i in range(3)
    ]
    backend = StubPoseBackend()
    build_artifact(_request([2 / 24.0]), history, backend, Path("x.mp4"))
    assert len(backend.seen_targets) == 1
    target = backend.seen_targets[0]
    assert target.frame_index == 1
    assert target.frame_number == 2
    assert target.source_frame == 2


def test_pose_target_source_frame_falls_back_to_history_index():
    # Legacy/stub histories without frame numbers: source frame == history index.
    target = PoseTarget(frame_index=7, box=Box(0.1, 0.1, 0.2, 0.2))
    assert target.frame_number is None
    assert target.source_frame == 7


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
        # Phase timings ride in the done sidecar (contract-safe extra keys).
        assert set(status["timings"]) == {"track_s", "pose_s", "total_s"}
        assert all(v >= 0 for v in status["timings"].values())
        # The scored-run invariant: no detections/ pose|orb file was written.
        assert not (path.parent / "detections").exists()


def test_run_job_stamps_request_setup_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)
        # A stale setup.json in the bundle must NOT override the request's hash.
        (root / "r" / "k" / "setup.json").write_text('{"setupHash": "stale"}')
        req = VitPoseRequest(
            video_path="analysis/r/k/k.mp4", route_folder="r", video_key="k",
            frames=(0.0,), climber_point=Point(0.50, 0.40), setup_hash="fresh",
        )
        path = run_vitpose_job(root, req, StubTracker(_history_two_people()), StubPoseBackend())
        assert json.loads(path.read_text())["setupHash"] == "fresh"


def test_run_job_falls_back_to_bundle_setup_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)
        (root / "r" / "k" / "setup.json").write_text('{"setupHash": "from-setup"}')
        # Request carries no setup_hash -> stamp the bundle's current setup.json.
        path = run_vitpose_job(root, _request([0.0]), StubTracker(_history_two_people()), StubPoseBackend())
        assert json.loads(path.read_text())["setupHash"] == "from-setup"


def test_run_job_omits_setup_hash_when_unresolvable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)  # no setup.json written
        path = run_vitpose_job(root, _request([0.0]), StubTracker(_history_two_people()), StubPoseBackend())
        assert "setupHash" not in json.loads(path.read_text())


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


def test_run_job_poses_climber_for_nonpanning_point_and_crop_setup():
    # Regression for issue #3: the maze-of-death iewxlKJNhC8 bundle is a NON-panning
    # setup carrying BOTH a climber tap and a climber crop. That shape reached the
    # real pose backend, which then crashed inside transformers' scipy_warp_affine
    # ("NameError: name 'inv' is not defined"). Drive that exact request shape through
    # the job with a stub backend to lock in that a climber IS selected and the pose
    # backend IS invoked -- the code path that led to the crash -- and that the run
    # ends in a `done` artifact with posed keypoints.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _make_bundle(root)
        backend = StubPoseBackend()
        req = VitPoseRequest(
            video_path="analysis/r/k/k.mp4", route_folder="r", video_key="k",
            frames=(0.0, 0.5, 1.0),
            climber_point=Point(0.51, 0.35),      # taps the climber (track 1)
            climber_crop=Box(0.40, 0.20, 0.25, 0.40),
            panning=False,
        )
        path = run_vitpose_job(root, req, StubTracker(_history_two_people()), backend)

        status = json.loads((path.parent / "vitpose.status.json").read_text())
        assert status["status"] == "done"
        # The climber was selected and handed to the pose backend (the crashing path).
        assert backend.seen_targets
        art = json.loads(path.read_text())
        assert any(f["keypoints"] for f in art["frames"])


# --------------------------------------------------------------------------- #
# scipy preflight (the real root cause: a missing scipy dies deep in transformers
# with a cryptic NameError 'inv'; the guard turns that into an actionable message)
# --------------------------------------------------------------------------- #

def test_require_scipy_raises_actionable_error_when_missing():
    try:
        _require_scipy(False)
    except ImportError as exc:
        assert "scipy" in str(exc).lower()
        assert "requirements.txt" in str(exc)
        return
    raise AssertionError("expected ImportError when scipy is unavailable")


def test_require_scipy_noop_when_available():
    _require_scipy(True)  # scipy present -> the pose path proceeds, no raise


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
