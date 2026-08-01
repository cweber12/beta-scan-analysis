"""Stub-backed tests for the crop trajectory (issue #169).

No MediaPipe and no video: the tracker takes a ``probe`` callable, so the entire state
machine — prediction, the search ladder, recovery, the appearance gate, giving up — is
exercised with a stub. Same property that lets ``test_vitpose_job.py`` run without torch.

These target the behaviour that decides whether a run is measurable, not the call sequence.
The recovery tests carry the most weight: a prototype without recovery reached only 59–100%
(median 81%) across six Bundles where full-frame detects 0%, and every missing frame went the
same way — the crop lost the climber and then sat on a stale location forever.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import crop_track as ct
from crop_track import CropTrack, CropTrackConfig, Probe, track, track_hash


def _config(**over) -> CropTrackConfig:
    return replace(CropTrackConfig(), **over)


def _frames(n: int, step: float = 0.5):
    """``n`` frames at ``step`` seconds; the frame payload is just its index."""
    return [(round(i * step, 3), i) for i in range(n)]


class Walker:
    """A climber walking steadily upward, visible except inside ``blind``.

    The probe only reports a hit when the climber is actually inside the crop it was handed,
    which is what makes the ladder and the prediction do real work rather than being trusted.
    """

    def __init__(self, blind=(), speed=0.02, appearance=(1.0, 0.0), start=(0.5, 0.9)):
        self.blind = set(blind)
        self.speed = speed
        self.appearance = appearance
        self.start = start
        self.seen: list[tuple[float, float]] = []

    def at(self, timestamp):
        return self.start[0], max(0.0, self.start[1] - self.speed * timestamp / 0.5)

    def __call__(self, timestamp, frame, cx, cy, half):
        self.seen.append((timestamp, half))
        if timestamp in self.blind:
            return None
        tx, ty = self.at(timestamp)
        if abs(tx - cx) <= half and abs(ty - cy) <= half:
            return Probe(tx, ty, self.appearance)
        return None


# --------------------------------------------------------------------------- #
# Arm identity — a trajectory difference must never hide inside a shared stamp
# --------------------------------------------------------------------------- #

def test_every_tracker_factor_moves_the_hash():
    """Two runs cropped by different trajectories are not comparable. If the tracker config
    can change without the stamp changing, they pool as one arm — #149 on a third axis."""

    base = track_hash(CropTrackConfig())
    for field, value in (
        ("half_candidates", (0.2,)), ("step", 0.25), ("ladder", (1.0, 2.0)), ("velocity_decay", 0.9),
        ("lost_after", 3), ("appearance_confident", 0.3), ("appearance_mismatch", 0.9),
        ("appearance_ema", 0.5),
    ):
        assert track_hash(_config(**{field: value})) != base, field


def test_identical_configs_share_a_hash():
    assert track_hash(CropTrackConfig()) == track_hash(CropTrackConfig())


def test_the_selected_crop_size_is_recorded_but_stays_out_of_the_arm_hash():
    """Crop size is chosen per Bundle because it has no global optimum, and that choice is a
    nuisance parameter rather than a factor — the trajectory is held fixed across every arm
    compared against it. But `crop_track_hash` feeds the *arm* identity, so if the winning
    value entered the hash, every Bundle would become its own arm and pooling would collapse
    entirely. The candidate set is stamped; the winner is only recorded."""

    walker = Walker()
    small = track(_frames(4), walker, _config(), *walker.at(0.0), half=0.15)
    large = track(_frames(4), walker, _config(), *walker.at(0.0), half=0.25)
    assert small.selected_half == 0.15 and large.selected_half == 0.25
    assert track_hash(small.config) == track_hash(large.config)
    assert small.to_artifact()["selectedHalf"] == 0.15
    assert "selectedHalf" not in small.to_artifact()["trackerConfig"]
    # ...and it survives the round trip, so an audit can see which size actually ran.
    assert CropTrack.from_artifact(small.to_artifact()).selected_half == 0.15


# --------------------------------------------------------------------------- #
# Tracking and recovery — the reason this module exists
# --------------------------------------------------------------------------- #

def test_seed_anchors_the_first_crop_and_detections_follow_the_climber():
    walker = Walker()
    out = track(_frames(6), walker, _config(), *walker.at(0.0))
    assert out.detected == 6
    assert all(b.source == ct.SRC_DETECTED for b in out.boxes)
    # The trajectory tracks the climber upward rather than sitting on the seed.
    assert out.boxes[-1].cy < out.boxes[0].cy


def test_a_gap_is_survived_and_the_track_recovers():
    """The failure the prototype could not handle: the climber vanishes for a few frames.
    Without recovery the crop sits on a stale location forever and every later frame is
    lost too — that is where all of the prototype's missing 19% went."""

    walker = Walker(blind={1.0, 1.5, 2.0})
    out = track(_frames(10), walker, _config(), *walker.at(0.0))
    by_ts = {b.timestamp: b for b in out.boxes}
    assert by_ts[1.0].source == ct.SRC_PREDICTED
    assert by_ts[2.0].source == ct.SRC_PREDICTED
    # ...and crucially, it comes back rather than staying lost.
    assert by_ts[2.5].source == ct.SRC_DETECTED
    assert out.boxes[-1].source == ct.SRC_DETECTED


def test_the_search_ladder_widens_on_misses_and_collapses_on_a_hit():
    """Widening is a response to failure, never a default: a bigger crop makes the climber
    smaller in frame, which is the exact size floor that makes full-frame detection fail."""

    walker = Walker(blind={0.5, 1.0})
    cfg = _config(ladder=(1.0, 2.0, 3.0), half_candidates=(0.15,))
    track(_frames(6), walker, cfg, *walker.at(0.0))
    halves = {}
    for ts, half in walker.seen:
        halves.setdefault(ts, []).append(half)
    base = cfg.half_candidates[0]
    assert halves[0.0] == [base], "no widening before any miss"
    assert max(halves[1.0]) > base, "must widen while missing"
    # After recovery the crop is back to base size on the next frame.
    assert halves[2.0][0] == base


def test_prediction_carries_velocity_through_a_gap():
    """A climber moves steadily, so a predicted crop should lead the stale position rather
    than freeze on it — that is what puts the climber back inside the crop on return."""

    walker = Walker(blind={1.0, 1.5})
    out = track(_frames(6), walker, _config(velocity_decay=1.0), *walker.at(0.0))
    by_ts = {b.timestamp: b for b in out.boxes}
    assert by_ts[1.0].source == ct.SRC_PREDICTED
    assert by_ts[1.0].cy < by_ts[0.5].cy, "prediction must move, not freeze"


def test_velocity_needs_two_detections_before_it_can_help():
    """A real limitation, pinned rather than hidden: velocity cannot be extrapolated from a
    single observation, so a climber lost immediately after the seed gets a frozen crop and
    only the search ladder to recover with. Widening is the whole defence in that window."""

    walker = Walker(blind={0.5, 1.0, 1.5})
    out = track(_frames(5), walker, _config(velocity_decay=1.0), *walker.at(0.0))
    by_ts = {b.timestamp: b for b in out.boxes}
    assert by_ts[1.0].cy == by_ts[0.0].cy, "no observed velocity to carry"
    # ...and the ladder is what has to find them instead.
    assert max(h for ts, h in walker.seen if ts == 1.0) > _config().half_candidates[0]


def test_the_track_gives_up_honestly_rather_than_faking_a_position():
    """After enough consecutive misses a predicted box is fantasy. An explicit `lost` lets
    stage B see that those timestamps were cropped on a guess."""

    walker = Walker(blind={round(i * 0.5, 3) for i in range(1, 20)})
    out = track(_frames(15), walker, _config(lost_after=4), *walker.at(0.0))
    sources = [b.source for b in out.boxes]
    assert ct.SRC_PREDICTED in sources
    assert sources[-1] == ct.SRC_LOST
    # Once lost it stops inventing motion.
    lost = [b for b in out.boxes if b.source == ct.SRC_LOST]
    assert len({(round(b.cx, 6), round(b.cy, 6)) for b in lost}) == 1


def test_every_frame_gets_exactly_one_entry():
    walker = Walker(blind={1.0})
    out = track(_frames(8), walker, _config(), *walker.at(0.0))
    assert len(out.boxes) == 8
    assert [b.timestamp for b in out.boxes] == [b[0] for b in _frames(8)]


# --------------------------------------------------------------------------- #
# Appearance — the guard that stops recovery adopting a bystander
# --------------------------------------------------------------------------- #

def test_a_mismatched_appearance_is_recorded_but_never_becomes_the_reference():
    """Recovery widens the search, and a wider search can reach a belayer. One suspect frame
    must not be able to drag the rolling reference onto the wrong person — measured on the
    planet-x pair, the reappearing climber sat at 0.35 while bystanders sat at 0.76–0.84."""

    class Impostor(Walker):
        def __call__(self, timestamp, frame, cx, cy, half):
            hit = super().__call__(timestamp, frame, cx, cy, half)
            if hit is not None and timestamp == 1.0:
                return Probe(hit.cx, hit.cy, (0.0, 1.0))   # opposite histogram
            return hit

    walker = Impostor()
    out = track(_frames(6), walker, _config(), *walker.at(0.0))
    by_ts = {b.timestamp: b for b in out.boxes}
    assert by_ts[1.0].appearance_dist is not None
    assert by_ts[1.0].appearance_dist > _config().appearance_mismatch
    # The next genuine frame is still recognised as the climber, which it would not be if
    # the impostor had been folded into the reference.
    assert by_ts[1.5].appearance_dist is not None
    assert by_ts[1.5].appearance_dist <= _config().appearance_confident


def test_a_featureless_probe_degrades_to_pure_geometry():
    """Histograms are optional; a probe that reports none must still track, exactly as a
    feature-less history reduces vitpose_job's stitcher to motion-only association."""

    class Featureless(Walker):
        def __call__(self, timestamp, frame, cx, cy, half):
            hit = super().__call__(timestamp, frame, cx, cy, half)
            return Probe(hit.cx, hit.cy, ()) if hit else None

    walker = Featureless()
    out = track(_frames(6), walker, _config(), *walker.at(0.0))
    assert out.detected == 6
    assert all(b.appearance_dist is None for b in out.boxes)


def test_bhattacharyya_bounds():
    assert ct.bhattacharyya((1.0, 0.0), (1.0, 0.0)) == 0.0
    assert ct.bhattacharyya((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert 0.0 < ct.bhattacharyya((0.5, 0.5), (0.9, 0.1)) < 1.0


# --------------------------------------------------------------------------- #
# The artifact — stage B reads this, so it has to survive a round trip
# --------------------------------------------------------------------------- #

def test_artifact_round_trips_with_its_config():
    walker = Walker(blind={1.0})
    out = track(_frames(6), walker, _config(half_candidates=(0.2,)), *walker.at(0.0))
    out.setup_hash = "sh"
    out.seed = {"x": 0.5, "y": 0.9, "t": 0.0}
    back = CropTrack.from_artifact(json.loads(json.dumps(out.to_artifact())))
    assert track_hash(back.config) == track_hash(out.config)
    assert [b.source for b in back.boxes] == [b.source for b in out.boxes]
    assert back.setup_hash == "sh"
    assert len(back.boxes) == len(out.boxes)


def test_artifact_reports_its_own_detection_rate():
    walker = Walker(blind={0.5, 1.0})
    out = track(_frames(10), walker, _config(), *walker.at(0.0))
    stats = out.to_artifact()["stats"]
    assert stats["frames"] == 10
    assert stats["detected"] == out.detected
    assert abs(stats["rate"] - out.detected / 10) < 1e-6


def test_nearest_maps_a_sparse_experimental_grid_onto_the_track():
    """Stage B samples 12·sqrt(n) scattered timestamps that will not line up with the
    tracking step, so every one has to find its nearest recorded crop."""

    walker = Walker()
    out = track(_frames(10), walker, _config(), *walker.at(0.0))
    assert out.nearest(0.0).timestamp == 0.0
    assert out.nearest(1.24).timestamp == 1.0
    assert out.nearest(1.26).timestamp == 1.5
    assert out.nearest(99.0).timestamp == 4.5     # past the end: still the best available
    assert CropTrack(config=_config()).nearest(1.0) is None


def test_load_and_write_round_trip_on_disk():
    tmp = Path(tempfile.mkdtemp())
    try:
        walker = Walker()
        out = track(_frames(4), walker, _config(), *walker.at(0.0))
        assert ct.load_crop_track(tmp) is None
        ct.write_crop_track(tmp, out)
        back = ct.load_crop_track(tmp)
        assert back is not None and len(back.boxes) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_crop_rect_is_clamped_into_frame():
    box = ct.CropBox(0.0, 0.05, 0.97, 0.15, ct.SRC_DETECTED)
    x0, y0, x1, y1 = box.rect()
    assert (x0, y0) == (0.0, 0.82)
    assert (x1, y1) == (0.2, 1.0)


def test_module_needs_no_mediapipe():
    assert "mediapipe" not in sys.modules, "crop_track must stay pure"


def _run_all():
    fns = [fn for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


if __name__ == "__main__":
    sys.exit(_run_all())
