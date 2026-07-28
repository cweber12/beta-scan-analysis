"""Tests for the corpus re-seed driver's frame grid.

The driver decides which timestamps a scaffold is posed on, and the scanner matches a
scaffold pose to one of its Detection Frames within **1 ms**, resting on a documented
assumption: "every Detection Frame timestamp is a 100 ms multiple by construction (the
uniform grid)" (``harnessGroundTruthScaffold.TIMESTAMP_EPSILON``).

Emitting timestamps measured from a raw climb start instead of from zero breaks that
silently and expensively: nothing matches, the scanner seeds every frame absent, and the
failure surfaces only when a human accepts empty truth. It cost a 662-of-662-posed bundle
its entire truth, and 33 of 89 scaffolds were off-grid before it was caught.

Runnable with pytest, or standalone: ``python -m tests.test_reset_scaffolds``.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "reset_scaffolds", Path(__file__).resolve().parent.parent / "scripts" / "reset_scaffolds.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _bundle(root: Path, setup: dict, duration: float | None = None) -> Path:
    b = root / "route" / "vid"
    b.mkdir(parents=True, exist_ok=True)
    (b / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    metadata: dict = {"route_folder": "route", "video_key": "vid"}
    if duration is not None:
        metadata["source_video"] = {"duration_seconds": duration}
    (b / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return b


def test_frames_are_always_on_the_hundred_millisecond_grid():
    """The invariant, over climb starts chosen to be maximally awkward."""

    with tempfile.TemporaryDirectory() as tmp:
        for start in (0.0, 2.87, 7.91, 2.75, 50.31, 1.82, 0.05, 0.99):
            root = Path(tmp) / f"s{start}"
            b = _bundle(root, {"setupHash": "sh",
                               "climberPoint": {"x": 0.5, "y": 0.9, "t": start},
                               "climbEnd": start + 5.0})
            frames = driver._frames_for(b, json.loads((b / "setup.json").read_text()))
            assert frames, f"no frames for start={start}"
            assert driver.off_grid(frames) == [], (
                f"start={start} produced off-grid timestamps, e.g. {frames[:3]}")
            # ...and each is genuinely a multiple of 100 ms, not merely near one.
            assert all(round(t * 10) == round(t * 10, 6) for t in frames)


def test_the_window_is_respected_while_snapping():
    """Snapping must not walk outside the climb window at either end."""

    with tempfile.TemporaryDirectory() as tmp:
        b = _bundle(Path(tmp), {"setupHash": "sh",
                                "climberPoint": {"x": 0.5, "y": 0.9, "t": 2.87},
                                "climbEnd": 6.94})
        frames = driver._frames_for(b, json.loads((b / "setup.json").read_text()))
        assert frames[0] == 2.9      # first grid point at or after 2.87
        assert frames[-1] == 6.9     # last at or before 6.94
        assert all(2.87 <= t <= 6.94 for t in frames)

        # A window whose bounds already sit on the grid keeps them exactly.
        b2 = _bundle(Path(tmp) / "exact", {"setupHash": "sh",
                                           "climberPoint": {"x": 0.5, "y": 0.9, "t": 3.0},
                                           "climbEnd": 5.0})
        exact = driver._frames_for(b2, json.loads((b2 / "setup.json").read_text()))
        assert exact[0] == 3.0 and exact[-1] == 5.0


def test_no_climb_end_falls_back_to_the_recorded_duration():
    with tempfile.TemporaryDirectory() as tmp:
        b = _bundle(Path(tmp), {"setupHash": "sh"}, duration=4.0)
        frames = driver._frames_for(b, json.loads((b / "setup.json").read_text()))
        assert frames[0] == 0.0 and frames[-1] == 4.0
        assert driver.off_grid(frames) == []

        # Neither a climb end nor a duration means there is nothing to ask for.
        b2 = _bundle(Path(tmp) / "bare", {"setupHash": "sh"})
        assert driver._frames_for(b2, json.loads((b2 / "setup.json").read_text())) == []


def test_off_grid_detects_what_the_old_code_produced():
    """The guard must catch the exact shape of the bug it exists for."""

    # What the driver used to emit for a climb start of 2.87.
    bad = [round(2.87 + i * 0.1, 3) for i in range(5)]
    assert driver.off_grid(bad) == bad
    # ...and passes a clean grid, including float-noisy multiples.
    assert driver.off_grid([0.0, 0.1, 0.2, 2.9, 69.0]) == []
    assert driver.off_grid([round(i * 0.1, 3) for i in range(700)]) == []


def test_a_plan_refuses_to_send_off_grid_frames(monkeypatch=None):
    """Belt and braces: even if the generator regressed, the plan would not ship it."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        b = _bundle(root, {"setupHash": "sh",
                           "climberPoint": {"x": 0.5, "y": 0.9, "t": 2.87},
                           "climbEnd": 9.0})
        (b / "vid.mp4").write_bytes(b"x")
        original = driver._frames_for
        driver._frames_for = lambda *_a, **_k: [2.87, 2.97]
        try:
            plan = driver._plan(b)
        finally:
            driver._frames_for = original
        assert "skip" in plan and "off the 0.1s grid" in plan["skip"]


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all reset-scaffold tests passed")


if __name__ == "__main__":
    _run_all()
