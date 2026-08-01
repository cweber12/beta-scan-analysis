"""Stub-backed tests for the harness MediaPipe detection module (PRD #156).

No MediaPipe required: everything that decides what a run *means* is pure, and the heavy
dependency sits behind a Protocol at the bottom of the module. Same property that lets
``test_vitpose_job.py`` run without torch.

These target the artifact and the contract, never the call sequence. The tests that matter
most are the configuration-hash ones: if two arms can share a stamp they pool, and the
experiment silently degrades into another observational corpus — which is the failure
PRD #156 exists to prevent, and the failure issue #149 already shipped once on the truth
side.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import mediapipe_job as mj
from mediapipe_job import (
    DetectionConfig,
    DetectRequest,
    Keypoint,
    PreprocessStep,
    build_orb_payload,
    build_pose_payload,
    config_hash,
    pass_requests,
    run_mediapipe_job,
)


def _config(**over) -> DetectionConfig:
    base = dict(mode=1, preprocess=(), crop=mj.CROP_NONE)
    base.update(over)
    return DetectionConfig(**base)


def _request(**over) -> DetectRequest:
    base = dict(
        video_path="analysis/r/k/k.mp4", route_folder="r", video_key="k",
        frames=(0.0, 0.5, 1.0), config=_config(), setup_hash="sh", repeats=3,
    )
    base.update(over)
    return DetectRequest(**base)


# --------------------------------------------------------------------------- #
# Stub seam — the whole job runs here with no MediaPipe installed
# --------------------------------------------------------------------------- #

class StubDetector:
    """A detector whose output is a pure function of its build index.

    Repeats are supposed to be *independent passes*, so a stub that returns the same
    thing every time could not tell an independent pass from a re-export. This one
    nudges every coordinate by its build index, which makes the difference visible in
    the written artifacts.
    """

    builds: list["StubDetector"] = []

    def __init__(self, config: DetectionConfig, undecodable: tuple[float, ...] = (),
                 empty: tuple[float, ...] = ()) -> None:
        self.config = config
        self.index = len(StubDetector.builds)
        self.calls: list[tuple[Path, tuple[float, ...]]] = []
        self.undecodable = undecodable
        self.empty = empty
        StubDetector.builds.append(self)

    def detect(self, video_path, timestamps, config, crop_track=None):
        self.calls.append((video_path, tuple(timestamps)))
        out = {}
        for t in timestamps:
            if t in self.undecodable:
                continue          # key absent: the decoder produced nothing
            if t in self.empty:
                out[t] = []       # key present, empty: the detector found nobody
                continue
            offset = self.index * 0.01
            out[t] = [Keypoint("left_wrist", 0.4 + offset, 0.5 + offset, 0.9)]
        return out


def _stub_factory(**over):
    StubDetector.builds = []
    return lambda config: StubDetector(config, **over)


def _bundle(tmp: Path, *, setup_hash: str | None = "bundle-hash") -> Path:
    """A minimal on-disk Bundle: the analysis root, one route, one video key."""

    bundle = tmp / "analysis" / "route" / "key"
    bundle.mkdir(parents=True)
    (bundle / "key.mp4").write_bytes(b"not really a video")
    if setup_hash is not None:
        (bundle / "setup.json").write_text(json.dumps({"setupHash": setup_hash}))
    return bundle


def _job_request(**over) -> DetectRequest:
    base = dict(
        video_path="route/key/key.mp4", route_folder="route", video_key="key",
        frames=(0.0, 0.1, 0.2), config=_config(), setup_hash=None, repeats=3,
    )
    base.update(over)
    return DetectRequest(**base)


def _pose_runs(bundle: Path) -> list[dict]:
    return [json.loads(p.read_text())
            for p in sorted((bundle / "detections").glob("*_pose.json"))]


# --------------------------------------------------------------------------- #
# Configuration identity — the arm stamp
# --------------------------------------------------------------------------- #

def test_identical_configs_share_a_hash():
    """Two runs differing in nothing must pool, or a repeat set cannot form."""
    assert config_hash(_config()) == config_hash(_config())


def test_every_factor_moves_the_config_hash():
    """Issue #149 on the detection side: a factor missing from the hash lets two arms
    share a stamp and be pooled as one, which is the confound the whole PRD exists to
    remove. Each factor is checked individually so a regression names itself."""

    base = config_hash(_config())
    assert config_hash(_config(mode=0)) != base
    assert config_hash(_config(mode=2)) != base
    assert config_hash(_config(crop=mj.CROP_ADAPTIVE)) != base
    assert config_hash(_config(preprocess=(PreprocessStep("contrast"),))) != base
    assert config_hash(replace(_config(), module_version="99")) != base


def test_preprocess_parameters_move_the_hash():
    """A step at a different strength is a different arm. Stamping only the step's *name*
    would pool 'contrast 1.2' with 'contrast 2.0' and read the difference as noise."""

    weak = _config(preprocess=(PreprocessStep("contrast", {"factor": 1.2}),))
    strong = _config(preprocess=(PreprocessStep("contrast", {"factor": 2.0}),))
    assert config_hash(weak) != config_hash(strong)
    # ...and parameter dict ordering is not itself a factor.
    a = _config(preprocess=(PreprocessStep("x", {"a": 1, "b": 2}),))
    b = _config(preprocess=(PreprocessStep("x", {"b": 2, "a": 1}),))
    assert config_hash(a) == config_hash(b)


def test_preprocess_order_is_a_factor():
    """Image transforms do not commute, so contrast-then-brightness is a different arm
    from brightness-then-contrast and must not collapse onto one stamp."""

    forward = _config(preprocess=(PreprocessStep("contrast"), PreprocessStep("brightness")))
    reverse = _config(preprocess=(PreprocessStep("brightness"), PreprocessStep("contrast")))
    assert config_hash(forward) != config_hash(reverse)


# --------------------------------------------------------------------------- #
# Preprocessing steps (issue #161) — the transform, not just the stamp
# --------------------------------------------------------------------------- #

def _frame(*values) -> "numpy.ndarray":
    import numpy
    return numpy.array([list(values)], dtype=numpy.uint8)


def test_each_step_alone_is_a_distinguishable_arm():
    """The AC that makes the factorial readable: baseline, contrast-only and
    brightness-only must be three arms, not two-and-a-half."""

    contrast = _config(preprocess=(PreprocessStep(mj.STEP_CONTRAST, {"factor": 1.5}),))
    brightness = _config(preprocess=(PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 20}),))
    stamps = {config_hash(_config()), config_hash(contrast), config_hash(brightness)}
    assert len(stamps) == 3
    # ...and the two-factor arm is a fourth, distinct from both of its own margins.
    both = _config(preprocess=(PreprocessStep(mj.STEP_CONTRAST, {"factor": 1.5}),
                               PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 20})))
    assert config_hash(both) not in stamps


def test_contrast_pivots_at_mid_grey_so_it_does_not_move_brightness():
    """A plain value*factor also raises the mean, which would tangle the two main effects
    at the source. Pivoting keeps mid-grey fixed and spreads the ends symmetrically."""

    out = mj.apply_preprocess(_frame(28, 128, 228),
                              (PreprocessStep(mj.STEP_CONTRAST, {"factor": 2.0}),))
    assert list(out[0]) == [0, 128, 255], "mid-grey must not move; the ends spread"
    # A pair straddling the pivot keeps its mean: the step spends its effect on spread and
    # none on level, which is what makes contrast and brightness separable factors.
    dark, bright = mj.apply_preprocess(
        _frame(100, 155), (PreprocessStep(mj.STEP_CONTRAST, {"factor": 1.5}),))[0]
    assert (int(dark) + int(bright)) / 2 == (100 + 155) / 2 == mj.CONTRAST_PIVOT


def test_brightness_is_additive_in_eight_bit_levels_and_clamps():
    out = mj.apply_preprocess(_frame(0, 100, 250),
                              (PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 20}),))
    assert list(out[0]) == [20, 120, 255]
    out = mj.apply_preprocess(_frame(10, 100),
                              (PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": -20}),))
    assert list(out[0]) == [0, 80]


def test_steps_apply_in_declared_order_and_do_not_commute():
    """Order is in the arm hash because the transforms genuinely differ once a value
    clips. If this ever became commutative in fact, the two stamps would be measuring one
    thing and the ordering factor would be reporting noise."""

    contrast = PreprocessStep(mj.STEP_CONTRAST, {"factor": 2.0})
    brightness = PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 60})
    forward = mj.apply_preprocess(_frame(100), (contrast, brightness))
    reverse = mj.apply_preprocess(_frame(100), (brightness, contrast))
    # 100 -> contrast 72 -> +60 -> 132, against 100 -> +60 -> 160 -> contrast 192. Contrast
    # applied second *amplifies the offset*; the two orders are 60 levels apart on this
    # pixel, so the ordering factor in the arm hash is measuring something real.
    assert list(forward[0]) == [132] and list(reverse[0]) == [192]
    # Each step is a uint8 -> uint8 function, so a two-step arm is exactly the composition
    # of the two one-step arms — what lets a factorial be read against its own margins.
    step_by_step = mj.apply_preprocess(
        mj.apply_preprocess(_frame(100), (contrast,)), (brightness,))
    assert list(forward[0]) == list(step_by_step[0])


def test_no_steps_returns_the_decoders_own_bytes():
    """The baseline arm must not round-trip through the transform at all — an identity
    that quietly rounded would make baseline non-reproducible against its own prior runs."""

    frame = _frame(1, 2, 3)
    assert mj.apply_preprocess(frame, ()) is frame


def test_a_misspelled_parameter_is_refused_rather_than_defaulted():
    """The subtle fabricated null: `factr` falls back to the default, so the arm stamps a
    strength its pixels never saw and the experiment reports that contrast does not
    matter."""

    for step in (PreprocessStep(mj.STEP_CONTRAST, {"factr": 1.5}),
                 PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 20, "factor": 2})):
        try:
            mj.step_amount(step)
        except ValueError as exc:
            assert "unknown parameter" in str(exc)
        else:
            raise AssertionError(f"{step} must be refused, not silently defaulted")


def test_the_identity_transform_is_refused_as_an_arm():
    """An arm whose transform does nothing is bytes identical to baseline under a
    different stamp. The control level is the absence of the step."""

    for step in (PreprocessStep(mj.STEP_CONTRAST, {"factor": 1.0}),
                 PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 0})):
        try:
            mj.step_amount(step)
        except ValueError as exc:
            assert "identity" in str(exc)
        else:
            raise AssertionError(f"{step} would duplicate baseline under a second stamp")


def test_a_parameter_outside_its_range_is_refused():
    for step in (PreprocessStep(mj.STEP_CONTRAST, {"factor": 0.0}),
                 PreprocessStep(mj.STEP_CONTRAST, {"factor": 99.0}),
                 PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 400})):
        try:
            mj.step_amount(step)
        except ValueError:
            continue
        raise AssertionError(f"{step} is outside the supported range and must be refused")


def test_option_helper_fixes_the_order_rather_than_leaving_it_to_typing():
    """Contrast-then-brightness and brightness-then-contrast are different arms, so a flag
    pair must not silently produce one or the other depending on argument order."""

    steps = mj.preprocess_from_options(contrast=1.5, brightness=20)
    assert [s.name for s in steps] == [mj.STEP_CONTRAST, mj.STEP_BRIGHTNESS]
    assert mj.preprocess_from_options() == ()
    assert [s.name for s in mj.preprocess_from_options(brightness=20)] == [mj.STEP_BRIGHTNESS]


def test_the_written_artifact_reconstructs_the_steps_and_their_order():
    """Step order has to be recoverable from the run itself — an arm hash says two runs
    differ, the artifact has to say how."""

    request = _request(config=_config(preprocess=(
        PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": -15}),
        PreprocessStep(mj.STEP_CONTRAST, {"factor": 1.4}))))
    payload = build_pose_payload(request, {}, pass_index=0)
    steps = payload["diagnostics"]["experiment"]["config"]["preprocess"]
    assert [s["name"] for s in steps] == [mj.STEP_BRIGHTNESS, mj.STEP_CONTRAST]
    assert steps[0]["params"] == {"delta": -15} and steps[1]["params"] == {"factor": 1.4}


def test_the_pixels_handed_to_the_model_are_the_transformed_ones():
    """The wiring, not the arithmetic: a transform that ran on a copy nobody posed would
    be the exact fabricated null the refusal exists to prevent, and every test above it
    would still pass."""

    import numpy

    seen: dict = {}

    class FakeMp:
        class ImageFormat:
            SRGB = "srgb"

        class Image:
            def __init__(self, image_format, data):
                seen["data"] = data

    class FakeCv2:
        COLOR_BGR2RGB = 0

        @staticmethod
        def cvtColor(image, code):
            return image

    class FakeLandmarker:
        @staticmethod
        def detect(image):
            return type("Result", (), {"pose_landmarks": []})()

    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    frame = numpy.full((64, 64, 3), 100, dtype=numpy.uint8)
    step = PreprocessStep(mj.STEP_BRIGHTNESS, {"delta": 20})

    detector._pose_region(FakeLandmarker, FakeMp, FakeCv2, [], frame, None, (step,))
    assert int(seen["data"][0][0][0]) == 120, "the model must see the transformed pixels"
    detector._pose_region(FakeLandmarker, FakeMp, FakeCv2, [], frame, None)
    assert int(seen["data"][0][0][0]) == 100, "...and baseline must see the decoder's own"


def test_the_crop_tracking_pass_is_never_preprocessed():
    """The trajectory is computed once per Bundle and read by every arm, which is what
    makes the experiment isolate pose quality *given* localization (#169). A contrast arm
    that tracked through its own filter would crop different pixels from baseline, and the
    difference would be part localization and part detection with no way to split them.

    Asserted at the signature, because ``build_crop_track``'s probe relies on the default
    and exercising it for real needs MediaPipe installed — which this suite must not."""

    import inspect

    default = inspect.signature(mj.MediaPipeDetector._pose_region).parameters["steps"].default
    assert default == (), "the tracking probe passes no steps and depends on this default"
    source = inspect.getsource(mj.MediaPipeDetector.build_crop_track)
    assert ".preprocess" not in source, (
        "the tracking pass must not reach for the arm's preprocessing steps")


def test_a_job_refuses_an_unrunnable_arm_before_it_decodes_anything():
    """A batch is hours; the arm is checked once, up front, rather than surfacing as the
    same error 84 times with the real reason buried in each."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _selectable(tmp, "r", "a")
        StubDetector.builds = []
        try:
            run_mediapipe_job(
                tmp / "analysis",
                _request(route_folder="r", video_key="a",
                         video_path=str(bundle / "a.mp4"),
                         config=_config(preprocess=(PreprocessStep("gamma"),))),
                _stub_factory())
        except NotImplementedError:
            pass
        else:
            raise AssertionError("an unimplemented step must stop the job")
        status = json.loads((bundle / mj.STATUS_NAME).read_text())
        assert status["status"] == "error" and status["passesWritten"] == 0
        assert not any(d.calls for d in StubDetector.builds), "nothing may have decoded"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_identity_is_json_stable():
    """The hash is taken over this, so it must contain no set/dict ordering nondeterminism."""
    import json
    cfg = _config(preprocess=(PreprocessStep("contrast", {"factor": 1.5}),))
    once = json.dumps(cfg.identity(), sort_keys=True)
    assert once == json.dumps(cfg.identity(), sort_keys=True)


# --------------------------------------------------------------------------- #
# Artifact shape — must be what the existing writer and readers expect
# --------------------------------------------------------------------------- #

def test_one_frame_per_requested_timestamp_echoed_verbatim():
    """Timestamps are echoed and never thinned. A dropped frame reads downstream as 'the
    Climber was absent' rather than 'the detector missed', which is a different finding."""

    req = _request(frames=(0.0, 0.5, 1.0))
    payload = build_pose_payload(req, {0.0: [Keypoint("nose", 0.5, 0.4, 0.9)]}, pass_index=0)
    assert [f["timestamp"] for f in payload["frames"]] == [0.0, 0.5, 1.0]
    assert payload["frames"][0]["source"] == "detected"
    assert payload["frames"][0]["keypoints"][0]["name"] == "nose"
    # Undetected frames survive as empty, not missing.
    assert payload["frames"][1]["keypoints"] == []
    assert payload["frames"][1]["source"] == "missing"


def test_run_carries_its_arm_and_pairing_provenance():
    req = _request()
    diag = build_pose_payload(req, {}, pass_index=2)["diagnostics"]
    assert diag["experiment"]["configHash"] == config_hash(req.config)
    assert diag["experiment"]["passIndex"] == 2
    assert diag["experiment"]["repeats"] == 3
    assert diag["experiment"]["config"]["mode"] == 1
    # setupHash rides on the payload: evaluation only scores a run against truth whose
    # setupHash matches, so a run without it cannot be scored at all.
    assert build_pose_payload(req, {}, pass_index=0)["setupHash"] == "sh"


def test_harness_runs_are_distinguishable_from_scanner_runs():
    """Same artifact *shape*, different origin — and the origin is recorded. A pooled
    number must never blend a browser-produced run with a harness-produced one, because
    whether those agree is the open parity question the PRD gates on."""

    diag = build_pose_payload(_request(), {}, pass_index=0)["diagnostics"]
    assert diag["origin"] == mj.ORIGIN
    assert diag["appVersion"].startswith(mj.ORIGIN)
    # The module's own version rides in the arm identity, so a module change is as
    # detectable as a scanner build change.
    assert diag["experiment"]["config"]["moduleVersion"] == mj.MODULE_VERSION


def test_orb_half_is_explicitly_not_computed():
    """The pose+ORB pairing is an invariant of the writer and every reader. This module has
    no cross-match to compute, so the ORB half says so — rather than being fabricated, or
    borrowed from another run and silently attributed to this pose."""

    orb = build_orb_payload(_request())
    assert orb["notComputed"] is True
    assert orb["matches"] == []
    assert "no ORB cross-match" in orb["notComputedReason"]
    assert orb["diagnostics"]["origin"] == mj.ORIGIN


# --------------------------------------------------------------------------- #
# Repeats — the variance floor has to be produced, not hoped for
# --------------------------------------------------------------------------- #

def test_repeats_default_to_one_because_the_detector_is_deterministic():
    """Reverses the original PRD decision, on measurement.

    #134 mandated three repeats, but it measured the *scanner's* detector, which genuinely
    scatters (PCK 0.0055 median). This one is bit-deterministic — three passes produce
    byte-identical output across all modes, videos and processes — so its floor is exactly
    0 and repeats buy provably nothing at N× the cost. Drift detection moved to the
    byte-identical canary (#168). Repeats stay a parameter for the day determinism breaks."""

    assert mj.DEFAULT_REPEATS == 1
    assert _request(repeats=5).repeats == 5, "a caller may still ask for more"


def test_repeat_passes_are_enumerated_and_distinguishable():
    req = _request(repeats=3)
    passes = pass_requests(req)
    assert len(passes) == 3
    payloads = [build_pose_payload(p, {}, pass_index=i) for i, p in enumerate(passes)]
    indices = [p["diagnostics"]["experiment"]["passIndex"] for p in payloads]
    assert indices == [0, 1, 2]
    # Same arm across passes: repeats measure noise, so they must not look like arms.
    hashes = {p["diagnostics"]["experiment"]["configHash"] for p in payloads}
    assert len(hashes) == 1


def test_zero_repeats_is_refused():
    try:
        pass_requests(_request(repeats=0))
    except ValueError:
        return
    raise AssertionError("repeats=0 must be refused, not silently produce no runs")


# --------------------------------------------------------------------------- #
# The job — N passes onto disk through the existing writer
# --------------------------------------------------------------------------- #

def test_job_writes_one_pose_orb_pair_per_pass():
    """The whole point of the slice: passes reach disk, in the scanner's artifact shape,
    through the writer the scanner posts through — so evaluate needs no change to read
    them."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        runs = run_mediapipe_job(tmp / "analysis", _job_request(repeats=3), _stub_factory())
        assert len(runs) == 3
        poses = sorted((bundle / "detections").glob("*_pose.json"))
        orbs = sorted((bundle / "detections").glob("*_orb.json"))
        assert len(poses) == 3 and len(orbs) == 3
        # Every pose file has the orb half of its pair — the writer's invariant.
        assert [p.name[:-len("_pose.json")] for p in poses] == \
               [o.name[:-len("_orb.json")] for o in orbs]
        env = json.loads(poses[0].read_text())
        assert env["type"] == "pose"
        assert env["data"]["diagnostics"]["origin"] == mj.ORIGIN
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_each_pass_is_a_fresh_detector_and_its_own_detection_sweep():
    """#134's finding, guarded: 27 of 33 apparent historical repeats were a single pass
    exported N times. A repeat that shares a detector is a continuation, not a repeat —
    MediaPipe's graph carries state — so each pass must be built and swept separately,
    and the written artifacts must differ when the detector's output does."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        run_mediapipe_job(tmp / "analysis", _job_request(repeats=3), _stub_factory())
        # Asserted on *sweeps*, not on how many detectors were constructed: the job also
        # builds one throwaway probe to read model identity, and the contract that matters
        # is "three independent detection passes", not the factory's call count.
        swept = [d for d in StubDetector.builds if d.calls]
        assert len(swept) == 3, "one detector per pass, never a shared one"
        assert all(len(d.calls) == 1 for d in swept), "one sweep per pass"
        # Distinct objects, not the same detector handed back three times.
        assert len({id(d) for d in swept}) == 3

        wrists = [
            env["data"]["frames"][0]["keypoints"][0]["x"] for env in _pose_runs(bundle)
        ]
        assert len(set(wrists)) == 3, (
            "three passes produced identical artifacts — a re-export would look like this"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_passes_get_distinct_run_ids_so_evaluation_records_cannot_collide():
    """Three passes inside one wall-clock second must still be three runs.

    ``save_detection_run`` bumps a colliding filename but not the ``run_ts`` inside the
    envelope, and evaluation names its record ``<run_ts>_vs_<truthHash>``. Sharing an id
    would leave three pose files and one scored record, each overwriting the last — the
    repeat set silently collapsing back into a single measurement."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        runs = run_mediapipe_job(tmp / "analysis", _job_request(repeats=3), _stub_factory())
        ids = [r["runTs"] for r in runs]
        assert len(set(ids)) == 3, ids
        # The id the *envelope* carries is what evaluate reads — not just the filename.
        assert sorted(env["run_ts"] for env in _pose_runs(bundle)) == sorted(ids)
        # The arm rides in the id, so a detections/ listing groups by arm unopened.
        assert all(config_hash(_config())[:8] in run_ts for run_ts in ids)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_written_run_carries_arm_module_version_and_setup_hash():
    """A run missing any of the three is unusable: without setupHash it cannot be paired
    with truth at all, and without the arm stamp it cannot be told from another arm."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp, setup_hash="from-the-bundle")
        run_mediapipe_job(tmp / "analysis", _job_request(setup_hash=None), _stub_factory())
        for env in _pose_runs(bundle):
            data = env["data"]
            # Falls back to the Bundle's own setup.json when the request omits it.
            assert data["setupHash"] == "from-the-bundle"
            experiment = data["diagnostics"]["experiment"]
            assert experiment["configHash"] == config_hash(_config())
            assert experiment["config"]["moduleVersion"] == mj.MODULE_VERSION
            assert data["diagnostics"]["appVersion"].startswith(mj.ORIGIN)
        # The ORB half is written for every pass, explicitly not computed.
        for path in sorted((bundle / "detections").glob("*_orb.json")):
            assert json.loads(path.read_text())["data"]["notComputed"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_bundle_with_no_setup_hash_is_refused_before_any_pass_runs():
    """An unscoreable run is worse than no run: it costs the same and reads as evidence.
    Better to fail loudly at the top than to write three runs truth can never pair with."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp, setup_hash=None)
        try:
            run_mediapipe_job(tmp / "analysis", _job_request(), _stub_factory())
        except ValueError as exc:
            assert "setupHash" in str(exc)
        else:
            raise AssertionError("a Bundle with no setupHash must not be run")
        assert not list((bundle / "detections").glob("*_pose.json"))
        assert json.loads((bundle / mj.STATUS_NAME).read_text())["status"] == "error"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Undecodable frames are not detection misses
# --------------------------------------------------------------------------- #

def test_undecodable_frames_are_distinguished_from_detection_misses():
    """A frame the decoder never produced and a frame the detector found nobody on are
    different findings. Folding the first into the second reports a container problem as
    a detection failure — the cheapest way to manufacture a regression that never was."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        run_mediapipe_job(
            tmp / "analysis",
            _job_request(repeats=1),
            _stub_factory(undecodable=(0.1,), empty=(0.2,)),
        )
        frames = _pose_runs(bundle)[0]["data"]["frames"]
        assert [f["timestamp"] for f in frames] == [0.0, 0.1, 0.2]
        assert frames[0]["source"] == mj.FRAME_DETECTED
        assert frames[1]["source"] == mj.FRAME_UNDECODABLE
        assert frames[2]["source"] == mj.FRAME_MISSING
        # Both empty either way: the artifact never drops a requested timestamp.
        assert frames[1]["keypoints"] == [] and frames[2]["keypoints"] == []
        status = json.loads((bundle / mj.STATUS_NAME).read_text())
        assert status["runs"][0]["framesUndecodable"] == 1
        assert status["runs"][0]["framesDetected"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_payload_without_decoded_set_keeps_the_pre_existing_missing_behaviour():
    payload = build_pose_payload(_request(), {0.0: [Keypoint("nose", 0.5, 0.4, 0.9)]},
                                 pass_index=0)
    assert [f["source"] for f in payload["frames"]] == [
        mj.FRAME_DETECTED, mj.FRAME_MISSING, mj.FRAME_MISSING]


# --------------------------------------------------------------------------- #
# Status sidecar — a crashed job must be diagnosable without re-running it
# --------------------------------------------------------------------------- #

def test_status_sidecar_reaches_a_terminal_done():
    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        run_mediapipe_job(tmp / "analysis", _job_request(repeats=2), _stub_factory())
        status = json.loads((bundle / mj.STATUS_NAME).read_text())
        assert status["status"] == "done"
        assert len(status["runs"]) == 2
        assert status["configHash"] == config_hash(_config())
        assert status["repeats"] == 2 and status["requestedFrames"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_failure_records_its_exception_type_and_traceback():
    """The sidecar is the only channel a caller has once the job is running. A terminal
    ``error`` that names only "it failed" costs a whole re-run to diagnose, which for a
    GPU-class sweep is the expensive half of the work."""

    class Exploding:
        def __init__(self, config): pass
        def detect(self, video_path, timestamps, config, crop_track=None):
            raise RuntimeError("decoder went away")

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        try:
            run_mediapipe_job(tmp / "analysis", _job_request(), lambda c: Exploding(c))
        except RuntimeError:
            pass                      # re-raised on purpose: the sidecar never swallows
        else:
            raise AssertionError("the job must re-raise after recording the failure")
        status = json.loads((bundle / mj.STATUS_NAME).read_text())
        assert status["status"] == "error"
        assert status["errorType"] == "RuntimeError"
        assert "decoder went away" in status["error"]
        assert "Traceback" in status["traceback"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_mid_batch_failure_still_names_the_runs_that_reached_disk():
    """Otherwise the Bundle holds runs the status sidecar does not account for, and the
    next reader cannot tell a partial batch from a complete one."""

    class FailsOnThirdPass:
        """Counts *sweeps*, not constructions — the job builds a throwaway probe for
        model identity, so counting builds would target the wrong pass."""

        sweeps = 0

        def __init__(self, config):
            pass

        def detect(self, video_path, timestamps, config, crop_track=None):
            FailsOnThirdPass.sweeps += 1
            if FailsOnThirdPass.sweeps == 3:
                raise RuntimeError("pass 3 died")
            return {t: [Keypoint("nose", 0.5, 0.5, 0.9)] for t in timestamps}

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        try:
            run_mediapipe_job(tmp / "analysis", _job_request(repeats=4),
                              lambda c: FailsOnThirdPass(c))
        except RuntimeError:
            pass
        status = json.loads((bundle / mj.STATUS_NAME).read_text())
        assert status["status"] == "error"
        assert status["passesWritten"] == 2
        assert len(status["runs"]) == 2
        assert len(list((bundle / "detections").glob("*_pose.json"))) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Bundle inputs — the grid and the pairing anchor come from the Bundle itself
# --------------------------------------------------------------------------- #

def test_timestamp_grid_is_taken_from_the_bundle_truth():
    """Running any other grid puts the run's samples between truth frames, and the
    nearest-frame join then scores a fraction of what was computed — silently."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        (bundle / "vitpose.json").write_text(json.dumps(
            {"frames": [{"timestamp": 0.0}, {"timestamp": 0.5}]}))
        assert mj.truth_timestamps(bundle) == (0.0, 0.5)
        # Human-reviewed truth wins over the scaffold, mirroring evaluate.load_truth.
        (bundle / "ground-truth.json").write_text(json.dumps(
            {"frames": [{"timestamp": 1.0}, {"timestamp": 1.5}, {"timestamp": 2.0}]}))
        assert mj.truth_timestamps(bundle) == (1.0, 1.5, 2.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_truthless_bundle_reports_an_empty_grid_rather_than_guessing_one():
    tmp = Path(tempfile.mkdtemp())
    try:
        assert mj.truth_timestamps(_bundle(tmp)) == ()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_ids_are_unique_even_against_ids_already_on_disk():
    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        detections = bundle / "detections"
        detections.mkdir()
        taken = mj.pass_run_ts("20260731-120000", _config(), 0)
        (detections / f"{taken}_pose.json").write_text("{}")
        assert mj._unique_run_ts(detections, taken) != taken
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The backend refuses what it cannot honour
# --------------------------------------------------------------------------- #

def test_the_backend_refuses_a_config_it_would_silently_ignore():
    """A factor in the arm identity that nothing implements would write a run stamped with
    a transform its pixels never saw — two arms, different stamps, identical output, read
    as 'the transform had no effect'. A fabricated null is worse than a crash.

    Contrast and brightness are implemented as of #161; the refusal has to survive intact
    for everything else, which is what this now checks (`equalize-hist` is a real candidate
    — the scanner side measured it blinding MediaPipe on the detection crop)."""

    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    detector._mode = 1
    for config, expected in (
        (_config(preprocess=(PreprocessStep("equalize-hist"),)), "equalize-hist"),
        (_config(preprocess=(PreprocessStep("gamma", {"value": 2.2}),)), "gamma"),
        (_config(crop=mj.CROP_ADAPTIVE), "adaptive"),
    ):
        try:
            detector._require_supported(config)
        except NotImplementedError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{config} must be refused, not silently ignored")
    # ...and a detector built for one mode must not run another's config, which would
    # stamp the wrong arm onto real output.
    try:
        detector._require_supported(_config(mode=2))
    except ValueError:
        pass
    else:
        raise AssertionError("a mode mismatch must be refused")


def test_unknown_detection_mode_is_refused_at_construction():
    try:
        mj.MediaPipeDetector(mode=7)
    except ValueError:
        return
    raise AssertionError("an unknown mode must not silently pick a model bundle")


# --------------------------------------------------------------------------- #
# Model pinning — the weights are part of the arm, and are never fetched at run time
# --------------------------------------------------------------------------- #

def test_model_sha_moves_the_config_hash():
    """`mode` names a bundle; it does not identify weights. Upstream publishes at a
    `latest` URL, so without the sha in the arm identity a republished bundle would arrive
    under an unchanged stamp and two arms built from different weights would pool as one —
    issue #149 verbatim, on the detection side."""

    base = config_hash(_config())
    assert config_hash(replace(_config(), model_sha="a" * 64)) != base
    assert config_hash(replace(_config(), model_sha="b" * 64)) != \
           config_hash(replace(_config(), model_sha="a" * 64))


def test_absent_model_sha_is_omitted_so_stub_hashes_stay_stable():
    """Omitted, not null: a stub-backed arm hashes exactly as it did before pinning
    existed, which is what keeps the pure-core tests meaningful."""

    assert "modelSha" not in _config().identity()
    assert mj.MODEL_LOCK_PATH.name == "mediapipe.lock.json"


def test_every_mode_is_pinned_in_the_lock_file():
    """A mode that resolves to no pin is a mode whose runs cannot be attributed to
    weights. All three must be pinned before any sweep, not discovered mid-batch."""

    for mode in mj.DETECTION_MODES:
        entry = mj.pinned_model(mode)
        assert len(entry["sha256"]) == 64, mode
        assert entry["size"] > 0, mode


def test_the_job_stamps_model_identity_from_the_detector_not_the_caller():
    """Read off the detector that will run, so the stamp records what ran rather than
    what someone declared — the discipline vitpose_job applies for issue #149."""

    class Pinned(StubDetector):
        model_sha = "c" * 64

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        StubDetector.builds = []
        run_mediapipe_job(tmp / "analysis", _job_request(repeats=1), lambda c: Pinned(c))
        stamped = json.loads(
            (bundle / mj.STATUS_NAME).read_text())["config"]
        assert stamped["modelSha"] == "c" * 64
        experiment = _pose_runs(bundle)[0]["data"]["diagnostics"]["experiment"]
        assert experiment["config"]["modelSha"] == "c" * 64
        # The run id, the sidecar and the artifact must all name the same arm.
        assert experiment["configHash"] == config_hash(
            replace(_config(), model_sha="c" * 64))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_bundle_that_does_not_match_its_pin_is_refused():
    """Never adopt silently. A run produced by weights the lock does not describe cannot
    be attributed to an arm, which is the failure the whole pinning scheme prevents."""

    tmp = Path(tempfile.mkdtemp())
    try:
        name = mj.MODEL_BUNDLES[1]
        (tmp / f"{name}.task").write_bytes(b"not the pinned weights")
        os.environ[mj.MODEL_DIR_ENV] = str(tmp)
        try:
            mj.resolve_model_bundle(1)
        except ValueError as exc:
            assert "does not match the pin" in str(exc)
            assert "fetch_mediapipe_models" in str(exc)
        else:
            raise AssertionError("an unpinned bundle must be refused, not run")
        # ...and an absent bundle names the fetch step rather than silently downloading.
        (tmp / f"{name}.task").unlink()
        try:
            mj.resolve_model_bundle(1)
        except FileNotFoundError as exc:
            assert "fetch_mediapipe_models" in str(exc)
        else:
            raise AssertionError("a missing bundle must be refused, not downloaded")
    finally:
        os.environ.pop(mj.MODEL_DIR_ENV, None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_nothing_is_fetched_at_run_time():
    """A batch must not depend on a network round trip, and an arm must not depend on
    what upstream happened to be serving that afternoon."""

    source = Path("mediapipe_job.py").read_text(encoding="utf-8")
    assert "urlopen" not in source, (
        "mediapipe_job must not download; fetching lives in "
        "scripts/fetch_mediapipe_models.py so an arm is pinned before it runs"
    )


# --------------------------------------------------------------------------- #
# Tracked crop (issue #169) — the arm, the coordinates, and the truth firewall
# --------------------------------------------------------------------------- #

class _Landmark:
    def __init__(self, x, y, v=0.9):
        self.x, self.y, self.visibility = x, y, v


class _Landmarker:
    def __init__(self, points): self._points = points
    def detect(self, image): return type("R", (), {"pose_landmarks": [self._points]})()


class _Mp:
    ImageFormat = type("F", (), {"SRGB": 1})
    class Image:
        def __init__(self, **kw): pass


class _Cv2:
    COLOR_BGR2RGB = 0
    def cvtColor(self, image, code): return image


def test_cropped_keypoints_land_in_full_frame_coordinates():
    """The load-bearing conversion. A cropped arm whose keypoints stayed in crop coordinates
    would be silently uncomparable with every uncropped arm, and the error would read as a
    detection-quality difference rather than a units bug."""

    import numpy as np
    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    landmarker = _Landmarker([_Landmark(0.5, 0.5)])       # dead centre of whatever it sees

    # Crop spanning x 0.2-0.6 and y 0.4-0.8: the crop's centre is (0.4, 0.6) full-frame.
    cropped = detector._pose_region(
        landmarker, _Mp, _Cv2(), ["nose"], frame, (0.2, 0.4, 0.6, 0.8))
    assert abs(cropped[0].x - 0.4) < 1e-9
    assert abs(cropped[0].y - 0.6) < 1e-9

    # Uncropped, the same landmark is simply the frame centre — same space, no conversion.
    full = detector._pose_region(landmarker, _Mp, _Cv2(), ["nose"], frame, None)
    assert (abs(full[0].x - 0.5), abs(full[0].y - 0.5)) < (1e-9, 1e-9)


def test_a_degenerate_crop_returns_nothing_rather_than_garbage():
    import numpy as np
    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    landmarker = _Landmarker([_Landmark(0.5, 0.5)])
    assert detector._pose_region(
        landmarker, _Mp, _Cv2(), ["nose"], frame, (0.5, 0.5, 0.51, 0.51)) == []


def test_the_crop_trajectory_joins_the_arm_identity():
    """Two arms cropped by trajectories from different tracker settings are not comparable.
    `crop` names a policy; the trajectory decides which pixels the detector actually saw."""

    base = config_hash(_config(crop=mj.CROP_TRACKED))
    assert config_hash(replace(_config(crop=mj.CROP_TRACKED), crop_track_hash="a" * 16)) != base
    assert config_hash(replace(_config(crop=mj.CROP_TRACKED), crop_track_hash="a" * 16)) != \
           config_hash(replace(_config(crop=mj.CROP_TRACKED), crop_track_hash="b" * 16))
    # Omitted when absent, so uncropped arms hash exactly as they did before #169.
    assert "cropTrackHash" not in _config().identity()


def test_tracked_crop_without_a_trajectory_is_refused_not_silently_full_frame():
    """Falling back to full frame would write runs stamped as a cropped arm that never saw a
    crop — and on 24% of Bundles full frame detects nothing, so the arm would read as a
    catastrophic regression that never happened."""

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        try:
            run_mediapipe_job(
                tmp / "analysis",
                _job_request(config=_config(crop=mj.CROP_TRACKED)),
                _stub_factory(),
            )
        except FileNotFoundError as exc:
            assert "crop-track.json" in str(exc)
        else:
            raise AssertionError("a tracked arm with no trajectory must be refused")
        assert json.loads((bundle / mj.STATUS_NAME).read_text())["status"] == "error"
        assert not list((bundle / "detections").glob("*_pose.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_detector_refuses_a_tracked_config_with_no_track_handed_to_it():
    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    detector._mode = 1
    try:
        detector.detect(Path("x.mp4"), (0.0,), _config(crop=mj.CROP_TRACKED))
    except ValueError as exc:
        assert "crop trajectory" in str(exc)
    else:
        raise AssertionError("tracked crop with no trajectory must not run full-frame")


def test_the_crop_path_never_reads_truth():
    """Seeding from ViTPose or Ground Truth hands the detector the answer: detection rate
    would approach 100% and mean nothing. Asserted against the source, not left to
    convention, because it is the path of least resistance and invisible in results."""

    import ast

    tree = ast.parse(Path("crop_track.py").read_text(encoding="utf-8"))
    # Docstrings are excluded on purpose: this module's prose *explains* the prohibition, so
    # a plain text search finds the very sentence forbidding the thing. Only executable
    # string constants count.
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
    for text in literals:
        low = text.lower()
        assert "vitpose" not in low and "ground-truth" not in low and "groundtruth" not in low, \
            f"crop_track must not reach for truth: {text!r}"
    # The only artifact it names is its own.
    assert "crop-track.json" in literals

    # ...and the tracker seed in the job comes from calibration, by name.
    job = Path("mediapipe_job.py").read_text(encoding="utf-8")
    assert "climberPoint" in job, "the tracker must seed from the setup tap"


# --------------------------------------------------------------------------- #
# Frame sampling (issue #159) — 12·√n, deterministic, never contiguous
# --------------------------------------------------------------------------- #

def test_sampling_keeps_twelve_root_n_frames():
    """Measured across 55 Bundles at median 0.0017 / p90 0.0056 |dPCK| against the full
    grid — a p90 essentially equal to #134's 0.0055 noise floor, which is what makes it a
    principled stopping point rather than a taste call."""

    import math
    for n in (76, 331, 701, 1811, 5977):
        kept = mj.sample_timestamps(tuple(float(i) for i in range(n)))
        assert len(kept) == min(n, int(12 * math.sqrt(n))), n
    # A short Bundle keeps everything rather than being padded or truncated.
    assert len(mj.sample_timestamps(tuple(range(76)))) == 76


def test_the_frame_set_is_a_pure_function_of_the_bundle():
    """Mode batches run on different days. A frame set chosen at batch time would hand the
    three modes three different frame sets, reintroducing across batches exactly the
    confound the design removes within one."""

    grid = tuple(round(i * 0.1, 1) for i in range(701))
    first = mj.sample_timestamps(grid)
    second = mj.sample_timestamps(tuple(list(grid)))
    assert first == second
    # ...and it is stable across separately-constructed requests, which is how a batch
    # actually reaches it.
    a = _request(frames=mj.sample_timestamps(grid))
    b = _request(frames=mj.sample_timestamps(grid))
    assert a.frames == b.frames


def test_sampling_spans_the_whole_grid_and_is_never_contiguous():
    """Eight contiguous 300-frame windows of one Bundle — same run, same truth — produced
    PCK from 0.104 to 0.839. That 0.735 spread is ~130x the noise floor, so a contiguous
    sample would swamp the experiment with frame-choice noise wearing an arm's name."""

    grid = tuple(float(i) for i in range(2000))
    kept = mj.sample_timestamps(grid)
    assert kept[0] == grid[0] and kept[-1] == grid[-1], "must span the full climb"
    gaps = {round(b - a, 6) for a, b in zip(kept, kept[1:])}
    assert min(gaps) > 1.0, "consecutive frames would be a contiguous stretch"
    # Evenly spread, not stride-and-truncate — which silently drops the video's tail.
    assert max(gaps) - min(gaps) <= 1.0


def test_sampling_handles_degenerate_grids():
    assert mj.sample_timestamps(()) == ()
    assert mj.sample_timestamps((4.0,)) == (4.0,)


# --------------------------------------------------------------------------- #
# Bundle selection (issue #159) — thresholded exclusion, every drop recorded
# --------------------------------------------------------------------------- #

def _truth(bundle: Path, n: int = 10, wrong: int = 0) -> None:
    frames = [{"frameIndex": i, "timestamp": i * 0.1, "state": "present",
               "review": "human-flagged-wrong" if i < wrong else "auto", "joints": {}}
              for i in range(n)]
    (bundle / "ground-truth.json").write_text(json.dumps({"frames": frames}))


def _selectable(tmp: Path, route: str, key: str, *, wrong: int = 0, truth: bool = True,
                video: bool = True, setup: bool = True) -> Path:
    bundle = tmp / "analysis" / route / key
    bundle.mkdir(parents=True)
    if video:
        (bundle / f"{key}.mp4").write_bytes(b"v")
    if setup:
        (bundle / "setup.json").write_text(json.dumps({"setupHash": f"sh-{key}"}))
    if truth:
        _truth(bundle, 10, wrong)
    return bundle


def test_selection_excludes_only_badly_wrong_truth_and_says_why():
    """A threshold, not all seven. `evaluate` already drops human-flagged-wrong frames from
    scoring (ADR 0004/0005), so excluding whole Bundles would discard ~9,990 good truth
    frames to remove 2,113 bad ones. But not zero either: wrong-*person* truth points at a
    specific other human, so an arm latching onto the same bystander gets rewarded — the
    one error that does not cancel between arms."""

    tmp = Path(tempfile.mkdtemp())
    try:
        _selectable(tmp, "r", "clean")
        _selectable(tmp, "r", "slightly", wrong=1)     # 10% — kept
        _selectable(tmp, "r", "badly", wrong=6)        # 60% — dropped
        sel = mj.select_bundles(tmp / "analysis")
        assert ("r", "clean") in sel.included
        assert ("r", "slightly") in sel.included, "10% wrong must not cost the whole Bundle"
        assert ("r", "badly") not in sel.included
        dropped = {e["videoKey"]: e for e in sel.excluded}
        assert dropped["badly"]["reason"] == "wrong-person-truth"
        assert dropped["badly"]["wrongShare"] == 0.6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selection_records_every_reason_a_bundle_cannot_run():
    """A batch that silently skipped a Bundle would produce a pooled number over a
    population nobody can reconstruct."""

    tmp = Path(tempfile.mkdtemp())
    try:
        _selectable(tmp, "r", "ok")
        _selectable(tmp, "r", "notruth", truth=False)
        _selectable(tmp, "r", "novideo", video=False)
        _selectable(tmp, "r", "nosetup", setup=False)
        sel = mj.select_bundles(tmp / "analysis")
        reasons = {e["videoKey"]: e["reason"] for e in sel.excluded}
        assert reasons == {"notruth": "no-truth", "novideo": "no-video",
                           "nosetup": "no-setup-hash"}
        assert sel.as_dict()["includedCount"] == 1
        assert sel.as_dict()["excludedCount"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_explicit_subset_is_still_filtered_and_a_missing_bundle_is_named():
    tmp = Path(tempfile.mkdtemp())
    try:
        _selectable(tmp, "r", "ok")
        _selectable(tmp, "r", "badly", wrong=9)
        sel = mj.select_bundles(tmp / "analysis",
                                only=[("r", "ok"), ("r", "badly"), ("r", "ghost")])
        assert sel.included == (("r", "ok"),)
        reasons = {e["videoKey"]: e["reason"] for e in sel.excluded}
        assert reasons == {"badly": "wrong-person-truth", "ghost": "no-bundle"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The batch — single-flight, per-Bundle isolation, progress
# --------------------------------------------------------------------------- #

def test_run_ids_carry_the_experiment_prefix():
    """So a detections/ listing is self-describing, a selective wipe is a glob rather than
    a JSON scan, and every aggregation has a trivially correct segregation key."""

    run_ts = mj.pass_run_ts("20260801-120000", _config(), 0)
    assert run_ts.startswith(mj.RUN_ID_PREFIX)

    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _bundle(tmp)
        run_mediapipe_job(tmp / "analysis", _job_request(repeats=1), _stub_factory())
        # Visible in the envelope run_ts, not only the filename — evaluate reads the
        # envelope, and a prefix that lived only in the filename would not segregate.
        assert _pose_runs(bundle)[0]["run_ts"].startswith(mj.RUN_ID_PREFIX)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_batch_sweeps_every_eligible_bundle_and_records_its_selection():
    tmp = Path(tempfile.mkdtemp())
    try:
        for key in ("a", "b"):
            _selectable(tmp, "r", key)
        _selectable(tmp, "r", "badly", wrong=9)
        result = mj.run_batch(tmp / "analysis", _config(), _stub_factory(), repeats=1)
        assert result["status"] == "done"
        assert result["bundlesRun"] == 2 and result["bundlesFailed"] == 0
        assert result["runsWritten"] == 2
        # The exclusion is visible in the batch's own record, not just in a log line.
        assert result["selection"]["excludedCount"] == 1
        assert result["selection"]["excluded"][0]["reason"] == "wrong-person-truth"
        # The batch reports the arm actually *written*, not the config it was handed —
        # model sha and crop trajectory are stamped per Bundle inside the job, so the
        # requested config is not yet an arm.
        assert result["armsWritten"] == [config_hash(_config())]
        assert result["armsMixed"] is False
        status = json.loads((tmp / "analysis" / mj.BATCH_STATUS_NAME).read_text())
        assert status["status"] == "done"
        assert [b["videoKey"] for b in status["bundles"]] == ["a", "b"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_each_bundle_is_sampled_to_twelve_root_n():
    tmp = Path(tempfile.mkdtemp())
    try:
        bundle = _selectable(tmp, "r", "a")
        _truth(bundle, 400)
        result = mj.run_batch(tmp / "analysis", _config(), _stub_factory(), repeats=1)
        entry = result["bundles"][0]
        assert entry["truthFrames"] == 400
        assert entry["sampledFrames"] == len(mj.sample_timestamps(tuple(range(400))))
        assert entry["sampledFrames"] < 400, "the whole point is not sampling everything"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_batch_flags_when_its_bundles_did_not_share_one_arm():
    """More than one arm in a batch means the Bundles did not share a trajectory or a
    model, so the runs are not one experimental condition. Surfaced by the batch rather
    than discovered at analysis time, where it would look like a real arm difference."""

    tmp = Path(tempfile.mkdtemp())
    try:
        for key in ("a", "b"):
            _selectable(tmp, "r", key)

        class Drifting(StubDetector):
            """Reports a different model per Bundle — the shape of a mid-batch re-pin.

            Keyed on ``index // 2`` because each Bundle builds two detectors: a probe the
            job reads model identity from, then the one that actually sweeps.
            """
            @property
            def model_sha(self):
                return ("a" if self.index // 2 == 0 else "b") * 64

        StubDetector.builds = []
        result = mj.run_batch(tmp / "analysis", _config(),
                              lambda c: Drifting(c), repeats=1)
        assert len(result["armsWritten"]) == 2
        assert result["armsMixed"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_one_bad_bundle_does_not_cost_the_rest_of_the_sweep():
    """A sweep takes hours. One unreadable video must not end it."""

    tmp = Path(tempfile.mkdtemp())
    try:
        for key in ("a", "boom", "c"):
            _selectable(tmp, "r", key)

        def factory(config):
            det = StubDetector(config)
            real = det.detect

            def detect(video_path, timestamps, config, crop_track=None):
                if "boom" in str(video_path):
                    raise RuntimeError("unreadable video")
                return real(video_path, timestamps, config, crop_track)

            det.detect = detect
            return det

        StubDetector.builds = []
        result = mj.run_batch(tmp / "analysis", _config(), factory, repeats=1)
        assert result["status"] == "done", "the batch itself completes"
        assert result["bundlesRun"] == 2 and result["bundlesFailed"] == 1
        failed = [b for b in result["bundles"] if b["status"] == "error"][0]
        assert failed["videoKey"] == "boom"
        assert failed["errorType"] == "RuntimeError"
        assert "Traceback" in failed["traceback"]
        # The other two still wrote their runs.
        assert result["runsWritten"] == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_second_batch_is_refused_while_one_is_running():
    """Two batches would interleave writes into one Bundle, share a base timestamp, race on
    run ids, and produce a repeat set whose members came from different arms."""

    tmp = Path(tempfile.mkdtemp())
    try:
        _selectable(tmp, "r", "a")
        seen = {}

        def factory(config):
            det = StubDetector(config)
            real = det.detect

            def detect(video_path, timestamps, config, crop_track=None):
                # Re-enter while the first batch holds the lock.
                assert mj.batch_is_running(tmp / "analysis")
                try:
                    mj.run_batch(tmp / "analysis", _config(), _stub_factory(), repeats=1)
                except RuntimeError as exc:
                    seen["refused"] = str(exc)
                return real(video_path, timestamps, config, crop_track)

            det.detect = detect
            return det

        StubDetector.builds = []
        mj.run_batch(tmp / "analysis", _config(), factory, repeats=1)
        assert "already running" in seen.get("refused", "")
        # ...and the lock is released afterwards, so the next batch can start.
        assert not mj.batch_is_running(tmp / "analysis")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_module_imports_and_runs_without_mediapipe_installed():
    """The property test_vitpose_job.py has with torch, and the reason this suite runs
    anywhere: everything that decides what a run *means* is pure, and the heavy dependency
    is reachable only through the backend's own methods."""

    assert "mediapipe" not in sys.modules, (
        "importing mediapipe_job must not import mediapipe"
    )
    # Every model bundle is named, so a mode can never resolve to nothing.
    assert set(mj.MODEL_BUNDLES) == set(mj.DETECTION_MODES)


def _run_all():
    fns = [fn for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


if __name__ == "__main__":
    sys.exit(_run_all())
