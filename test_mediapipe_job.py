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

    def detect(self, video_path, timestamps, config):
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

def test_repeats_default_to_a_floor_producing_count():
    """Issue #134: the historical corpus has six genuine repeat groups and therefore no
    usable floor. A batch must produce its own by default, not on request."""

    assert mj.DEFAULT_REPEATS >= 3
    assert _request().repeats == mj.DEFAULT_REPEATS or _request().repeats == 3


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
        def detect(self, video_path, timestamps, config):
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

        def detect(self, video_path, timestamps, config):
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
    """Preprocessing and crop are already part of the arm identity but nothing implements
    them. Running them as a no-op would write a run stamped 'contrast 1.5' whose pixels
    never saw contrast — two arms, different stamps, identical output, read as 'the
    transform had no effect'. A fabricated null is worse than a crash."""

    detector = mj.MediaPipeDetector.__new__(mj.MediaPipeDetector)
    detector._mode = 1
    for config, expected in (
        (_config(preprocess=(PreprocessStep("contrast", {"factor": 1.5}),)), "contrast"),
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
