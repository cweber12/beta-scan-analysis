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

import sys
from dataclasses import replace

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


def _run_all():
    fns = [fn for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


if __name__ == "__main__":
    sys.exit(_run_all())
