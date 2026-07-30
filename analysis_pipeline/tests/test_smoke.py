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


def _write_scaffold(video_dir: Path, samples: list[float], posed: list[float],
                    seed_found: bool | None = None) -> None:
    """A ViTPose scaffold + status sidecar — the evidence absence provenance is derived
    from (issue #101): which timestamps were sampled, which carry a posed Climber, and
    whether seeding succeeded at all."""

    video_dir.mkdir(parents=True, exist_ok=True)
    frames = [{"timestamp": t,
               "keypoints": _kp_list(_TRUTH_JOINTS) if t in posed else []}
              for t in samples]
    (video_dir / "vitpose.json").write_text(
        json.dumps({"version": 1, "frames": frames}), encoding="utf-8")
    if seed_found is not None:
        (video_dir / "vitpose.status.json").write_text(
            json.dumps({"status": "done", "seedDebug": {"seedFound": seed_found}}),
            encoding="utf-8")


def _evaluate(root, **kw):
    """``evaluate`` with the truth-sufficiency floor lowered to fit these fixtures.

    Issue #101 gates a real Bundle at 20 truth-present fit frames — the floor that
    quarantines a Bundle whose near-perfect fit rests on eleven frames. The synthetic
    Bundles here carry three to five frames by design, so every one of them would
    quarantine and each test would stop being about its own subject. The production
    floor itself is asserted, unpatched, in
    ``test_truth_sufficiency_floor_quarantines_a_thin_bundle``.
    """

    from analysis_pipeline import evaluate as ev

    original = ev.CONFORMANCE_MIN_FIT_FRAMES
    ev.CONFORMANCE_MIN_FIT_FRAMES = 2
    try:
        return ev.evaluate(root, **kw)
    finally:
        ev.CONFORMANCE_MIN_FIT_FRAMES = original


def _write_run(video_dir: Path, stem: str, *, video_hash: str, setup_hash: str,
               config: dict, labels: dict, det_rate: float, written_at: str,
               overlay_quality: float | None = None, bad_stretches: list | None = None,
               provenance: bool = False) -> None:
    det = video_dir / "detections"
    det.mkdir(parents=True, exist_ok=True)
    # When provenance is requested, tag frames with a source + per-frame region
    # stats (the Phase 2 export contract) so the exported-stats path is exercised.
    sources = ["raw", "raw", "interpolated", "flipDiscarded"]
    frames = []
    for i in range(4):
        fr = {"timestamp": round(i * 1.0, 1),
              "keypoints": [{"name": "nose", "x": 0.5, "y": 0.5, "score": 0.9},
                            {"name": "left_shoulder", "x": 0.4, "y": 0.6, "score": 0.8}]}
        if provenance:
            fr["source"] = sources[i]
            fr["climber"] = {"mean": 70.0 + i, "stdDev": 25.0, "sharpness": 90.0 + i}
            fr["wall"] = {"mean": 80.0, "stdDev": 20.0, "sharpness": 100.0 + i}
        frames.append(fr)
    result_pose = {"sampledFrames": 4, "detectedFrames": int(det_rate * 4),
                   "detectionRate": det_rate, "flippedFrames": 0,
                   "goodFrames": 3, "confidence": {"avg": 0.88, "min": 0.7},
                   "avgKeypointCount": 20.0}
    result: dict = {"pose": result_pose, "badStretches": bad_stretches or []}
    if overlay_quality is not None:
        result["overlayQuality"] = overlay_quality
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
                "result": result,
            },
            "frames": frames,
        },
    }
    orb = {"video_key": video_dir.name, "run_ts": stem, "type": "orb",
           "data": {"referenceFrameMeta": {"refKeypointCount": 2000,
                                           "wall": {"sharpness": 100.0}}, "summary": {}}}
    (det / f"{stem}_pose.json").write_text(json.dumps(pose), encoding="utf-8")
    (det / f"{stem}_orb.json").write_text(json.dumps(orb), encoding="utf-8")
    # metadata.json now carries only source/structural facts; the condition labels
    # live in setup.json.analysisInputs (scanner-written at calibration).
    md = {"route_folder": video_dir.parent.name, "video_key": video_dir.name}
    (video_dir / "metadata.json").write_text(json.dumps(md), encoding="utf-8")
    (video_dir / "setup.json").write_text(
        json.dumps({"climberCrop": {"x": 0.4, "y": 0.5, "w": 0.1, "h": 0.4},
                    "wallCrop": {"x": 0.3, "y": 0.2, "w": 0.3, "h": 0.6},
                    "setupHash": setup_hash,
                    "analysisInputs": labels}), encoding="utf-8")


def _write_video_stats(video_dir: Path, setup_hash: str) -> None:
    """A minimal phase-2 video-stats.json + phase-1 metadata block (issue #23)."""
    doc = {
        "version": 1, "setupHash": setup_hash, "source": "endpoint",
        "regionStats": {
            "panningFlagged": False,
            "wall": {"luma": {"mean": 140.0}, "rmsContrast": 0.12,
                     "texture": {"edgeDensity": 0.08, "laplacianVar": 210.0},
                     "hue": {"meanDeg": 30.0, "concentration": 0.9},
                     "saturation": {"mean": 60.0, "std": 12.0}},
            "climberWall": {"deltaE": 32.5, "lumaSeparation": 18.0},
            "shadow": {"fraction": {"mean": 0.22, "std": 0.02},
                       "inOutLumaRatio": 0.55,
                       "blobs": {"count": 3, "largestFraction": 0.4},
                       "drift": {"range": 0.05}},
        },
        "suggestions": {"shadows": "patchy"},
        "cameraAngle": {"estimate": "level", "source": "vitpose"},
    }
    (video_dir / "video-stats.json").write_text(json.dumps(doc), encoding="utf-8")
    metadata_path = video_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["video_stats"] = {
        "luma": {"mean": 120.0, "std": 30.0, "p5": 40.0, "p95": 220.0},
        "clippedHighlightFraction": 0.01, "crushedShadowFraction": 0.0,
        "rmsContrast": 0.15, "sharpness": {"mean": 180.0, "min": 90.0},
        "frameDiff": {"mean": 0.02, "max": 0.05},
        "exposureDrift": {"slopePerMinute": 1.2, "range": 6.0},
        "colorCast": {"rOverG": 1.05, "bOverG": 0.92}, "bitsPerPixel": 0.11,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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
    # video B, C: distinct. vidB exercises the Phase 2 export contract
    # (overlayQuality + badStretches + per-frame provenance/region stats).
    _write_run(root / "routeB" / "vidB", "20260102-000001", video_hash="hb", setup_hash="sb",
               config=cfg, labels={**base_labels, "route_orientation": "head-on"},
               det_rate=1.0, written_at="2026-01-02T00:00:00",
               overlay_quality=0.82, bad_stretches=[{"startSec": 1.0, "endSec": 1.5, "reason": "flip"}],
               provenance=True)
    _write_run(root / "routeC" / "vidC", "20260103-000001", video_hash="hc", setup_hash="sc",
               config=cfg, labels={**base_labels, "route_orientation": "right",
                                   "camera_stability": "moving"},
               det_rate=0.9, written_at="2026-01-03T00:00:00")
    # Video Stats (issue #23): vidB's stats match its run's setupHash (fresh);
    # vidC's were computed under an older calibration (stale). vidA has none.
    _write_video_stats(root / "routeB" / "vidB", setup_hash="sb")
    _write_video_stats(root / "routeC" / "vidC", setup_hash="sc_OLD")


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

        # New pose-outcome columns (ADR 0001), populated only for vidB.
        assert "out_overlayQuality" in run_df.columns
        assert "out_badStretchSeconds" in run_df.columns
        vidb = run_df.set_index("video_key").loc["vidB"]
        assert abs(float(vidb["out_overlayQuality"]) - 0.82) < 1e-9
        assert abs(float(vidb["out_badStretchSeconds"]) - 0.5) < 1e-9
        assert run_df.set_index("video_key").loc["vidA"]["out_overlayQuality"] is None \
            or pd.isna(run_df.set_index("video_key").loc["vidA"]["out_overlayQuality"])

        # Video Stats predictor columns (issue #23) + the staleness flag.
        by_key = run_df.set_index("video_key")
        assert by_key.loc["vidB", "vs_climberWallDeltaE"] == 32.5
        assert by_key.loc["vidB", "vs_shadowBlobCount"] == 3
        assert by_key.loc["vidB", "src_sharpnessMean"] == 180.0
        assert by_key.loc["vidB", "src_bitsPerPixel"] == 0.11
        assert by_key.loc["vidB", "vs_cameraAngle"] == "level"
        assert by_key.loc["vidB", "vs_stale"] == False  # noqa: E712 — pandas object col
        assert by_key.loc["vidC", "vs_stale"] == True  # noqa: E712 — computed under sc_OLD
        assert by_key.loc["vidA", "vs_stale"] is None or pd.isna(by_key.loc["vidA", "vs_stale"])
        assert pd.isna(by_key.loc["vidA", "vs_climberWallDeltaE"])

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

        # Per-frame provenance columns exist; vidB carries real source tags and the
        # exported region stats (so raw_detected is a real 0/1 outcome there).
        assert {"source", "raw_detected"}.issubset(frame_df.columns)
        vidb_frames = frame_df[frame_df["video_key"] == "vidB"]
        assert set(vidb_frames["source"]) == {"raw", "interpolated", "flipDiscarded"}
        assert vidb_frames["raw_detected"].sum() == 2  # two "raw" frames
        assert vidb_frames["wall_sharpness"].notna().all()  # from the export, not decode

        corr = stats.within_run_correlations(frame_df)
        assert set(["predictor", "outcome", "mean_r", "n_runs"]).issubset(corr.columns) or corr.empty
        # velocity/coverage are constant here -> may be empty; ensure it runs without error
        assert isinstance(corr, pd.DataFrame)


def test_cliffs_delta_bounds():
    assert stats.cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    assert stats.cliffs_delta([1, 2], [3, 4, 5]) == -1.0
    assert stats.cliffs_delta([2, 2], [2, 2]) == 0.0
    assert stats.cliffs_delta([], [1]) is None


def _write_matrix(path: Path, keys_routes: dict[str, str], same_hi=0.7, cross_lo=0.03) -> None:
    """Fabricate an orb_match_matrix.json over the given {key: route} videos."""

    pairs = []
    for rk, rr in keys_routes.items():
        for qk, qr in keys_routes.items():
            same = rr == qr
            ratio = 1.0 if rk == qk else (same_hi if same else cross_lo)
            pairs.append({
                "trainKey": rk, "trainRoute": rr, "queryKey": qk, "queryRoute": qr,
                "sameRoute": same, "matches": 100, "inliers": int(round(ratio * 100)),
                "inlierRatio": ratio, "homographyValid": same, "reprojErrorPx": 3.0 if same else None,
            })
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")


def test_crossmatch_reducers():
    from analysis_pipeline import crossmatch

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mpath = root / "orb_match_matrix.json"
        # two routes, two videos each -> off-diagonal same-route pairs exist
        _write_matrix(mpath, {"r1a": "route1", "r1b": "route1",
                              "r2a": "route2", "r2b": "route2"})
        df = crossmatch.load_match_matrix(mpath)
        assert len(df) == 16

        sep = crossmatch.separation_stats(df)
        assert sep["available"]
        assert sep["same_mean"] > sep["cross_mean"]
        assert sep["auc"] == 1.0  # perfectly separable
        assert sep["n_same"] == 4 and sep["n_cross"] == 8

        thr = crossmatch.best_threshold(df)
        assert thr["available"] and thr["f1"] == 1.0

        mtx = crossmatch.ordered_matrix(df)
        assert mtx["available"] and len(mtx["keys"]) == 4
        assert len(mtx["values"]) == 4 and len(mtx["values"][0]) == 4

        # missing / malformed file -> empty, no crash
        assert crossmatch.load_match_matrix(root / "nope.json").empty
        assert not crossmatch.separation_stats(crossmatch.load_match_matrix(root / "nope.json"))["available"]


def test_pipeline_end_to_end_renders_report():
    from analysis_pipeline import cli

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        out = Path(tmp) / "reports"
        out.mkdir(parents=True, exist_ok=True)
        _build_corpus(root)
        _write_matrix(out / "orb_match_matrix.json",
                      {"vidA": "routeA", "vidB": "routeB", "vidC": "routeC"})

        outputs = cli.run(root, out, decode=False, matrix=out / "orb_match_matrix.json")
        html_text = outputs["html"].read_text(encoding="utf-8")
        for header in ("Corpus quality overview", "Per-video failure cards",
                       "ORB cross-match", "Per-frame failure timeline"):
            assert header in html_text, f"missing report section: {header}"


# --------------------------------------------------------------------------- #
# evaluate subcommand (issue #6)
# --------------------------------------------------------------------------- #

# A truth skeleton with a torso length of exactly 0.3 (shoulder-mid (0.5,0.4) to
# hip-mid (0.5,0.7)) -> PCK@0.5-torso threshold is 0.15.
_TRUTH_JOINTS = {
    "nose": (0.5, 0.2),
    "left_shoulder": (0.4, 0.4), "right_shoulder": (0.6, 0.4),
    "left_elbow": (0.35, 0.5), "right_elbow": (0.65, 0.5),
    "left_wrist": (0.3, 0.6), "right_wrist": (0.7, 0.6),
    "left_hip": (0.4, 0.7), "right_hip": (0.6, 0.7),
    "left_knee": (0.4, 0.85), "right_knee": (0.6, 0.85),
    "left_ankle": (0.4, 0.95), "right_ankle": (0.6, 0.95),
}


def _kp_list(joints: dict) -> list:
    return [{"name": n, "x": x, "y": y, "score": 0.9} for n, (x, y) in joints.items()]


def _write_pose_run(video_dir: Path, stem: str, setup_hash: str, frames: list,
                    app_version: str = "",
                    detector_attempts: list[dict] | None = None,
                    config: dict | None = None,
                    detector_code_hash: str | None = None) -> None:
    det = video_dir / "detections"
    det.mkdir(parents=True, exist_ok=True)
    diagnostics = {"appVersion": app_version} if app_version else {}
    if detector_code_hash is not None:
        # None omits the key (a record predating #130); "" writes an explicit null, the
        # scanner's "derivation failed" value. Both must read as unknown provenance.
        diagnostics["detectorCodeHash"] = detector_code_hash or None
    if config is not None:
        diagnostics["config"] = config
    data = {"setupHash": setup_hash, "diagnostics": diagnostics, "frames": frames}
    if detector_attempts is not None:
        data["detectorAttempts"] = detector_attempts
    env = {"video_key": video_dir.name, "route_folder": video_dir.parent.name,
           "run_ts": stem, "type": "pose",
           "data": data}
    (det / f"{stem}_pose.json").write_text(json.dumps(env), encoding="utf-8")


def _write_bundle_meta(video_dir: Path, setup_hash: str) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "metadata.json").write_text(
        json.dumps({"route_folder": video_dir.parent.name, "video_key": video_dir.name}),
        encoding="utf-8")
    (video_dir / "setup.json").write_text(
        json.dumps({"setupHash": setup_hash}), encoding="utf-8")


def _ground_truth_doc(setup_hash: str | None) -> dict:
    """Truth with the mix of frames the edge cases need."""

    frames = [
        {"frameIndex": 1, "timestamp": 1.0, "state": "present",
         "review": "auto", "joints": {n: {"x": x, "y": y, "occluded": False}
                                      for n, (x, y) in _TRUTH_JOINTS.items()}},
        {"frameIndex": 2, "timestamp": 2.0, "state": "present",
         "review": "auto", "joints": {n: {"x": x, "y": y, "occluded": False}
                                      for n, (x, y) in _TRUTH_JOINTS.items()}},
        {"frameIndex": 3, "timestamp": 3.0, "state": "absent",
         "review": "human-flagged-absent", "joints": {}},
        # torso-undefined: right_hip missing so shoulder/hip mid can't be formed
        {"frameIndex": 4, "timestamp": 4.0, "state": "present", "review": "auto",
         "joints": {n: {"x": x, "y": y, "occluded": False}
                    for n, (x, y) in _TRUTH_JOINTS.items() if n != "right_hip"}},
        # scanner-missing: present, torso defined, but no scanner frame near ts=9
        {"frameIndex": 9, "timestamp": 9.0, "state": "present", "review": "auto",
         "joints": {n: {"x": x, "y": y, "occluded": False}
                    for n, (x, y) in _TRUTH_JOINTS.items()}},
    ]
    doc: dict = {"version": 1, "jointSet": list(_TRUTH_JOINTS),
                 "frames": frames, "groundTruthHash": "abcdef1234567890"}
    if setup_hash is not None:
        doc["setupHash"] = setup_hash
    return doc


def _ground_truth_doc_with_annotations(setup_hash: str) -> dict:
    doc = _ground_truth_doc(setup_hash)
    doc["detectionAnnotations"] = [
        {"startFrame": 2, "endFrame": 3, "failureClass": "wrong-subject",
         "distractor": "tree_bush", "setupHash": setup_hash},
        {"startFrame": 1, "endFrame": 1, "failureClass": "distorted",
         "distractor": "gear", "setupHash": f"{setup_hash}_STALE"},
    ]
    return doc


def _scanner_frames_for_pck() -> list:
    """@1.0 matches truth exactly; @2.0 offsets nose and thins left_wrist;
    @3.0 hallucinates a pose on the truth-absent frame; @4.0 matches the
    torso-undefined truth frame (coverage evidence, not PCK/distance)."""

    f1 = {"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)}
    off = dict(_TRUTH_JOINTS)
    off["nose"] = (0.7, 0.2)      # 0.2 > 0.15 threshold -> nose wrong here
    off.pop("left_wrist")          # thinned scanner joint -> a miss
    f2 = {"timestamp": 2.0, "keypoints": _kp_list(off)}
    f3 = {"timestamp": 3.0, "keypoints": _kp_list(_TRUTH_JOINTS)}  # truth absent here
    f4 = {"timestamp": 4.0, "keypoints": _kp_list(_TRUTH_JOINTS)}  # torso-undefined
    return [f1, f2, f3, f4]


def test_evaluate_pck_exact_and_edge_cases():
    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeE" / "vidE"
        _write_bundle_meta(vdir, setup_hash="sh_match")
        # legacy ground-truth (no self setupHash) -> effective hash = setup.json's
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000001", "sh_match", _scanner_frames_for_pck())

        summary = _evaluate(root)
        assert len(summary.written) == 1 and not summary.skipped
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))

        # Record shape / provenance header.
        assert rec["schemaVersion"] == ev.SCHEMA_VERSION == 14
        assert rec["metrics"] == ["pck@0.5-torso", "normDistMedian", "normDistP90",
                                  "presence2x2", "jointCoverage"]
        assert rec["setupHash"] == "sh_match"
        assert rec["truthSource"] == "ground-truth"
        assert rec["truthHash"] == "abcdef1234567890"
        assert rec["truthSetupHashSource"] == "setup.json"
        assert rec["jointSet"] == ev.COCO_CORE_JOINTS

        counts = rec["counts"]
        # The t3 frame is a deprecated manual absent flag (ADR 0005): excluded from
        # scoring and reported in agreementSkipped. The rest are auto; there are no
        # flagged-wrong seeds, and nothing is accuracy-tier evidence.
        # The one absent frame carries an absence reason (issue #101); with no scaffold
        # in this synthetic bundle there is nothing to derive from, so it reads
        # `unknown` — never silently promoted to a confirmed absence.
        assert counts == {"truthFramesTotal": 5, "truthFramesPresent": 4,
                          "truthFramesAbsent": 1, "truthFramesVerified": 0,
                          "truthFramesOutOfScope": 0,
                          "absenceReasons": {"out-of-scope": 0, "not-sampled": 0,
                                             "untracked": 0, "confirmed-absent": 0,
                                             "unknown": 1},
                          "review": {"auto": 4, "flaggedWrong": 0, "flaggedAbsent": 1},
                          "agreementSkipped": {"flaggedWrong": 0, "flaggedAbsent": 1,
                                               "outOfScope": 0}}
        assert rec["climbWindow"] == {"start": None, "end": None}
        assert rec["scannerFrameIntervalSec"] == 1.0
        assert rec["joinToleranceSec"] == 0.5

        agr = rec["agreement"]
        # Frame accounting: t1/t2/t4 matched present, t9 has no scanner sample within
        # tolerance (unobserved). The t3 manual-absent frame is excluded entirely, so
        # it never reaches the presence 2x2 despite the scanner hallucinating there.
        assert agr["frames"] == {
            "truthFrames": 4, "verifiedFrames": 0,
            "matchedPresent": 3, "matchedAbsent": 0,
            "unmatchedPresent": 1, "unmatchedAbsent": 0,
            "lowVisibility": 0, "torsoUndefined": 1, "scoreable": 2,
            "unconfirmedAbsent": 0}
        assert agr["presence"] == {"presentDetected": 3, "presentUndetected": 0,
                                   "absentDetected": 0, "absentUndetected": 0}

        # Visible-joint histogram over the 3 matched-present frames (measure-first,
        # excludes nothing): t1/t2 carry all 13 truth joints, t4 drops right_hip
        # (torso-undefined) -> 12. Positional list over 0..13; sum == matchedPresent.
        assert len(agr["visibleJoints"]) == 14
        assert agr["visibleJoints"][13] == 2 and agr["visibleJoints"][12] == 1
        assert sum(agr["visibleJoints"]) == agr["frames"]["matchedPresent"] == 3
        assert agr["frames"]["lowVisibility"] == 0  # gate disabled in v1

        pj = agr["perJoint"]
        # nose wrong in frame2 -> 1/2; its normalized dists are [0, 0.2/0.3].
        assert pj["nose"]["pck"] == {"correct": 1, "total": 2, "value": 0.5}
        assert pj["nose"]["normDist"] == {"n": 2, "median": 0.333333, "p90": 0.6}
        assert pj["nose"]["coverage"] == {"emitted": 3, "frames": 3, "rate": 1.0}
        # left_wrist thinned in frame2 -> a PCK miss AND a coverage gap; the one
        # emitted observation is exact so its distances collapse to zero.
        assert pj["left_wrist"]["pck"] == {"correct": 1, "total": 2, "value": 0.5}
        assert pj["left_wrist"]["normDist"] == {"n": 1, "median": 0.0, "p90": 0.0}
        assert pj["left_wrist"]["coverage"] == {"emitted": 2, "frames": 3,
                                                "rate": 0.666667}
        # every other core joint matched exactly on both scoreable frames.
        for name in ev.COCO_CORE_JOINTS:
            if name in ("nose", "left_wrist"):
                continue
            assert pj[name]["pck"] == {"correct": 2, "total": 2, "value": 1.0}, name
            assert pj[name]["normDist"] == {"n": 2, "median": 0.0, "p90": 0.0}, name
            assert pj[name]["coverage"] == {"emitted": 3, "frames": 3,
                                            "rate": 1.0}, name

        agg = agr["aggregate"]
        assert agg["pck"] == {"correct": 24, "total": 26, "value": 0.923077}
        assert agg["normDist"] == {"n": 25, "median": 0.0, "p90": 0.0}
        assert agg["coverage"] == {"emitted": 38, "frames": 39, "rate": 0.974359}

        # Accuracy tier: no trustworthy human attestation exists (ADR 0005 retired
        # manual-absent as evidence), so the block is present with explicit zero
        # counts and null metrics — represented, never dropped.
        acc = rec["accuracy"]
        assert acc["frames"] == {
            "truthFrames": 0, "verifiedFrames": 0,
            "matchedPresent": 0, "matchedAbsent": 0,
            "unmatchedPresent": 0, "unmatchedAbsent": 0,
            "lowVisibility": 0, "torsoUndefined": 0, "scoreable": 0,
            "unconfirmedAbsent": 0}
        assert sum(acc["visibleJoints"]) == 0
        assert acc["presence"] == {"presentDetected": 0, "presentUndetected": 0,
                                   "absentDetected": 0, "absentUndetected": 0}
        assert acc["perJoint"]["nose"] == {
            "pck": {"correct": 0, "total": 0, "value": None},
            "normDist": {"n": 0, "median": None, "p90": None},
            "coverage": {"emitted": 0, "frames": 0, "rate": None}}
        assert acc["aggregate"]["pck"]["value"] is None
        assert acc["aggregate"]["normDist"] == {"n": 0, "median": None, "p90": None}

        # Conformance block (issue #15). The fixture's scanner is identity on truth
        # except one wrong nose, so the fit stays near-identity: a single off joint
        # does not trip the gate (x r² ≈ 0.94 > 0.9, y a perfect identity). Thresholds
        # echo the module constants so a record captures the gate it was judged under.
        conf = rec["conformance"]
        assert set(conf) == {"x", "y", "n", "conforms", "reasons", "cause",
                             "causeEvidence", "thresholds"}
        assert conf["thresholds"] == {
            "slopeMin": ev.CONFORMANCE_SLOPE_MIN, "slopeMax": ev.CONFORMANCE_SLOPE_MAX,
            "r2Min": ev.CONFORMANCE_R2_MIN, "r2MinX": ev.CONFORMANCE_R2_MIN_X,
            "minPoints": ev.CONFORMANCE_MIN_POINTS,
            # The gate's own truth-sufficiency floor is echoed beside the cause-split
            # floor it used to be conflated with (issue #101). ``_evaluate`` lowered
            # the gate for this miniature fixture, and the record captures what it
            # was actually judged under rather than the module default.
            "minFitFramesGate": 2,
            "minFitFrames": ev.NONCONFORMANCE_MIN_FIT_FRAMES,
            "minAcceptedShare": ev.NONCONFORMANCE_MIN_ACCEPTED_SHARE,
            "rateMismatchMinRatio": ev.RATE_MISMATCH_MIN_RATIO}
        assert conf["n"] == 37  # matched-present truth joints with a scanner pred
        assert conf["conforms"] is True and conf["reasons"] == []
        assert conf["y"] == {"slope": 1.0, "intercept": 0.0, "r2": 1.0}
        assert ev.record_conforms(rec) is True

        # Idempotent filename: rerun overwrites, no second file.
        summary2 = _evaluate(root)
        assert len(summary2.written) == 1
        assert summary2.written[0].record_path == summary.written[0].record_path
        assert len(list((vdir / "evaluations").glob("*.json"))) == 1


def test_evaluate_conformance_gate_and_pooled_quarantine():
    """Issue #15: a near-identity scanner↔truth fit conforms; a mis-tracked bundle
    (fit slope outside the band) is flagged non-conforming and dropped from every
    pooled trend derivation, while its record stays on disk and is named in the
    shame list."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    def _truth_doc(hash_: str) -> dict:
        frames = [
            {"frameIndex": i, "timestamp": float(i), "state": "present", "review": "auto",
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in _TRUTH_JOINTS.items()}}
            for i in (1, 2, 3)
        ]
        return {"version": 1, "jointSet": list(_TRUTH_JOINTS), "frames": frames,
                "groundTruthHash": hash_, "setupHash": "sh"}

    def _scanner_frames(transform) -> list:
        return [{"timestamp": float(i),
                 "keypoints": _kp_list({n: transform(x, y)
                                        for n, (x, y) in _TRUTH_JOINTS.items()})}
                for i in (1, 2, 3)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        good = root / "routeC" / "vidGood"
        _write_bundle_meta(good, setup_hash="sh")
        (good / "ground-truth.json").write_text(
            json.dumps(_truth_doc("aaaa0000")), encoding="utf-8")
        _write_pose_run(good, "20260101-000001", "sh",
                        _scanner_frames(lambda x, y: (x, y)))  # identity
        bad = root / "routeC" / "vidBad"
        _write_bundle_meta(bad, setup_hash="sh")
        (bad / "ground-truth.json").write_text(
            json.dumps(_truth_doc("bbbb1111")), encoding="utf-8")
        _write_pose_run(bad, "20260101-000002", "sh",
                        _scanner_frames(lambda x, y: (2 * x, 2 * y)))  # slope 2 → off-band

        summary = _evaluate(root)
        assert len(summary.written) == 2
        recs = {}
        for p in summary.written:
            r = json.loads(p.record_path.read_text(encoding="utf-8"))
            recs[r["videoKey"]] = r

        gc = recs["vidGood"]["conformance"]
        assert gc["n"] >= ev.CONFORMANCE_MIN_POINTS
        assert gc["conforms"] is True and gc["reasons"] == []
        assert gc["x"]["slope"] == 1.0 and gc["x"]["r2"] == 1.0
        assert gc["y"]["slope"] == 1.0 and gc["y"]["r2"] == 1.0
        assert ev.record_conforms(recs["vidGood"]) is True

        bc = recs["vidBad"]["conformance"]
        assert bc["conforms"] is False
        assert "x-nonconforming" in bc["reasons"] and "y-nonconforming" in bc["reasons"]
        assert bc["x"]["slope"] == 2.0 and bc["x"]["r2"] == 1.0  # a clean line, wrong slope
        assert ev.record_conforms(recs["vidBad"]) is False

        # Pooled quarantine: only the clean bundle feeds the pooled derivations, and
        # the mis-tracked one is accounted for by name.
        ctx = trends.build_trend_context(root)
        assert ctx["eval_count"] == 1 and ctx["eval_count_total"] == 2
        assert ctx["quarantined_count"] == 1
        assert {r.video_key for r in ctx["eval_records"]} == {"vidGood"}
        q = ctx["quarantined_bundles"][0]
        assert q["video_key"] == "vidBad" and "x-nonconforming" in q["reasons"]


def test_conformance_x_axis_has_looser_r2_floor():
    """Issue #16: the r² floor is asymmetric — looser on x than y — because a climber's
    narrow horizontal spread depresses x-r² even when the x-slope is at identity and y
    fits clean. An in-band slope with x-r² between the two floors conforms on x but not
    on y; genuine mis-tracking (r²≈0) and an off-band slope still fail on either axis."""

    from analysis_pipeline import evaluate as ev

    # The two floors straddle a value the narrow-x false positives land in (0.79–0.87).
    assert ev.CONFORMANCE_R2_MIN_X < ev.CONFORMANCE_R2_MIN
    assert ev._axis_r2_min("x") == ev.CONFORMANCE_R2_MIN_X
    assert ev._axis_r2_min("y") == ev.CONFORMANCE_R2_MIN

    slope_ok = 0.95  # inside [CONFORMANCE_SLOPE_MIN, CONFORMANCE_SLOPE_MAX]
    r2_between = (ev.CONFORMANCE_R2_MIN_X + ev.CONFORMANCE_R2_MIN) / 2  # e.g. 0.825
    borderline = (slope_ok, 0.0, r2_between)
    assert ev._axis_conforms(borderline, "x") is True   # passes the looser x floor
    assert ev._axis_conforms(borderline, "y") is False  # fails the strict y floor

    # Genuine wrong-subject: r²≈0 fails on both axes even with an in-band slope.
    wild = (slope_ok, 0.0, 0.05)
    assert ev._axis_conforms(wild, "x") is False
    assert ev._axis_conforms(wild, "y") is False

    # The slope band is symmetric: an off-band slope fails regardless of a perfect r².
    off_band = (2.0, 0.0, 1.0)
    assert ev._axis_conforms(off_band, "x") is False
    assert ev._axis_conforms(off_band, "y") is False

    # A degenerate (None) fit never conforms on either axis.
    assert ev._axis_conforms(None, "x") is False
    assert ev._axis_conforms(None, "y") is False


def test_nonconformance_cause_splits_sparse_match_from_suspected_mistrack():
    """Issue #88: a non-conforming record is annotated with *why* it failed the #15 gate.

    Two bundles fail the gate identically (scanner = 2×truth, slope off-band on both
    axes) and differ only in how much the detector supplied: one accepted every attempt,
    the other accepted 40% of them. Only the first is a truth-mis-track suspect, and only
    it reaches the truth-repair worklist. The gate verdict itself is untouched — both are
    still non-conforming and still quarantined out of the pooled metrics."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report
    from analysis_pipeline import trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}

    def _truth_doc(hash_: str, n_frames: int) -> dict:
        return {"version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": hash_,
                "setupHash": "sh88",
                "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                            "review": "auto", "joints": present}
                           for i in range(1, n_frames + 1)]}

    def _attempts(n_frames: int, accepted_upto: int, scale: float) -> list[dict]:
        """Frames 1..accepted_upto are accepted (pose scaled off identity); the rest miss."""
        out = []
        for i in range(1, n_frames + 1):
            if i <= accepted_upto:
                kps = _kp_list({n: (scale * x, scale * y)
                                for n, (x, y) in _TRUTH_JOINTS.items()})
                out.append({"timestamp": float(i), "status": "accepted",
                            "rawKeypoints": kps, "acceptedKeypoints": kps})
            else:
                out.append({"timestamp": float(i), "status": "missing",
                            "reacquireAttempted": True,
                            "rawKeypoints": [], "acceptedKeypoints": []})
        return out

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        # Mis-track suspect: 24 present frames, every attempt accepted, fit at slope 2.
        mistrack = root / "route88" / "vidMistrack"
        _write_bundle_meta(mistrack, setup_hash="sh88")
        (mistrack / "ground-truth.json").write_text(
            json.dumps(_truth_doc("88mistrack00", 24)), encoding="utf-8")
        _write_pose_run(mistrack, "20260101-000088", "sh88", [],
                        detector_attempts=_attempts(24, 24, 2.0))
        # Sparse match: the same off-band fit over the same 24 accepted frames, but 36
        # further attempts missed -> accepted share 0.4. The fit-frame floor is cleared,
        # so this is the *share* floor firing on its own.
        sparse = root / "route88" / "vidSparse"
        _write_bundle_meta(sparse, setup_hash="sh88")
        (sparse / "ground-truth.json").write_text(
            json.dumps(_truth_doc("88sparse0000", 60)), encoding="utf-8")
        _write_pose_run(sparse, "20260101-000089", "sh88", [],
                        detector_attempts=_attempts(60, 24, 2.0))
        # Control: same evidence volume, identity fit -> conforms, so no cause at all.
        clean = root / "route88" / "vidClean"
        _write_bundle_meta(clean, setup_hash="sh88")
        (clean / "ground-truth.json").write_text(
            json.dumps(_truth_doc("88clean0000", 24)), encoding="utf-8")
        _write_pose_run(clean, "20260101-000090", "sh88", [],
                        detector_attempts=_attempts(24, 24, 1.0))

        summary = _evaluate(root)
        recs = {p.video_key: json.loads(p.record_path.read_text(encoding="utf-8"))
                for p in summary.written}
        assert recs["vidMistrack"]["schemaVersion"] >= 11  # the cause split landed in v11

        mc = recs["vidMistrack"]["conformance"]
        assert mc["conforms"] is False  # the #15 verdict is unchanged by the split
        assert mc["cause"] == ev.NONCONFORMANCE_SUSPECTED_MISTRACK
        assert mc["causeEvidence"] == {"fitFrames": 24, "presentAttempts": 24,
                                       "acceptedAttempts": 24, "acceptedShare": 1.0,
                                       # No scaffold on disk, so the sampling grids
                                       # cannot be compared: unknown, never "agree".
                                       "scaffoldStepSec": None, "truthStepSec": 1.0,
                                       "samplingRatio": None}
        assert ev.record_nonconformance_cause(recs["vidMistrack"]) == "suspected-mistrack"

        sc = recs["vidSparse"]["conformance"]
        assert sc["conforms"] is False
        assert sc["cause"] == ev.NONCONFORMANCE_SPARSE_MATCH
        # Same fit, same fit-frame count as the mis-track bundle: only the share differs.
        assert sc["causeEvidence"]["fitFrames"] == 24
        assert sc["causeEvidence"]["acceptedShare"] == 0.4
        assert sc["x"]["slope"] == mc["x"]["slope"] == 2.0

        # A conforming record carries no cause, and its evidence is still reported.
        cc = recs["vidClean"]["conformance"]
        assert cc["conforms"] is True and cc["cause"] is None
        assert cc["causeEvidence"]["acceptedShare"] == 1.0
        assert ev.record_nonconformance_cause(recs["vidClean"]) is None

        # The volume floor fires independently of the share floor: plenty accepted, but
        # too few frames for the fit to indict the truth.
        assert ev._nonconformance_cause(
            {"fitFrames": ev.NONCONFORMANCE_MIN_FIT_FRAMES - 1,
             "acceptedShare": 1.0}) == ev.NONCONFORMANCE_SPARSE_MATCH
        # Fail-open: a pre-v11 non-conforming record keeps its pre-#88 place.
        assert ev.record_nonconformance_cause(
            {"conformance": {"conforms": False}}) == ev.NONCONFORMANCE_SUSPECTED_MISTRACK

        # Trend seam: the gate still quarantines both, grouped by cause, and only the
        # mis-track suspect feeds the truth-repair worklist.
        ctx = trends.build_trend_context(root)
        assert ctx["quarantined_count"] == 2
        assert ctx["quarantine_cause_counts"] == {"rate-mismatch": 0,
                                                  "sparse-match": 1,
                                                  "suspected-mistrack": 1}
        assert ctx["truth_repair_count"] == 1
        worklist = ctx["truth_repair_worklist"]
        assert [r["video_key"] for r in worklist] == ["vidMistrack"]
        assert worklist[0]["accepted_share"] == 1.0
        by_video = {r["video_key"]: r for r in ctx["quarantined_bundles"]}
        assert by_video["vidSparse"]["cause"] == "sparse-match"
        assert by_video["vidSparse"]["fit_frames"] == 24

        csvs = trends.write_trend_tables(Path(tmp) / "reports", ctx)
        repair_csv = pd.read_csv(csvs["eval_truth_repair_worklist.csv"])
        assert list(repair_csv["video_key"]) == ["vidMistrack"]
        quarantine_csv = pd.read_csv(csvs["eval_quarantined_bundles.csv"])
        assert set(quarantine_csv["cause"]) == {"sparse-match", "suspected-mistrack"}

        # Report seam: the section is grouped by cause and names the worklist scope.
        html = report._quarantine_table(ctx["quarantined_bundles"])
        assert "sparse-match" in html and "suspected-mistrack" in html
        assert "vidSparse" in html and "vidMistrack" in html
        assert "truth-repair worklist" in html


def test_evaluate_setuphash_mismatch_is_skipped():
    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeF" / "vidF"
        _write_bundle_meta(vdir, setup_hash="sh_truth")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        # A stale run whose setupHash != the setup.json the truth was authored under,
        # AND whose frames sample a disjoint time span (t≈100s) so it never overlaps a
        # scorable truth frame — the #44 best-overlap fallback finds nothing to recover.
        stale = [{"timestamp": 100.0 + i, "keypoints": _kp_list(_TRUTH_JOINTS)}
                 for i in range(4)]
        _write_pose_run(vdir, "20260101-000009", "sh_STALE", stale)

        summary = _evaluate(root)
        assert not summary.written
        assert not summary.loose
        assert len(summary.skipped) == 1
        assert "setupHash mismatch" in summary.skipped[0].reason
        assert not (vdir / "evaluations").exists()


def test_evaluate_loose_overlap_pairing_fallback():
    """Issue #44 deliverable 4: a bundle whose only setupHash-matched run samples a
    disjoint time span (n=0 overlap) is recovered by loose-pairing the run with the most
    timestamp overlap — even one whose setupHash differs — stamped loosePaired and held
    out of the trusted pool. Mirrors the IE4T94qX55g n=0 case."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeLP" / "vidLP"
        _write_bundle_meta(vdir, setup_hash="sh_cur")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        # Matched run (sh_cur) samples t≈100s — never overlaps truth (t1..t9) -> n=0.
        matched = [{"timestamp": 100.0 + i, "keypoints": _kp_list(_TRUTH_JOINTS)}
                   for i in range(3)]
        _write_pose_run(vdir, "20260101-000001", "sh_cur", matched)
        # Stale run (sh_OLD) overlaps truth at t1/t2/t4 -> the best-overlap candidate.
        _write_pose_run(vdir, "20260101-000002", "sh_OLD", _scanner_frames_for_pck())

        summary = _evaluate(root)
        # The matched-but-disjoint run writes a normal (n=0) record; the stale
        # overlapping run is recovered as a loose pairing.
        assert len(summary.written) == 2
        assert len(summary.loose) == 1
        loose = summary.loose[0]
        assert loose.run_ts == "20260101-000002"

        rec = json.loads(loose.record_path.read_text(encoding="utf-8"))
        assert rec["loosePaired"] is True
        assert rec["setupHash"] == "sh_OLD"  # the run's own hash, not the truth's
        assert rec["truthSetupHashSource"] == "loose-overlap"
        assert "best-overlap" in rec["loosePairReason"]
        assert rec["agreement"]["frames"]["matchedPresent"] == 3
        assert ev.record_conforms(rec) is True   # a clean identity fit
        assert ev.record_trusted(rec) is False   # ...but loose -> never trusted

        # Pooled trends hold the loose record out of the trusted pool and name it.
        ctx = trends.build_trend_context(root)
        assert ctx["loose_count"] == 1
        assert ctx["loose_bundles"][0]["video_key"] == "vidLP"
        # Neither the n=0 matched record nor the loose one feeds trusted pooling here
        # (the matched record has no scored joints; the loose one is excluded).
        assert all(not r.data.get("loosePaired") for r in ctx["eval_records"])


def test_evaluate_detection_annotations_override_and_ignore_stale():
    """Issue #45: active detectionAnnotations override the auto class, stale ranges are
    ignored, and the human distractor surfaces in pooled frame-quality trends."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeDA" / "vidDA"
        setup_hash = "sh_ann"
        _write_bundle_meta(vdir, setup_hash=setup_hash)
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc_with_annotations(setup_hash)), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000060", setup_hash, _scanner_frames_for_pck())

        summary = _evaluate(root)
        assert len(summary.written) == 1
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))

        fq = rec["frameQuality"]
        by_t = {e["t"]: e for e in fq["frames"]}
        assert by_t[1.0]["class"] == "ok"
        assert by_t[1.0]["autoClass"] == "ok"
        assert by_t[1.0]["distractor"] is None
        assert by_t[2.0]["class"] == "wrong-subject"
        assert by_t[2.0]["autoClass"] == "ok"
        assert by_t[2.0]["distractor"] == "tree_bush"
        # The annotation range covers frame indices 2-3, but frame 3 is the deprecated
        # manual-absent flag (ADR 0005) and is excluded from scoring entirely — so it
        # yields no frameQuality entry for an annotation to override. Only frame 2 of
        # the range is scorable.
        assert 3.0 not in by_t
        assert by_t[1.0]["annotationSetupHash"] is None
        assert by_t[2.0]["annotationSetupHash"] == setup_hash
        assert fq["classCounts"] == {"ok": 2, "wrong-subject": 1,
                                      "hallucination-fp": 0,
                                      "flipped-rotated": 0, "distorted": 0}

        ctx = trends.build_trend_context(root)
        classes = ctx["frame_quality_classes"].set_index("class")
        assert classes.loc["ok", "n"] == 2
        assert classes.loc["wrong-subject", "n"] == 1
        distractors = ctx["frame_quality_distractors"].set_index("distractor")
        assert distractors.loc["tree_bush", "n"] == 1
        assert "gear" not in distractors.index
        assert ctx["frame_quality_flagged"] == 1

        from analysis_pipeline import report

        # The frame-quality section, not the whole page: build_report_html additionally
        # needs the correlation context that only the `analysis` command assembles.
        assert "Distractor frequency" in report._frame_quality_html(ctx)


def test_frame_quality_classification_one_per_class():
    """Issue #44 deliverable 1: each matched, scanner-detected frame is sorted into one
    auto class from the scanner↔truth geometry, plus a cross-cutting frozen-stale flag.
    One synthetic frame per class (ok / hallucination-fp / wrong-subject / distorted /
    flipped-rotated) + a frozen duplicate."""

    from analysis_pipeline import evaluate as ev

    cy = sum(y for _, y in _TRUTH_JOINTS.values()) / len(_TRUTH_JOINTS)
    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": "fq00fq00fq00fq00",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present", "review": "auto", "joints": present},
            {"frameIndex": 2, "timestamp": 2.0, "state": "present", "review": "auto", "joints": present},
            {"frameIndex": 3, "timestamp": 3.0, "state": "absent", "review": "auto", "joints": {}},
            {"frameIndex": 4, "timestamp": 4.0, "state": "present", "review": "auto", "joints": present},
            {"frameIndex": 5, "timestamp": 5.0, "state": "present", "review": "auto", "joints": present},
            {"frameIndex": 6, "timestamp": 6.0, "state": "present", "review": "auto", "joints": present},
        ],
    }
    # t1/t2 exact (both ok; a 2-frame near-duplicate is NOT a sustained freeze under the
    # issue #68 run-length rule, so neither is frozenStale). t3 hallucination on an absent
    # frame. t4 shifted +0.35 in x (centroid ≈1.17 torso → wrong-subject). t5 zig-zag
    # x-perturbation (centroid ≈0, residual ≈0.67 torso → distorted). t6 reflected
    # vertically about the truth centroid (nose below hips → flipped-rotated).
    exact = _kp_list(_TRUTH_JOINTS)
    nudged = _kp_list({n: (x + 0.05, y + 0.05) for n, (x, y) in _TRUTH_JOINTS.items()})
    shifted = _kp_list({n: (x + 0.35, y) for n, (x, y) in _TRUTH_JOINTS.items()})
    zig = _kp_list({n: (x + (0.2 if i % 2 == 0 else -0.2), y)
                    for i, (n, (x, y)) in enumerate(_TRUTH_JOINTS.items())})
    flipped = _kp_list({n: (x, 2 * cy - y) for n, (x, y) in _TRUTH_JOINTS.items()})
    scanner = [
        {"timestamp": 1.0, "keypoints": exact},
        {"timestamp": 2.0, "keypoints": exact},   # 2-frame duplicate -> NOT frozen (#68)
        {"timestamp": 3.0, "keypoints": nudged},  # hallucination (truth absent), distinct
        {"timestamp": 4.0, "keypoints": shifted},
        {"timestamp": 5.0, "keypoints": zig},
        {"timestamp": 6.0, "keypoints": flipped},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeFQ" / "vidFQ"
        _write_bundle_meta(vdir, setup_hash="sh_fq")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000050", "sh_fq", scanner)

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        fq = rec["frameQuality"]
        assert fq["detectedFrames"] == 6
        assert fq["classCounts"] == {"ok": 2, "wrong-subject": 1, "hallucination-fp": 1,
                                     "flipped-rotated": 1, "distorted": 1}
        assert fq["flaggedCount"] == 4
        assert fq["heldPoseCount"] == 0
        assert fq["frozenStaleCount"] == 0  # #68: a 2-frame duplicate is not a sustained freeze
        assert fq["thresholds"]["wrongSubjectCentroid"] == ev.FQ_WRONG_SUBJECT_CENTROID
        assert fq["thresholds"]["frozenMinRun"] == ev.FQ_FROZEN_MIN_RUN

        by_t = {e["t"]: e for e in fq["frames"]}
        assert by_t[1.0]["class"] == "ok" and by_t[1.0]["heldPose"] is False
        assert by_t[1.0]["frozenStale"] is False
        assert by_t[2.0]["class"] == "ok" and by_t[2.0]["heldPose"] is False
        assert by_t[2.0]["frozenStale"] is False
        assert by_t[3.0]["class"] == "hallucination-fp"
        assert by_t[4.0]["class"] == "wrong-subject"
        assert by_t[5.0]["class"] == "distorted"
        assert by_t[6.0]["class"] == "flipped-rotated"
        # Every entry carries a crop placeholder for the exporter (deliverable 2).
        assert all("crop" in e for e in fq["frames"])


def test_frozen_stale_requires_sustained_run():
    """Issue #68: only non-anchor repeats inside a sustained near-identical run are held.
    Source provenance decides whether a held pose is a raw frozen-stale failure."""

    from analysis_pipeline import evaluate as ev

    assert ev.FQ_FROZEN_MIN_RUN == 3  # the scenarios below are keyed to this default

    def pose(x: float) -> dict[str, tuple[float, float]]:
        return {"nose": (x, 0.5), "left_hip": (x, 0.7), "right_hip": (x + 0.02, 0.7)}

    still = pose(0.30)              # a held/static pose
    moved = pose(0.50)             # clearly displaced (> FQ_FROZEN_EPS)

    # Empty / single frame: never frozen.
    assert ev._frozen_flags([]) == []
    assert ev._frozen_flags([still]) == [False]

    # A 2-frame duplicate is below the run floor → not frozen.
    assert ev._frozen_flags([still, still]) == [False, False]

    # A 3-frame identical run -> the anchor is fresh; only repeats are held.
    assert ev._frozen_flags([still, still, still]) == [False, True, True]

    # A sustained run bracketed by motion: only repeats in the sustained stretch flag.
    seq = [moved, still, still, still, moved, pose(0.55)]
    assert ev._frozen_flags(seq) == [False, False, True, True, False, False]

    # Two short pauses (2 frames each) separated by motion → neither reaches the floor.
    seq2 = [still, still, moved, pose(0.52), pose(0.52)]
    assert ev._frozen_flags(seq2) == [False, False, False, False, False]


def test_frame_quality_splits_held_pose_from_raw_frozen_stale():
    """Issue #68 v8: held poses are neutral diagnostics; frozenStale is raw-only."""

    from analysis_pipeline import evaluate as ev

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    frames = [
        {"frameIndex": i, "timestamp": float(i), "state": "present",
         "review": "auto", "joints": present}
        for i in range(1, 6)
    ]
    doc = {"version": 1, "jointSet": list(_TRUTH_JOINTS),
           "groundTruthHash": "heldheldheld0001", "frames": frames}
    scanner = [
        {"timestamp": 1.0, "source": "raw", "keypoints": _kp_list(_TRUTH_JOINTS)},
        {"timestamp": 2.0, "source": "raw", "keypoints": _kp_list(_TRUTH_JOINTS)},
        {"timestamp": 3.0, "source": "interpolated", "keypoints": _kp_list(_TRUTH_JOINTS)},
        {"timestamp": 4.0, "source": "filled", "keypoints": _kp_list(_TRUTH_JOINTS)},
        {"timestamp": 5.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeHeld" / "vidHeld"
        _write_bundle_meta(vdir, setup_hash="sh_held")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000060", "sh_held", scanner)

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        fq = rec["frameQuality"]
        assert fq["heldPoseCount"] == 4
        assert fq["frozenStaleCount"] == 1

        by_t = {e["t"]: e for e in fq["frames"]}
        assert by_t[1.0]["source"] == "raw"
        assert by_t[1.0]["heldPose"] is False and by_t[1.0]["frozenStale"] is False
        assert by_t[2.0]["source"] == "raw"
        assert by_t[2.0]["heldPose"] is True and by_t[2.0]["frozenStale"] is True
        assert by_t[3.0]["source"] == "interpolated"
        assert by_t[3.0]["heldPose"] is True and by_t[3.0]["frozenStale"] is False
        assert by_t[4.0]["source"] == "filled"
        assert by_t[4.0]["heldPose"] is True and by_t[4.0]["frozenStale"] is False
        assert by_t[5.0]["source"] is None
        assert by_t[5.0]["heldPose"] is True and by_t[5.0]["frozenStale"] is False


def test_evaluate_prefers_detector_attempts_over_dense_frames():
    """Issue #73: attempt-bearing runs use detectorAttempts[] as detector evidence.

    The dense playback frames below all carry keypoints, but their sources are
    post-processed legacy playback values. They must not turn missing/rejected attempts
    into detector successes or frame-quality events.
    """

    from analysis_pipeline import evaluate as ev

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1,
        "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "attempt73attempt73",
        "setupHash": "sh_attempt",
        "frames": [
            {"frameIndex": i, "timestamp": float(i), "state": "present",
             "review": "auto", "joints": present}
            for i in (1, 2, 3, 4)
        ],
    }

    exact = _kp_list(_TRUTH_JOINTS)
    cy = sum(y for _, y in _TRUTH_JOINTS.values()) / len(_TRUTH_JOINTS)
    flipped = _kp_list({n: (x, 2 * cy - y) for n, (x, y) in _TRUTH_JOINTS.items()})
    shifted = _kp_list({n: (x + 0.25, y) for n, (x, y) in _TRUTH_JOINTS.items()})
    dense = [
        {"timestamp": 1.0, "source": "interpolated", "keypoints": exact},
        {"timestamp": 2.0, "source": "filled", "keypoints": exact},
        {"timestamp": 3.0, "source": "smoothed", "keypoints": exact},
        {"timestamp": 4.0, "source": "constrained", "keypoints": exact},
    ]
    crop = {"x": 0.2, "y": 0.25, "w": 0.3, "h": 0.4}
    full_frame = {"x": 0, "y": 0, "w": 1, "h": 1}
    attempts = [
        {"timestamp": 1.0, "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact,
         "acceptedKeypoints": exact, "candidateCount": 1, "selectionMethod": "tracked",
         "searchConditions": {"mean": 50, "stdDev": 20, "sharpness": 100,
                              "flags": {"tooDark": False}}},
        {"timestamp": 2.0, "status": "missing", "rawKeypoints": [],
         "acceptedKeypoints": [], "candidateCount": 0},
        {"timestamp": 3.0, "status": "flipRejected", "initialSearchRegion": crop,
         "detectionRegion": full_frame, "reacquireAttempted": True,
         "reacquired": False, "rawKeypoints": flipped,
         "acceptedKeypoints": [], "candidateCount": 1, "rejectedCandidateCount": 1,
         "selectionMethod": "best",
         "searchConditions": {"mean": 20, "stdDev": 5, "sharpness": 10,
                              "flags": {"tooDark": True, "lowContrast": True}}},
        {"timestamp": 4.0, "status": "qualityRejected", "initialSearchRegion": crop,
         "detectionRegion": full_frame, "reacquireAttempted": True,
         "reacquired": True, "rawKeypoints": shifted,
         "acceptedKeypoints": [], "candidateCount": 1, "rejectedCandidateCount": 1,
         "selectionMethod": "best",
         "searchConditions": {"mean": 30, "stdDev": 8, "sharpness": 18,
                              "flags": {"tooDark": True}}},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeATT" / "vidATT"
        _write_bundle_meta(vdir, setup_hash="sh_attempt")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(
            vdir,
            "20260101-000073",
            "sh_attempt",
            dense,
            detector_attempts=attempts,
        )

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        agr = rec["agreement"]
        assert agr["frames"]["matchedPresent"] == 4
        assert agr["presence"] == {"presentDetected": 1, "presentUndetected": 3,
                                   "absentDetected": 0, "absentUndetected": 0}
        assert agr["perJoint"]["nose"]["pck"] == {"correct": 1, "total": 4, "value": 0.25}
        assert agr["perJoint"]["nose"]["coverage"] == {"emitted": 1, "frames": 4,
                                                        "rate": 0.25}

        fq = rec["frameQuality"]
        assert fq["detectorEvidence"] == "attempts"
        assert fq["detectorAttemptStatusCounts"] == {
            "accepted": 1,
            "flipRejected": 1,
            "missing": 1,
            "qualityRejected": 1,
            "unknown": 0,
        }
        assert fq["classCounts"] == {"ok": 1, "wrong-subject": 0,
                                     "hallucination-fp": 0, "flipped-rotated": 1,
                                     "distorted": 1}
        by_t = {e["t"]: e for e in fq["frames"]}
        assert set(by_t) == {1.0, 3.0, 4.0}
        assert by_t[1.0]["class"] == "ok"
        assert by_t[1.0]["detectorAttemptStatus"] == "accepted"
        assert by_t[1.0]["acceptedKeypoints"] == exact
        assert by_t[3.0]["class"] == "flipped-rotated"
        assert by_t[3.0]["detectorAttemptStatus"] == "flipRejected"
        assert by_t[3.0]["rawKeypoints"] == flipped
        assert by_t[4.0]["class"] == "distorted"
        assert by_t[4.0]["detectorAttemptStatus"] == "qualityRejected"
        assert by_t[4.0]["rawKeypoints"] == shifted

        from analysis_pipeline import trends

        ctx = trends.build_trend_context(root)
        attempt_runs = ctx["detection_error_attempt_runs"]
        assert len(attempt_runs) == 1
        ar = attempt_runs.iloc[0]
        assert ar["attempt_evidence"] == "attempts"
        assert ar["flagged_frames"] == 2
        assert ar["flagged_rate"] == 2 / 3
        assert ar["attempt_reacquire_attempted_count"] == 2
        assert ar["attempt_reacquire_succeeded_count"] == 1
        assert ar["attempt_reacquire_failed_count"] == 1
        assert ar["attempt_full_frame_reacquire_success_count"] == 1
        assert ar["attempt_initial_search_region_area_mean"] == 0.12
        assert ar["attempt_search_luma_mean_mean"] == (50 + 20 + 30) / 3
        assert ar["attempt_search_flag_too_dark_rate"] == 2 / 4

        worklist = ctx["frame_quality_worklist"].set_index("t")
        assert worklist.loc[3.0]["detector_attempt_status"] == "flipRejected"
        assert worklist.loc[3.0]["reacquire_failed"] == True  # noqa: E712
        assert worklist.loc[4.0]["reacquire_succeeded"] == True  # noqa: E712
        assert worklist.loc[4.0]["detection_region_area"] == 1.0


def test_rejection_correctness_verdicts_and_pooled_rate():
    """Issue #85: every flip/quality-rejected Detector Attempt's discarded raw pose is
    scored against truth, per-frame verdicts land in the record, and the pooled
    over-rejection rate reaches ``build_trend_context`` per Run.

    One bundle crafts all three verdicts (plus both truthUnknown mechanisms) so the
    pooled rate and the per-gate split are each pinned; a second, legacy frames-only
    bundle proves an unmeasured Run reports ``None`` rather than a zero rate."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report
    from analysis_pipeline import trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    # t3's truth drops right_hip, so no torso can be formed -> the rejection is not
    # truth-checkable however good the raw pose looks.
    no_torso = {n: {"x": x, "y": y, "occluded": False}
                for n, (x, y) in _TRUTH_JOINTS.items() if n != "right_hip"}
    doc = {
        "version": 1,
        "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "rejection85rejection",
        "setupHash": "sh_rej",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present",
             "review": "auto", "joints": present},
            {"frameIndex": 2, "timestamp": 2.0, "state": "present",
             "review": "auto", "joints": present},
            {"frameIndex": 3, "timestamp": 3.0, "state": "present",
             "review": "auto", "joints": no_torso},
            {"frameIndex": 4, "timestamp": 4.0, "state": "present",
             "review": "auto", "joints": present},
            {"frameIndex": 5, "timestamp": 5.0, "state": "absent",
             "review": "auto", "joints": {}},
        ],
    }

    exact = _kp_list(_TRUTH_JOINTS)
    cy = sum(y for _, y in _TRUTH_JOINTS.values()) / len(_TRUTH_JOINTS)
    flipped = _kp_list({n: (x, 2 * cy - y) for n, (x, y) in _TRUTH_JOINTS.items()})
    attempts = [
        # t1 — the flip gate discarded a raw pose that sat right on truth: over-rejection.
        {"timestamp": 1.0, "status": "flipRejected", "rawKeypoints": exact,
         "acceptedKeypoints": [], "rejectedCandidateCount": 1},
        # t2 — genuinely upside-down raw pose: the gate was right.
        {"timestamp": 2.0, "status": "flipRejected", "rawKeypoints": flipped,
         "acceptedKeypoints": [], "rejectedCandidateCount": 1},
        # t3 — truth has no torso, so the geometry is unnormalizable: truthUnknown.
        {"timestamp": 3.0, "status": "qualityRejected", "rawKeypoints": exact,
         "acceptedKeypoints": [], "rejectedCandidateCount": 1},
        # t4 — a rejection with nothing to inspect: truthUnknown, and no frameQuality
        # entry at all, so the pooled counts must come from the pairs, not the entries.
        {"timestamp": 4.0, "status": "flipRejected", "rawKeypoints": [],
         "acceptedKeypoints": []},
        # t5 — a pose where truth says no Climber: correctly rejected.
        {"timestamp": 5.0, "status": "qualityRejected", "rawKeypoints": exact,
         "acceptedKeypoints": [], "rejectedCandidateCount": 1},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeREJ" / "vidREJ"
        _write_bundle_meta(vdir, setup_hash="sh_rej")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000085", "sh_rej", [],
                        detector_attempts=attempts)
        # A legacy frames-only bundle alongside it: no attempts, so no rejections.
        legacy = root / "routeREJL" / "vidREJL"
        _write_bundle_meta(legacy, setup_hash="sh_legacy")
        (legacy / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(legacy, "20260101-000086", "sh_legacy", _scanner_frames_for_pck())

        summary = _evaluate(root)
        assert len(summary.written) == 2
        by_run = {p.run_ts: json.loads(p.record_path.read_text(encoding="utf-8"))
                  for p in summary.written}
        rec = by_run["20260101-000085"]
        assert rec["schemaVersion"] >= 9  # rejection scoring landed in v9

        rc = rec["frameQuality"]["rejectionCorrectness"]
        assert rc["rejected"] == 5
        assert rc["verdictCounts"] == {"goodPoseRejected": 1, "badPoseRejected": 2,
                                       "truthUnknown": 2}
        assert rc["truthCheckable"] == 3
        assert rc["overRejectionRate"] == round(1 / 3, 6)
        # t5's rejection is correct by construction (no Climber there), so the
        # Climber-present denominator drops it: 1 good of 2 rather than 1 of 3.
        assert rc["truthAbsent"] == 1
        assert rc["truthPresentCheckable"] == 2
        assert rc["overRejectionRateTruthPresent"] == 0.5
        # Each gate is measured on its own — the corpus baseline is a flip-gate rate.
        assert rc["byStatus"]["flipRejected"]["rejected"] == 3
        assert rc["byStatus"]["flipRejected"]["overRejectionRate"] == 0.5
        assert rc["byStatus"]["flipRejected"]["overRejectionRateTruthPresent"] == 0.5
        assert rc["byStatus"]["qualityRejected"]["rejected"] == 2
        assert rc["byStatus"]["qualityRejected"]["overRejectionRate"] == 0.0
        # The quality gate's only checkable rejection was the Climber-absent one, so
        # there is no Climber-present evidence about that gate at all.
        assert rc["byStatus"]["qualityRejected"]["overRejectionRateTruthPresent"] is None
        assert rc["thresholds"]["minJointAgreement"] == ev.REJECTION_MIN_JOINT_AGREEMENT
        assert rc["thresholds"]["pckTorsoFraction"] == ev.PCK_TORSO_FRACTION

        by_t = {e["t"]: e for e in rec["frameQuality"]["frames"]}
        assert set(by_t) == {1.0, 2.0, 3.0, 5.0}  # t4 had no raw pose to classify
        assert by_t[1.0]["rejectionVerdict"] == "goodPoseRejected"
        assert by_t[1.0]["rejectionReason"] == "raw-pose-agrees-truth"
        assert by_t[1.0]["rejectionCentroidDist"] == 0.0
        assert by_t[1.0]["rejectionJointAgreement"] == 1.0
        assert by_t[1.0]["rejectionRawClass"] == "ok"
        assert by_t[2.0]["rejectionVerdict"] == "badPoseRejected"
        assert by_t[2.0]["rejectionRawClass"] == "flipped-rotated"
        assert by_t[3.0]["rejectionVerdict"] == "truthUnknown"
        assert by_t[3.0]["rejectionReason"] == "truth-ungeometric"
        assert by_t[3.0]["rejectionJointAgreement"] is None
        assert by_t[5.0]["rejectionVerdict"] == "badPoseRejected"
        assert by_t[5.0]["rejectionReason"] == "truth-absent"

        # Legacy record: same schema, no rejections, and an explicitly *unmeasured* rate.
        legacy_rec = by_run["20260101-000086"]
        legacy_rc = legacy_rec["frameQuality"]["rejectionCorrectness"]
        assert legacy_rc["rejected"] == 0
        assert legacy_rc["overRejectionRate"] is None
        assert legacy_rec["frameQuality"]["detectorEvidence"] == "legacy-frames"
        assert all(e["rejectionVerdict"] is None
                   for e in legacy_rec["frameQuality"]["frames"])

        # ...and the trend seam. Every attempt was rejected, so no accepted pose feeds
        # the #15 fit and the attempt record is quarantined — the per-frame pools
        # deliberately span all records, quarantined included, so it still lands here.
        ctx = trends.build_trend_context(root)
        runs = ctx["detection_error_attempt_runs"].set_index("run_ts")
        row = runs.loc["20260101-000085"]
        assert row["rejected_attempts"] == 5
        assert row["good_pose_rejected"] == 1
        assert row["bad_pose_rejected"] == 2
        assert row["rejection_truth_unknown"] == 2
        assert row["rejection_truth_checkable"] == 3
        assert row["rejection_truth_absent"] == 1
        assert row["over_rejection_rate"] == round(1 / 3, 6)
        assert row["over_rejection_rate_truth_present"] == 0.5
        assert row["flip_over_rejection_rate"] == 0.5
        legacy_row = runs.loc["20260101-000086"]
        assert legacy_row["rejected_attempts"] == 0
        assert pd.isna(legacy_row["over_rejection_rate"])

        totals = ctx["rejection_correctness"]
        assert totals["rejected_attempts"] == 5
        assert totals["good_pose_rejected"] == 1
        assert totals["truth_checkable"] == 3
        assert totals["truth_absent"] == 1
        assert totals["truth_present_checkable"] == 2
        assert totals["over_rejection_rate"] == 1 / 3
        assert totals["over_rejection_rate_truth_present"] == 0.5
        # The legacy run has no rate at all, so it neither dilutes nor pads the mean.
        assert totals["runs_with_checkable_rejections"] == 1
        assert totals["over_rejection_rate_run_mean"] == round(1 / 3, 6)

        worklist = ctx["frame_quality_worklist"].set_index("t")
        assert worklist.loc[1.0]["rejection_verdict"] == "goodPoseRejected"
        assert worklist.loc[1.0]["rejection_joint_agreement"] == 1.0

        # The CSV export carries the per-run rate for lighter agents / notebooks...
        csvs = trends.write_trend_tables(Path(tmp) / "reports", ctx)
        run_csv = pd.read_csv(csvs["eval_detection_error_attempt_runs.csv"])
        assert "over_rejection_rate" in run_csv.columns
        assert "flip_over_rejection_rate" in run_csv.columns
        assert run_csv.set_index("run_ts").loc[
            "20260101-000085", "over_rejection_rate"] == round(1 / 3, 6)

        # ...and the report section states it.
        html = report._detection_error_attempt_html(ctx)
        assert "over-rejection rate (pooled)" in html
        assert "over-rejection rate (Climber present)" in html
        assert "truth-checkable rejections" in html
        assert "0.33" in html


def test_crop_quality_iou_and_miss_causes():
    """Issue #86: every matched Detector Attempt is scored against a truth bbox, and each
    missing attempt gets a cause class.

    All four causes are crafted, including the case that decides the taxonomy: a miss
    whose crop excluded the Climber but which *also* ran a full-frame reacquire is NOT
    crop-misplaced — everywhere was searched, so the crop cannot be what lost them — yet
    its crop-placement failure is still recorded."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report
    from analysis_pipeline import trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1,
        "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "crop86crop86crop86",
        "setupHash": "sh_crop",
        "frames": [
            {"frameIndex": i, "timestamp": float(i), "state": "present",
             "review": "auto", "joints": present}
            for i in (1, 2, 3, 4)
        ] + [
            {"frameIndex": 5, "timestamp": 5.0, "state": "absent",
             "review": "auto", "joints": {}},
        ],
    }

    # _TRUTH_JOINTS spans x 0.3..0.7, y 0.2..0.95 -> extent 0.4 x 0.75, pad = 0.075
    # -> bbox (0.225, 0.125) .. (0.775, 1.025), i.e. w=0.55 h=0.90, area 0.495.
    bbox_area = 0.55 * 0.90
    on_truth = {"x": 0.225, "y": 0.125, "w": 0.55, "h": 0.90}   # exactly the truth bbox
    elsewhere = {"x": 0.0, "y": 0.0, "w": 0.15, "h": 0.15}      # nowhere near the Climber
    full_frame = {"x": 0, "y": 0, "w": 1, "h": 1}
    clean = {"mean": 120, "stdDev": 40, "sharpness": 200,
             "flags": {"isBlurry": False, "isUnderexposed": False}}
    adverse = {"mean": 20, "stdDev": 5, "sharpness": 10,
               "flags": {"isBlurry": True, "isUnderexposed": True}}
    exact = _kp_list(_TRUTH_JOINTS)
    attempts = [
        # t1 accepted, crop sits exactly on truth -> IoU 1.0, contained.
        {"timestamp": 1.0, "status": "accepted", "initialSearchRegion": on_truth,
         "detectionRegion": on_truth, "rawKeypoints": exact, "acceptedKeypoints": exact,
         "searchConditions": clean},
        # t2 missing, crop excluded the Climber, NO reacquire -> crop-misplaced.
        {"timestamp": 2.0, "status": "missing", "initialSearchRegion": elsewhere,
         "reacquireAttempted": False, "rawKeypoints": [], "acceptedKeypoints": [],
         "searchConditions": clean},
        # t3 missing, crop excluded the Climber, but full-frame reacquire ran and failed
        # -> NOT crop-misplaced. Conditions fired, so: adverse-conditions.
        {"timestamp": 3.0, "status": "missing", "initialSearchRegion": elsewhere,
         "reacquireAttempted": True, "reacquired": False, "rawKeypoints": [],
         "acceptedKeypoints": [], "searchConditions": adverse},
        # t4 missing, everywhere searched, conditions clean -> unexplained.
        {"timestamp": 4.0, "status": "missing", "initialSearchRegion": on_truth,
         "reacquireAttempted": True, "reacquired": False, "rawKeypoints": [],
         "acceptedKeypoints": [], "searchConditions": clean},
        # t5 missing on a Climber-absent frame -> climber-absent (a correct miss).
        {"timestamp": 5.0, "status": "missing", "initialSearchRegion": full_frame,
         "reacquireAttempted": True, "rawKeypoints": [], "acceptedKeypoints": [],
         "searchConditions": clean},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeCQ" / "vidCQ"
        _write_bundle_meta(vdir, setup_hash="sh_crop")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000086", "sh_crop", [],
                        detector_attempts=attempts)
        # Legacy frames-only bundle: no attempts, so nothing to score.
        legacy = root / "routeCQL" / "vidCQL"
        _write_bundle_meta(legacy, setup_hash="sh_cqlegacy")
        (legacy / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(legacy, "20260101-000087", "sh_cqlegacy", _scanner_frames_for_pck())

        summary = _evaluate(root)
        by_run = {p.run_ts: json.loads(p.record_path.read_text(encoding="utf-8"))
                  for p in summary.written}
        rec = by_run["20260101-000086"]
        assert rec["schemaVersion"] >= 10  # cropQuality landed in v10

        cq = rec["cropQuality"]
        assert cq["matchedAttempts"] == 5
        assert cq["missingAttempts"] == 4
        # No attempt here carries missReason or candidateCount, so the fail-open
        # fallbacks (adverse-conditions / unexplained) still classify — the pre-evidence
        # behavior, retained by design.
        assert cq["missCauseCounts"] == {"climber-absent": 1, "crop-misplaced": 1,
                                        "identity-gated": 0, "no-candidates": 0,
                                        "adverse-conditions": 1, "unexplained": 1}
        assert cq["thresholds"]["truthBboxPad"] == ev.TRUTH_BBOX_PAD
        assert cq["thresholds"]["reacquireSearchesFullFrame"] is True

        by_t = {e["t"]: e for e in cq["frames"]}
        # A crop placed exactly on the truth bbox scores IoU 1 and full containment.
        assert by_t[1.0]["initialSearchRegionIou"] == 1.0
        assert by_t[1.0]["detectionRegionIou"] == 1.0
        assert by_t[1.0]["initialCropContainment"] == 1.0
        assert by_t[1.0]["cropContainedTruth"] is True
        assert by_t[1.0]["missCause"] is None
        assert by_t[1.0]["truthBbox"] == {"x": 0.225, "y": 0.125, "w": 0.55, "h": 0.9}
        # A crop nowhere near the Climber: zero overlap on both measures.
        assert by_t[2.0]["missCause"] == "crop-misplaced"
        assert by_t[2.0]["initialSearchRegionIou"] == 0.0
        assert by_t[2.0]["initialCropContainment"] == 0.0
        assert by_t[2.0]["cropContainedTruth"] is False
        # The decisive case: crop excluded the Climber, but everywhere was searched.
        assert by_t[3.0]["missCause"] == "adverse-conditions"
        assert by_t[3.0]["cropContainedTruth"] is False  # still recorded
        assert by_t[3.0]["firedSearchFlags"] == ["is_blurry", "is_underexposed"]
        assert by_t[4.0]["missCause"] == "unexplained"
        assert by_t[4.0]["searchFlagsFired"] is False
        # Climber absent -> no bbox to score against, and the miss is correct.
        assert by_t[5.0]["missCause"] == "climber-absent"
        assert by_t[5.0]["truthBbox"] is None
        assert by_t[5.0]["initialSearchRegionIou"] is None
        assert by_t[5.0]["initialCropContainment"] is None

        # A full-frame crop containing the bbox: containment 1, but IoU only bbox/frame.
        assert cq["cropContainedTruth"] == {"contained": 2, "scored": 4,
                                            "rate": 0.5}
        assert cq["initialSearchRegionIou"]["n"] == 4
        assert cq["initialSearchRegionIou"]["median"] == round((1.0 + 0.0) / 2, 6)

        # Legacy record: block present, nothing scored — not "zero misplaced crops".
        legacy_cq = by_run["20260101-000087"]["cropQuality"]
        assert legacy_cq["matchedAttempts"] == 0
        assert legacy_cq["missingAttempts"] == 0
        assert legacy_cq["cropContainedTruth"]["rate"] is None
        assert legacy_cq["frames"] == []

        # ...and the trend seam.
        ctx = trends.build_trend_context(root)
        causes = ctx["crop_quality_miss_causes"].set_index("miss_cause")
        assert int(causes.loc["crop-misplaced", "n"]) == 1
        assert int(causes.loc["adverse-conditions", "crop_missed_truth"]) == 1
        totals = ctx["crop_quality"]
        assert totals["matched_attempts"] == 5
        assert totals["missing_attempts"] == 4
        # Two of four scorable crops excluded the Climber (t2, t3).
        assert totals["crop_missed_truth"] == 2
        assert totals["crop_missed_truth_rate"] == 0.5
        assert totals["miss_cause_counts"]["unexplained"] == 1

        runs = ctx["detection_error_attempt_runs"].set_index("run_ts")
        row = runs.loc["20260101-000086"]
        assert row["missing_attempts"] == 4
        assert row["miss_crop_misplaced_count"] == 1
        assert row["miss_crop_misplaced_share"] == 0.25
        assert row["crop_contained_truth_rate"] == 0.5
        legacy_row = runs.loc["20260101-000087"]
        assert legacy_row["missing_attempts"] == 0
        assert pd.isna(legacy_row["crop_contained_truth_rate"])

        csvs = trends.write_trend_tables(Path(tmp) / "reports", ctx)
        attempts_csv = pd.read_csv(csvs["eval_crop_quality_attempts.csv"])
        assert len(attempts_csv) == 5
        assert "initial_search_region_iou" in attempts_csv.columns
        assert "eval_crop_quality_miss_causes.csv" in csvs

        html = report._detection_error_attempt_html(ctx)
        assert "Missing-attempt causes" in html
        assert "crop-misplaced" in html
        assert "attempts whose crop excluded Climber" in html


def test_miss_reason_splits_the_residual_and_retro_derives():
    """Reply handoff 2026-07-25: the scanner's ``missReason`` — or its ``candidateCount``
    retro-derivation on older streams — splits the old ``unexplained`` residual into
    ``identity-gated`` (a gate rejection) vs ``no-candidates`` (a detector failure).
    Fail-open when neither signal exists, and never outranking ``climber-absent`` or
    ``crop-misplaced``: candidates found inside a crop that excluded the Climber were
    not the Climber, so the crop still owns that miss."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends
    from analysis_pipeline.detector_attempts import miss_reason, parse_detector_attempts

    # Parser + helper seam first: the absent/empty reacquireSteps distinction, the
    # authored-field-wins rule, and the retro-derivation.
    parsed = parse_detector_attempts({"detectorAttempts": [
        {"timestamp": 1.0, "status": "missing", "missReason": "identity-gated",
         "reacquireSteps": [], "bestUnselectedCandidateScore": 0.5, "candidateCount": 0},
        {"timestamp": 2.0, "status": "missing",
         "reacquireSteps": [{"region": {"x": 0, "y": 0, "w": 1, "h": 1}, "found": False}]},
    ]})
    assert parsed is not None
    assert parsed[0]["reacquireSteps"] == []          # ran no reacquire — not legacy
    assert parsed[0]["bestUnselectedCandidateScore"] == 0.5
    assert parsed[1]["reacquireSteps"] == [
        {"region": {"x": 0, "y": 0, "w": 1, "h": 1}, "found": False}]
    assert miss_reason(parsed[0]) == "identity-gated"  # authored wins over derivation
    assert miss_reason({"missReason": "bogus", "candidateCount": 2}) == "identity-gated"
    assert miss_reason({"missReason": None, "candidateCount": 0}) == "no-candidates"
    assert miss_reason({}) is None                     # neither signal: fail open
    legacy = parse_detector_attempts({"detectorAttempts": [
        {"timestamp": 1.0, "status": "missing"}]})
    assert legacy is not None and legacy[0]["reacquireSteps"] is None  # legacy payload

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1,
        "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "reason13reason13",
        "setupHash": "sh_reason",
        "frames": [
            {"frameIndex": i, "timestamp": float(i), "state": "present",
             "review": "auto", "joints": present}
            for i in (1, 2, 3, 4, 5, 7)
        ] + [
            {"frameIndex": 6, "timestamp": 6.0, "state": "absent",
             "review": "auto", "joints": {}},
        ],
    }
    on_truth = {"x": 0.225, "y": 0.125, "w": 0.55, "h": 0.90}
    elsewhere = {"x": 0.0, "y": 0.0, "w": 0.15, "h": 0.15}
    full_frame = {"x": 0, "y": 0, "w": 1, "h": 1}
    clean = {"mean": 120, "stdDev": 40, "sharpness": 200,
             "flags": {"isBlurry": False, "isUnderexposed": False}}
    adverse = {"mean": 20, "stdDev": 5, "sharpness": 10,
               "flags": {"isBlurry": True, "isUnderexposed": True}}
    miss = {"status": "missing", "reacquireAttempted": True, "reacquired": False,
            "rawKeypoints": [], "acceptedKeypoints": [], "initialSearchRegion": on_truth,
            "searchConditions": clean}
    attempts = [
        # t1 authored identity-gated, with the gate-tuning score.
        {**miss, "timestamp": 1.0, "missReason": "identity-gated", "candidateCount": 3,
         "bestUnselectedCandidateScore": 0.91},
        # t2 authored no-candidates, with today's one-rung reacquire trace.
        {**miss, "timestamp": 2.0, "missReason": "no-candidates", "candidateCount": 0,
         "reacquireSteps": [{"region": full_frame, "found": False}]},
        # t3 no missReason, candidateCount > 0, adverse flags — the derivation outranks
        # the adverse-conditions fallback.
        {**miss, "timestamp": 3.0, "candidateCount": 2, "searchConditions": adverse},
        # t4 no missReason, candidateCount == 0 -> no-candidates by derivation.
        {**miss, "timestamp": 4.0, "candidateCount": 0},
        # t5 neither signal -> unexplained, the pre-evidence residual, retained.
        {**miss, "timestamp": 5.0},
        # t6 Climber absent outranks an authored reason: the miss is correct.
        {**miss, "timestamp": 6.0, "missReason": "no-candidates", "candidateCount": 0,
         "initialSearchRegion": full_frame},
        # t7 crop excluded the Climber and nothing else was searched: crop-misplaced
        # outranks the gated reading — the gated candidates were not the Climber.
        {**miss, "timestamp": 7.0, "reacquireAttempted": False,
         "initialSearchRegion": elsewhere, "candidateCount": 2},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeMR" / "vidMR"
        _write_bundle_meta(vdir, setup_hash="sh_reason")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000101", "sh_reason", [],
                        detector_attempts=attempts)

        summary = _evaluate(root)
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))
        assert rec["schemaVersion"] >= 13

        cq = rec["cropQuality"]
        assert cq["missCauseCounts"] == {"climber-absent": 1, "crop-misplaced": 1,
                                         "identity-gated": 2, "no-candidates": 2,
                                         "adverse-conditions": 0, "unexplained": 1}
        by_t = {e["t"]: e for e in cq["frames"]}
        assert by_t[1.0]["missCause"] == "identity-gated"
        assert by_t[1.0]["missReason"] == "identity-gated"
        assert by_t[1.0]["bestUnselectedCandidateScore"] == 0.91
        assert by_t[2.0]["missCause"] == "no-candidates"
        assert by_t[3.0]["missCause"] == "identity-gated"   # derived
        assert by_t[3.0]["missReason"] == "identity-gated"
        assert by_t[4.0]["missCause"] == "no-candidates"    # derived
        assert by_t[5.0]["missCause"] == "unexplained"
        assert by_t[5.0]["missReason"] is None
        assert by_t[6.0]["missCause"] == "climber-absent"
        assert by_t[7.0]["missCause"] == "crop-misplaced"
        assert by_t[7.0]["missReason"] == "identity-gated"  # still recorded, not the cause

        ctx = trends.build_trend_context(root)
        causes = ctx["crop_quality_miss_causes"].set_index("miss_cause")
        assert int(causes.loc["identity-gated", "n"]) == 2
        assert int(causes.loc["no-candidates", "n"]) == 2
        assert causes.loc["identity-gated",
                          "median_best_unselected_candidate_score"] == 0.91
        runs = ctx["detection_error_attempt_runs"].set_index("run_ts")
        row = runs.loc["20260101-000101"]
        assert row["miss_identity_gated_count"] == 2
        assert row["miss_no_candidates_count"] == 2
        attempts_rows = ctx["crop_quality_attempts"]
        assert "miss_reason" in attempts_rows.columns
        assert "best_unselected_candidate_score" in attempts_rows.columns


def test_crop_export_selection_and_writes():
    """Issue #44 deliverable 2: crops are budgeted worst-first (flagged, then ok to fill
    the cap); with decode off nothing is written and every crop path stays None; with a
    frame reader injected, PNGs land in crops/ and the selected frameQuality entries get
    their crop path stamped in place."""

    import numpy as np

    from analysis_pipeline import crops

    entries = [
        {"t": 1.0, "class": "ok", "crop": None},
        {"t": 2.0, "class": "wrong-subject", "crop": None},
        {"t": 3.0, "class": "distorted", "crop": None},
        {"t": 4.0, "class": "ok", "crop": None},
    ]
    # Selection: flagged first (worst-first budget), then ok to fill the cap.
    assert [e["class"] for e in crops._select_for_crop(entries, 3)] == \
        ["wrong-subject", "distorted", "ok"]
    assert crops._select_for_crop(entries, 1)[0]["class"] == "wrong-subject"
    assert crops._select_for_crop(entries, 0) == []

    pose_frames = [{"timestamp": e["t"], "keypoints": [
        {"name": "nose", "x": 0.4, "y": 0.3}, {"name": "left_hip", "x": 0.5, "y": 0.6}]}
        for e in entries]

    with tempfile.TemporaryDirectory() as tmp:
        vdir = Path(tmp) / "routeCR" / "vidCR"
        vdir.mkdir(parents=True)

        # decode off -> best-effort no-op: nothing written, no crops/ dir, paths None.
        fq_off = {"frames": [dict(e) for e in entries]}
        assert crops.export_run_crops(vdir, "20260101-000001", pose_frames, fq_off,
                                      decode=False) == 0
        assert all(e["crop"] is None for e in fq_off["frames"])
        assert not (vdir / "crops").exists()

        # Injected reader -> writes PNGs for the selected frames, stamps their paths.
        gray = np.full((100, 120), 128, dtype=np.uint8)
        fq = {"frames": [dict(e) for e in entries]}
        n = crops.export_run_crops(vdir, "20260101-000001", pose_frames, fq,
                                   decode=True, cap=3, frame_reader=lambda t: gray)
        assert n == 3
        assert len(list((vdir / "crops").glob("*.png"))) == 3
        by_t = {e["t"]: e for e in fq["frames"]}
        assert by_t[2.0]["crop"].startswith("crops/") and by_t[3.0]["crop"]
        assert by_t[1.0]["crop"]           # ok at t1 selected to fill the cap
        assert by_t[4.0]["crop"] is None   # cap reached before this ok


def test_frame_quality_aggregation_pools_all_records():
    """Issue #44 deliverable 3: per-frame classes are pooled across ALL records —
    quarantined ones included — into a class-frequency table + worst-first worklist,
    an independent pool from the conforming-only trusted metrics."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"

        # Good (conforming) bundle: t1/t2 ok (enough present joints to clear the
        # conformance point floor), t3 hallucination on an auto-absent frame. The t2
        # scanner is nudged off t1 so it is not a frozen duplicate.
        good = root / "routeFA" / "vidGood"
        _write_bundle_meta(good, setup_hash="sh_g")
        (good / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": "aaaaaaaa11111111",
            "frames": [
                {"frameIndex": 1, "timestamp": 1.0, "state": "present", "review": "auto", "joints": present},
                {"frameIndex": 2, "timestamp": 2.0, "state": "present", "review": "auto", "joints": present},
                {"frameIndex": 3, "timestamp": 3.0, "state": "absent", "review": "auto", "joints": {}},
            ]}), encoding="utf-8")
        _write_pose_run(good, "20260101-000001", "sh_g", [
            {"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
            {"timestamp": 2.0, "keypoints": _kp_list(
                {n: (x + 0.02, y) for n, (x, y) in _TRUTH_JOINTS.items()})},
            {"timestamp": 3.0, "keypoints": _kp_list(_TRUTH_JOINTS)}])

        # Bad (non-conforming, quarantined) bundle: scanner is 2x truth -> wrong-subject
        # on every frame. Quarantined out of the trusted pool but still mined here.
        bad = root / "routeFA" / "vidBad"
        _write_bundle_meta(bad, setup_hash="sh_b")
        (bad / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": "bbbbbbbb22222222",
            "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                        "review": "auto", "joints": present} for i in (1, 2, 3)]}),
            encoding="utf-8")
        _write_pose_run(bad, "20260101-000002", "sh_b", [
            {"timestamp": float(i), "keypoints": _kp_list(
                {n: (2 * x, 2 * y) for n, (x, y) in _TRUTH_JOINTS.items()})}
            for i in (1, 2, 3)])

        # Held-but-ok bundle: all frames are geometrically correct and non-raw. These
        # should increase held-pose diagnostics but must not inflate the failure worklist.
        held_ok = root / "routeFA" / "vidHeldOk"
        _write_bundle_meta(held_ok, setup_hash="sh_h")
        (held_ok / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": "cccccccc33333333",
            "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                        "review": "auto", "joints": present} for i in (1, 2, 3)]}),
            encoding="utf-8")
        _write_pose_run(held_ok, "20260101-000003", "sh_h", [
            {"timestamp": float(i), "source": "interpolated",
             "keypoints": _kp_list(_TRUTH_JOINTS)}
            for i in (1, 2, 3)])

        summary = _evaluate(root)
        assert len(summary.written) == 3

        ctx = trends.build_trend_context(root)
        # Trusted pool excludes the quarantined bundle; the frame-quality pool keeps it.
        assert ctx["eval_count"] == 2 and ctx["quarantined_count"] == 1
        assert ctx["frame_quality_detected"] == 9  # 3 good + 3 bad + 3 held-ok
        assert ctx["frame_quality_flagged"] == 4   # 1 hallucination + 3 wrong-subject
        assert ctx["frame_quality_held"] == 4       # 2 bad repeats + 2 held-ok repeats
        assert ctx["frame_quality_frozen"] == 0     # no held frame has source == raw

        classes = ctx["frame_quality_classes"].set_index("class")["n"].to_dict()
        assert classes["ok"] == 5
        assert classes["hallucination-fp"] == 1
        assert classes["wrong-subject"] == 3

        wl = ctx["frame_quality_worklist"]
        assert len(wl) == 4  # all flagged frames; held-ok diagnostics stay out
        assert wl.iloc[0]["class"] == "hallucination-fp"  # worst class first
        assert {"crop", "source", "held_pose", "frozen_stale"}.issubset(wl.columns)
        assert set(wl["class"]) == {"hallucination-fp", "wrong-subject"}


def test_hallucination_split_by_truth_presence():
    """Issue #69: ``hallucination-fp`` conflates a real false positive (a pose on a
    truth-*absent* frame) with a tracking miss (a pose on a truth-*present* frame that
    reads as a false detection). Every frameQuality frame carries ``truthPresent``, the
    record pools ``hallucinationSplit``, and both sub-cases survive into the pooled class
    table, the report, and the CSV."""

    from analysis_pipeline import cli
    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    setup_hash = "sh_hs"
    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS), "setupHash": setup_hash,
        "groundTruthHash": "hs00hs00hs00hs00",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present", "review": "auto",
             "joints": present},
            {"frameIndex": 2, "timestamp": 2.0, "state": "present", "review": "auto",
             "joints": present},
            {"frameIndex": 3, "timestamp": 3.0, "state": "absent", "review": "auto",
             "joints": {}},
            {"frameIndex": 4, "timestamp": 4.0, "state": "absent", "review": "auto",
             "joints": {}},
        ],
        # The tracking-miss half: a human called frame 2 a hallucination even though the
        # Climber is in it (the scanner locked onto a spectator). The auto classifier
        # cannot produce this — it only sets hallucination-fp on absent frames — so the
        # annotation is what makes the truth-present sub-case reachable at all.
        "detectionAnnotations": [
            {"startFrame": 2, "endFrame": 2, "failureClass": "hallucination-fp",
             "distractor": "spectator", "setupHash": setup_hash},
        ],
    }
    # Each pose is distinct enough that no run of >= FQ_FROZEN_MIN_RUN near-identical
    # poses forms — the split is being tested, not the held-pose flag.
    scanner = [
        {"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
        {"timestamp": 2.0, "keypoints": _kp_list(
            {n: (x + 0.05, y) for n, (x, y) in _TRUTH_JOINTS.items()})},
        {"timestamp": 3.0, "keypoints": _kp_list(
            {n: (x + 0.15, y) for n, (x, y) in _TRUTH_JOINTS.items()})},
        {"timestamp": 4.0, "keypoints": _kp_list(
            {n: (x + 0.25, y) for n, (x, y) in _TRUTH_JOINTS.items()})},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeHS" / "vidHS"
        _write_bundle_meta(vdir, setup_hash=setup_hash)
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000069", setup_hash, scanner)
        # A scaffold on the same grid as truth, holding the Climber up to t=2 and
        # losing them after: the two absent frames are a trailing run, not an
        # interior gap, so they are *confirmed* absences and the split can claim
        # them (issue #101).
        _write_scaffold(vdir, samples=[1.0, 2.0, 3.0, 4.0], posed=[1.0, 2.0],
                        seed_found=True)

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec["schemaVersion"] == ev.SCHEMA_VERSION >= 12
        fq = rec["frameQuality"]
        assert fq["classCounts"]["hallucination-fp"] == 3
        assert fq["classCounts"]["ok"] == 1

        by_t = {e["t"]: e for e in fq["frames"]}
        # Presence is recorded on every frame, not just the hallucinations.
        assert by_t[1.0]["class"] == "ok" and by_t[1.0]["truthPresent"] is True
        assert by_t[2.0]["class"] == "hallucination-fp"
        assert by_t[2.0]["autoClass"] == "ok"       # human override, present frame
        assert by_t[2.0]["truthPresent"] is True    # ...so: a tracking miss
        assert by_t[3.0]["class"] == "hallucination-fp"
        assert by_t[3.0]["truthPresent"] is False   # ...a real false positive
        assert by_t[4.0]["truthPresent"] is False
        # ...and the absence is one the harness can actually claim (issue #101).
        assert by_t[3.0]["absenceReason"] == ev.ABSENCE_CONFIRMED
        assert by_t[1.0]["absenceReason"] is None   # present frame: no such question

        split = fq["hallucinationSplit"]
        assert split["total"] == 3
        assert split[ev.HALLUCINATION_TRUTH_ABSENT] == 2
        assert split[ev.HALLUCINATION_TRUTH_PRESENT] == 1
        assert split["unconfirmedAbsent"] == 0
        assert split["truthAbsentShare"] == round(2 / 3, 6)
        assert split["truthPresentShare"] == round(1 / 3, 6)

        ctx = trends.build_trend_context(root)
        classes = ctx["frame_quality_classes"].set_index("class")
        assert classes.loc["hallucination-fp", "n"] == 3
        assert classes.loc["hallucination-fp", "truth_absent"] == 2
        assert classes.loc["hallucination-fp", "truth_present"] == 1
        assert classes.loc["hallucination-fp", "truth_unknown"] == 0
        assert classes.loc["hallucination-fp", "truth_absent_share"] == 2 / 3
        # A non-hallucination class is split too — the axis is per-frame, not per-class.
        assert classes.loc["ok", "truth_present"] == 1

        pooled = ctx["frame_quality_hallucination"]
        assert (pooled["total"], pooled["truth_absent"], pooled["truth_present"]) == (3, 2, 1)

        # The worklist carries presence so a flagged frame can be triaged without a join.
        wl = ctx["frame_quality_worklist"].set_index("t")
        assert bool(wl.loc[2.0, "truth_present"]) is True
        assert bool(wl.loc[3.0, "truth_present"]) is False

        out = Path(tmp) / "reports"
        cli.main(["analysis", str(root), "-o", str(out), "--no-decode"])

        html = (out / "report.html").read_text(encoding="utf-8")
        assert "real false positives" in html and "tracking misses" in html
        assert "66.7%" in html  # the truth-absent share, stated in the split callout

        csv_text = (out / "eval_frame_quality_classes.csv").read_text(encoding="utf-8")
        header = csv_text.splitlines()[0]
        assert {"truth_absent", "truth_present", "truth_unknown",
                "truth_absent_share"}.issubset(set(header.split(",")))


def test_hallucination_split_reads_old_frames_as_unknown_and_unconfirmed():
    """Fail-open on both axes, one per schema bump.

    Issue #69: a frame from a record written before schema v12 recorded no presence and
    reads as *unknown* rather than being counted on either side. Issue #101: a frame
    whose absence carries no reason — a pre-v14 record — is an absence the harness
    cannot confirm, and an unconfirmed absence is not evidence of a false positive, so
    it is held out of the split instead of being promoted to one."""

    import pandas as pd

    from analysis_pipeline import trends

    df = pd.DataFrame([
        {"class": "hallucination-fp", "truth_present": None,      # pre-v12 frame
         "absence_reason": None, "held_pose": 0, "frozen_stale": 0},
        {"class": "hallucination-fp", "truth_present": False,     # pre-v14 frame
         "absence_reason": None, "held_pose": 0, "frozen_stale": 0},
        {"class": "hallucination-fp", "truth_present": False,     # v14, confirmed
         "absence_reason": "confirmed-absent", "held_pose": 0, "frozen_stale": 0},
        {"class": "hallucination-fp", "truth_present": False,     # v14, not confirmed
         "absence_reason": "not-sampled", "held_pose": 0, "frozen_stale": 0},
        {"class": "ok", "truth_present": True, "absence_reason": None,
         "held_pose": 0, "frozen_stale": 0},
    ])
    classes = trends._frame_quality_classes(df).set_index("class")
    assert classes.loc["hallucination-fp", "truth_unknown"] == 1        # presence unknown
    assert classes.loc["hallucination-fp", "truth_absent"] == 1         # confirmed only
    assert classes.loc["hallucination-fp", "truth_absent_unconfirmed"] == 2
    assert classes.loc["hallucination-fp", "truth_present"] == 0
    # Shares are over the frames the harness can actually claim.
    assert classes.loc["hallucination-fp", "truth_absent_share"] == 1.0

    pooled = trends._hallucination_split_totals(df)
    assert pooled == {"total": 4, "truth_present": 0, "truth_absent": 1,
                      "truth_absent_unconfirmed": 2, "truth_unknown": 1,
                      "truth_present_share": 0.0, "truth_absent_share": 1.0}

    # An all-unknown pool reports no split at all rather than a fabricated 0%.
    legacy = trends._hallucination_split_totals(
        df[df["truth_present"].isna()].copy())
    assert legacy["truth_unknown"] == 1
    assert legacy["truth_absent_share"] is None
    assert trends._hallucination_split_totals(pd.DataFrame())["total"] == 0

    # The absence-reason breakdown names every reason, marks which one counts, and
    # covers every absent frame exactly once.
    reasons = trends._absence_reason_counts(df).set_index("reason")
    assert reasons.loc["confirmed-absent", "n"] == 1
    assert reasons.loc["not-sampled", "n"] == 1
    assert reasons.loc["unknown", "n"] == 1        # the pre-v14 absent frame
    assert reasons.loc["out-of-scope", "n"] == 0   # keyed even at zero
    assert int(reasons["n"].sum()) == 3            # the three absent frames
    assert bool(reasons.loc["confirmed-absent", "counts_as_absent"]) is True
    assert bool(reasons.loc["untracked", "counts_as_absent"]) is False


def test_attempt_funnel_pools_and_distributes_over_runs():
    """Issue #87: the attempt funnel reports each status pooled over attempts *and*
    distributed over runs, plus reacquire effectiveness and condition-flag rates by
    status — all at the Run unit, with no pooled-attempt CIs.

    The two runs are deliberately lopsided (a 4-attempt run that mostly works, and one
    that mostly misses) so the pooled share and the run median disagree: that gap is the
    thing the run-unit columns exist to show."""

    from analysis_pipeline import cli
    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    exact = _kp_list(_TRUTH_JOINTS)
    crop = {"x": 0.2, "y": 0.1, "w": 0.6, "h": 0.9}
    full_frame = {"x": 0, "y": 0, "w": 1, "h": 1}

    def truth_doc(hash_: str) -> dict:
        return {"version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": hash_,
                "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                            "review": "auto", "joints": present} for i in (1, 2, 3, 4)]}

    def conditions(too_dark: bool) -> dict:
        return {"mean": 40, "stdDev": 12, "sharpness": 60, "flags": {"tooDark": too_dark}}

    dense = [{"timestamp": float(i), "keypoints": exact} for i in (1, 2, 3, 4)]

    # Healthy run: 2 accepted, 1 missing, 1 flipRejected. Never reacquires, so it must
    # contribute no reacquire-success value at all rather than a zero.
    healthy = [
        {"timestamp": 1.0, "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact, "acceptedKeypoints": exact,
         "candidateCount": 1, "searchConditions": conditions(False)},
        {"timestamp": 2.0, "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact, "acceptedKeypoints": exact,
         "candidateCount": 1, "searchConditions": conditions(False)},
        {"timestamp": 3.0, "status": "missing", "initialSearchRegion": crop,
         "rawKeypoints": [], "acceptedKeypoints": [], "candidateCount": 0,
         "searchConditions": conditions(True)},
        {"timestamp": 4.0, "status": "flipRejected", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact, "acceptedKeypoints": [],
         "candidateCount": 1, "rejectedCandidateCount": 1,
         "searchConditions": conditions(False)},
    ]
    # Collapsing run: 3 missing (each with a failed full-frame reacquire) and one
    # accepted that a reacquire recovered — 75% missing makes it a tail run.
    collapsing = [
        {"timestamp": float(i), "status": "missing", "initialSearchRegion": crop,
         "detectionRegion": full_frame, "reacquireAttempted": True, "reacquired": False,
         "rawKeypoints": [], "acceptedKeypoints": [], "candidateCount": 0,
         "searchConditions": conditions(True)}
        for i in (1, 2, 3)
    ] + [
        {"timestamp": 4.0, "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": full_frame, "reacquireAttempted": True, "reacquired": True,
         "rawKeypoints": exact, "acceptedKeypoints": exact, "candidateCount": 1,
         "searchConditions": conditions(False)},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        out = Path(tmp) / "reports"

        for name, attempts, hash_ in (("vidHealthy", healthy, "11111111aaaaaaaa"),
                                      ("vidCollapse", collapsing, "22222222bbbbbbbb")):
            vdir = root / "routeFUN" / name
            _write_bundle_meta(vdir, setup_hash=f"sh_{name}")
            (vdir / "ground-truth.json").write_text(json.dumps(truth_doc(hash_)),
                                                    encoding="utf-8")
            _write_pose_run(vdir, f"20260101-0000{name[3:5]}", f"sh_{name}", dense,
                            detector_attempts=attempts)

        # A legacy frames-only bundle: in the corpus, but it has no attempt stream to
        # funnel, so it must not appear in any funnel table.
        legacy = root / "routeFUN" / "vidLegacy"
        _write_bundle_meta(legacy, setup_hash="sh_legacy")
        (legacy / "ground-truth.json").write_text(
            json.dumps(truth_doc("33333333cccccccc")), encoding="utf-8")
        _write_pose_run(legacy, "20260101-0000LG", "sh_legacy", dense)

        assert len(_evaluate(root).written) == 3
        ctx = trends.build_trend_context(root)

        # Attempt-backed runs only — the legacy run is in the corpus but not the funnel.
        runs = ctx["attempt_funnel_runs"]
        assert len(runs) == 2
        assert set(runs["video_key"]) == {"vidHealthy", "vidCollapse"}
        assert ctx["evidence_generation_funnel"]["label"] == "attempts"
        assert ctx["evidence_generation_funnel"]["mixed"] is False

        # Status mix: every status is a row even at zero, in funnel order.
        status = ctx["attempt_funnel_status"]
        assert list(status["status"]) == ["accepted", "missing", "flipRejected",
                                          "qualityRejected", "unknown"]
        by_status = status.set_index("status")
        assert int(by_status.loc["accepted", "attempts"]) == 3
        assert int(by_status.loc["missing", "attempts"]) == 4
        assert int(by_status.loc["flipRejected", "attempts"]) == 1
        assert by_status.loc["missing", "share"] == 0.5          # 4 of 8 attempts
        assert by_status.loc["qualityRejected", "attempts"] == 0
        assert by_status.loc["qualityRejected", "share"] == 0.0
        assert by_status.loc["qualityRejected", "runs_with_any"] == 0

        # ...and the run-unit distribution the pooled share hides: 25% vs 75% missing.
        assert by_status.loc["missing", "run_share_median"] == 0.5
        assert abs(by_status.loc["missing", "run_share_p90"] - 0.7) < 1e-9
        assert by_status.loc["missing", "run_share_max"] == 0.75
        assert int(by_status.loc["missing", "tail_runs"]) == 1   # only the collapsing run
        assert int(by_status.loc["accepted", "tail_runs"]) == 0

        # Reacquire: a run that never reacquired contributes no rate, not a zero.
        stats = ctx["attempt_funnel_run_stats"].set_index("metric")
        assert stats.loc["attempt_count", "median"] == 4.0
        assert int(stats.loc["attempt_reacquire_success_rate", "n_runs"]) == 1
        assert stats.loc["attempt_reacquire_success_rate", "median"] == 0.25
        assert int(stats.loc["attempt_reacquire_attempt_rate", "n_runs"]) == 2

        # Condition flags split by what happened next: tooDark fires on every miss and
        # on nothing that was accepted.
        flags = ctx["attempt_funnel_flags"].set_index(["flag", "status"])
        assert flags.loc[("too_dark", "missing"), "rate"] == 1.0
        assert int(flags.loc[("too_dark", "missing"), "attempts_scored"]) == 4
        assert int(flags.loc[("too_dark", "missing"), "n_runs"]) == 2
        assert flags.loc[("too_dark", "missing"), "run_rate_median"] == 1.0
        assert flags.loc[("too_dark", "accepted"), "rate"] == 0.0
        assert int(flags.loc[("too_dark", "accepted"), "attempts_scored"]) == 3
        assert flags.loc[("too_dark", "accepted"), "run_rate_p90"] == 0.0
        assert flags.loc[("too_dark", "flipRejected"), "rate"] == 0.0

        totals = ctx["attempt_funnel"]
        assert totals["runs"] == 2 and totals["attempts"] == 8
        assert totals["reacquire_attempted"] == 4 and totals["reacquire_succeeded"] == 1
        assert totals["reacquire_success_rate"] == 0.25
        assert totals["tail_runs_missing"] == 1
        assert totals["missing_share_run_median"] == 0.5

        # A collapsing run fails the #15 gate — and must still be in the funnel, since a
        # collapsed funnel is exactly what quarantines it.
        assert ctx["quarantined_count"] >= 1
        assert "vidCollapse" in set(runs["video_key"])

        # The section renders the funnel, its evidence generation, and the CSVs land.
        html_frag = report._attempt_funnel_html(ctx)
        for text in ("Status mix", "Tail runs", "flipRejected", "50.0%",
                     "Search-condition flags by status", "too_dark"):
            assert text in html_frag, text
        assert "evidence: attempts" in report._evidence_generation_html(
            ctx["evidence_generation_funnel"])

        outputs = cli.run(root, out, decode=False)
        html_text = outputs["html"].read_text(encoding="utf-8")
        assert "Detector Attempt funnel (run unit)" in html_text
        for name in ("eval_attempt_funnel_status.csv", "eval_attempt_funnel_runs.csv",
                     "eval_attempt_funnel_run_stats.csv", "eval_attempt_funnel_flags.csv"):
            assert (out / name).exists(), name
        status_csv = pd.read_csv(out / "eval_attempt_funnel_status.csv")
        assert list(status_csv["status"])[:2] == ["accepted", "missing"]
        assert {"share", "run_share_median", "run_share_p90", "tail_runs"}.issubset(
            status_csv.columns)


def test_evidence_generation_dedup_prefers_attempt_backed_record():
    """Issue #89: a video+truth pairing carrying both an attempt-backed record and the
    legacy-frames record it superseded is pooled **once**, from the attempt-backed side.

    The legacy record stays on disk and readable; only the aggregation passes it over,
    and the report accounts for it by name. A pairing with no attempt-backed record is
    untouched — a legacy-only corpus aggregates exactly as before."""

    from analysis_pipeline import cli
    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    exact = _kp_list(_TRUTH_JOINTS)
    truth_doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "dddddddd44444444",
        "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                    "review": "auto", "joints": present} for i in (1, 2, 3)],
    }
    dense = [{"timestamp": float(i), "keypoints": exact} for i in (1, 2, 3)]
    crop = {"x": 0.2, "y": 0.1, "w": 0.6, "h": 0.9}
    attempts = [
        {"timestamp": float(i), "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact, "acceptedKeypoints": exact,
         "candidateCount": 1, "selectionMethod": "tracked"}
        for i in (1, 2, 3)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        out = Path(tmp) / "reports"

        # The double batch: one video scanned twice against the same truth — a morning
        # legacy-frames run and an afternoon attempt-backed one.
        both = root / "routeGEN" / "vidBoth"
        _write_bundle_meta(both, setup_hash="sh_both")
        (both / "ground-truth.json").write_text(json.dumps(truth_doc), encoding="utf-8")
        # The morning run carries a distinct config so the two generations survive
        # discovery's byte-identical dedup as two observations, as they do in the
        # real corpus.
        _write_pose_run(both, "20260724-090000", "sh_both", dense, app_version="aaa1111",
                        config={"frameStep": 10, "frameIntervalMs": 100})
        _write_pose_run(both, "20260724-150000", "sh_both", dense, app_version="aaa1111",
                        detector_attempts=attempts)

        # A legacy-only video: two runs, neither attempt-backed, nothing to supersede.
        legacy = root / "routeGEN" / "vidLegacy"
        _write_bundle_meta(legacy, setup_hash="sh_legacy")
        (legacy / "ground-truth.json").write_text(
            json.dumps({**truth_doc, "groundTruthHash": "eeeeeeee55555555"}),
            encoding="utf-8")
        _write_pose_run(legacy, "20260724-090001", "sh_legacy", dense, app_version="aaa1111",
                        config={"frameStep": 10, "frameIntervalMs": 100})
        _write_pose_run(legacy, "20260724-150001", "sh_legacy", dense, app_version="aaa1111")

        summary = _evaluate(root)
        assert len(summary.written) == 4  # every run is still evaluated and committed

        ctx = trends.build_trend_context(root)
        assert ctx["eval_count_on_disk"] == 4
        assert ctx["eval_count_total"] == 3       # the superseded legacy run is dropped
        assert ctx["eval_count"] == 3             # ...and all three conform

        # Named, not merely absent: which record, and what superseded it.
        assert ctx["superseded_count"] == 1
        sup = ctx["superseded_records"][0]
        assert (sup["route_folder"], sup["video_key"]) == ("routeGEN", "vidBoth")
        assert sup["run_ts"] == "20260724-090000"
        assert sup["evidence_generation"] == "legacy-frames"
        assert sup["superseded_by"] == "20260724-150000"
        assert sup["truth_hash"] == "dddddddd44444444"

        # ...and the record it names is untouched on disk.
        assert (both / "evaluations" / "20260724-090000_vs_dddddddd.json").exists()

        # The pairing is counted once, from the attempt-backed side; the legacy-only
        # video keeps both of its runs.
        runs = ctx["detection_error_attempt_runs"].set_index("run_ts")
        assert set(runs.index) == {"20260724-150000", "20260724-090001", "20260724-150001"}
        assert runs.loc["20260724-150000", "attempt_evidence"] == "attempts"
        assert (runs.loc[["20260724-090001", "20260724-150001"],
                         "attempt_evidence"] == "unknown").all()

        # Version tracking sees three records, not four — a change of evidence
        # generation can no longer masquerade as a scanner change.
        assert list(ctx["version_overview"]["n_records"]) == [3]

        frame_rows = ctx["frame_joint_df"]
        assert set(frame_rows[frame_rows["video_key"] == "vidBoth"]["run_ts"]) == {
            "20260724-150000"}

        # Issue #101: the same dedup now runs *before* the frame table is built — the
        # superseded legacy run contributes no frame rows (so no decode is ever spent
        # on it), while the legacy-only video keeps every run it has.
        frame_records = discover_runs(root)
        assert {r.run_ts for r in frame_records if r.video_key == "vidBoth"} == {
            "20260724-090000", "20260724-150000"}
        frame_df = build_frame_table(frame_records, decode=False)
        assert set(frame_df[frame_df["video_key"] == "vidBoth"]["run_ts"]) == {
            "20260724-150000"}
        assert set(frame_df[frame_df["video_key"] == "vidLegacy"]["run_ts"]) == {
            "20260724-090001", "20260724-150001"}

        # Every pooled section can state what it is made of, and a mixed pool says so.
        pooled = ctx["evidence_generation_trusted"]
        assert pooled["counts"] == {"attempts": 1, "legacy-frames": 2, "unknown": 0}
        assert pooled["mixed"] is True
        assert pooled["n_records"] == 3
        badge = report._evidence_generation_html(pooled)
        assert "evidence: MIXED" in badge
        assert "legacy-frames" in badge and "attempts" in badge

        # A single-generation pool names its generation instead.
        single = trends._evidence_generation_summary(
            [r for r in ctx["eval_records"] if r.video_key == "vidBoth"], "test")
        assert single["mixed"] is False and single["label"] == "attempts"
        assert "evidence: attempts" in report._evidence_generation_html(single)

        # The report accounts for the superseded record, and the CSV exports it.
        outputs = cli.run(root, out, decode=False)
        html_text = outputs["html"].read_text(encoding="utf-8")
        assert "Superseded records (#89 evidence-generation dedup)" in html_text
        assert "20260724-090000" in html_text
        sup_csv = pd.read_csv(out / "eval_superseded_records.csv")
        assert list(sup_csv["run_ts"]) == ["20260724-090000"]
        assert list(sup_csv["superseded_by"]) == ["20260724-150000"]


def test_evidence_generation_dedup_is_scoped_to_one_truth_revision():
    """Issue #89: the pairing key includes the truth revision. An attempt-backed record
    scored against a *different* truth supersedes nothing — the two records were never
    measuring the same thing, and #10's mixed-truth guard already refuses to compare
    them."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    exact = _kp_list(_TRUTH_JOINTS)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeGEN2" / "vidRevised"
        _write_bundle_meta(vdir, setup_hash="sh_rev")
        (vdir / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS),
            "groundTruthHash": "ffffffff66666666",
            "frames": [{"frameIndex": i, "timestamp": float(i), "state": "present",
                        "review": "auto", "joints": present} for i in (1, 2, 3)]}),
            encoding="utf-8")
        _write_pose_run(
            vdir, "20260724-150000", "sh_rev",
            [{"timestamp": float(i), "keypoints": exact} for i in (1, 2, 3)],
            detector_attempts=[
                {"timestamp": float(i), "status": "accepted", "rawKeypoints": exact,
                 "acceptedKeypoints": exact, "candidateCount": 1}
                for i in (1, 2, 3)])
        assert len(_evaluate(root).written) == 1

        # A legacy record for the same video, scored against the truth revision this
        # bundle has since replaced.
        (vdir / "evaluations" / "20260101-000001_vs_99990000.json").write_text(
            json.dumps({"schemaVersion": 7, "routeFolder": "routeGEN2",
                        "videoKey": "vidRevised", "runTs": "20260101-000001",
                        "truthHash": "9999000099990000",
                        "frameQuality": {"detectorEvidence": "legacy-frames",
                                         "frames": []}}),
            encoding="utf-8")

        ctx = trends.build_trend_context(root)
        assert ctx["superseded_count"] == 0
        assert ctx["eval_count_total"] == 2


def _absence_bundle(root: Path, *, setup: dict, truth_frames: list,
                    scanner: list, samples: list[float], posed: list[float],
                    seed_found: bool | None = None, name: str = "vidABS") -> Path:
    """A bundle wired for absence provenance: calibration, truth, scaffold, one Run."""

    vdir = root / "routeABS" / name
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "metadata.json").write_text(
        json.dumps({"route_folder": "routeABS", "video_key": name}), encoding="utf-8")
    (vdir / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    (vdir / "ground-truth.json").write_text(json.dumps({
        "version": 1, "jointSet": list(_TRUTH_JOINTS), "setupHash": setup["setupHash"],
        "groundTruthHash": f"ab5{name}"[:16].ljust(16, "0"),
        "frames": truth_frames}), encoding="utf-8")
    _write_pose_run(vdir, "20260101-000101", setup["setupHash"], scanner)
    _write_scaffold(vdir, samples=samples, posed=posed, seed_found=seed_found)
    return vdir


def _truth_frame(i: int, t: float, present: bool) -> dict:
    joints = ({n: {"x": x, "y": y, "occluded": False}
               for n, (x, y) in _TRUTH_JOINTS.items()} if present else {})
    return {"frameIndex": i, "timestamp": t,
            "state": "present" if present else "absent",
            "review": "auto", "joints": joints}


def test_absence_reason_is_derived_from_on_disk_evidence():
    """Issue #101: every absent truth frame carries *why* it is absent, derived from the
    climb window, the scaffold's sampling grid and its tracking-gap structure — never
    authored into Ground Truth, which stays pure keypoints.

    One frame per reason, in one bundle. Truth is on a 0.5 s grid while the scaffold
    sampled at 1 Hz — the shape of the real defect: t=0.5 is before the climb start
    (out-of-scope), t=2.5 falls between the scaffold's samples (not-sampled), t=4 sits
    inside a gap the scaffold held the Climber on both sides of (untracked), and t=7 is
    a trailing absence after the last posed frame (confirmed-absent)."""

    from analysis_pipeline import evaluate as ev

    truth = [
        _truth_frame(1, 0.5, False),    # before climbStart=1.0
        _truth_frame(2, 1.0, True),
        _truth_frame(3, 1.5, True),
        _truth_frame(4, 2.0, True),
        _truth_frame(5, 2.5, False),    # between the scaffold's 1 s samples
        _truth_frame(6, 3.0, True),
        _truth_frame(7, 3.5, True),
        _truth_frame(8, 4.0, False),    # interior gap: posed at 3 and 5
        _truth_frame(9, 4.5, True),
        _truth_frame(10, 5.0, True),
        _truth_frame(11, 7.0, False),   # trailing: nothing posed after 5
    ]
    scanner = [{"timestamp": f["timestamp"], "keypoints": _kp_list(_TRUTH_JOINTS)}
               for f in truth]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _absence_bundle(
            root,
            setup={"setupHash": "sh_abs", "climberPoint": {"x": 0.5, "y": 0.9, "t": 1.0},
                   "climbEnd": 8.0},
            truth_frames=truth, scanner=scanner,
            samples=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], posed=[2.0, 3.0, 5.0],
            seed_found=True)

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec["schemaVersion"] == ev.SCHEMA_VERSION == 14
        assert rec["counts"]["absenceReasons"] == {
            ev.ABSENCE_OUT_OF_SCOPE: 1, ev.ABSENCE_NOT_SAMPLED: 1,
            ev.ABSENCE_UNTRACKED: 1, ev.ABSENCE_CONFIRMED: 1, ev.ABSENCE_UNKNOWN: 0}
        assert rec["climbWindow"] == {"start": 1.0, "end": 8.0}

        # Each reason is visible per frame, not only in the totals.
        by_t = {e["t"]: e for e in rec["frameQuality"]["frames"]}
        assert by_t[2.5]["absenceReason"] == ev.ABSENCE_NOT_SAMPLED
        assert by_t[4.0]["absenceReason"] == ev.ABSENCE_UNTRACKED
        assert by_t[7.0]["absenceReason"] == ev.ABSENCE_CONFIRMED
        assert by_t[2.0]["absenceReason"] is None      # present: no such question
        assert 0.5 not in by_t                         # out of scope: not scored at all

        # Only the confirmed absence reaches the presence 2×2; the other two matched
        # absences are held out and counted, never dropped.
        agr = rec["agreement"]
        assert agr["presence"]["absentDetected"] == 1
        assert agr["frames"]["unconfirmedAbsent"] == 2


def test_untracked_absence_when_the_scaffold_never_seeded():
    """A scaffold whose seeding failed outright makes *every* absence a tracking
    failure — reading those as a departed Climber would blame the scanner for the
    harness's own miss (issue #101)."""

    from analysis_pipeline import evaluate as ev

    truth = [_truth_frame(1, 1.0, True), _truth_frame(2, 2.0, False),
             _truth_frame(3, 3.0, False)]
    scanner = [{"timestamp": f["timestamp"], "keypoints": _kp_list(_TRUTH_JOINTS)}
               for f in truth]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _absence_bundle(root, setup={"setupHash": "sh_seedfail"},
                        truth_frames=truth, scanner=scanner,
                        samples=[1.0, 2.0, 3.0], posed=[], seed_found=False)
        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_UNTRACKED] == 2
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_CONFIRMED] == 0
        assert rec["agreement"]["presence"]["absentDetected"] == 0


def test_absence_reason_is_unknown_without_a_scaffold_to_derive_from():
    """Fail-open: with no scaffold on disk there is nothing to derive from, so an absent
    frame reads ``unknown`` and stays out of the presence 2×2 — never silently promoted
    to a confirmed absence (issue #101)."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeNS" / "vidNS"
        _write_bundle_meta(vdir, setup_hash="sh_ns")
        (vdir / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS), "setupHash": "sh_ns",
            "groundTruthHash": "ns00ns00ns00ns00",
            "frames": [_truth_frame(1, 1.0, True), _truth_frame(2, 2.0, False)]}),
            encoding="utf-8")
        _write_pose_run(vdir, "20260101-000102", "sh_ns",
                        [{"timestamp": t, "keypoints": _kp_list(_TRUTH_JOINTS)}
                         for t in (1.0, 2.0)])

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_UNKNOWN] == 1
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_CONFIRMED] == 0
        assert rec["agreement"]["presence"]["absentDetected"] == 0
        assert rec["agreement"]["frames"]["unconfirmedAbsent"] == 1


def test_climb_window_excludes_out_of_scope_frames_from_scoring_and_the_fit():
    """Issue #101: a Climber walking away from a finished problem is out of scope, not a
    detection failure. Out-of-window truth frames are excluded from scoring *and* from
    the conformance fit — so they cannot influence whether a Bundle is quarantined — and
    the count is surfaced rather than silently applied."""

    from analysis_pipeline import evaluate as ev

    # Six in-window present frames, plus two post-topout frames where the scanner is
    # tracking something else entirely (joints far from truth).
    truth = [_truth_frame(i, float(i), True) for i in range(1, 7)]
    truth += [_truth_frame(7, 7.0, True), _truth_frame(8, 8.0, True)]
    scanner = [{"timestamp": float(i), "keypoints": _kp_list(_TRUTH_JOINTS)}
               for i in range(1, 7)]
    wrong = {n: (x * 0.2 + 0.05, y * 0.2 + 0.05) for n, (x, y) in _TRUTH_JOINTS.items()}
    scanner += [{"timestamp": float(i), "keypoints": _kp_list(wrong)} for i in (7, 8)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _absence_bundle(
            root,
            setup={"setupHash": "sh_win", "climberPoint": {"x": 0.5, "y": 0.9, "t": 1.0},
                   "climbEnd": 6.0},
            truth_frames=truth, scanner=scanner,
            samples=[float(i) for i in range(1, 9)],
            posed=[float(i) for i in range(1, 7)], seed_found=True)

        rec = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec["counts"]["truthFramesOutOfScope"] == 2
        assert rec["counts"]["agreementSkipped"]["outOfScope"] == 2
        # Scored over the six in-window frames only...
        assert rec["agreement"]["frames"]["matchedPresent"] == 6
        # ...and the fit is clean, which it would not be with the two wrong-subject
        # post-topout frames dragged in.
        assert rec["conformance"]["y"]["slope"] == 1.0
        assert rec["conformance"]["conforms"] is True

        # With the end marker removed the bundle behaves exactly as it did before the
        # window existed: every frame scored, and the fit sees the bad ones.
        setup_path = root / "routeABS" / "vidABS" / "setup.json"
        setup_path.write_text(json.dumps({"setupHash": "sh_win"}), encoding="utf-8")
        rec2 = json.loads(_evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        assert rec2["counts"]["truthFramesOutOfScope"] == 0
        assert rec2["climbWindow"] == {"start": None, "end": None}
        assert rec2["agreement"]["frames"]["matchedPresent"] == 8
        assert rec2["conformance"]["conforms"] is False


def test_truth_sufficiency_floor_quarantines_a_thin_bundle():
    """Issue #101: the conformance gate gains a floor counted in **frames**.

    The motivating Bundle fit near-perfectly on 11 truth-present frames out of 633 and
    passed, because ``CONFORMANCE_MIN_POINTS`` counts joint-pairs and 11 frames × 11
    joints clears 20 comfortably. This runs the real, unpatched gate: a thin bundle
    quarantines on ``insufficient-frames`` even with a flawless fit, while the same
    bundle padded past the floor conforms."""

    from analysis_pipeline import evaluate as ev

    def _bundle(root: Path, n_frames: int, name: str) -> dict:
        truth = [_truth_frame(i, float(i), True) for i in range(1, n_frames + 1)]
        scanner = [{"timestamp": float(i), "keypoints": _kp_list(_TRUTH_JOINTS)}
                   for i in range(1, n_frames + 1)]
        _absence_bundle(root, setup={"setupHash": f"sh_{name}"}, truth_frames=truth,
                        scanner=scanner, samples=[float(i) for i in range(1, n_frames + 1)],
                        posed=[float(i) for i in range(1, n_frames + 1)],
                        seed_found=True, name=name)
        written = {p.video_key: p for p in ev.evaluate(root).written}
        return json.loads(written[name].record_path.read_text(encoding="utf-8"))

    assert ev.CONFORMANCE_MIN_FIT_FRAMES == 20
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        thin = _bundle(root, 11, "vidTHIN")
        # A flawless fit over plenty of joint-pairs...
        assert thin["conformance"]["y"]["r2"] == 1.0
        assert thin["conformance"]["n"] >= ev.CONFORMANCE_MIN_POINTS
        # ...and still quarantined, on frames.
        assert thin["conformance"]["causeEvidence"]["fitFrames"] == 11
        assert thin["conformance"]["conforms"] is False
        assert "insufficient-frames" in thin["conformance"]["reasons"]
        assert thin["conformance"]["thresholds"]["minFitFramesGate"] == 20

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        thick = _bundle(root, 25, "vidTHICK")
        assert thick["conformance"]["causeEvidence"]["fitFrames"] == 25
        assert thick["conformance"]["conforms"] is True
        assert thick["conformance"]["reasons"] == []


def test_rate_mismatch_is_its_own_nonconformance_cause():
    """Issue #101: a scaffold sampled far coarser than the truth grid is a *data* defect
    — it routes to regenerating the scaffold, not to the truth-repair worklist and not
    to the detector worklist, so it must not be labelled ``sparse-match``."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    # Truth on a 0.1 s grid; the scaffold sampled at 1 Hz — the measured corpus defect.
    truth, scanner = [], []
    for i in range(1, 41):
        t = round(i * 0.1, 1)
        posed_here = abs(t - round(t)) < 1e-9
        truth.append(_truth_frame(i, t, posed_here))
        scanner.append({"timestamp": t, "keypoints": _kp_list(_TRUTH_JOINTS)})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _absence_bundle(root, setup={"setupHash": "sh_rate"}, truth_frames=truth,
                        scanner=scanner, samples=[float(i) for i in range(1, 5)],
                        posed=[float(i) for i in range(1, 5)], seed_found=True,
                        name="vidRATE")
        rec = json.loads(ev.evaluate(root).written[0].record_path.read_text(encoding="utf-8"))

        conf = rec["conformance"]
        assert conf["conforms"] is False           # only 4 present frames survive
        assert conf["cause"] == ev.NONCONFORMANCE_RATE_MISMATCH
        assert conf["causeEvidence"]["samplingRatio"] == 10.0
        assert conf["causeEvidence"]["scaffoldStepSec"] == 1.0
        assert conf["causeEvidence"]["truthStepSec"] == 0.1
        # The absences it fabricates are named as such, not counted as departures.
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_NOT_SAMPLED] == 36
        assert rec["counts"]["absenceReasons"][ev.ABSENCE_CONFIRMED] == 0

        # It quarantines under its own cause, and never reaches the truth-repair
        # worklist — re-seeding truth would repair nothing here.
        ctx = trends.build_trend_context(root)
        assert ctx["quarantine_cause_counts"][ev.NONCONFORMANCE_RATE_MISMATCH] == 1
        assert ctx["truth_repair_count"] == 0

        # The report names the cause and tells the reader what to do about it...
        from analysis_pipeline import report
        quarantine_html = report._quarantine_table(ctx["quarantined_bundles"])
        assert "rate-mismatch" in quarantine_html
        assert "regenerate the scaffold" in quarantine_html.lower()
        # ...and the absence breakdown is rendered rather than left in a CSV.
        absence_html = report._absence_reason_html(ctx)
        assert "not-sampled" in absence_html
        assert "scaffold artifact" in absence_html
        assert "regenerate those scaffolds" in absence_html


def test_rate_mismatch_is_reported_even_when_the_bundle_conforms():
    """Issue #101: ``rate-mismatch`` is a non-conformance *cause*, so it only speaks
    when a record also fails the gate — and a Bundle can under-sample its truth grid
    tenfold while fitting cleanly on the frames it did sample. On the real corpus that
    is 17 records. The defect must stay visible anyway, because the fix is the same."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    # Truth on a 0.1 s grid, scaffold at 1 Hz — but enough posed frames that the fit is
    # clean and the truth-sufficiency floor is cleared, so the gate passes.
    truth, scanner = [], []
    for i in range(1, 251):
        t = round(i * 0.1, 1)
        posed_here = abs(t - round(t)) < 1e-9
        truth.append(_truth_frame(i, t, posed_here))
        scanner.append({"timestamp": t, "keypoints": _kp_list(_TRUTH_JOINTS)})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _absence_bundle(root, setup={"setupHash": "sh_ratepass"}, truth_frames=truth,
                        scanner=scanner, samples=[float(i) for i in range(1, 26)],
                        posed=[float(i) for i in range(1, 26)], seed_found=True,
                        name="vidRATEPASS")
        rec = json.loads(ev.evaluate(root).written[0].record_path.read_text(encoding="utf-8"))

        # It conforms — so the quarantine cause is silent by construction...
        assert rec["conformance"]["conforms"] is True
        assert rec["conformance"]["cause"] is None
        assert rec["conformance"]["causeEvidence"]["samplingRatio"] == 10.0

        ctx = trends.build_trend_context(root)
        assert ctx["quarantine_cause_counts"][ev.NONCONFORMANCE_RATE_MISMATCH] == 0
        # ...and the record is surfaced regardless, named with its ratio.
        assert ctx["rate_mismatch_count"] == 1
        row = ctx["rate_mismatch_records"][0]
        assert row["video_key"] == "vidRATEPASS"
        assert (row["sampling_ratio"], row["conforms"]) == (10.0, True)
        assert "still <em>pass</em>" in report._absence_reason_html(ctx)

def test_frame_table_sequential_reader_and_memo():
    """Issue #101 decode contract, asserted through the injected frame-reader seam:
    the decode path requests timestamps in sequential order, and a repeated
    video/timestamp is read exactly once however many Runs sample it — with no
    video file and no codec involved."""

    import numpy as np

    class LogReader:
        """Fake frame reader: records every access, serves a synthetic gray frame."""

        def __init__(self):
            self.calls: list[tuple[str, float]] = []

        def read_gray(self, video_path, t):
            self.calls.append((Path(video_path).name, t))
            return np.full((8, 8), 128, dtype=np.uint8)

        def close(self):
            pass

    labels = {"route_orientation": "head-on", "camera_angle": "level",
              "shadows": "high", "climber_contrast": "low", "wall_contrast": "medium",
              "motion_blur": "low", "occlusion": "unknown", "camera_stability": "steady"}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vdir = root / "routeSEQ" / "vidSEQ"
        # Two legacy runs over the same video (no exported stats, no attempts), with
        # distinct configs: the 1 s run samples t=0..3, the 2 s run t=0 and t=2 — a
        # strict subset, so every one of its reads should be a memo hit.
        _write_run(vdir, "20260101-000001", video_hash="hseq", setup_hash="sseq",
                   config={"frameStep": 10, "frameIntervalMs": 100}, labels=labels,
                   det_rate=1.0, written_at="2026-01-01T00:00:00")
        _write_run(vdir, "20260101-000002", video_hash="hseq", setup_hash="sseq",
                   config={"frameStep": 20, "frameIntervalMs": 100}, labels=labels,
                   det_rate=1.0, written_at="2026-01-01T00:01:00")
        # The (gitignored, here empty) binary only needs to exist for the decode
        # path to engage; the fake reader never opens it.
        (vdir / "vidSEQ.mp4").write_bytes(b"")

        records = discover_runs(root)
        assert len(records) == 2

        reader = LogReader()
        frame_df = build_frame_table(records, decode=True, frame_reader=reader)
        assert len(frame_df) == 4 + 2  # t=0,1,2,3 and t=0,2

        # Sequential access, one read per (video, timestamp) — the second run's
        # samples are served from the crop-stat memo, not re-read.
        assert reader.calls == [("vidSEQ.mp4", 0.0), ("vidSEQ.mp4", 1.0),
                                ("vidSEQ.mp4", 2.0), ("vidSEQ.mp4", 3.0)]

        # Both runs still carry decode-derived crop stats, memo hits included.
        assert frame_df["climber_luma_mean"].notna().all()
        assert frame_df["wall_luma_mean"].notna().all()


def test_sequential_reader_holds_one_video_open_at_a_time():
    """Regression: the sequential reader must not accumulate open captures.

    Reading forward means holding decoder state between calls, and the first
    implementation kept a capture per video and released them all at the end. On the
    real corpus that is 76 live FFmpeg decoders over 632 MB of media — enough memory
    pressure to hang the run, on a different video each time. Driven here through a
    stub cv2, so the bound is asserted with no video files and no codec."""

    from analysis_pipeline import frames as frames_mod

    class FakeCap:
        def __init__(self, path):
            self.path = path
            self.released = False

        def isOpened(self):
            return "broken" not in self.path

        def get(self, prop):
            return 30.0

        def grab(self):
            return True

        def read(self):
            return True, "frame"

        def set(self, prop, value):
            return True

        def release(self):
            self.released = True

    class FakeCv2:
        CAP_PROP_FPS = 5
        CAP_PROP_POS_MSEC = 0
        CAP_PROP_POS_FRAMES = 1

        def __init__(self):
            self.opened = []

        def VideoCapture(self, path):
            cap = FakeCap(path)
            self.opened.append(cap)
            return cap

        def cvtColor(self, frame, code):
            return frame

        COLOR_BGR2GRAY = 6

    fake = FakeCv2()
    original = frames_mod.cv2
    frames_mod.cv2 = fake
    try:
        reader = frames_mod.SequentialFrameReader()
        for video in ("a.mp4", "b.mp4", "c.mp4"):
            for t in (0.0, 1.0, 2.0):
                assert reader.read_gray(video, t) == "frame"
            # Never more than one live capture, however many videos have been read.
            assert reader.open_videos == 1

        assert len(fake.opened) == 3                      # one capture per video...
        assert [c.released for c in fake.opened] == [True, True, False]  # ...prior ones freed

        # A video that fails to open is remembered, not retried per frame, and does
        # not evict the video currently being read.
        assert reader.read_gray("broken.mp4", 0.0) is None
        assert reader.read_gray("broken.mp4", 1.0) is None
        assert len(fake.opened) == 4
        assert reader.open_videos == 1

        reader.close()
        assert reader.open_videos == 0
        assert all(c.released for c in fake.opened if c.isOpened())
    finally:
        frames_mod.cv2 = original


def test_stale_truth_detected_when_the_scaffold_moves_underneath_it():
    """Issue #101 follow-up: Ground Truth is authored *from* the scaffold, so when the
    scaffold is regenerated the truth on disk keeps describing the old one — and every
    frame the new scaffold poses that the old truth calls absent becomes a phantom
    absence.

    Nothing else catches this. ``setupHash`` tracks *calibration*, and a re-seed does not
    change the calibration, so the truth still pairs as current on both sides. The
    fixture reproduces exactly that: matching setupHashes, truth that looks accepted, and
    a scaffold that has moved on."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    def bundle(root, name, truth_present, scaffold_posed, total):
        vdir = root / "routeST" / name
        _write_bundle_meta(vdir, setup_hash="sh_stale")
        # Truth and setup agree on setupHash — the existing staleness test passes.
        (vdir / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS), "setupHash": "sh_stale",
            "groundTruthHash": f"st{name}"[:16].ljust(16, "0"),
            "frames": [_truth_frame(i, float(i), i < truth_present)
                       for i in range(total)]}), encoding="utf-8")
        _write_scaffold(vdir, samples=[float(i) for i in range(total)],
                        posed=[float(i) for i in range(scaffold_posed)], seed_found=True)
        return vdir

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        # Adrift: truth records almost nothing, the scaffold poses nearly everything.
        bundle(root, "vidDrift", truth_present=5, scaffold_posed=200, total=220)
        # Healthy: truth broadly matches what the scaffold poses.
        bundle(root, "vidFresh", truth_present=195, scaffold_posed=200, total=220)
        # Small shortfall from ordinary human editing must NOT trip it.
        bundle(root, "vidEdited", truth_present=95, scaffold_posed=100, total=120)

        drift = ev.scaffold_truth_drift(root / "routeST" / "vidDrift")
        assert drift["drifted"] is True
        assert (drift["truthPresent"], drift["scaffoldPosed"]) == (5, 200)
        assert drift["scaffoldSeedHash"] is None  # stub scaffold carries none
        assert ev.scaffold_truth_drift(root / "routeST" / "vidFresh")["drifted"] is False
        assert ev.scaffold_truth_drift(root / "routeST" / "vidEdited")["drifted"] is False

        # A bundle with no authored truth cannot drift from the scaffold — it *is*
        # scored against the scaffold.
        (root / "routeST" / "vidFresh" / "ground-truth.json").unlink()
        assert ev.scaffold_truth_drift(root / "routeST" / "vidFresh") is None

        rows = trends._stale_truth_worklist(root)
        assert [r["video_key"] for r in rows] == ["vidDrift"]
        assert rows[0]["shortfall"] == 195

        html = report._stale_truth_html({"stale_truth_bundles": rows})
        assert "vidDrift" in html and "phantom absences" in html
        assert "re-accept these in the scanner" in html.lower()
        # ...and an untroubled corpus says so rather than rendering an empty table.
        assert "No bundle's truth has fallen behind" in report._stale_truth_html({})


def _run_keyed(n: int, runs: int) -> dict[str, list]:
    """Run-identity columns spreading ``n`` frames evenly over ``runs`` runs."""

    idx = [i % runs for i in range(n)]
    return {
        "route_folder": [f"route{i}" for i in idx],
        "video_key": [f"vid{i}" for i in idx],
        "run_ts": [f"2026010{i}-000000" for i in idx],
    }


def test_frame_quality_condition_bands_flagged_rate():
    """Issue #44 deliverable 3: the flagged-frame rate is banded against a Video Stats
    condition via the same qcut + bootstrap machinery as the geometric trends."""

    from analysis_pipeline import trends

    # 30 rows over 3 runs: the lowest-luma tercile is all flagged, the rest clean.
    df = pd.DataFrame({
        **_run_keyed(30, 3),
        "vs_wall_luma_mean": [float(i) for i in range(30)],
        "flagged": [1 if i < 10 else 0 for i in range(30)],
    })
    bands = trends._frame_quality_condition_bands(df, bins=3)
    assert not bands.empty
    assert set(bands["condition"]) == {"wall_luma_mean"}
    assert sorted(bands["band"]) == [1, 2, 3]
    by_band = bands.set_index("band")["flagged_rate"].to_dict()
    assert by_band[1] == 1.0            # darkest tercile: every frame flagged
    assert by_band[2] == 0.0 and by_band[3] == 0.0

    assert trends._frame_quality_condition_bands(pd.DataFrame()).empty


def _approx(a: object, b: object, tol: float = 1e-9) -> bool:
    """Float equality for the share/rate assertions. This suite runs without pytest, so
    there is no ``pytest.approx`` to lean on."""

    return abs(float(a) - float(b)) <= tol  # type: ignore[arg-type]


def _funnel_run_row(route: str, attempts: int, missing: int, conforming: bool,
                    cause: str | None) -> dict:
    """One per-run funnel row: ``attempts`` attempts of which ``missing`` were missed."""

    return {
        "route_folder": route, "video_key": f"vid_{route}",
        "run_ts": "20260101-000000",
        "conforming": conforming, "nonconformance_cause": cause,
        "attempt_count": attempts,
        "attempt_status_accepted_count": attempts - missing,
        "attempt_status_accepted_rate": (attempts - missing) / attempts,
        "attempt_status_missing_count": missing,
        "attempt_status_missing_rate": missing / attempts,
    }


def test_conformance_is_a_covariate_on_failure_modes_not_a_filter():
    """Issue #132: the #15 gate must not filter the attempt funnel.

    The gate exists to keep a mis-fit truth out of *truth-fit* metrics. Applied to the
    funnel it selects on the failure being measured — most sharply for ``sparse-match``,
    which by definition is the detector supplying too little to fit. So the funnel pools
    every run and reports conformance as a dimension.

    The corpus here is deliberately the shape that broke cross-batch comparison: one clean
    conforming run and one collapsed ``sparse-match`` run holding a minority of the
    attempts and the overwhelming majority of the misses."""

    from analysis_pipeline import trends

    funnel = pd.DataFrame([
        _funnel_run_row("clean", attempts=800, missing=8, conforming=True, cause=None),
        _funnel_run_row("collapsed", attempts=200, missing=180, conforming=False,
                        cause="sparse-match"),
    ])
    out = trends._attempt_funnel_conformance(funnel)
    by_pop = out.set_index("population")

    # The pooled row is over ALL runs: 188/1000, not the conforming pool's 8/800.
    assert by_pop.loc["all", "attempts"] == 1000
    assert _approx(by_pop.loc["all", "missing_share"], 0.188)
    assert _approx(by_pop.loc["conforming", "missing_share"], 0.01)

    # ...and the gate's selectivity is visible rather than baked in: the quarantined run
    # is a fifth of the attempts and almost all of the misses. This pair of columns is the
    # whole point of the breakout.
    assert _approx(by_pop.loc["non-conforming", "share_of_attempts"], 0.2)
    assert _approx(by_pop.loc["non-conforming", "share_of_missing"], 180 / 188)

    # The cause rows partition the non-conforming pool, and every cause is named even at
    # zero so "no mis-track suspects this batch" is distinguishable from "never split".
    causes = out[out["kind"] == "cause"].set_index("population")
    assert set(causes.index) == set(trends.NONCONFORMANCE_CAUSES)
    assert _approx(causes.loc["sparse-match", "share_of_missing"], 180 / 188)
    assert causes.loc["suspected-mistrack", "runs"] == 0

    # Construction invariant: the `all` row is the same arithmetic as the section's
    # headline tiles, so the breakout can never contradict the number above it.
    totals = trends._attempt_funnel_totals(
        funnel, trends._attempt_funnel_status_table(funnel))
    assert totals["attempts"] == by_pop.loc["all", "attempts"]
    assert _approx(totals["status_shares"]["missing"], by_pop.loc["all", "missing_share"])

    # A pool with no conformance verdict at all yields the pooled row alone rather than a
    # fabricated split.
    bare = trends._attempt_funnel_conformance(funnel.drop(columns=["conforming"]))
    assert list(bare["population"]) == ["all"]
    assert trends._attempt_funnel_conformance(pd.DataFrame()).empty


def test_miss_cause_mix_breaks_out_by_conformance():
    """Issue #132: the miss-cause mix read off the conforming pool is not the corpus's
    mix, because the quarantined runs hold most of the misses. Both are reported."""

    from analysis_pipeline import trends

    rows = []
    # Conforming run: 10 misses, all identity-gated.
    rows += [{"route_folder": "r1", "video_key": "v1", "run_ts": "20260101-000000",
              "conforming": True, "nonconformance_cause": None,
              "miss_cause": "identity-gated"} for _ in range(10)]
    # Quarantined run: 90 misses, all no-candidates.
    rows += [{"route_folder": "r2", "video_key": "v2", "run_ts": "20260101-000000",
              "conforming": False, "nonconformance_cause": "sparse-match",
              "miss_cause": "no-candidates"} for _ in range(90)]
    crop_df = pd.DataFrame(rows)

    out = trends._miss_cause_conformance(crop_df).set_index("population")
    # Gated, the corpus would read 100% identity-gated. Pooled, it reads 90% no-candidates
    # — the number a scanner change would actually be judged on.
    assert _approx(out.loc["conforming", "identity_gated_share"], 1.0)
    assert _approx(out.loc["all", "identity_gated_share"], 0.1)
    assert _approx(out.loc["all", "no_candidates_share"], 0.9)
    assert _approx(out.loc["non-conforming", "share_of_misses"], 0.9)
    assert out.loc["all", "misses"] == 100
    assert out.loc["all", "runs"] == 2

    assert trends._miss_cause_conformance(pd.DataFrame()).empty
    # Nothing scored as a miss -> no table rather than a table of zeros.
    assert trends._miss_cause_conformance(
        crop_df.assign(miss_cause=None)).empty


def test_rejection_and_crop_breakouts_reuse_the_section_totals():
    """Issue #132: the rejection and crop breakouts are built by running the section's own
    totals function once per population, so the ``all`` row cannot drift from the headline
    tiles it sits under. Asserted rather than assumed — two code paths computing the same
    pooled number is exactly how the report has disagreed with itself before."""

    from analysis_pipeline import trends

    run_df = pd.DataFrame([
        {"route_folder": "r1", "video_key": "v1", "run_ts": "20260101-000000",
         "conforming": True, "nonconformance_cause": None,
         "rejected_attempts": 20, "good_pose_rejected": 8, "bad_pose_rejected": 12,
         "rejection_truth_absent": 2, "rejection_truth_unknown": 0,
         "over_rejection_rate": 0.4},
        {"route_folder": "r2", "video_key": "v2", "run_ts": "20260101-000000",
         "conforming": False, "nonconformance_cause": "suspected-mistrack",
         "rejected_attempts": 10, "good_pose_rejected": 1, "bad_pose_rejected": 9,
         "rejection_truth_absent": 4, "rejection_truth_unknown": 0,
         "over_rejection_rate": 0.1},
    ])
    rej = trends._rejection_conformance(run_df).set_index("population")
    headline = trends._rejection_totals(run_df)
    assert _approx(rej.loc["all", "over_rejection_rate"], headline["over_rejection_rate"])
    assert _approx(rej.loc["all", "over_rejection_rate"], 9 / 30)
    # The gate would have reported 0.4 — and it inverts the ranking of the two runs.
    assert _approx(rej.loc["conforming", "over_rejection_rate"], 0.4)
    assert _approx(rej.loc["non-conforming", "over_rejection_rate"], 0.1)

    crop_df = pd.DataFrame([
        {"route_folder": "r1", "video_key": "v1", "run_ts": "20260101-000000",
         "conforming": True, "nonconformance_cause": None, "miss_cause": None,
         "crop_contained_truth": True, "initial_crop_containment": 1.0,
         "initial_search_region_iou": 0.6},
        {"route_folder": "r2", "video_key": "v2", "run_ts": "20260101-000000",
         "conforming": False, "nonconformance_cause": "sparse-match",
         "miss_cause": "no-candidates",
         "crop_contained_truth": False, "initial_crop_containment": 0.0,
         "initial_search_region_iou": 0.0},
    ])
    crop = trends._crop_conformance(crop_df).set_index("population")
    crop_headline = trends._crop_totals(crop_df)
    assert _approx(crop.loc["all", "crop_missed_truth_rate"],
                   crop_headline["crop_missed_truth_rate"])
    assert _approx(crop.loc["all", "crop_missed_truth_rate"], 0.5)
    # Gated, the crop looks flawless; pooled, half the scored crops excluded the Climber.
    assert _approx(crop.loc["conforming", "crop_missed_truth_rate"], 0.0)
    assert _approx(crop.loc["non-conforming", "crop_missed_truth_rate"], 1.0)

    assert trends._rejection_conformance(pd.DataFrame()).empty
    assert trends._crop_conformance(pd.DataFrame()).empty


def test_condition_band_cis_are_computed_at_the_run_unit():
    """Issue #70: frames within a run are pseudo-replicated, so a band's CI resamples
    runs, not frames — the pooled rate is unchanged but the interval widens to match the
    handful of runs it actually rests on, and the per-run dispersion travels with it."""

    from analysis_pipeline import trends

    # 120 frames from 4 runs. Two runs fail on every frame, two on none: the pooled rate
    # is 0.5, but the evidence is 4 runs, so the interval must span most of [0, 1].
    n_runs, per_run = 4, 30
    df = pd.DataFrame({
        **_run_keyed(n_runs * per_run, n_runs),
        "tier": ["agreement"] * (n_runs * per_run),
        "size_frac": [0.5] * (n_runs * per_run),
        # _run_keyed cycles run index i % 4, so runs 0 and 1 fail, runs 2 and 3 do not.
        "failure": [1 if (i % n_runs) < 2 else 0 for i in range(n_runs * per_run)],
    })
    stats = trends._run_unit_rate(df, "failure")
    assert stats == {
        "n": 120, "n_runs": 4, "rate": 0.5,
        "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
        "run_rate_median": 0.5, "run_rate_p90": stats["run_rate_p90"],
    }
    # Every run is all-or-nothing, so a 4-run resample can land on 0.0 or 1.0.
    assert stats["ci_low"] == 0.0 and stats["ci_high"] == 1.0
    assert stats["run_rate_p90"] == 1.0

    # The frame-pooled bootstrap on the same rows would claim ~±0.09 — that gap is the
    # whole point of #70.
    frame_pooled = trends._bootstrap_rate(df["failure"].tolist())
    assert frame_pooled[0] == 0.5
    assert (frame_pooled[2] - frame_pooled[1]) < 0.3

    # And the band builders carry the run-unit columns through.
    bands = trends._condition_bands(
        df.assign(size_frac=[float(i) for i in range(len(df))]), "size_frac", bins=3)
    assert not bands.empty
    assert {"n_runs", "run_rate_median", "run_rate_p90"} <= set(bands.columns)
    assert (bands["ci_low"] <= bands["failure_rate"]).all()
    assert (bands["failure_rate"] <= bands["ci_high"]).all()

    # No run identity -> no band, rather than a frame-pooled CI wearing a run-unit label.
    assert trends._run_unit_rate(df.drop(columns=list(trends._RUN_KEY_COLS)),
                                "failure") is None


def test_evaluate_vitpose_fallback_when_no_ground_truth():
    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeG" / "vidG"
        _write_bundle_meta(vdir, setup_hash="sh_vit")
        vitpose = {"version": 1, "frames": [
            {"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
            {"timestamp": 2.0, "keypoints": []},  # seeded absent
        ]}
        (vdir / "vitpose.json").write_text(json.dumps(vitpose), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000010", "sh_vit",
                        [{"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
                         {"timestamp": 2.0, "keypoints": _kp_list(_TRUTH_JOINTS)}])

        summary = _evaluate(root)
        assert len(summary.written) == 1
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))
        assert rec["truthSource"] == "vitpose"
        assert rec["setupHash"] == "sh_vit"
        # vitpose hash is content-derived (no groundTruthHash), 64-hex sha256.
        assert len(rec["truthHash"]) == 64
        assert rec["counts"]["truthFramesAbsent"] == 1  # the empty-keypoints frame
        agr = rec["agreement"]
        assert agr["frames"]["matchedPresent"] == 1
        # the scanner posed the seeded-absent frame -> presence false positive.
        assert agr["presence"]["absentDetected"] == 1
        # perfect match on the one scored frame.
        assert agr["perJoint"]["nose"]["pck"] == {"correct": 1, "total": 1,
                                                  "value": 1.0}
        assert agr["perJoint"]["nose"]["normDist"] == {"n": 1, "median": 0.0,
                                                       "p90": 0.0}
        # vitpose truth is a machine seed: never accuracy-tier evidence.
        assert rec["accuracy"]["frames"]["truthFrames"] == 0


def test_evaluate_review_provenance_routing():
    """Issue #11 / ADR 0005: both flagged-wrong seeds and deprecated manual absent
    flags are excluded from every tier and surface only in skip accounting; a frame
    whose review field is missing degrades to auto."""

    from analysis_pipeline import evaluate as ev

    # Frames at 1s spacing: t1 auto (present, exact), t2 human-flagged-wrong (present
    # but its seed joints are deliberately off — must never be scored), t3 legacy
    # (no review field -> auto), t4 human-flagged-absent (deprecated, excluded).
    bad = {n: (x + 5.0, y + 5.0) for n, (x, y) in _TRUTH_JOINTS.items()}
    doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "beef1234beef5678",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present", "review": "auto",
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in _TRUTH_JOINTS.items()}},
            {"frameIndex": 2, "timestamp": 2.0, "state": "present",
             "review": "human-flagged-wrong",
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in bad.items()}},
            {"frameIndex": 3, "timestamp": 3.0, "state": "present",  # no review field
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in _TRUTH_JOINTS.items()}},
            {"frameIndex": 4, "timestamp": 4.0, "state": "absent",
             "review": "human-flagged-absent", "joints": {}},
        ],
    }
    scanner = [
        {"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},  # exact on t1
        {"timestamp": 2.0, "keypoints": _kp_list(_TRUTH_JOINTS)},  # correct, but t2 seed is bad
        {"timestamp": 3.0, "keypoints": _kp_list(_TRUTH_JOINTS)},  # exact on t3
        {"timestamp": 4.0, "keypoints": _kp_list(_TRUTH_JOINTS)},  # hallucination on absent t4
    ]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeR" / "vidR"
        _write_bundle_meta(vdir, setup_hash="sh_r")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000020", "sh_r", scanner)

        summary = _evaluate(root)
        assert len(summary.written) == 1
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))

        # Per-category counts and skip accounting: both flag classes are skipped.
        assert rec["counts"]["review"] == {"auto": 2, "flaggedWrong": 1,
                                            "flaggedAbsent": 1}
        assert rec["counts"]["agreementSkipped"] == {"flaggedWrong": 1,
                                                     "flaggedAbsent": 1,
                                                     "outOfScope": 0}
        assert rec["counts"]["truthFramesVerified"] == 0

        # Agreement excludes both the flagged-wrong seed and the manual absent flag:
        # only t1 and t3 (auto) are scoreable present frames, and the bad t2 joints
        # never enter PCK — a perfect 2/2 despite the scanner "matching" the seed.
        agr = rec["agreement"]
        assert agr["frames"]["scoreable"] == 2
        assert agr["aggregate"]["pck"]["value"] == 1.0
        # t4's manual-absent flag is excluded, so its hallucination is NOT scored.
        assert agr["presence"]["absentDetected"] == 0

        # Accuracy tier is empty: no trustworthy human attestation exists (ADR 0005).
        acc = rec["accuracy"]
        assert acc["frames"]["truthFrames"] == 0
        assert acc["presence"] == {"presentDetected": 0, "presentUndetected": 0,
                                   "absentDetected": 0, "absentUndetected": 0}
        assert acc["aggregate"]["pck"]["value"] is None


def test_evaluate_legacy_ground_truth_without_review_all_auto():
    """Issue #11: a ground-truth file with no review field on any frame degrades to
    all-auto — agreement-tier evidence, empty accuracy tier, nothing skipped."""

    from analysis_pipeline import evaluate as ev

    doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS),
        "groundTruthHash": "1eac1eac1eac1eac",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present",
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in _TRUTH_JOINTS.items()}},
            {"frameIndex": 2, "timestamp": 2.0, "state": "absent", "joints": {}},
        ],
    }
    scanner = [{"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
               {"timestamp": 2.0, "keypoints": []}]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeL" / "vidL"
        _write_bundle_meta(vdir, setup_hash="sh_l")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000021", "sh_l", scanner)

        rec = json.loads(_evaluate(root).written[0].record_path
                         .read_text(encoding="utf-8"))
        assert rec["counts"]["review"] == {"auto": 2, "flaggedWrong": 0,
                                           "flaggedAbsent": 0}
        assert rec["counts"]["truthFramesVerified"] == 0
        assert rec["counts"]["agreementSkipped"] == {"flaggedWrong": 0,
                                                    "flaggedAbsent": 0,
                                                    "outOfScope": 0}
        assert rec["agreement"]["frames"]["truthFrames"] == 2
        assert rec["accuracy"]["frames"]["truthFrames"] == 0


def test_analysis_report_includes_eval_trend_sections():
    from analysis_pipeline import cli
    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        out = Path(tmp) / "reports"

        # One scored bundle -> evaluation record exists.
        v_ok = root / "routeT" / "vidT"
        _write_bundle_meta(v_ok, setup_hash="sh_ok")
        (v_ok / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(v_ok, "20260101-010101", "sh_ok", _scanner_frames_for_pck())

        # One truthless bundle -> appears in shame list.
        v_no_truth = root / "routeT" / "vidNoTruth"
        _write_bundle_meta(v_no_truth, setup_hash="sh_nt")

        # One stale setup run that still overlaps truth -> appears in the stale shame
        # list AND is recovered by the #44 best-overlap fallback (loose-paired).
        v_stale = root / "routeT" / "vidStale"
        _write_bundle_meta(v_stale, setup_hash="sh_truth")
        (v_stale / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(v_stale, "20260101-020202", "sh_old", _scanner_frames_for_pck())

        # Seed committed evaluation records once, then run analysis. vidT is a trusted
        # record; vidStale is loose-paired (setupHash mismatch, but overlaps truth).
        summary = _evaluate(root)
        assert len(summary.written) == 2
        assert len(summary.loose) == 1

        outputs = cli.run(root, out, decode=False)
        html_text = outputs["html"].read_text(encoding="utf-8")
        for header in (
            "Low-confidence truth (visible-joint measurement)",
            "Per-frame detection quality (auto-flagged classes)",
            "Detector Attempt funnel (run unit)",
            "Scanner version regression (build identity run-over-run)",
            "Build-identity conflicts",
            "Per-joint failure ranking (frame/joint unit)",
            "Within-video frame-level conditions vs error",
            "Cross-video descriptive splits",
            "Shame lists",
            "Superseded records (#89 evidence-generation dedup)",
            "Loose-paired bundles (#44 best-overlap fallback)",
            # Issue #132: the gate's two roles are each stated where they apply.
            "How the #15 conformance gate is applied",
            "Conformance breakout (covariate, not a filter)",
            "Miss-cause mix by conformance",
            "Rejection correctness by conformance",
            "Crop placement by conformance",
        ):
            assert header in html_text, f"missing report section: {header}"

        # Every truth-fit section says out loud that it quarantines (#132) — four of them:
        # version regression, joint ranking, condition bands, cross-video splits.
        assert html_text.count("Truth-fit metric — quarantined pool") == 4

        # Issue #133: the same four, plus the trend-accounting header, each name the
        # accuracy tier's missing input. A tier-bearing section read on its own must not
        # leave "why is everything tagged agreement?" to inference.
        assert html_text.count("accuracy: NOT COMPUTABLE") == 5
        assert "missing input" in html_text

        assert "routeT/vidNoTruth" in html_text
        assert "routeT/vidStale" in html_text  # stale shame list + loose table
        assert (out / "eval_joint_ranking.csv").exists()
        assert (out / "eval_low_confidence_worklist.csv").exists()


def test_empty_accuracy_tier_names_its_missing_input():
    """Issue #133: an unmeasured accuracy tier must never read as a measured-and-poor one.

    Nothing renders an *empty* accuracy row — the tier produces none at all — so without
    this the reader sees only agreement rows and infers a detection problem. ADR 0010
    made the emptiness permanent and structural, so the note states the cause rather
    than leaving it to be re-derived.

    The verified branch must still work: the note is a report of the corpus, not an
    assertion about it, so if attested frames ever appear it says so.
    """

    from analysis_pipeline import report

    empty = report._accuracy_tier_html({"verified_frames_total": 0, "verified_records": 0})
    assert "NOT COMPUTABLE" in empty
    # It must name the *input*, not just the absence — that distinction is the issue.
    assert "0 verified truth frames" in empty
    assert "missing input" in empty
    assert "ADR 0010" in empty
    # And it must say what the surviving numbers actually are.
    assert "agreement" in empty and "scaffold" in empty

    # Absent/None ctx keys degrade to the empty branch rather than raising.
    assert "NOT COMPUTABLE" in report._accuracy_tier_html({})
    assert "NOT COMPUTABLE" in report._accuracy_tier_html({"verified_frames_total": None})

    # Fails open: verified truth is reported, never overridden by the ADR 0010 wording.
    scored = report._accuracy_tier_html(
        {"verified_frames_total": 12, "verified_records": 3})
    assert "NOT COMPUTABLE" not in scored
    assert "12 verified frame(s)" in scored and "3 record(s)" in scored


def test_low_confidence_visible_measurement_and_worklist():
    """Occluded truth joints (low ViTPose confidence) shrink a frame's visible-joint
    count. v1 measures the distribution and lists the thinnest frames worst-first,
    but excludes nothing from scoring."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    # t1: all 13 joints confident. t2: both wrists occluded -> 11 visible (they are
    # dropped from the scored joints but the frame is still scoreable). Both matched.
    occ = {"left_wrist", "right_wrist"}
    doc = {
        "version": 1, "jointSet": list(_TRUTH_JOINTS), "groundTruthHash": "c0ffee00c0ffee00",
        "frames": [
            {"frameIndex": 1, "timestamp": 1.0, "state": "present", "review": "auto",
             "joints": {n: {"x": x, "y": y, "occluded": False}
                        for n, (x, y) in _TRUTH_JOINTS.items()}},
            {"frameIndex": 2, "timestamp": 2.0, "state": "present", "review": "auto",
             "joints": {n: {"x": x, "y": y, "occluded": n in occ}
                        for n, (x, y) in _TRUTH_JOINTS.items()}},
        ],
    }
    scanner = [{"timestamp": 1.0, "keypoints": _kp_list(_TRUTH_JOINTS)},
               {"timestamp": 2.0, "keypoints": _kp_list(_TRUTH_JOINTS)}]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeLC" / "vidLC"
        _write_bundle_meta(vdir, setup_hash="sh_lc")
        (vdir / "ground-truth.json").write_text(json.dumps(doc), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000030", "sh_lc", scanner)

        summary = _evaluate(root)
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))
        vj = rec["agreement"]["visibleJoints"]
        assert vj[13] == 1 and vj[11] == 1
        assert sum(vj) == rec["agreement"]["frames"]["matchedPresent"] == 2
        # Occluded wrists were dropped from scoring but the frame was NOT excluded.
        assert rec["agreement"]["frames"]["lowVisibility"] == 0
        assert rec["agreement"]["frames"]["scoreable"] == 2

        ctx = trends.build_trend_context(root)
        hist = ctx["visible_histogram"]
        assert hist[13] == 1 and hist[11] == 1 and sum(hist) == 2

        wl = ctx["low_conf_worklist"]
        assert not wl.empty and len(wl) == 2
        # Worst-first: the 11-visible frame leads and names its occluded joints.
        top = wl.iloc[0]
        assert int(top["visible"]) == 11
        assert "left_wrist" in top["occluded_joints"] and "right_wrist" in top["occluded_joints"]


# --------------------------------------------------------------------------- #
# appVersion run-over-run regression tracking (issue #10)
# --------------------------------------------------------------------------- #

def test_version_regression_delta_isolated_to_injected_joint():
    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeV" / "vidV"
        _write_bundle_meta(vdir, setup_hash="sh_v")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        # v1 (aaa1111): nose off by 0.2 in y (norm 0.667 > 0.5 -> wrong) on every
        # scoreable frame; every other joint exact. v2 (bbb2222): all exact —
        # a known injected improvement on exactly one joint. Both versions
        # sample t=9.0 too so no truth frame is left unmatched (an unmatched
        # present frame counts as a miss in the frame/joint rows). The offset is
        # on y (not x) so this one-joint miss keeps the whole-bundle conformance
        # fit near-identity (issue #15) — it stays in the pooled corpus rather
        # than being quarantined as if the truth tracked the wrong subject.
        bad = dict(_TRUTH_JOINTS)
        bad["nose"] = (0.5, 0.4)
        frames_v1 = [{"timestamp": t, "keypoints": _kp_list(bad)}
                     for t in (1.0, 2.0, 9.0)]
        frames_v2 = [{"timestamp": t, "keypoints": _kp_list(_TRUTH_JOINTS)}
                     for t in (1.0, 2.0, 9.0)]
        _write_pose_run(vdir, "20260101-000001", "sh_v", frames_v1,
                        app_version="aaa1111")
        _write_pose_run(vdir, "20260102-000001", "sh_v", frames_v2,
                        app_version="bbb2222")

        summary = _evaluate(root)
        assert len(summary.written) == 2

        ctx = trends.build_trend_context(root)
        overview = ctx["version_overview"]
        # Ordered by first-seen run timestamp.
        assert list(overview["app_version"]) == ["aaa1111", "bbb2222"]
        assert list(overview["n_records"]) == [1, 1]

        deltas = ctx["version_deltas"]
        assert not deltas.empty
        assert set(deltas["tier"]) == {"agreement"}  # no verified truth frames
        assert (deltas["from_version"] == "aaa1111").all()
        assert (deltas["to_version"] == "bbb2222").all()

        by_joint = deltas.set_index("joint")
        nose = by_joint.loc["nose"]
        assert nose["pck_from"] == 0.0 and nose["pck_to"] == 1.0
        assert nose["pck_delta"] == 1.0
        # Degenerate p=0 vs p=1 -> every bootstrap draw is +1: CI excludes 0.
        assert nose["pck_ci_low"] == 1.0 and nose["pck_ci_high"] == 1.0
        assert abs(nose["med_delta"] - (-0.666667)) < 1e-4

        # The injected improvement shows on nose only; every other joint is flat.
        for joint in ev.COCO_CORE_JOINTS:
            if joint == "nose":
                continue
            row = by_joint.loc[joint]
            assert row["pck_delta"] == 0.0, joint
            assert row["med_delta"] == 0.0, joint

        # Pooled row reflects the three recovered nose observations (36/39 -> 39/39).
        pooled = by_joint.loc["(all joints)"]
        assert abs(pooled["pck_delta"] - 3 / 39) < 1e-9
        assert ctx["version_flags"] == []


def test_version_regression_never_deltas_across_truth_revisions():
    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeW" / "vidW"
        _write_bundle_meta(vdir, setup_hash="sh_w")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        # Current-truth run from the newer version evaluates normally; the older
        # version's run is setup-stale so evaluate skips it today.
        _write_pose_run(vdir, "20260102-000001", "sh_w", _scanner_frames_for_pck(),
                        app_version="bbb2222")
        _write_pose_run(vdir, "20260101-000001", "sh_OLD", _scanner_frames_for_pck(),
                        app_version="aaa1111")
        summary = _evaluate(root)
        assert len(summary.written) == 1

        # The older version's committed record was evaluated against a different
        # (since-revised) truth: same video, disjoint truthHash.
        old_rec = {"schemaVersion": 2, "routeFolder": "routeW", "videoKey": "vidW",
                   "runTs": "20260101-000001", "truthHash": "ffff0000ffff0000"}
        (vdir / "evaluations" / "20260101-000001_vs_ffff0000.json").write_text(
            json.dumps(old_rec), encoding="utf-8")

        ctx = trends.build_trend_context(root)
        assert list(ctx["version_overview"]["app_version"]) == ["aaa1111", "bbb2222"]
        # Never delta'd across truth revisions: no rows, and the pair is flagged.
        assert ctx["version_deltas"].empty
        assert any("mixed truth" in f for f in ctx["version_flags"])
        assert any("routeW/vidW" in f for f in ctx["version_flags"])


# --------------------------------------------------------------------------- #
# per-run build identity: detectorCodeHash (issue #130)
# --------------------------------------------------------------------------- #

def test_build_identity_conflict_detected_on_runs_no_record_scored():
    """One appVersion stamping two detectorCodeHash values is flagged — and is found
    on runs that no evaluation record scored, which is where the real corpus's only
    conflict lives. A detector reading scored records alone would fire on nothing."""

    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeH" / "vidH"
        _write_bundle_meta(vdir, setup_hash="sh_h")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        # One scored run so the bundle enters the pose cache at all...
        _write_pose_run(vdir, "20260101-000001", "sh_h", _scanner_frames_for_pck(),
                        app_version="aaa1111", detector_code_hash="1111aaaa1111")
        # ...then the hot-reload pair, deliberately setup-stale so evaluate skips them.
        # Same stamp, different code: the c305954 signature.
        _write_pose_run(vdir, "20260101-000002", "sh_STALE", _scanner_frames_for_pck(),
                        app_version="aaa1111", detector_code_hash="2222bbbb2222")
        _write_pose_run(vdir, "20260101-000003", "sh_STALE", _scanner_frames_for_pck(),
                        app_version="aaa1111", detector_code_hash="1111aaaa1111")
        summary = _evaluate(root)
        assert len(summary.written) == 1  # only the sh_h run scored

        ctx = trends.build_trend_context(root)
        conflicts = ctx["build_conflicts"]
        assert not conflicts.empty
        assert set(conflicts["app_version"]) == {"aaa1111"}
        assert set(conflicts["detector_code_hash"]) == {"1111aaaa1111", "2222bbbb2222"}
        # Run counts come from every pose run on disk, not the scored subset.
        assert conflicts.set_index("detector_code_hash").loc["1111aaaa1111", "n_runs"] == 2
        assert conflicts.set_index("detector_code_hash").loc["2222bbbb2222", "n_runs"] == 1
        assert any("2 distinct detectorCodeHash" in f for f in ctx["version_flags"])


def test_build_identity_absent_hash_is_never_a_conflict():
    """Fail-open: null and missing hashes are unknown provenance, not contradiction.
    A stamp covering one hashed run and one unhashed run must not be flagged."""

    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeN" / "vidN"
        _write_bundle_meta(vdir, setup_hash="sh_n")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        _write_pose_run(vdir, "20260101-000001", "sh_n", _scanner_frames_for_pck(),
                        app_version="aaa1111", detector_code_hash="1111aaaa1111")
        # Explicit null — the scanner's "derivation failed" value.
        _write_pose_run(vdir, "20260101-000002", "sh_STALE", _scanner_frames_for_pck(),
                        app_version="aaa1111", detector_code_hash="")
        # Key absent entirely — a record predating the field.
        _write_pose_run(vdir, "20260101-000003", "sh_STALE", _scanner_frames_for_pck(),
                        app_version="aaa1111")
        _evaluate(root)

        ctx = trends.build_trend_context(root)
        assert ctx["build_conflicts"].empty
        assert not any("detectorCodeHash" in f for f in ctx["version_flags"])


def test_version_regression_splits_one_appversion_by_detector_code_hash():
    """A stamp that hot-reloaded mid-batch splits into its real behavioural groups
    instead of averaging them — the grouping half of #130."""

    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeS" / "vidS"
        _write_bundle_meta(vdir, setup_hash="sh_s")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        bad = dict(_TRUTH_JOINTS)
        bad["nose"] = (0.5, 0.4)
        frames_v1 = [{"timestamp": t, "keypoints": _kp_list(bad)} for t in (1.0, 2.0, 9.0)]
        frames_v2 = [{"timestamp": t, "keypoints": _kp_list(_TRUTH_JOINTS)}
                     for t in (1.0, 2.0, 9.0)]
        # One appVersion, two detector builds — the stamp alone would pool these and
        # average a real behavioural delta into nothing.
        _write_pose_run(vdir, "20260101-000001", "sh_s", frames_v1,
                        app_version="aaa1111", detector_code_hash="1111aaaa1111")
        _write_pose_run(vdir, "20260102-000001", "sh_s", frames_v2,
                        app_version="aaa1111", detector_code_hash="2222bbbb2222")
        summary = _evaluate(root)
        assert len(summary.written) == 2

        ctx = trends.build_trend_context(root)
        overview = ctx["version_overview"]
        assert list(overview["app_version"]) == [
            "aaa1111·1111aaaa1111", "aaa1111·2222bbbb2222"]
        assert list(overview["n_records"]) == [1, 1]

        # The injected one-joint improvement survives as a delta between the two
        # builds; grouping by the stamp alone would have shown no transition at all.
        deltas = ctx["version_deltas"]
        assert not deltas.empty
        assert (deltas["from_version"] == "aaa1111·1111aaaa1111").all()
        assert (deltas["to_version"] == "aaa1111·2222bbbb2222").all()
        nose = deltas.set_index("joint").loc["nose"]
        assert nose["pck_from"] == 0.0 and nose["pck_to"] == 1.0


def test_version_regression_pools_two_commits_sharing_one_hash():
    """The benefit that runs the other way: different appVersion, same
    detectorCodeHash is a commit that did not touch detection, so its runs stay one
    behavioural group instead of reading as a version boundary."""

    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeP2" / "vidP2"
        _write_bundle_meta(vdir, setup_hash="sh_p2")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        frames = _scanner_frames_for_pck()
        _write_pose_run(vdir, "20260101-000001", "sh_p2", frames,
                        app_version="aaa1111", detector_code_hash="1111aaaa1111")
        _write_pose_run(vdir, "20260102-000001", "sh_p2", frames,
                        app_version="bbb2222", detector_code_hash="1111aaaa1111")
        summary = _evaluate(root)
        assert len(summary.written) == 2

        ctx = trends.build_trend_context(root)
        overview = ctx["version_overview"]
        # One row, not two: both commits pooled, and the label names both.
        assert len(overview) == 1
        assert overview.iloc[0]["app_version"] == "aaa1111+bbb2222·1111aaaa1111"
        assert overview.iloc[0]["n_records"] == 2
        # No transition to delta, because no behavioural boundary exists.
        assert ctx["version_deltas"].empty
        assert ctx["build_conflicts"].empty


def test_version_regression_unhashed_corpus_renders_exactly_as_before():
    """The 495-of-499 case. With no hash anywhere, labels stay bare appVersions and
    the section is byte-identical to its pre-#130 output."""

    from analysis_pipeline import trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeL" / "vidL"
        _write_bundle_meta(vdir, setup_hash="sh_l2")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")

        frames = _scanner_frames_for_pck()
        _write_pose_run(vdir, "20260101-000001", "sh_l2", frames, app_version="aaa1111")
        _write_pose_run(vdir, "20260102-000001", "sh_l2", frames, app_version="bbb2222")
        _evaluate(root)

        ctx = trends.build_trend_context(root)
        overview = ctx["version_overview"]
        assert list(overview["app_version"]) == ["aaa1111", "bbb2222"]
        assert list(overview["detector_code_hash"]) == ["", ""]
        assert ctx["build_conflicts"].empty
        assert ctx["version_flags"] == []


# --------------------------------------------------------------------------- #
# stale-run orphan pruning (issue #32)
# --------------------------------------------------------------------------- #

def test_evaluate_prune_removes_stale_run_orphan_keeps_history():
    """A record whose run is setupHash-skipped AND whose truthHash8 is no longer
    current is a stale-run orphan and is pruned; a superseded-truth record whose run
    still pairs is truth-revision history and is retained; the live record stays."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeP" / "vidP"
        _write_bundle_meta(vdir, setup_hash="sh_cur")
        # Current truth self-reports groundTruthHash "abcdef1234567890" -> hash8 abcdef12.
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        # A live run that pairs under the current setup.
        _write_pose_run(vdir, "20260101-000001", "sh_cur", _scanner_frames_for_pck())
        # A stale run whose setupHash no longer matches -> evaluate skips it this run.
        _write_pose_run(vdir, "20260101-000002", "sh_STALE", _scanner_frames_for_pck())

        # First pass writes the live record for run ...0001 vs abcdef12.
        summary0 = _evaluate(root)
        assert len(summary0.written) == 1
        eval_dir = vdir / "evaluations"
        live_name = "20260101-000001_vs_abcdef12.json"
        assert (eval_dir / live_name).exists()

        # Seed two extra records on disk:
        #  - an orphan for the stale run against an OLD truth hash (run no longer
        #    pairs, hash not current) -> must be pruned.
        orphan_name = "20260101-000002_vs_deadbeef.json"
        (eval_dir / orphan_name).write_text(json.dumps({"stale": True}), encoding="utf-8")
        #  - truth-revision history for the LIVE run against an old truth hash (run
        #    still pairs) -> must be retained.
        history_name = "20260101-000001_vs_99998888.json"
        (eval_dir / history_name).write_text(json.dumps({"old": True}), encoding="utf-8")

        # Dry run: reports the orphan, deletes nothing.
        dry = _evaluate(root, prune=False)
        assert len(dry.orphans) == 1
        assert dry.orphans[0].record_path.name == orphan_name
        assert not dry.orphans[0].removed
        assert not dry.pruned
        assert (eval_dir / orphan_name).exists()  # still there after dry run

        # Prune: deletes only the orphan; history and the live record survive.
        wet = _evaluate(root, prune=True)
        assert len(wet.pruned) == 1
        assert wet.pruned[0].record_path.name == orphan_name
        assert not (eval_dir / orphan_name).exists()
        assert (eval_dir / history_name).exists()
        assert (eval_dir / live_name).exists()


# --------------------------------------------------------------------------- #
# targeted evaluate modes: all vs un-analyzed (issue #57)
# --------------------------------------------------------------------------- #

def _stamp_sentinel(record_path: Path, key: str = "_sentinel") -> None:
    """Tamper an on-disk record so a later rewrite is detectable by the key vanishing."""
    doc = json.loads(record_path.read_text(encoding="utf-8"))
    doc[key] = True
    record_path.write_text(json.dumps(doc), encoding="utf-8")


def _has_sentinel(record_path: Path, key: str = "_sentinel") -> bool:
    return json.loads(record_path.read_text(encoding="utf-8")).get(key) is True


def test_evaluate_mode_all_default_is_full_sweep():
    """Default and explicit mode='all' are the same full sweep: every run is (re)scored,
    nothing is skipped, and a re-run overwrites in place (idempotent)."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeM" / "vidM"
        _write_bundle_meta(vdir, setup_hash="sh_match")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000001", "sh_match", _scanner_frames_for_pck())

        # Default == explicit 'all', and neither skips anything.
        s_default = _evaluate(root)
        assert len(s_default.written) == 1 and not s_default.analyzed_skipped
        rec_path = s_default.written[0].record_path

        # A full sweep always rewrites, even when the record already exists: stamp a
        # sentinel and confirm mode='all' clobbers it.
        _stamp_sentinel(rec_path)
        s_all = _evaluate(root, mode=ev.EVAL_MODE_ALL)
        assert len(s_all.written) == 1 and not s_all.analyzed_skipped
        assert not _has_sentinel(rec_path)  # rewritten in place

        # An unknown mode is rejected rather than silently treated as 'all'.
        try:
            _evaluate(root, mode="nope")
        except ValueError as e:
            assert "unknown evaluate mode" in str(e)
        else:
            raise AssertionError("expected ValueError for an unknown mode")


def test_evaluate_mode_unanalyzed_skips_analyzed_processes_new():
    """un-analyzed mode skips a bundle whose matched runs already carry current-truth
    records (leaving them untouched), but processes a bundle again once a new run — or a
    revised truth — makes it un-analyzed."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeU" / "vidU"
        _write_bundle_meta(vdir, setup_hash="sh_match")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000001", "sh_match", _scanner_frames_for_pck())

        # A fresh corpus has nothing analyzed yet, so un-analyzed processes it in full —
        # equivalent to a full sweep on the first pass.
        first = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert len(first.written) == 1 and not first.analyzed_skipped
        rec_path = first.written[0].record_path

        # Second un-analyzed pass: the matched run already has a current-truth record, so
        # the bundle is skipped and its record is left byte-for-byte untouched.
        _stamp_sentinel(rec_path)
        second = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert not second.written
        assert second.analyzed_skipped == ["routeU/vidU"]
        assert _has_sentinel(rec_path)  # never rewritten

        # A full sweep still rewrites the same bundle (clobbers the sentinel).
        third = _evaluate(root, mode=ev.EVAL_MODE_ALL)
        assert len(third.written) == 1 and not _has_sentinel(rec_path)

        # A NEW run makes the bundle un-analyzed again -> un-analyzed reprocesses it and
        # scores both runs.
        _write_pose_run(vdir, "20260101-000002", "sh_match", _scanner_frames_for_pck())
        fourth = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert not fourth.analyzed_skipped
        assert {p.run_ts for p in fourth.written} == {"20260101-000001", "20260101-000002"}

        # A truth revision (new groundTruthHash) also un-analyzes the bundle: the prior
        # records are keyed on the old hash, so the new hash has none.
        revised = _ground_truth_doc(setup_hash=None)
        revised["groundTruthHash"] = "0000face0000face"
        (vdir / "ground-truth.json").write_text(json.dumps(revised), encoding="utf-8")
        fifth = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert not fifth.analyzed_skipped and len(fifth.written) == 2


def test_evaluate_mode_unanalyzed_preserves_loose_fallback():
    """Issue #57 x #44: the best-overlap loose fallback fires identically under
    un-analyzed mode — a fresh loose-eligible bundle is recovered, and a re-run leaves the
    loose record untouched once the matched run is analyzed."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeUL" / "vidUL"
        _write_bundle_meta(vdir, setup_hash="sh_cur")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        # Matched run (sh_cur) samples a disjoint span -> n=0; stale run (sh_OLD) overlaps
        # truth and is the best-overlap loose candidate — exactly the #44 fixture.
        matched = [{"timestamp": 100.0 + i, "keypoints": _kp_list(_TRUTH_JOINTS)}
                   for i in range(3)]
        _write_pose_run(vdir, "20260101-000001", "sh_cur", matched)
        _write_pose_run(vdir, "20260101-000002", "sh_OLD", _scanner_frames_for_pck())

        # Driven entirely via un-analyzed from scratch: still writes the n=0 matched record
        # AND recovers the loose pairing.
        summary = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert len(summary.written) == 2 and len(summary.loose) == 1
        loose = summary.loose[0]
        assert loose.run_ts == "20260101-000002"

        # Re-run: the matched run now has a current-truth record, so the whole bundle is
        # skipped and the loose record is left in place, untouched.
        _stamp_sentinel(loose.record_path)
        again = _evaluate(root, mode=ev.EVAL_MODE_UNANALYZED)
        assert not again.written
        assert again.analyzed_skipped == ["routeUL/vidUL"]
        assert _has_sentinel(loose.record_path)


def test_evaluate_mode_unanalyzed_prune_interaction():
    """Issue #57: pruning is orthogonal to mode — an already-analyzed bundle is skipped for
    scoring, yet un-analyzed + --prune still removes a stale-run orphan while retaining the
    live record and truth-revision history, exactly as a full sweep would."""

    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeUP" / "vidUP"
        _write_bundle_meta(vdir, setup_hash="sh_cur")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        _write_pose_run(vdir, "20260101-000001", "sh_cur", _scanner_frames_for_pck())
        # A stale run whose setupHash no longer matches -> never analyzed, never counts
        # toward the un-analyzed gate.
        _write_pose_run(vdir, "20260101-000002", "sh_STALE", _scanner_frames_for_pck())

        # Analyze the live run once (full sweep) -> current-truth record on disk.
        first = _evaluate(root)
        assert len(first.written) == 1
        eval_dir = vdir / "evaluations"
        live_name = "20260101-000001_vs_abcdef12.json"
        assert (eval_dir / live_name).exists()

        # Seed a stale-run orphan (run no longer pairs, old truth hash) + truth-revision
        # history for the live run (still pairs, old truth hash).
        orphan_name = "20260101-000002_vs_deadbeef.json"
        (eval_dir / orphan_name).write_text(json.dumps({"stale": True}), encoding="utf-8")
        history_name = "20260101-000001_vs_99998888.json"
        (eval_dir / history_name).write_text(json.dumps({"old": True}), encoding="utf-8")

        result = _evaluate(root, prune=True, mode=ev.EVAL_MODE_UNANALYZED)
        # The bundle is skipped for scoring...
        assert not result.written
        assert result.analyzed_skipped == ["routeUP/vidUP"]
        # ...but pruning still fires: the orphan is removed, live + history survive.
        assert len(result.pruned) == 1
        assert result.pruned[0].record_path.name == orphan_name
        assert not (eval_dir / orphan_name).exists()
        assert (eval_dir / live_name).exists()
        assert (eval_dir / history_name).exists()


def _basis_corpus(root: Path, versions_by_run: dict[str, int | None] | None = None,
                  builds: dict[str, tuple[str, str | None]] | None = None) -> None:
    """A two-run corpus for the #131 basis tests, optionally with the on-disk
    ``schemaVersion`` rewritten per run to stage a mixed-basis pool.

    Rewriting after evaluation rather than faking a record wholesale keeps the fixture
    honest: every record is one the pipeline actually wrote, and only the basis stamp
    differs — which is exactly the real case (records written under an older schema, still
    on disk, still pooled)."""

    present = {n: {"x": x, "y": y, "occluded": False} for n, (x, y) in _TRUTH_JOINTS.items()}
    exact = _kp_list(_TRUTH_JOINTS)
    crop = {"x": 0.2, "y": 0.1, "w": 0.6, "h": 0.9}
    frames = [1, 2, 3]
    attempts = [
        {"timestamp": float(i), "status": "accepted", "initialSearchRegion": crop,
         "detectionRegion": crop, "rawKeypoints": exact, "acceptedKeypoints": exact,
         "candidateCount": 1, "selectionMethod": "tracked"}
        for i in frames
    ]
    builds = builds or {"20260729-100000": ("bbb2222", None),
                        "20260729-110000": ("bbb2222", None)}
    for i, (run_ts, (app_version, code_hash)) in enumerate(sorted(builds.items())):
        vid = root / "routeBASIS" / f"vid{i}"
        _write_bundle_meta(vid, setup_hash=f"sh_basis{i}")
        (vid / "ground-truth.json").write_text(json.dumps({
            "version": 1, "jointSet": list(_TRUTH_JOINTS),
            "groundTruthHash": f"basis{i:03d}0000{i:04d}",
            "frames": [{"frameIndex": f, "timestamp": float(f), "state": "present",
                        "review": "auto", "joints": present} for f in frames],
        }), encoding="utf-8")
        _write_pose_run(vid, run_ts, f"sh_basis{i}",
                        [{"timestamp": float(f), "keypoints": exact} for f in frames],
                        app_version=app_version, detector_attempts=attempts,
                        detector_code_hash=code_hash)

    _evaluate(root)

    if versions_by_run:
        for path in root.glob("*/*/evaluations/*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            run_ts = str(data.get("runTs") or "")
            if run_ts not in versions_by_run:
                continue
            version = versions_by_run[run_ts]
            if version is None:
                data.pop("schemaVersion", None)
            else:
                data["schemaVersion"] = version
            path.write_text(json.dumps(data), encoding="utf-8")


def test_measurement_basis_names_schema_and_build_set():
    """Issue #131: every pooled section states the schema version(s) and build
    identities behind it, and a clean single-basis pool says so without warning.

    The acceptance is that a reader cannot compare two sections resting on different
    bases without being told — so the basis has to be *in* the section, not inferable."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        # Two builds, one hashed and one not: the build set is genuinely plural, which is
        # the corpus's real condition and must be named rather than averaged away.
        _basis_corpus(root, builds={
            "20260729-100000": ("bbb2222", None),
            "20260729-110000": ("ccc3333", "feda64515c1b"),
        })
        ctx = trends.build_trend_context(root)

        for key in ("measurement_basis_trusted", "measurement_basis_frames",
                    "measurement_basis_funnel"):
            basis = ctx[key]
            assert basis["n_records"] == 2, key
            # Records the pipeline just wrote are on the frozen basis by construction.
            assert basis["schema_versions"] == [str(ev.BASELINE_CYCLE_SCHEMA)], key
            assert basis["schema_mixed"] is False, key
            assert basis["on_basis"] == 2 and basis["off_basis"] == 0, key
            assert basis["cycle_broken"] is False, key
            # The build set is established and plural, and names both halves of the
            # hashed identity (#130's _build_label).
            assert basis["build_set_known"] is True, key
            assert basis["n_builds"] == 2 and basis["build_mixed"] is True, key
            assert "ccc3333·feda64515c1b" in basis["build_label"], key
            assert "bbb2222" in basis["build_label"], key

        html = report._measurement_basis_html(ctx["measurement_basis_trusted"])
        assert f"schemaVersion {ev.BASELINE_CYCLE_SCHEMA}" in html
        assert "MIXED SCHEMA" not in html
        assert "class='sub'" in html  # clean basis renders calm, not as a warning
        assert "2 build identities" in html

        # ...and the top-of-report declaration names the frozen cycle.
        banner = report._basis_banner_html(ctx)
        assert f"v{ev.BASELINE_CYCLE_SCHEMA}" in banner and "BASIS BROKEN" not in banner


def test_mixed_schema_pool_is_loudly_flagged():
    """Issue #131 acceptance: a pooled population spanning schema versions is loudly
    flagged rather than silently averaged.

    Flagged, not refused — a corpus mid-migration legitimately spans bases, and refusing
    to report would destroy the accounting that shows what the mixture is. What must not
    happen is the blend being invisible, which is how the 88/12 miss split survived four
    baselines."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    older = ev.BASELINE_CYCLE_SCHEMA - 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _basis_corpus(root, versions_by_run={"20260729-100000": older})
        ctx = trends.build_trend_context(root)

        basis = ctx["measurement_basis_trusted"]
        assert basis["n_records"] == 2
        assert basis["schema_versions"] == [str(older), str(ev.BASELINE_CYCLE_SCHEMA)]
        assert basis["schema_mixed"] is True
        assert basis["on_basis"] == 1 and basis["off_basis"] == 1
        assert basis["schema_counts"] == {str(older): 1, str(ev.BASELINE_CYCLE_SCHEMA): 1}
        # The cycle itself is not broken — the *writer* still matches the freeze. A mixed
        # pool and a mid-cycle bump are different failures and must not be conflated.
        assert basis["cycle_broken"] is False

        html = report._measurement_basis_html(basis)
        assert "MIXED SCHEMA" in html
        assert "class='warn'" in html          # cannot be skimmed past
        assert "1 of 2 record(s) are" in html   # names how much is off-basis
        assert "--mode all" in html             # ...and what to do about it

        # The mixture is stated on every pooled section, not just the first.
        for key in ("measurement_basis_frames", "measurement_basis_funnel"):
            assert ctx[key]["schema_mixed"] is True, key
            assert "MIXED SCHEMA" in report._measurement_basis_html(ctx[key]), key


def test_mid_cycle_schema_bump_demands_a_full_rescore():
    """Issue #131: bumping the writer's schema without moving the cycle freeze is
    permitted but never silent — it demands a re-score of the *whole* compared
    population, because scoring only the new batch leaves it straddling two bases."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _basis_corpus(root)
        # Simulate the bump at the writer while the cycle freeze stays put. Patched on
        # trends, which binds the value at import.
        original = trends.SCHEMA_VERSION
        trends.SCHEMA_VERSION = ev.BASELINE_CYCLE_SCHEMA + 1
        try:
            ctx = trends.build_trend_context(root)
        finally:
            trends.SCHEMA_VERSION = original

        basis = ctx["measurement_basis_trusted"]
        assert basis["cycle_broken"] is True
        assert basis["writer_schema"] == ev.BASELINE_CYCLE_SCHEMA + 1
        assert basis["frozen_schema"] == ev.BASELINE_CYCLE_SCHEMA

        html = report._measurement_basis_html(basis)
        assert "MID-CYCLE SCHEMA BUMP" in html
        assert "evaluate --mode all" in html
        assert "class='warn'" in html

        banner = report._basis_banner_html(ctx)
        assert "BASIS BROKEN" in banner and "--mode all" in banner

        # Unpatched, the same corpus is clean — the flag tracks the constants, not the data.
        assert trends.build_trend_context(root)["measurement_basis_trusted"]["cycle_broken"] is False


def test_unstamped_record_reads_as_unknown_basis_never_as_frozen():
    """Issue #131: a record that does not say what basis it was written on reads as
    *unknown*, and an unknown basis is itself a mixture when pooled with a stamped one.

    Collapsing unknown into the frozen version would let the exact contamination this
    slice exists to surface read as clean."""

    from analysis_pipeline import evaluate as ev
    from analysis_pipeline import report, trends

    # The reader normalizes anything unparseable to None rather than guessing.
    assert ev.record_schema_version({"schemaVersion": 14}) == 14
    assert ev.record_schema_version({"schemaVersion": "14"}) == 14
    assert ev.record_schema_version({}) is None
    assert ev.record_schema_version({"schemaVersion": None}) is None
    assert ev.record_schema_version({"schemaVersion": "v14"}) is None
    assert ev.record_schema_version({"schemaVersion": True}) is None
    assert ev.record_schema_version({"schemaVersion": {"n": 14}}) is None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        _basis_corpus(root, versions_by_run={"20260729-100000": None})
        basis = trends.build_trend_context(root)["measurement_basis_trusted"]

        assert basis["schema_mixed"] is True
        # unknown sorts last: it is not a version and must not sort among them.
        assert basis["schema_versions"] == [str(ev.BASELINE_CYCLE_SCHEMA), "unknown"]
        assert basis["schema_counts"]["unknown"] == 1
        assert basis["on_basis"] == 1 and basis["off_basis"] == 1
        assert "unstamped" in report._measurement_basis_html(basis)


def _run_all():
    """Every ``test_*`` in this module, discovered rather than listed.

    A hand-maintained list silently drops a test the moment someone forgets to add it:
    ``test_evaluate_detection_annotations_override_and_ignore_stale`` was absent from it
    and had been failing unnoticed. Discovery makes that impossible.
    """

    fns = [fn for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
    _legacy_order = [test_discovery_dedup_prune_and_stats, test_cliffs_delta_bounds,
           test_crossmatch_reducers, test_pipeline_end_to_end_renders_report,
           test_evaluate_pck_exact_and_edge_cases,
           test_evaluate_conformance_gate_and_pooled_quarantine,
           test_conformance_x_axis_has_looser_r2_floor,
           test_nonconformance_cause_splits_sparse_match_from_suspected_mistrack,
           test_evaluate_setuphash_mismatch_is_skipped,
           test_evaluate_loose_overlap_pairing_fallback,
           test_frame_quality_classification_one_per_class,
           test_frozen_stale_requires_sustained_run,
           test_frame_quality_splits_held_pose_from_raw_frozen_stale,
           test_evaluate_prefers_detector_attempts_over_dense_frames,
           test_rejection_correctness_verdicts_and_pooled_rate,
           test_crop_quality_iou_and_miss_causes,
           test_miss_reason_splits_the_residual_and_retro_derives,
           test_crop_export_selection_and_writes,
           test_frame_quality_aggregation_pools_all_records,
           test_hallucination_split_by_truth_presence,
           test_hallucination_split_reads_old_frames_as_unknown_and_unconfirmed,
           test_attempt_funnel_pools_and_distributes_over_runs,
           test_evidence_generation_dedup_prefers_attempt_backed_record,
           test_evidence_generation_dedup_is_scoped_to_one_truth_revision,
           test_frame_table_sequential_reader_and_memo,
           test_frame_quality_condition_bands_flagged_rate,
           test_condition_band_cis_are_computed_at_the_run_unit,
           test_evaluate_vitpose_fallback_when_no_ground_truth,
           test_evaluate_prune_removes_stale_run_orphan_keeps_history,
           test_evaluate_mode_all_default_is_full_sweep,
           test_evaluate_mode_unanalyzed_skips_analyzed_processes_new,
           test_evaluate_mode_unanalyzed_preserves_loose_fallback,
           test_evaluate_mode_unanalyzed_prune_interaction,
           test_analysis_report_includes_eval_trend_sections,
           test_low_confidence_visible_measurement_and_worklist,
           test_version_regression_delta_isolated_to_injected_joint,
           test_version_regression_never_deltas_across_truth_revisions,
           test_build_identity_conflict_detected_on_runs_no_record_scored,
           test_build_identity_absent_hash_is_never_a_conflict,
           test_version_regression_splits_one_appversion_by_detector_code_hash,
           test_version_regression_pools_two_commits_sharing_one_hash,
           test_version_regression_unhashed_corpus_renders_exactly_as_before]
    assert set(_legacy_order) <= set(fns), "a listed test vanished from the module"
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all smoke tests passed")


if __name__ == "__main__":
    _run_all()
