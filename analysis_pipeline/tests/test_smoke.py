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


def _write_pose_run(video_dir: Path, stem: str, setup_hash: str, frames: list) -> None:
    det = video_dir / "detections"
    det.mkdir(parents=True, exist_ok=True)
    env = {"video_key": video_dir.name, "route_folder": video_dir.parent.name,
           "run_ts": stem, "type": "pose",
           "data": {"setupHash": setup_hash, "diagnostics": {}, "frames": frames}}
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

        summary = ev.evaluate(root)
        assert len(summary.written) == 1 and not summary.skipped
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))

        # Record shape / provenance header.
        assert rec["schemaVersion"] == ev.SCHEMA_VERSION == 2
        assert rec["metrics"] == ["pck@0.5-torso", "normDistMedian", "normDistP90",
                                  "presence2x2", "jointCoverage"]
        assert rec["setupHash"] == "sh_match"
        assert rec["truthSource"] == "ground-truth"
        assert rec["truthHash"] == "abcdef1234567890"
        assert rec["truthSetupHashSource"] == "setup.json"
        assert rec["jointSet"] == ev.COCO_CORE_JOINTS

        counts = rec["counts"]
        assert counts == {"truthFramesTotal": 5, "truthFramesPresent": 4,
                          "truthFramesAbsent": 1, "truthFramesVerified": 0}
        assert rec["scannerFrameIntervalSec"] == 1.0
        assert rec["joinToleranceSec"] == 0.5

        agr = rec["agreement"]
        # Frame accounting: t1/t2/t4 matched present, t3 matched absent, t9 has no
        # scanner sample within tolerance (unobserved, not undetected).
        assert agr["frames"] == {
            "truthFrames": 5, "verifiedFrames": 0,
            "matchedPresent": 3, "matchedAbsent": 1,
            "unmatchedPresent": 1, "unmatchedAbsent": 0,
            "torsoUndefined": 1, "scoreable": 2}
        # Presence 2x2: the scanner hallucinated a full pose on the absent frame.
        assert agr["presence"] == {"presentDetected": 3, "presentUndetected": 0,
                                   "absentDetected": 1, "absentUndetected": 0}

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

        # Accuracy tier: no human-verified frames exist, so the block is present
        # with explicit zero counts and null metrics — represented, never dropped.
        acc = rec["accuracy"]
        assert acc["frames"] == {
            "truthFrames": 0, "verifiedFrames": 0,
            "matchedPresent": 0, "matchedAbsent": 0,
            "unmatchedPresent": 0, "unmatchedAbsent": 0,
            "torsoUndefined": 0, "scoreable": 0}
        assert acc["presence"] == {"presentDetected": 0, "presentUndetected": 0,
                                   "absentDetected": 0, "absentUndetected": 0}
        assert acc["perJoint"]["nose"] == {
            "pck": {"correct": 0, "total": 0, "value": None},
            "normDist": {"n": 0, "median": None, "p90": None},
            "coverage": {"emitted": 0, "frames": 0, "rate": None}}
        assert acc["aggregate"]["pck"]["value"] is None
        assert acc["aggregate"]["normDist"] == {"n": 0, "median": None, "p90": None}

        # Idempotent filename: rerun overwrites, no second file.
        summary2 = ev.evaluate(root)
        assert len(summary2.written) == 1
        assert summary2.written[0].record_path == summary.written[0].record_path
        assert len(list((vdir / "evaluations").glob("*.json"))) == 1


def test_evaluate_setuphash_mismatch_is_skipped():
    from analysis_pipeline import evaluate as ev

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "analysis"
        vdir = root / "routeF" / "vidF"
        _write_bundle_meta(vdir, setup_hash="sh_truth")
        (vdir / "ground-truth.json").write_text(
            json.dumps(_ground_truth_doc(setup_hash=None)), encoding="utf-8")
        # A stale run whose setupHash != the setup.json the truth was authored under.
        _write_pose_run(vdir, "20260101-000009", "sh_STALE", _scanner_frames_for_pck())

        summary = ev.evaluate(root)
        assert not summary.written
        assert len(summary.skipped) == 1
        assert "setupHash mismatch" in summary.skipped[0].reason
        assert not (vdir / "evaluations").exists()


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

        summary = ev.evaluate(root)
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


def _run_all():
    fns = [test_discovery_dedup_prune_and_stats, test_cliffs_delta_bounds,
           test_crossmatch_reducers, test_pipeline_end_to_end_renders_report,
           test_evaluate_pck_exact_and_edge_cases,
           test_evaluate_setuphash_mismatch_is_skipped,
           test_evaluate_vitpose_fallback_when_no_ground_truth]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all smoke tests passed")


if __name__ == "__main__":
    _run_all()
