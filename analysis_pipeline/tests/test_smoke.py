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


def _write_pose_run(video_dir: Path, stem: str, setup_hash: str, frames: list,
                    app_version: str = "") -> None:
    det = video_dir / "detections"
    det.mkdir(parents=True, exist_ok=True)
    diagnostics = {"appVersion": app_version} if app_version else {}
    env = {"video_key": video_dir.name, "route_folder": video_dir.parent.name,
           "run_ts": stem, "type": "pose",
           "data": {"setupHash": setup_hash, "diagnostics": diagnostics,
                    "frames": frames}}
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
        assert rec["schemaVersion"] == ev.SCHEMA_VERSION == 6
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
        assert counts == {"truthFramesTotal": 5, "truthFramesPresent": 4,
                          "truthFramesAbsent": 1, "truthFramesVerified": 0,
                          "review": {"auto": 4, "flaggedWrong": 0, "flaggedAbsent": 1},
                          "agreementSkipped": {"flaggedWrong": 0, "flaggedAbsent": 1}}
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
            "lowVisibility": 0, "torsoUndefined": 1, "scoreable": 2}
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
            "lowVisibility": 0, "torsoUndefined": 0, "scoreable": 0}
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
        assert set(conf) == {"x", "y", "n", "conforms", "reasons", "thresholds"}
        assert conf["thresholds"] == {
            "slopeMin": ev.CONFORMANCE_SLOPE_MIN, "slopeMax": ev.CONFORMANCE_SLOPE_MAX,
            "r2Min": ev.CONFORMANCE_R2_MIN, "r2MinX": ev.CONFORMANCE_R2_MIN_X,
            "minPoints": ev.CONFORMANCE_MIN_POINTS}
        assert conf["n"] == 37  # matched-present truth joints with a scanner pred
        assert conf["conforms"] is True and conf["reasons"] == []
        assert conf["y"] == {"slope": 1.0, "intercept": 0.0, "r2": 1.0}
        assert ev.record_conforms(rec) is True

        # Idempotent filename: rerun overwrites, no second file.
        summary2 = ev.evaluate(root)
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

        summary = ev.evaluate(root)
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

        summary = ev.evaluate(root)
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

        summary = ev.evaluate(root)
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
    # t1/t2 exact (ok; t2 is a frozen duplicate of t1). t3 hallucination on an absent
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
        {"timestamp": 2.0, "keypoints": exact},   # identical -> frozen
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

        rec = json.loads(ev.evaluate(root).written[0].record_path.read_text(encoding="utf-8"))
        fq = rec["frameQuality"]
        assert fq["detectedFrames"] == 6
        assert fq["classCounts"] == {"ok": 2, "wrong-subject": 1, "hallucination-fp": 1,
                                     "flipped-rotated": 1, "distorted": 1}
        assert fq["flaggedCount"] == 4
        assert fq["frozenStaleCount"] == 1
        assert fq["thresholds"]["wrongSubjectCentroid"] == ev.FQ_WRONG_SUBJECT_CENTROID

        by_t = {e["t"]: e for e in fq["frames"]}
        assert by_t[1.0]["class"] == "ok" and by_t[1.0]["frozenStale"] is False
        assert by_t[2.0]["class"] == "ok" and by_t[2.0]["frozenStale"] is True
        assert by_t[3.0]["class"] == "hallucination-fp"
        assert by_t[4.0]["class"] == "wrong-subject"
        assert by_t[5.0]["class"] == "distorted"
        assert by_t[6.0]["class"] == "flipped-rotated"
        # Every entry carries a crop placeholder for the exporter (deliverable 2).
        assert all("crop" in e for e in fq["frames"])


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

        summary = ev.evaluate(root)
        assert len(summary.written) == 1
        rec = json.loads(summary.written[0].record_path.read_text(encoding="utf-8"))

        # Per-category counts and skip accounting: both flag classes are skipped.
        assert rec["counts"]["review"] == {"auto": 2, "flaggedWrong": 1,
                                            "flaggedAbsent": 1}
        assert rec["counts"]["agreementSkipped"] == {"flaggedWrong": 1,
                                                     "flaggedAbsent": 1}
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

        rec = json.loads(ev.evaluate(root).written[0].record_path
                         .read_text(encoding="utf-8"))
        assert rec["counts"]["review"] == {"auto": 2, "flaggedWrong": 0,
                                           "flaggedAbsent": 0}
        assert rec["counts"]["truthFramesVerified"] == 0
        assert rec["counts"]["agreementSkipped"] == {"flaggedWrong": 0,
                                                    "flaggedAbsent": 0}
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
        summary = ev.evaluate(root)
        assert len(summary.written) == 2
        assert len(summary.loose) == 1

        outputs = cli.run(root, out, decode=False)
        html_text = outputs["html"].read_text(encoding="utf-8")
        for header in (
            "Low-confidence truth (visible-joint measurement)",
            "Scanner version regression (appVersion run-over-run)",
            "Per-joint failure ranking (frame/joint unit)",
            "Within-video frame-level conditions vs error",
            "Cross-video descriptive splits",
            "Shame lists",
            "Loose-paired bundles (#44 best-overlap fallback)",
        ):
            assert header in html_text, f"missing report section: {header}"

        assert "routeT/vidNoTruth" in html_text
        assert "routeT/vidStale" in html_text  # stale shame list + loose table
        assert (out / "eval_joint_ranking.csv").exists()
        assert (out / "eval_low_confidence_worklist.csv").exists()


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

        summary = ev.evaluate(root)
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

        summary = ev.evaluate(root)
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
        summary = ev.evaluate(root)
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
        summary0 = ev.evaluate(root)
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
        dry = ev.evaluate(root, prune=False)
        assert len(dry.orphans) == 1
        assert dry.orphans[0].record_path.name == orphan_name
        assert not dry.orphans[0].removed
        assert not dry.pruned
        assert (eval_dir / orphan_name).exists()  # still there after dry run

        # Prune: deletes only the orphan; history and the live record survive.
        wet = ev.evaluate(root, prune=True)
        assert len(wet.pruned) == 1
        assert wet.pruned[0].record_path.name == orphan_name
        assert not (eval_dir / orphan_name).exists()
        assert (eval_dir / history_name).exists()
        assert (eval_dir / live_name).exists()


def _run_all():
    fns = [test_discovery_dedup_prune_and_stats, test_cliffs_delta_bounds,
           test_crossmatch_reducers, test_pipeline_end_to_end_renders_report,
           test_evaluate_pck_exact_and_edge_cases,
           test_evaluate_conformance_gate_and_pooled_quarantine,
           test_conformance_x_axis_has_looser_r2_floor,
           test_evaluate_setuphash_mismatch_is_skipped,
           test_evaluate_loose_overlap_pairing_fallback,
           test_frame_quality_classification_one_per_class,
           test_evaluate_vitpose_fallback_when_no_ground_truth,
           test_evaluate_prune_removes_stale_run_orphan_keeps_history,
           test_analysis_report_includes_eval_trend_sections,
           test_low_confidence_visible_measurement_and_worklist,
           test_version_regression_delta_isolated_to_injected_joint,
           test_version_regression_never_deltas_across_truth_revisions]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all smoke tests passed")


if __name__ == "__main__":
    _run_all()
