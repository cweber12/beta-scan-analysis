"""API-layer smoke tests for app.py — dependency-free (no httpx/TestClient).

The route handlers are plain functions over Pydantic models, so we exercise them
directly with the network/ffmpeg-touching core functions monkeypatched. This keeps
the test dependency footprint at zero (stdlib ``unittest.mock`` only) while still
covering schema validation, response shaping, error mapping, and the static-page
wiring introduced when the homepage moved to ``static/index.html``.

Runnable with pytest, or standalone: ``python -m tests.test_api``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

import numpy as np

import app as app_module
import video_stats
from app import (
    DetectionRequest,
    DownloadRequest,
    ImportRequest,
    VideoStatsRequest,
    compute_video_stats,
    get_contract,
    create_download_bundle,
    create_import_bundle,
    get_routes,
    homepage,
    list_route_folders,
    push_detections,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _expect_raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _stub_download_result(source_type: str) -> SimpleNamespace:
    video_path = Path("analysis/routeA/vid_20260101-000001/vid_20260101-000001.mp4")
    return SimpleNamespace(
        timestamp="20260101-000001",
        route_folder="routeA",
        source_type=source_type,
        video_path=video_path,
        video_key="vid_20260101-000001",
    )


def _fake_build_bundle(download_result, analysis_root, source_extras=None):
    # Echo source_extras into source_video so a test can assert they flow through
    # (e.g. requested_resolution on download). Labels are no longer handled here.
    video_dir = download_result.video_path.parent
    source_video = {"title": "The Mandala", "video_id": "abc123"}
    if source_extras:
        source_video.update(source_extras)
    return {
        "video_key": download_result.video_key,
        "video_dir": video_dir,
        "metadata_path": video_dir / "metadata.json",
        "frame_path": video_dir / "final_frame.png",
        "detections_dir": video_dir / "detections",
        "metadata": {"source_video": source_video},
    }


# --------------------------------------------------------------------------- #
# Static-page wiring
# --------------------------------------------------------------------------- #

def test_homepage_serves_static_file():
    response = homepage()
    assert isinstance(response, FileResponse)
    served = Path(response.path)
    assert served == app_module.STATIC_DIR / "index.html"
    assert served.is_file(), "static/index.html must exist on disk"


def test_static_index_content_and_css_unescape():
    html = (app_module.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "<title>Climb Video Analyzer</title>" in html
    # The Python-string escape (\\203A) must have been undone to a plain CSS escape.
    assert 'content: "\\203A "' in html
    assert "\\\\203A" not in html


# --------------------------------------------------------------------------- #
# Route-folder listing
# --------------------------------------------------------------------------- #

def test_get_routes_lists_directories():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "the-mandala").mkdir()
        (root / "midnight-lightning").mkdir()
        (root / "notes.txt").write_text("ignore me", encoding="utf-8")
        with patch.object(app_module, "ANALYSIS_DIR", root):
            assert list_route_folders() == ["midnight-lightning", "the-mandala"]
            assert get_routes() == {"routes": ["midnight-lightning", "the-mandala"]}


# --------------------------------------------------------------------------- #
# Download / import response shaping
# --------------------------------------------------------------------------- #

def test_create_download_bundle_shapes_response():
    payload = DownloadRequest(url="https://youtu.be/abc123", route_folder="routeA",
                              resolution=1080)
    with patch.object(app_module, "download_video",
                      lambda *a, **k: _stub_download_result("youtube")), \
         patch.object(app_module, "build_analysis_bundle", _fake_build_bundle):
        resp = create_download_bundle(payload)

    assert resp["video_key"] == "vid_20260101-000001"
    assert resp["source_type"] == "youtube"
    assert resp["source_title"] == "The Mandala"
    assert resp["source_video_id"] == "abc123"
    # Condition labels are no longer collected or returned by the harness.
    assert "analysis_inputs" not in resp


def test_create_import_bundle_shapes_response():
    payload = ImportRequest(local_path="downloads/Midnight_Lightning_V8.mp4",
                            route_folder="routeA")
    with patch.object(app_module, "import_local_video",
                      lambda *a, **k: _stub_download_result("local")), \
         patch.object(app_module, "build_analysis_bundle", _fake_build_bundle):
        resp = create_import_bundle(payload)

    assert resp["source_type"] == "local"
    assert resp["video_key"] == "vid_20260101-000001"
    assert "analysis_inputs" not in resp


def test_create_download_bundle_maps_core_error_to_400():
    payload = DownloadRequest(url="https://youtu.be/abc123", route_folder="routeA")

    def _boom(*a, **k):
        raise RuntimeError("download failed")

    with patch.object(app_module, "download_video", _boom):
        try:
            create_download_bundle(payload)
        except HTTPException as exc:
            assert exc.status_code == 400
            assert "download failed" in exc.detail
        else:
            raise AssertionError("expected HTTPException")


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #

def test_push_detections_rejects_shallow_path():
    payload = DetectionRequest(video_path="video.mp4", pose={}, orb={})
    try:
        push_detections(payload)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "route/video_key" in exc.detail
    else:
        raise AssertionError("expected HTTPException for a path without route/video_key")


def test_push_detections_derives_route_and_key():
    payload = DetectionRequest(
        video_path="analysis/routeA/vid_20260101-000001/vid_20260101-000001.mp4",
        pose={"detected": True},
        orb={"keypoints": 2000},
    )
    captured = {}

    def _fake_save(analysis_root, route_folder, video_key, pose, orb):
        captured.update(route_folder=route_folder, video_key=video_key)
        return {"route_folder": route_folder, "video_key": video_key, "ok": True}

    with patch.object(app_module, "save_detection_run", _fake_save):
        resp = push_detections(payload)

    assert captured == {"route_folder": "routeA", "video_key": "vid_20260101-000001"}
    assert resp["ok"] is True


# --------------------------------------------------------------------------- #
# Contract endpoint (cross-program drift check)
# --------------------------------------------------------------------------- #

def test_contract_advertises_endpoints_and_versions():
    import vitpose_job

    contract = get_contract()
    assert contract["apiVersion"] == app_module.API_VERSION
    # Endpoint list is derived from the live route table — the cross-program
    # surface the scanner depends on must all be advertised.
    for path in ("/api/contract", "/api/detections", "/api/vitpose",
                 "/api/video-stats"):
        assert path in contract["endpoints"], path
    assert all(p.startswith("/api/") for p in contract["endpoints"])
    assert contract["artifacts"] == {
        "vitpose": vitpose_job.ARTIFACT_VERSION,
        "videoStats": video_stats.VIDEO_STATS_VERSION,
    }


def test_contract_reports_suggestion_fit_state():
    # With the baked corpus fit, suggestions are advertised as available...
    contract = get_contract()
    assert contract["suggestions"]["available"] is True
    assert contract["suggestions"]["fitDate"]
    assert contract["suggestions"]["corpusSize"] > 0
    # ...and with thresholds unfit, the scanner is told not to prefill.
    with patch.object(video_stats, "SUGGESTION_THRESHOLDS", None):
        unfit = get_contract()
    assert unfit["suggestions"]["available"] is False
    assert unfit["suggestions"]["fitDate"] is None


# --------------------------------------------------------------------------- #
# Video Stats endpoint (issue #23)
# --------------------------------------------------------------------------- #

def _fake_samples(*_a, **_k):
    """Two flat synthetic frames — enough for the pure stats core to run on."""
    frame = np.full((60, 80, 3), 150, dtype=np.uint8)
    return [frame, frame], [0.0, 1.0]


def _make_stats_bundle(root, *, with_setup=True, with_video=True):
    bundle = root / "routeA" / "vidA"
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text(
        json.dumps({"route_folder": "routeA", "video_key": "vidA",
                    "source_video": {"filesize": 1_000_000, "width": 80, "height": 60,
                                     "fps": 30, "duration_seconds": 10}}),
        encoding="utf-8")
    if with_setup:
        (bundle / "setup.json").write_text(
            json.dumps({"wallCrop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
                        "climberCrop": {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
                        "climberPoint": {"x": 0.5, "y": 0.5, "t": 0.0},
                        "panning": False, "setupHash": "sh_setup"}),
            encoding="utf-8")
    if with_video:
        (bundle / "vidA.mp4").write_bytes(b"\x00" * 16)
    return bundle


def test_video_stats_missing_bundle_maps_404():
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(app_module, "ANALYSIS_DIR", Path(tmp)):
        payload = VideoStatsRequest(route_folder="nope", video_key="missing")
        try:
            compute_video_stats(payload)
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("expected 404 for a missing bundle")


def test_video_stats_requires_wall_crop():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_stats_bundle(root, with_setup=False)
        with patch.object(app_module, "ANALYSIS_DIR", root):
            payload = VideoStatsRequest(route_folder="routeA", video_key="vidA")
            try:
                compute_video_stats(payload)
            except HTTPException as exc:
                assert exc.status_code == 400
                assert "wall crop" in exc.detail
            else:
                raise AssertionError("expected 400 without a wall crop")


def test_video_stats_computes_writes_and_stamps_hash():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = _make_stats_bundle(root)
        with patch.object(app_module, "ANALYSIS_DIR", root), \
             patch.object(video_stats, "sample_video_frames", _fake_samples):
            # Crops omitted from the payload -> fall back to setup.json.
            resp = compute_video_stats(
                VideoStatsRequest(route_folder="routeA", video_key="vidA"))

        assert resp["setupHash"] == "sh_setup"
        assert resp["regionStats"]["wall"]["luma"]["mean"] > 0
        assert resp["regionStats"]["climberWall"] is not None
        assert isinstance(resp["suggestions"], dict)

        artifact = json.loads((bundle / "video-stats.json").read_text(encoding="utf-8"))
        assert artifact["setupHash"] == "sh_setup"
        assert artifact["source"] == "endpoint"
        assert artifact["regionStats"] == resp["regionStats"]

        # Phase-1 self-heal: metadata.json gained the video_stats block, with the
        # compression proxy derived from the recorded source facts.
        metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["video_stats"]["sampledFrames"] == 2
        assert metadata["video_stats"]["bitsPerPixel"] is not None


def test_video_stats_payload_overrides_and_camelcase():
    # camelCase field names (setup.json casing) must be accepted...
    payload = VideoStatsRequest.model_validate({
        "routeFolder": "routeA", "videoKey": "vidA",
        "wallCrop": {"x": 0, "y": 0, "w": 1, "h": 1},
        "setupHash": "sh_payload",
    })
    assert payload.route_folder == "routeA"
    assert payload.wall_crop.w == 1
    # ...and payload geometry/hash must win over setup.json's.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = _make_stats_bundle(root)
        with patch.object(app_module, "ANALYSIS_DIR", root), \
             patch.object(video_stats, "sample_video_frames", _fake_samples):
            resp = compute_video_stats(payload)
        assert resp["setupHash"] == "sh_payload"
        artifact = json.loads((bundle / "video-stats.json").read_text(encoding="utf-8"))
        assert artifact["setupHash"] == "sh_payload"


def test_video_stats_decode_failure_maps_500():
    def _boom(*_a, **_k):
        raise RuntimeError("could not open video")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_stats_bundle(root)
        with patch.object(app_module, "ANALYSIS_DIR", root), \
             patch.object(video_stats, "sample_video_frames", _boom):
            try:
                compute_video_stats(
                    VideoStatsRequest(route_folder="routeA", video_key="vidA"))
            except HTTPException as exc:
                assert exc.status_code == 500
                assert "could not open video" in exc.detail
            else:
                raise AssertionError("expected 500 on decode failure")


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #

def test_schema_validation_rejects_bad_payloads():
    _expect_raises(ValidationError, lambda: DownloadRequest(url="https://youtu.be/x"))
    _expect_raises(ValidationError,
                   lambda: DownloadRequest(url="https://youtu.be/x",
                                           route_folder="routeA", resolution=50))
    _expect_raises(ValidationError, lambda: ImportRequest(route_folder="routeA"))


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_all():
    fns = [
        test_homepage_serves_static_file,
        test_static_index_content_and_css_unescape,
        test_get_routes_lists_directories,
        test_create_download_bundle_shapes_response,
        test_create_import_bundle_shapes_response,
        test_create_download_bundle_maps_core_error_to_400,
        test_push_detections_rejects_shallow_path,
        test_push_detections_derives_route_and_key,
        test_contract_advertises_endpoints_and_versions,
        test_contract_reports_suggestion_fit_state,
        test_video_stats_missing_bundle_maps_404,
        test_video_stats_requires_wall_crop,
        test_video_stats_computes_writes_and_stamps_hash,
        test_video_stats_payload_overrides_and_camelcase,
        test_video_stats_decode_failure_maps_500,
        test_schema_validation_rejects_bad_payloads,
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all api smoke tests passed")


if __name__ == "__main__":
    _run_all()
