"""API-layer smoke tests for app.py — dependency-free (no httpx/TestClient).

The route handlers are plain functions over Pydantic models, so we exercise them
directly with the network/ffmpeg-touching core functions monkeypatched. This keeps
the test dependency footprint at zero (stdlib ``unittest.mock`` only) while still
covering schema validation, response shaping, error mapping, and the static-page
wiring introduced when the homepage moved to ``static/index.html``.

Runnable with pytest, or standalone: ``python -m tests.test_api``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError

import app as app_module
from app import (
    DetectionRequest,
    DownloadRequest,
    ImportRequest,
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


def _fake_build_bundle(download_result, analysis_root, user_metadata):
    # Echo user_metadata back through analysis_inputs so the test can assert that
    # _analysis_inputs (and the per-endpoint extras) flow through _bundle_response.
    video_dir = download_result.video_path.parent
    return {
        "video_key": download_result.video_key,
        "video_dir": video_dir,
        "metadata_path": video_dir / "metadata.json",
        "frame_path": video_dir / "final_frame.png",
        "detections_dir": video_dir / "detections",
        "metadata": {
            "source_video": {"title": "The Mandala", "video_id": "abc123"},
            "analysis_inputs": user_metadata,
        },
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
                              resolution=1080, shadows="high")
    with patch.object(app_module, "download_video",
                      lambda *a, **k: _stub_download_result("youtube")), \
         patch.object(app_module, "build_analysis_bundle", _fake_build_bundle):
        resp = create_download_bundle(payload)

    assert resp["video_key"] == "vid_20260101-000001"
    assert resp["source_type"] == "youtube"
    assert resp["source_title"] == "The Mandala"
    assert resp["source_video_id"] == "abc123"
    # _analysis_inputs projection + the download-only extra flow through.
    assert resp["analysis_inputs"]["shadows"] == "high"
    assert resp["analysis_inputs"]["requested_resolution"] == 1080


def test_create_import_bundle_shapes_response():
    payload = ImportRequest(local_path="downloads/Midnight_Lightning_V8.mp4",
                            route_folder="routeA")
    with patch.object(app_module, "import_local_video",
                      lambda *a, **k: _stub_download_result("local")), \
         patch.object(app_module, "build_analysis_bundle", _fake_build_bundle):
        resp = create_import_bundle(payload)

    assert resp["source_type"] == "local"
    assert resp["analysis_inputs"]["imported_from"] == \
        "downloads/Midnight_Lightning_V8.mp4"


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
        test_schema_validation_rejects_bad_payloads,
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all api smoke tests passed")


if __name__ == "__main__":
    _run_all()
