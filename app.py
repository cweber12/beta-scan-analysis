from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import vitpose_job
from youtube_core import (
    build_analysis_bundle,
    download_video,
    generate_timestamp,
    import_local_video,
    save_detection_run,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis"
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Pre-load the ViTPose/YOLO models once at boot on a background thread, so the
    # first calibration is fast and startup isn't blocked. Best-effort: a missing
    # ML dependency just means the first real request pays the load cost instead.
    def _warm() -> None:
        try:
            vitpose_job.warm_backends()
        except Exception:  # noqa: BLE001 — pre-warm is best-effort
            pass

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Climb Video Analyzer", lifespan=_lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class AnalysisMetadata(BaseModel):
    # route_folder is the one manual field the harness still owns: it's structural
    # (it files the bundle at analysis/<route_folder>/<video_key>/). The descriptive
    # condition labels (orientation, contrast, blur, occlusion, notes, ...) are no
    # longer collected here — the scanner writes them into setup.json.analysisInputs
    # at calibration.
    route_folder: str = Field(..., min_length=1)


class DownloadRequest(AnalysisMetadata):
    url: str = Field(..., min_length=5)
    resolution: int = Field(default=720, ge=144, le=4320)


class ImportRequest(AnalysisMetadata):
    local_path: str = Field(..., min_length=1)


class DetectionRequest(BaseModel):
    # Path to the video the detector ran on, e.g.
    # "analysis/<route>/<video_key>/<video_key>.mp4". Route and video_key are derived
    # from the folder structure: video_key is the parent folder, route its grandparent.
    video_path: str = Field(..., min_length=1)
    pose: Any = Field(...)
    orb: Any = Field(...)


class NormPoint(BaseModel):
    x: float
    y: float


class NormCrop(BaseModel):
    x: float
    y: float
    w: float
    h: float


class VitPoseFrame(BaseModel):
    timestamp: float = Field(..., ge=0)


class VitPoseJobRequest(BaseModel):
    # Cross-program contract with beta-scanner (its HARNESS_API_BASE points here).
    # See docs/adr/0003. Coordinates are full-frame-normalized [0, 1].
    video_path: str = Field(..., min_length=1)
    route_folder: str = Field(..., min_length=1)
    video_key: str = Field(..., min_length=1)
    climber_point: NormPoint | None = None
    climber_crop: NormCrop | None = None
    wall_crop: NormCrop | None = None  # accepted for contract parity; ignored for pose
    panning: bool = False
    # Hash of the setup.json this job runs under; stamped into vitpose.json as the
    # provenance anchor. Optional: the job falls back to the bundle's setup.json.
    setup_hash: str | None = None
    frames: list[VitPoseFrame] = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Route-folder listing
# --------------------------------------------------------------------------- #

def list_route_folders() -> list[str]:
    routes: set[str] = set()
    if ANALYSIS_DIR.exists():
        for child in ANALYSIS_DIR.iterdir():
            if child.is_dir() and child.name.strip():
                routes.add(child.name)
    return sorted(routes)


# --------------------------------------------------------------------------- #
# Response shaping
# --------------------------------------------------------------------------- #

def _bundle_response(
    download_result, source_extras: dict[str, object] | None = None
) -> dict[str, object]:
    bundle = build_analysis_bundle(download_result, ANALYSIS_DIR, source_extras)
    source_video = bundle["metadata"]["source_video"]
    return {
        "timestamp": download_result.timestamp,
        "route_folder": download_result.route_folder,
        "source_type": download_result.source_type,
        "video_key": bundle["video_key"],
        "video_path": str(download_result.video_path),
        "analysis_video_dir": str(bundle["video_dir"]),
        "metadata_path": str(bundle["metadata_path"]),
        "frame_path": str(bundle["frame_path"]),
        "detections_dir": str(bundle["detections_dir"]),
        "source_title": source_video.get("title"),
        "source_video_id": source_video.get("video_id"),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/routes")
def get_routes() -> dict[str, list[str]]:
    return {"routes": list_route_folders()}


@app.post("/api/download")
def create_download_bundle(payload: DownloadRequest) -> dict[str, object]:
    try:
        download_result = download_video(
            payload.url,
            ANALYSIS_DIR,
            payload.resolution,
            route_folder=payload.route_folder,
            timestamp=generate_timestamp(),
        )
        return _bundle_response(
            download_result,
            {"requested_resolution": payload.resolution},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/import")
def create_import_bundle(payload: ImportRequest) -> dict[str, object]:
    try:
        download_result = import_local_video(
            Path(payload.local_path),
            ANALYSIS_DIR,
            route_folder=payload.route_folder,
            timestamp=generate_timestamp(),
        )
        # imported_from is already recorded in the source_video block by the core.
        return _bundle_response(download_result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/detections")
def push_detections(payload: DetectionRequest) -> dict[str, object]:
    video_path = Path(payload.video_path)
    video_key = video_path.parent.name
    route_folder = video_path.parent.parent.name

    if not video_key or not route_folder:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not derive route/video_key from video_path; expected "
                ".../<route>/<video_key>/<file>."
            ),
        )

    try:
        result = save_detection_run(
            ANALYSIS_DIR,
            route_folder,
            video_key,
            payload.pose,
            payload.orb,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


# --------------------------------------------------------------------------- #
# ViTPose++ Ground Truth scaffold (see docs/adr/0003)
# --------------------------------------------------------------------------- #

def _to_vitpose_request(payload: VitPoseJobRequest) -> vitpose_job.VitPoseRequest:
    point = (
        vitpose_job.Point(payload.climber_point.x, payload.climber_point.y)
        if payload.climber_point is not None
        else None
    )
    crop = (
        vitpose_job.Box(
            payload.climber_crop.x, payload.climber_crop.y,
            payload.climber_crop.w, payload.climber_crop.h,
        )
        if payload.climber_crop is not None
        else None
    )
    return vitpose_job.VitPoseRequest(
        video_path=payload.video_path,
        route_folder=payload.route_folder,
        video_key=payload.video_key,
        frames=tuple(f.timestamp for f in payload.frames),
        climber_point=point,
        climber_crop=crop,
        panning=payload.panning,
        setup_hash=payload.setup_hash,
    )


# The pose/track models are shared singletons; serialize jobs so two background
# threads never run inference on them at once (this is a local, single-user tool).
_vitpose_lock = threading.Lock()


def _run_vitpose_safely(request: vitpose_job.VitPoseRequest, job_id: str) -> None:
    # Failures are already recorded to the status sidecar inside run_vitpose_job; the
    # thread just needs to not crash the interpreter on an unhandled exception.
    try:
        with _vitpose_lock:
            vitpose_job.run_vitpose_job(
                ANALYSIS_DIR,
                request,
                vitpose_job.default_tracker(),
                vitpose_job.default_pose_backend(),
                job_id=job_id,
            )
    except Exception:  # noqa: BLE001 — surfaced via vitpose.status.json
        pass




@app.post("/api/vitpose")
def start_vitpose_job(payload: VitPoseJobRequest) -> JSONResponse:
    # Validate synchronously so a bad path/bundle fails fast with 4xx (per contract);
    # the model run itself is offloaded to a daemon thread and polled via the artifact.
    try:
        bundle_dir = vitpose_job.bundle_dir_for(
            ANALYSIS_DIR, payload.route_folder, payload.video_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not bundle_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No bundle for route={payload.route_folder!r} "
                f"video_key={payload.video_key!r}."
            ),
        )

    try:
        video_path = vitpose_job.resolve_video_path(ANALYSIS_DIR, payload.video_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not video_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Video not found: {payload.video_path}"
        )

    request = _to_vitpose_request(payload)
    job_id = uuid.uuid4().hex
    thread = threading.Thread(
        target=_run_vitpose_safely, args=(request, job_id), daemon=True
    )
    thread.start()

    return JSONResponse(status_code=202, content={"jobId": job_id, "status": "accepted"})
