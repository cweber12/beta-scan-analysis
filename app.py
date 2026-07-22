from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import video_stats
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
    t: float | None = None


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
    #
    # The scanner-side relay sends snake_case, but tolerate a camelCase `setupHash`
    # sent straight to this service (matching setup.json's own casing). setup_hash
    # stays canonical in the model so storage and logs are consistent.
    model_config = ConfigDict(populate_by_name=True)

    video_path: str = Field(..., min_length=1)
    route_folder: str = Field(..., min_length=1)
    video_key: str = Field(..., min_length=1)
    # Seed contract of record (scanner branch feat/harness-vitpose-seed-region):
    # `seed_tap` anchors the Climber, `seed_region` gates the seed — decoupled from
    # the Climber Crop. `climber_point`/`climber_crop` remain as legacy aliases for
    # older clients; `_to_vitpose_request` prefers the new fields when both are sent.
    seed_tap: NormPoint | None = Field(default=None, alias="seedTap")
    seed_region: NormCrop | None = Field(default=None, alias="seedRegion")
    climber_point: NormPoint | None = None
    climber_crop: NormCrop | None = None
    wall_crop: NormCrop | None = None  # accepted for contract parity; ignored for pose
    panning: bool = False
    # Hash of the setup.json this job runs under; stamped into vitpose.json as the
    # provenance anchor. Optional: the job falls back to the bundle's setup.json.
    # Accepts `setup_hash` (canonical) or `setupHash`.
    setup_hash: str | None = Field(default=None, alias="setupHash")
    frames: list[VitPoseFrame] = Field(..., min_length=1)


class VideoStatsRequest(BaseModel):
    # Phase-2 Video Stats trigger (issue #23): the scanner POSTs its freshly drawn
    # calibration crops here mid-calibration and gets back region stats + suggested
    # labels to prefill analysisInputs. Crop geometry is optional — omitted fields
    # fall back to the bundle's just-saved setup.json. Tolerates camelCase field
    # names (matching setup.json's casing) alongside canonical snake_case.
    model_config = ConfigDict(populate_by_name=True)

    route_folder: str = Field(..., min_length=1, alias="routeFolder")
    video_key: str = Field(..., min_length=1, alias="videoKey")
    climber_crop: NormCrop | None = Field(default=None, alias="climberCrop")
    wall_crop: NormCrop | None = Field(default=None, alias="wallCrop")
    climber_point: NormPoint | None = Field(default=None, alias="climberPoint")
    panning: bool | None = None
    setup_hash: str | None = Field(default=None, alias="setupHash")


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


# Bump when a breaking change to any cross-program contract ships (endpoint
# payload shape, artifact schema, label vocabulary). Additive changes don't bump.
API_VERSION = 1


@app.get("/api/contract")
def get_contract() -> dict[str, object]:
    """What this harness speaks — probed by the scanner at startup (drift check).

    The scanner gates features on this instead of assuming: prefill only runs if
    /api/video-stats is advertised AND suggestions.available is true; a missing
    endpoint or apiVersion mismatch surfaces as a visible "harness out of date"
    warning rather than a silent 404 mid-calibration.
    """
    thresholds = video_stats.SUGGESTION_THRESHOLDS or {}
    return {
        "service": "beta-scan-analysis-harness",
        "apiVersion": API_VERSION,
        # Derived from the live route table so this can never drift from reality.
        "endpoints": sorted(
            {r.path for r in app.routes if r.path.startswith("/api/")}
        ),
        "artifacts": {
            "vitpose": vitpose_job.ARTIFACT_VERSION,
            "videoStats": video_stats.VIDEO_STATS_VERSION,
        },
        # Additive feature flags the scanner gates on (no apiVersion bump). decoupledSeed
        # signals that POST /api/vitpose accepts seed_tap + seed_region as the seed
        # contract of record (with legacy climber_point/climber_crop alias support).
        "capabilities": {
            "decoupledSeed": True,
        },
        "suggestions": {
            "available": bool(thresholds),
            "fitDate": thresholds.get("fitDate"),
            "corpusSize": thresholds.get("corpusSize"),
            "labeledBundles": thresholds.get("labeledBundles"),
        },
    }


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
    # Resolve the decoupled seed contract: the new `seed_tap`/`seed_region` fields are
    # the contract of record and win over the legacy `climber_point`/`climber_crop`
    # aliases when both are present. Legacy-only clients still seed as before.
    tap_src = payload.seed_tap if payload.seed_tap is not None else payload.climber_point
    region_src = payload.seed_region if payload.seed_region is not None else payload.climber_crop
    seed_tap = (
        vitpose_job.Point(tap_src.x, tap_src.y, tap_src.t) if tap_src is not None else None
    )
    seed_region = (
        vitpose_job.Box(region_src.x, region_src.y, region_src.w, region_src.h)
        if region_src is not None
        else None
    )
    return vitpose_job.VitPoseRequest(
        video_path=payload.video_path,
        route_folder=payload.route_folder,
        video_key=payload.video_key,
        frames=tuple(f.timestamp for f in payload.frames),
        seed_tap=seed_tap,
        seed_region=seed_region,
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


# --------------------------------------------------------------------------- #
# Video Stats — phase-2 region stats + suggested labels (issue #23)
# --------------------------------------------------------------------------- #

def _find_bundle_video(bundle_dir: Path, video_key: str) -> Path | None:
    canonical = bundle_dir / f"{video_key}.mp4"
    if canonical.is_file():
        return canonical
    for path in sorted(bundle_dir.iterdir()):
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            return path
    return None


def _read_bundle_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


@app.post("/api/video-stats")
def compute_video_stats(payload: VideoStatsRequest) -> dict[str, object]:
    """Compute region-aware stats for a bundle's calibration crops, synchronously.

    Crop geometry missing from the payload falls back to the bundle's setup.json
    (the scanner POSTs right after saving it). Writes video-stats.json stamped with
    the setupHash and returns stats + suggested labels for the prefill flow.
    """
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

    setup = _read_bundle_json(bundle_dir / "setup.json")
    wall_crop = (
        payload.wall_crop.model_dump() if payload.wall_crop else setup.get("wallCrop")
    )
    if not wall_crop:
        raise HTTPException(
            status_code=400,
            detail="No wall crop in the request or the bundle's setup.json.",
        )
    climber_crop = (
        payload.climber_crop.model_dump()
        if payload.climber_crop
        else setup.get("climberCrop")
    )
    if payload.climber_point is not None:
        climber_point_t = payload.climber_point.t
    else:
        climber_point_t = (setup.get("climberPoint") or {}).get("t")
    panning = payload.panning if payload.panning is not None else bool(setup.get("panning"))
    setup_hash = payload.setup_hash or setup.get("setupHash")

    video_path = _find_bundle_video(bundle_dir, payload.video_key)
    if video_path is None:
        raise HTTPException(
            status_code=404, detail=f"No video binary in bundle {bundle_dir.name!r}."
        )

    try:
        frames, timestamps = video_stats.sample_video_frames(video_path)
        region_stats = video_stats.compute_region_stats(
            frames,
            timestamps,
            wall_crop,
            climber_crop=climber_crop,
            climber_point_t=climber_point_t,
            panning=panning,
        )

        # Suggestions blend phase-1 (motion blur, stability) with phase-2 stats.
        # Self-heal a bundle that predates phase-1 from the frames already decoded.
        metadata_path = bundle_dir / "metadata.json"
        metadata = _read_bundle_json(metadata_path)
        source_stats = metadata.get("video_stats")
        if source_stats is None and metadata:
            source_stats = video_stats.build_source_stats_block(
                video_path, metadata.get("source_video"), frames, timestamps
            )
            video_stats.write_source_stats(bundle_dir, source_stats)

        suggestions = video_stats.suggest_labels(source_stats, region_stats)
        artifact_path = video_stats.write_region_stats(
            bundle_dir, region_stats, suggestions, setup_hash, source="endpoint"
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — decode/compute failure -> explicit 500
        raise HTTPException(
            status_code=500, detail=f"Video stats extraction failed: {exc}"
        ) from exc

    return {
        "routeFolder": payload.route_folder,
        "videoKey": payload.video_key,
        "setupHash": setup_hash,
        "artifactPath": str(artifact_path),
        "regionStats": region_stats,
        "suggestions": suggestions,
    }
