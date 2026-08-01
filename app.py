from __future__ import annotations

import json
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

import cycle_integrity
import mediapipe_job
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
    # Climb window (issue #101). `climb_start` is the frozen **setup tap**'s timestamp
    # — the scanner must send the *setup* tap here, never the re-seed tap — and
    # `climb_end` the explicit end-of-climb marker. Both optional: omitted, the job
    # falls back to the bundle's setup.json and, failing that, behaves as it does today.
    climb_start: float | None = Field(default=None, alias="climbStart", ge=0)
    climb_end: float | None = Field(default=None, alias="climbEnd", ge=0)
    # Re-run even when the seed is unchanged (bypasses the idempotence skip).
    force: bool = False
    # Person-detector inference resolution / confidence floor. Omitted leaves the
    # backend defaults alone, so existing clients and existing seed hashes are
    # unchanged. Raise `detector_imgsz` when the Climber is too small to detect.
    detector_imgsz: int | None = Field(default=None, alias="detectorImgsz", ge=320, le=4096)
    detector_conf: float | None = Field(default=None, alias="detectorConf", gt=0, lt=1)
    panning: bool = False
    # Hash of the setup.json this job runs under; stamped into vitpose.json as the
    # provenance anchor. Optional: the job falls back to the bundle's setup.json.
    # Accepts `setup_hash` (canonical) or `setupHash`.
    setup_hash: str | None = Field(default=None, alias="setupHash")
    frames: list[VitPoseFrame] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_climb_window(self) -> "VitPoseJobRequest":
        if (
            self.climb_start is not None
            and self.climb_end is not None
            and self.climb_end <= self.climb_start
        ):
            raise ValueError("climb_end must be greater than climb_start.")
        return self


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
        # splitTaps signals that the harness treats the setup tap and the ViTPose seed
        # tap as *separate* values (issue #101): it reads the climb start from the
        # frozen setup tap in setup.json, takes the re-seed tap from `seed_tap` only,
        # accepts `climb_start`/`climb_end`, and skips a job whose seed is unchanged.
        # A scanner that has not adopted the split keeps working — the harness then
        # sees one tap and behaves as it does today.
        "capabilities": {
            "decoupledSeed": True,
            "splitTaps": True,
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
        climb_start=payload.climb_start,
        climb_end=payload.climb_end,
        panning=payload.panning,
        setup_hash=payload.setup_hash,
        force=payload.force,
        detector_imgsz=payload.detector_imgsz,
        detector_conf=payload.detector_conf,
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
    # Seed idempotence (issue #101): when the seed tap, seed region, climb window and
    # video binary all match the scaffold already on disk, re-running would rewrite the
    # same artifact. Report the skip synchronously — the scanner polls for an artifact
    # that is already there, so a 202 would make it wait for a job that never runs.
    start, end = vitpose_job.resolve_climb_window(request, bundle_dir)
    request = replace(request, climb_start=start, climb_end=end)
    job_id = uuid.uuid4().hex
    if vitpose_job.seed_is_unchanged(bundle_dir, request, video_path):
        # Stamp the sidecar too, not just the response body. A client written against
        # the old contract knows only the 202 + poll-the-sidecar flow; if the skip lived
        # solely in a 200 it has never seen, it would poll a sidecar that never reaches a
        # terminal state and hang the whole batch on an *unchanged* seed. Writing the
        # terminal status makes the new response safe for a scanner that hasn't adopted it.
        vitpose_job.write_skip_status(bundle_dir, job_id, "unchanged-seed")
        return JSONResponse(status_code=200, content={
            "jobId": job_id,
            "status": "skipped",
            "reason": "unchanged-seed",
            "seedHash": vitpose_job.artifact_seed_hash(bundle_dir),
            "artifactPath": str(bundle_dir / vitpose_job.ARTIFACT_NAME),
        })

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


# --------------------------------------------------------------------------- #
# Harness MediaPipe batch (PRD #156, issue #159; see docs/adr/0012)
# --------------------------------------------------------------------------- #

class BundleRef(BaseModel):
    route_folder: str = Field(..., min_length=1, alias="routeFolder")
    video_key: str = Field(..., min_length=1, alias="videoKey")

    model_config = ConfigDict(populate_by_name=True)


class MediaPipeBatchRequest(BaseModel):
    """One experimental sweep: an arm, a Bundle selection, and a repeat count.

    Deliberately *not* a cross-program contract — nothing in the scanner calls this. It is
    the analyst's handle on the experiment, mirroring ``POST /api/vitpose``'s shape so the
    service has one job idiom rather than two.
    """

    model_config = ConfigDict(populate_by_name=True)

    mode: int = Field(default=1, alias="mode")
    crop: str = Field(default=mediapipe_job.CROP_NONE)
    # Preprocessing steps (issue #161), one field per factor. Omitted means the step is
    # absent — which is the control level, and is a different arm from the step present at
    # its identity value (that one is refused; see mediapipe_job.step_amount).
    contrast: float | None = None
    brightness: float | None = None
    # Defaults to 1 because this detector is bit-deterministic — see DEFAULT_REPEATS.
    repeats: int = Field(default=mediapipe_job.DEFAULT_REPEATS, ge=1, le=25)
    # Omitted means "every eligible Bundle"; an explicit list is the smoke-batch path.
    bundles: list[BundleRef] | None = None

    @model_validator(mode="after")
    def _check_arm(self) -> "MediaPipeBatchRequest":
        if self.mode not in mediapipe_job.DETECTION_MODES:
            raise ValueError(
                f"mode must be one of {mediapipe_job.DETECTION_MODES} "
                "(0 lite, 1 full, 2 heavy)."
            )
        if self.crop not in mediapipe_job.CROP_POLICIES:
            raise ValueError(f"crop must be one of {mediapipe_job.CROP_POLICIES}.")
        # Validated here, synchronously, so an unrunnable arm is a 422 before the sweep
        # starts rather than the same error 84 times inside a batch nobody is watching.
        mediapipe_job.preprocess_from_options(self.contrast, self.brightness)
        return self

    def arm(self) -> mediapipe_job.DetectionConfig:
        return mediapipe_job.DetectionConfig(
            mode=self.mode, crop=self.crop,
            preprocess=mediapipe_job.preprocess_from_options(self.contrast, self.brightness))


def _run_batch_safely(request: MediaPipeBatchRequest, job_id: str,
                      cycle_id: str | None) -> None:
    # Failures are already recorded to the batch sidecar inside run_batch; this only stops
    # a daemon thread's traceback from going nowhere.
    try:
        mediapipe_job.run_batch(
            ANALYSIS_DIR,
            request.arm(),
            mediapipe_job.default_detector_factory,
            only=[(b.route_folder, b.video_key) for b in request.bundles]
            if request.bundles is not None else None,
            repeats=request.repeats,
            job_id=job_id,
            cycle_id=cycle_id,
        )
    except Exception:  # noqa: BLE001 — surfaced via mediapipe-batch.status.json
        pass


@app.post("/api/mediapipe/batch")
def start_mediapipe_batch(payload: MediaPipeBatchRequest) -> JSONResponse:
    """Start a sweep; return 202 immediately and report through the batch sidecar."""

    # Single-flight, refused synchronously (PRD #156 user story 40). Two batches would
    # interleave writes into one Bundle and produce repeat sets whose members came from
    # different arms. A 409 rather than a queue: the caller wants to be told no.
    if mediapipe_job.batch_is_running(ANALYSIS_DIR):
        raise HTTPException(
            status_code=409,
            detail="A MediaPipe batch is already running; refusing to start a second.",
        )

    selection = mediapipe_job.select_bundles(
        ANALYSIS_DIR,
        [(b.route_folder, b.video_key) for b in payload.bundles]
        if payload.bundles is not None else None,
    )
    if not selection.included:
        # A batch over nothing is a caller error, not a job — report it synchronously
        # rather than leaving a poller waiting on a sweep that will do nothing.
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No eligible bundles for this batch.",
                "selection": selection.as_dict(),
            },
        )

    # Which cycle this sweep will fall inside, or null (issue #168). Reported, never
    # gated: a batch outside a cycle is legitimate — a probe, a one-off re-run — it simply
    # cannot be certified against drift afterwards, and that is worth knowing before the
    # sweep rather than when someone tries to publish the comparison.
    #
    # Resolved *before* the thread starts and handed to it, so the 202 and the batch
    # sidecar cannot disagree about which cycle the sweep belongs to — and so the
    # association survives the response, which is where it used to end (issue #176).
    cycle = cycle_integrity.open_cycle_doc(ANALYSIS_DIR)
    cycle_id = cycle["cycleId"] if cycle else None

    job_id = uuid.uuid4().hex
    threading.Thread(
        target=_run_batch_safely, args=(payload, job_id, cycle_id), daemon=True
    ).start()

    return JSONResponse(status_code=202, content={
        "jobId": job_id,
        "status": "accepted",
        "configHash": mediapipe_job.config_hash(payload.arm()),
        "arm": payload.arm().identity(),
        "repeats": payload.repeats,
        "selection": selection.as_dict(),
        "statusPath": str(mediapipe_job.batch_status_path(ANALYSIS_DIR)),
        "cycle": cycle_id,
    })
