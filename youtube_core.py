from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yt_dlp


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

# The video_key (a sanitized filename stem) is used twice in the canonical path
# — once as the folder name and again as the copied file's name — so a long
# source filename is effectively doubled. Windows enforces MAX_PATH (260 chars)
# unless long-path support is enabled, and a raw source title can easily push the
# destination past that limit, surfacing as "[WinError 3] The system cannot find
# the path specified". Cap the stem so canonical paths stay comfortably short.
MAX_KEY_STEM_LENGTH = 48


@dataclass(frozen=True)
class DownloadResult:
    video_path: Path
    info: dict[str, Any]
    timestamp: str
    route_folder: str
    video_key: str
    source_type: str


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def sanitize_route_folder(route_folder: str) -> str:
    cleaned = sanitize_filename(route_folder.strip())
    return cleaned or "uncategorized"


def _truncate_stem(stem: str, limit: int = MAX_KEY_STEM_LENGTH) -> str:
    """Trim a sanitized filename stem so canonical paths stay under MAX_PATH.

    Trailing separators left by the cut are stripped so keys don't end in "_" or ".".
    """
    if len(stem) <= limit:
        return stem
    return stem[:limit].rstrip("._-") or stem[:limit]


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _find_ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    if sys.platform.startswith("win"):
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            candidates = [
                Path(local_appdata) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
                Path(local_appdata) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
                Path(local_appdata) / "Microsoft" / "WinGet" / "Packages",
            ]

            for candidate in candidates[:2]:
                if candidate.is_file():
                    return str(candidate)

            packages_root = candidates[2]
            if packages_root.is_dir():
                for candidate in packages_root.rglob("ffmpeg.exe"):
                    if "Gyan.FFmpeg" in str(candidate):
                        return str(candidate)

    return None


def _find_ffprobe_executable() -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    # ffprobe ships alongside ffmpeg; derive it from the resolved ffmpeg path.
    ffmpeg = _find_ffmpeg_executable()
    if ffmpeg:
        candidate = Path(ffmpeg).with_name(
            "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"
        )
        if candidate.is_file():
            return str(candidate)

    return None


def probe_video_metadata(video_path: Path) -> dict[str, Any]:
    """Best-effort technical metadata for a local file via ffprobe.

    Returns null-valued fields if ffprobe is unavailable or the probe fails, so
    a local import never hard-fails purely on metadata extraction.
    """
    empty = {
        "width": None,
        "height": None,
        "fps": None,
        "duration_seconds": None,
        "filesize": None,
    }

    try:
        filesize = video_path.stat().st_size
    except OSError:
        filesize = None
    empty["filesize"] = filesize

    ffprobe = _find_ffprobe_executable()
    if not ffprobe:
        return empty

    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        probe = json.loads(completed.stdout or "{}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return empty

    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )

    fps = None
    rate = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if rate and "/" in rate:
        num, _, den = rate.partition("/")
        try:
            num_f, den_f = float(num), float(den)
            if den_f:
                fps = round(num_f / den_f, 3)
        except ValueError:
            fps = None

    duration = None
    raw_duration = (probe.get("format") or {}).get("duration") or video_stream.get(
        "duration"
    )
    if raw_duration is not None:
        try:
            duration = round(float(raw_duration), 3)
        except (TypeError, ValueError):
            duration = None

    return {
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
        "duration_seconds": duration,
        "filesize": filesize,
    }


def _build_format_selector(max_height: int) -> tuple[str, str | None]:
    ffmpeg_executable = _find_ffmpeg_executable()

    if ffmpeg_executable:
        format_selector = (
            f"bestvideo[ext=mp4][height<={max_height}]"
            "+bestaudio[ext=m4a]/best[ext=mp4]/best"
        )
    else:
        format_selector = f"best[ext=mp4][height<={max_height}]/best[height<={max_height}]/best"

    return format_selector, ffmpeg_executable


def download_video(
    url: str,
    analysis_root: Path,
    max_height: int,
    route_folder: str = "uncategorized",
    timestamp: str | None = None,
) -> DownloadResult:
    """Download a YouTube video straight into its analysis key folder.

    The canonical home is analysis/<route>/<video_key>/<video_key><ext>, where
    video_key is <video_id>_<ts>. Bytes are first fetched into a staging dir so the
    video_id (only known after extraction) can name the final folder.
    """
    safe_route_folder = sanitize_route_folder(route_folder)
    route_dir = analysis_root / safe_route_folder
    route_dir.mkdir(parents=True, exist_ok=True)
    # Unique per-call staging dir on the same filesystem as the final folder, so the
    # move is a cheap rename and concurrent downloads to one route can't collide.
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging_", dir=route_dir))

    effective_timestamp = timestamp or generate_timestamp()

    format_selector, ffmpeg_executable = _build_format_selector(max_height)

    if not ffmpeg_executable:
        print(
            "Warning: ffmpeg not found on PATH; downloading single-file format fallback "
            "(quality/availability may vary).",
            file=sys.stderr,
        )

    ydl_opts = {
        "outtmpl": str(staging_dir / "%(id)s.%(ext)s"),
        "format": format_selector,
        "noplaylist": True,
        "quiet": False,
    }

    if ffmpeg_executable:
        ydl_opts["ffmpeg_location"] = str(Path(ffmpeg_executable).parent)
        ydl_opts["merge_output_format"] = "mp4"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            staged_path = Path(ydl.prepare_filename(info))

        # Canonical key is <video_id>_<ts>: unique, filesystem-safe, and the shared
        # key that the analysis folder and every detection file also use.
        video_id = sanitize_filename(str(info.get("id") or "unknown"))
        video_key = f"{video_id}_{effective_timestamp}"
        video_dir = route_dir / video_key
        video_dir.mkdir(parents=True, exist_ok=True)

        canonical_path = _next_available_path(
            video_dir / f"{video_key}{staged_path.suffix}"
        )
        staged_path.rename(canonical_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    if not canonical_path.exists() or canonical_path.stat().st_size < 100_000:
        raise RuntimeError("Download appears invalid: file missing or too small.")

    return DownloadResult(
        video_path=canonical_path,
        info=info,
        timestamp=effective_timestamp,
        route_folder=safe_route_folder,
        video_key=video_key,
        source_type="youtube",
    )


def import_local_video(
    local_path: Path,
    analysis_root: Path,
    route_folder: str = "uncategorized",
    timestamp: str | None = None,
) -> DownloadResult:
    """Copy a local video into its analysis key folder (non-destructive).

    Canonical home is analysis/<route>/<video_key>/<video_key><ext>, where video_key
    is <sanitized-filename-stem>_<ts>. The source file is left untouched.
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"No file at local path: {local_path}")

    suffix = local_path.suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video extension {suffix!r}. Allowed: {allowed}."
        )

    safe_route_folder = sanitize_route_folder(route_folder)
    effective_timestamp = timestamp or generate_timestamp()

    stem = _truncate_stem(sanitize_filename(local_path.stem)) or "video"
    video_key = f"{stem}_{effective_timestamp}"
    video_dir = analysis_root / safe_route_folder / video_key
    video_dir.mkdir(parents=True, exist_ok=True)

    canonical_path = _next_available_path(video_dir / f"{video_key}{suffix}")
    try:
        shutil.copy2(local_path, canonical_path)
    except OSError as exc:
        # On Windows a destination over MAX_PATH (260) surfaces as WinError 3
        # ("cannot find the path specified"), which reads like a missing source
        # file. Point at the real culprit: the copy target we couldn't create.
        raise OSError(
            f"Failed to write imported video to {canonical_path} "
            f"({len(str(canonical_path))} chars): {exc}"
        ) from exc

    if not canonical_path.exists() or canonical_path.stat().st_size < 100_000:
        raise RuntimeError("Imported file appears invalid: missing or too small.")

    probed = probe_video_metadata(canonical_path)
    info = {
        "original_filename": local_path.name,
        "imported_from": str(local_path),
        **probed,
    }

    return DownloadResult(
        video_path=canonical_path,
        info=info,
        timestamp=effective_timestamp,
        route_folder=safe_route_folder,
        video_key=video_key,
        source_type="local",
    )


# Write every decoded frame to the same file, so whatever remains is the *last*
# frame in the window. No -frames:v cap: that would keep the first frame instead.
_FRAME_OUTPUT_ARGS = ["-update", "1"]


def _run_frame_extraction(ffmpeg: str, video_path: Path, frame_path: Path, args: list[str]) -> bool:
    """Run one ffmpeg frame grab; return True only if a non-empty PNG lands.

    ffmpeg exits 0 even when a seek overshoots the last frame and nothing is
    encoded, so a clean return code is not proof of success — the file check is.
    """
    command = [ffmpeg, "-y", *args, "-i", str(video_path), *_FRAME_OUTPUT_ARGS, str(frame_path)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return False
    return frame_path.exists() and frame_path.stat().st_size > 0


def extract_last_frame(video_path: Path, frame_path: Path) -> Path:
    ffmpeg = _find_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract the final video frame.")

    frame_path.parent.mkdir(parents=True, exist_ok=True)

    # Seek near the end for a cheap grab, widening the window on each miss: a tight
    # window can overshoot the last decodable frame (given the file's keyframe
    # spacing) and encode nothing. The final, seek-free pass decodes the whole file
    # and keeps the last frame — reliable but the most expensive, so it's the last
    # resort.
    for attempt in (["-sseof", "-1"], ["-sseof", "-3"], ["-sseof", "-10"], []):
        if _run_frame_extraction(ffmpeg, video_path, frame_path, attempt):
            return frame_path

    raise RuntimeError("Could not extract the final frame from the video.")


def _paired_detection_paths(detections_dir: Path, run_ts: str) -> tuple[Path, str]:
    # Allocate a single run stem shared by the pose and orb files so they stay
    # visibly paired. If this exact second is already taken, bump BOTH together.
    stem = run_ts
    counter = 1
    while (
        (detections_dir / f"{stem}_pose.json").exists()
        or (detections_dir / f"{stem}_orb.json").exists()
    ):
        stem = f"{run_ts}_{counter}"
        counter += 1
    return detections_dir, stem


def save_detection_run(
    analysis_root: Path,
    route_folder: str,
    video_key: str,
    pose: Any,
    orb: Any,
    run_ts: str | None = None,
) -> dict[str, Any]:
    """Append one pose+orb detection run to an existing video's analysis folder.

    Raises FileNotFoundError if the video folder does not exist and ValueError if
    the resolved path escapes the analysis root.
    """
    safe_route = sanitize_route_folder(route_folder)
    safe_key = sanitize_filename(video_key.strip())
    if not safe_key:
        raise ValueError("video_key is empty after sanitization.")

    analysis_root = analysis_root.resolve()
    video_dir = (analysis_root / safe_route / safe_key).resolve()

    # Reject any path that escapes the analysis root (e.g. traversal via ../).
    if analysis_root not in video_dir.parents:
        raise ValueError("Resolved detection path escapes the analysis root.")

    if not video_dir.is_dir():
        raise FileNotFoundError(
            f"No analysis folder for route={safe_route!r} video_key={safe_key!r}. "
            "Create the video bundle before pushing detections."
        )

    detections_dir = video_dir / "detections"
    detections_dir.mkdir(exist_ok=True)

    effective_ts = run_ts or generate_timestamp()
    detections_dir, stem = _paired_detection_paths(detections_dir, effective_ts)

    written: dict[str, str] = {}
    for detection_type, blob in (("pose", pose), ("orb", orb)):
        envelope = {
            "video_key": safe_key,
            "route_folder": safe_route,
            "run_ts": effective_ts,
            "written_at": datetime.now().isoformat(timespec="seconds"),
            "type": detection_type,
            "data": blob,
        }
        path = detections_dir / f"{stem}_{detection_type}.json"
        path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written[detection_type] = str(path)

    return {
        "route_folder": safe_route,
        "video_key": safe_key,
        "run_ts": effective_ts,
        "detections_dir": str(detections_dir),
        "pose_path": written["pose"],
        "orb_path": written["orb"],
    }


def _build_source_video_block(download_result: DownloadResult) -> dict[str, Any]:
    info = download_result.info
    ext = download_result.video_path.suffix.lstrip(".") or None

    if download_result.source_type == "local":
        return {
            "source_type": "local",
            "url": None,
            "video_id": None,
            "title": Path(info.get("original_filename", "")).stem or None,
            "uploader": None,
            "channel": None,
            "channel_id": None,
            "upload_date": None,
            "format_id": None,
            "ext": ext,
            "original_filename": info.get("original_filename"),
            "imported_from": info.get("imported_from"),
            "duration_seconds": info.get("duration_seconds"),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "filesize": info.get("filesize"),
        }

    return {
        "source_type": "youtube",
        "url": info.get("webpage_url") or info.get("original_url"),
        "video_id": info.get("id"),
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "duration_seconds": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "filesize": info.get("filesize") or info.get("filesize_approx"),
        "format_id": info.get("format_id"),
        "ext": info.get("ext") or ext,
    }


def build_analysis_bundle(
    download_result: DownloadResult,
    analysis_root: Path,
    user_metadata: dict[str, Any],
) -> dict[str, Any]:
    # The video already lives canonically at analysis/<route>/<video_key>/<video_key><ext>.
    # We build the rest of the bundle (frame + metadata + detections dir) around it.
    video_key = download_result.video_key
    video_dir = download_result.video_path.parent
    route_analysis_dir = video_dir.parent

    detections_dir = video_dir / "detections"
    detections_dir.mkdir(exist_ok=True)

    # Extract the final frame directly from the canonical video in the analysis folder.
    frame_path = video_dir / "final_frame.png"
    extract_last_frame(download_result.video_path, frame_path)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_key": video_key,
        "route_folder": download_result.route_folder,
        "source_type": download_result.source_type,
        "analysis_video_dir": str(video_dir),
        "source_video_path": str(download_result.video_path),
        "final_frame": str(frame_path),
        "detections_dir": str(detections_dir),
        "source_video": _build_source_video_block(download_result),
        "analysis_inputs": user_metadata,
    }

    metadata_path = video_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "route_dir": route_analysis_dir,
        "video_dir": video_dir,
        "video_key": video_key,
        "source_video_path": download_result.video_path,
        "frame_path": frame_path,
        "detections_dir": detections_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }
