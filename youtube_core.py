from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yt_dlp


@dataclass(frozen=True)
class DownloadResult:
    video_path: Path
    info: dict[str, Any]
    timestamp: str
    route_folder: str


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def sanitize_route_folder(route_folder: str) -> str:
    cleaned = sanitize_filename(route_folder.strip())
    return cleaned or "uncategorized"


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
    output_dir: Path,
    max_height: int,
    route_folder: str = "uncategorized",
    timestamp: str | None = None,
) -> DownloadResult:
    safe_route_folder = sanitize_route_folder(route_folder)
    target_dir = output_dir / safe_route_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    effective_timestamp = timestamp or generate_timestamp()

    format_selector, ffmpeg_executable = _build_format_selector(max_height)

    if not ffmpeg_executable:
        print(
            "Warning: ffmpeg not found on PATH; downloading single-file format fallback "
            "(quality/availability may vary).",
            file=sys.stderr,
        )

    ydl_opts = {
        "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
        "format": format_selector,
        "noplaylist": True,
        "quiet": False,
    }

    if ffmpeg_executable:
        ydl_opts["ffmpeg_location"] = str(Path(ffmpeg_executable).parent)
        ydl_opts["merge_output_format"] = "mp4"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original_path = Path(ydl.prepare_filename(info))

    safe_name = sanitize_filename(original_path.name)
    safe_path = original_path.with_name(safe_name)

    if safe_path.exists() and original_path != safe_path:
        safe_path = _next_available_path(safe_path)

    if original_path != safe_path:
        original_path.rename(safe_path)

    timestamped_path = safe_path.with_name(f"{effective_timestamp}_{safe_path.name}")
    timestamped_path = _next_available_path(timestamped_path)
    safe_path.rename(timestamped_path)

    if not timestamped_path.exists() or timestamped_path.stat().st_size < 100_000:
        raise RuntimeError("Download appears invalid: file missing or too small.")

    return DownloadResult(
        video_path=timestamped_path,
        info=info,
        timestamp=effective_timestamp,
        route_folder=safe_route_folder,
    )


def extract_last_frame(video_path: Path, frame_path: Path) -> Path:
    ffmpeg = _find_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to extract the final video frame.")

    frame_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(frame_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    if not frame_path.exists() or frame_path.stat().st_size == 0:
        raise RuntimeError("Could not extract the final frame from the video.")

    return frame_path


def build_analysis_bundle(
    download_result: DownloadResult,
    analysis_root: Path,
    user_metadata: dict[str, Any],
) -> dict[str, Any]:
    route_analysis_dir = analysis_root / download_result.route_folder
    route_analysis_dir.mkdir(parents=True, exist_ok=True)

    info = download_result.info
    timestamp = download_result.timestamp

    copied_video = route_analysis_dir / download_result.video_path.name
    copied_video = _next_available_path(copied_video)
    shutil.copy2(download_result.video_path, copied_video)

    frame_path = route_analysis_dir / f"{timestamp}_final_frame.png"
    frame_path = _next_available_path(frame_path)
    extract_last_frame(copied_video, frame_path)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_route_dir": str(route_analysis_dir),
        "downloaded_video": str(copied_video),
        "final_frame": str(frame_path),
        "source_video": {
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
            "ext": info.get("ext"),
        },
        "analysis_inputs": user_metadata,
    }

    metadata_path = route_analysis_dir / f"{timestamp}_metadata.json"
    metadata_path = _next_available_path(metadata_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "route_dir": route_analysis_dir,
        "video_path": copied_video,
        "frame_path": frame_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }
