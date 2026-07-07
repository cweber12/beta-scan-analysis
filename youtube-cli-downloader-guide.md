# YouTube Download Review + Clean CLI Setup

This document has two goals:
1. Review how YouTube video downloading currently works in this repository.
2. Provide a simple, clean Python CLI program you can copy into a fresh repository and run from a terminal with a YouTube share URL argument.

## Current Download Flow (This Repository)

### Frontend trigger
- UI component: `frontend/src/components/upload/DownloadYouTube.jsx`
- Sends `POST` request to `${VITE_API_BASE_URL_P}/api/download-youtube`
- Body is `FormData` with:
  - `url` (required)
  - `resolution` is not currently sent by the frontend, so backend default applies.

### Backend endpoint
- API route: `backend_process/app/routers/youtube_download.py`
- Endpoint: `POST /download-youtube` (mounted under `/api` in `backend_process/app/main.py`)
- Behavior:
  - Downloads with `yt-dlp`
  - Uses `format = bestvideo[ext=mp4][height<=resolution]/mp4`
  - Sanitizes filename
  - Verifies file can be opened with OpenCV
  - Returns JSON payload with:
    - `file_path`
    - `video_url` (served from `/static/...`)
    - `actual_height`
    - optional `warning`

## Review Findings (Important)

1. Download format can produce video-only streams.
- Current format prefers `bestvideo...` without explicitly merging audio.
- Result: downloaded MP4 may not contain audio on some videos.
- Suggested fix in API code:
  - use `bestvideo+bestaudio/best` style selection when audio is needed.

2. Fallback logic message does not match behavior.
- Code says: "Defaulting to 720p" when requested resolution is above available.
- Actual fallback still asks for `height<=720`, which can still return lower resolutions if 720 is unavailable.
- Suggested fix:
  - message should state "using highest available <= 720p".

3. Unused import present.
- `urllib.parse` is imported but not used.
- Low severity cleanup.

4. OpenCV validation adds heavy dependency for simple downloading.
- For a clean downloader tool, OpenCV is unnecessary and increases install friction.
- Better in a standalone CLI:
  - validate by existence/size and let `yt-dlp` handle media concerns.

## Clean Repository: Minimal Python CLI Downloader

Use this when you only need a terminal command like:

```bash
python download_youtube.py "https://youtu.be/VIDEO_ID"
```

### 1. Create files

Create `download_youtube.py` with:

```python
import argparse
import re
import sys
from pathlib import Path

import yt_dlp


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\\w\\-.]", "_", name)


def download_video(url: str, output_dir: Path, max_height: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download best mp4 video + audio when available, then fall back gracefully.
    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "format": f"bestvideo[ext=mp4][height<={max_height}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        original_path = Path(ydl.prepare_filename(info))

    safe_name = sanitize_filename(original_path.name)
    safe_path = original_path.with_name(safe_name)

    if original_path != safe_path:
        original_path.rename(safe_path)

    if not safe_path.exists() or safe_path.stat().st_size < 100_000:
        raise RuntimeError("Download appears invalid: file missing or too small.")

    return safe_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a YouTube video from a share URL.")
    parser.add_argument("url", help="YouTube share URL, e.g. https://youtu.be/VIDEO_ID")
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="Maximum height in pixels (default: 720)",
    )
    parser.add_argument(
        "--out",
        default="downloads",
        help="Output folder (default: downloads)",
    )

    args = parser.parse_args()

    try:
        output_file = download_video(args.url, Path(args.out), args.resolution)
        print(f"Download complete: {output_file}")
        return 0
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `requirements.txt` with:

```txt
yt-dlp>=2025.3.27
```

### 2. Install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run from terminal with URL argument

```bash
python download_youtube.py "https://youtu.be/VIDEO_ID"
```

Optional custom resolution and output folder:

```bash
python download_youtube.py "https://youtu.be/VIDEO_ID" --resolution 1080 --out temp_uploads
```

## Notes

- `yt-dlp` may need `ffmpeg` available on your system `PATH` for best mux/format support.
- Prefer legal/authorized content downloads that comply with YouTube Terms and local law.
