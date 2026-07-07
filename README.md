# Climb Video Analyzer

Local Python app for downloading climbing videos, collecting analysis metadata, and exporting a final frame for ORB and pose-detection workflows.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install `ffmpeg` on Windows for the best download and frame-extraction behavior:

```powershell
winget install --id Gyan.FFmpeg -e
```

## Run the web app

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser, paste a YouTube URL, fill in the climbing metadata, and submit.

## Output folders

- `downloads/<route_folder>/` stores the downloaded video.
- Downloaded filenames are prefixed with the request timestamp.
- `analysis/<route_folder>/` stores a copied video, metadata file, and final frame for each run.
- Analysis filenames use the same timestamp prefix as the downloaded video from that request.

## CLI fallback

The original terminal downloader still works:

```powershell
python download_youtube.py "https://youtu.be/IyFjR9qRiJY?si=29UoeHV7aIYpaFV1"
```

Optional route grouping in CLI:

```powershell
python download_youtube.py "https://youtu.be/IyFjR9qRiJY?si=29UoeHV7aIYpaFV1" --route moonboard-v4
```

## Notes

- The metadata form is tuned for climbing footage: route orientation, camera angle, shadows, climber contrast, wall contrast, motion blur, occlusion, stability, and notes.
- If `ffmpeg` is on your `PATH`, the downloader can merge best video + audio and extract the final frame.
- If `ffmpeg` is not installed, the CLI falls back to single-file formats, but the analysis bundle needs `ffmpeg` for final-frame extraction.
- Use only for content you are authorized to download, and comply with local laws and platform terms.
