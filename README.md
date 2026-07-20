# Climb Video Analyzer

Local Python app for collecting climbing videos — from YouTube or your local
filesystem — into self-contained analysis bundles, ready for ORB and pose-detection
workflows.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

To reproduce the exact known-good environment instead (pinned versions, including the
CUDA torch wheels), install from the lockfile:

```powershell
pip install -r requirements.lock
```

Install `ffmpeg` on Windows for the best download and frame-extraction behavior (it also
provides `ffprobe`, used to read technical metadata from local imports):

```powershell
winget install --id Gyan.FFmpeg -e
```

## Run the web app

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. Pick a **Source**:

- **YouTube URL** — paste a share URL and choose a resolution; the app downloads it.
- **Local file** — paste a path to a video already on this machine (e.g.
  `downloads/Midnight_Lightning_V8.mp4`); the app **copies** it in, leaving the original
  untouched.

Enter the **route folder** and submit. That's the only field the harness collects — it
files the bundle at `analysis/<route_folder>/<video_key>/`. The climbing-condition
labels (route orientation, contrast, shadows, blur, occlusion, notes, …) are no longer
entered here; the Beta Scanner captures them during **calibration** and writes them into
`setup.json` as `analysisInputs`.

## Output folders

The canonical key for a video is `<video_key>` — unique, filesystem-safe, and shared by
the video file, its analysis folder, and every detection file. For YouTube it's
`<video_id>_<timestamp>`; for local imports it's `<sanitized-filename>_<timestamp>`.

Each video is a **self-contained bundle** — the video lives right next to its metadata:

```text
analysis/<route>/<video_key>/
    <video_key>.mp4        # canonical video; run detection on this
    final_frame.png        # last frame, extracted via ffmpeg
    metadata.json          # source/technical info + video path (no condition labels)
    setup.json             # scanner calibration: crops + analysisInputs (condition labels)
    detections/            # created empty at ingest time
        <run_ts>_pose.json
        <run_ts>_orb.json
```

`metadata.json` records a `source_type` of `"youtube"` or `"local"`. Video binaries are
git-ignored; the `metadata.json`, `final_frame.png`, and detection JSON are tracked.

## Detection endpoint

After a separate program runs pose + ORB detection on a video, it pushes the results to
that video's analysis folder:

```json
POST /api/detections
{
  "video_path": "analysis/<route>/<video_key>/<video_key>.mp4",
  "pose": { ... },
  "orb":  { ... }
}
```

The video key is derived from the video's parent folder and the route from its
grandparent. The server writes `detections/<run_ts>_pose.json` and
`detections/<run_ts>_orb.json` (both sharing one run timestamp), each a self-describing
envelope wrapping the verbatim detector output. Both `pose` and `orb` are required. The
endpoint returns 404 if the video's analysis folder does not exist — ingest the video
first. Re-running detection appends a new timestamped pair; nothing is overwritten.

## Notes

- The upload page takes only a **YouTube URL / local path** and a **route folder**. The
  climbing-condition labels (route orientation, camera angle, shadows, climber/wall
  contrast, motion blur, occlusion, stability, notes) are captured by the Beta Scanner at
  calibration and land in `setup.json` as `analysisInputs`; the correlation pipeline reads
  them from there.
- `ffmpeg` merges best video + audio for YouTube downloads and extracts the final frame;
  `ffprobe` reads width/height/fps/duration for local imports.
- Use only for content you are authorized to download, and comply with local laws and
  platform terms.
