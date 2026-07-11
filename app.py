from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from youtube_core import (
    build_analysis_bundle,
    download_video,
    generate_timestamp,
    import_local_video,
    save_detection_run,
)


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis"

app = FastAPI(title="Climb Video Analyzer")


class AnalysisMetadata(BaseModel):
    route_folder: str = Field(..., min_length=1)
    route_orientation: str = Field(default="unknown")
    camera_angle: str = Field(default="unknown")
    shadows: str = Field(default="unknown")
    climber_contrast: str = Field(default="unknown")
    wall_contrast: str = Field(default="unknown")
    motion_blur: str = Field(default="unknown")
    occlusion: str = Field(default="unknown")
    camera_stability: str = Field(default="unknown")
    notes: str = Field(default="")


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


def list_route_folders() -> list[str]:
    routes: set[str] = set()
    if ANALYSIS_DIR.exists():
        for child in ANALYSIS_DIR.iterdir():
            if child.is_dir() and child.name.strip():
                routes.add(child.name)
    return sorted(routes)


def render_homepage() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Climb Video Analyzer</title>
  <style>
    :root {
      --bg: #0a0e14;
      --bg-grid: rgba(56, 189, 197, 0.04);
      --panel: #10151f;
      --panel-2: #0d121b;
      --border: #1e2733;
      --border-strong: #2a3644;
      --text: #c6d0dc;
      --text-dim: #7c8a9c;
      --text-faint: #55627180;
      --accent: #38e1c7;
      --accent-dim: #1f8f81;
      --accent-glow: rgba(56, 225, 199, 0.18);
      --danger: #ff6b6b;
      --mono: "SFMono-Regular", "JetBrains Mono", "Consolas", ui-monospace, monospace;
      --sans: "Inter", -apple-system, system-ui, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    html { color-scheme: dark; }

    body {
      margin: 0;
      min-height: 100vh;
      padding: 32px 20px 56px;
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.5;
      background-color: var(--bg);
      background-image:
        linear-gradient(var(--bg-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px),
        radial-gradient(1200px 600px at 78% -10%, rgba(56, 225, 199, 0.06), transparent 60%);
      background-size: 40px 40px, 40px 40px, auto;
    }

    .shell {
      max-width: 960px;
      margin: 0 auto;
    }

    header {
      display: flex;
      align-items: baseline;
      gap: 14px;
      padding-bottom: 16px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }

    .logo {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      font-family: var(--mono);
      font-size: 1.15rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      color: #eef4fa;
    }

    .logo .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 10px 1px var(--accent-glow);
    }

    .tagline {
      font-family: var(--mono);
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-dim);
    }

    .panel {
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 22px;
    }

    fieldset {
      border: none;
      margin: 0;
      padding: 0;
    }

    fieldset + fieldset {
      margin-top: 22px;
      padding-top: 22px;
      border-top: 1px solid var(--border);
    }

    legend {
      display: block;
      width: 100%;
      padding: 0;
      margin-bottom: 14px;
      font-family: var(--mono);
      font-size: 0.68rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent);
    }

    legend .idx { color: var(--text-faint); margin-right: 8px; }

    form { display: block; }

    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .row-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .row + .row,
    .row + .row-3,
    .row-3 + .row,
    .row-3 + .row-3,
    .row + .local-only,
    .local-only + .row {
      margin-top: 14px;
    }

    label {
      display: grid;
      gap: 6px;
      font-family: var(--mono);
      font-size: 0.68rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-dim);
    }

    input, select {
      font-family: var(--sans);
      font-size: 0.9rem;
      letter-spacing: normal;
      text-transform: none;
      color: var(--text);
      padding: 9px 11px;
      background: #0b1017;
      border: 1px solid var(--border-strong);
      border-radius: 6px;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }

    input::placeholder { color: var(--text-faint); }

    input:hover, select:hover { border-color: #384a5c; }

    input:focus, select:focus {
      border-color: var(--accent-dim);
      box-shadow: 0 0 0 3px var(--accent-glow);
      background: #0c131c;
    }

    select {
      appearance: none;
      background-image:
        linear-gradient(45deg, transparent 50%, var(--text-dim) 50%),
        linear-gradient(135deg, var(--text-dim) 50%, transparent 50%);
      background-position:
        calc(100% - 16px) calc(50% - 2px),
        calc(100% - 11px) calc(50% - 2px);
      background-size: 5px 5px, 5px 5px;
      background-repeat: no-repeat;
      padding-right: 30px;
    }

    option { background: #0b1017; color: var(--text); }

    .actions {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-top: 24px;
      padding-top: 22px;
      border-top: 1px solid var(--border);
    }

    button {
      font-family: var(--mono);
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #04110e;
      padding: 11px 26px;
      background: var(--accent);
      border: 1px solid var(--accent);
      border-radius: 6px;
      cursor: pointer;
      transition: box-shadow 0.15s ease, transform 0.05s ease, background 0.15s ease;
    }

    button:hover {
      box-shadow: 0 0 18px -2px var(--accent-glow);
      background: #4dead2;
    }

    button:active { transform: translateY(1px); }

    button:disabled {
      cursor: not-allowed;
      color: var(--text-dim);
      background: #161d28;
      border-color: var(--border-strong);
      box-shadow: none;
    }

    #status {
      font-family: var(--mono);
      font-size: 0.76rem;
      letter-spacing: 0.06em;
      color: var(--text-dim);
      min-height: 1.2em;
    }

    #status::before {
      content: "\\203A ";
      color: var(--accent);
    }

    #status.is-error { color: var(--danger); }
    #status.is-error::before { color: var(--danger); }
    #status.is-ok { color: var(--accent); }

    .output {
      margin-top: 22px;
    }

    .output-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 9px 14px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-bottom: none;
      border-radius: 8px 8px 0 0;
      font-family: var(--mono);
      font-size: 0.66rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-dim);
    }

    .dots { display: inline-flex; gap: 6px; }
    .dots i {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--border-strong);
      display: inline-block;
    }

    pre {
      margin: 0;
      padding: 16px;
      background: #080b11;
      border: 1px solid var(--border);
      border-radius: 0 0 8px 8px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 0.8rem;
      line-height: 1.55;
      color: #9fb2c4;
      max-height: 420px;
      overflow: auto;
    }

    @media (max-width: 720px) {
      .row,
      .row-3 {
        grid-template-columns: 1fr;
      }
      header { flex-direction: column; align-items: flex-start; gap: 6px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <span class="logo"><span class="dot"></span>Climb Video Analyzer</span>
      <span class="tagline">Detection bundle intake</span>
    </header>

    <form id="download-form" class="panel">
      <fieldset>
        <legend><span class="idx">01</span>Source</legend>
        <div class="row">
          <label>
            Source
            <select id="source_type" name="source_type">
              <option value="youtube" selected>YouTube URL</option>
              <option value="local">Local file</option>
            </select>
          </label>
          <label>
            Route folder
            <input id="route_folder" name="route_folder" list="route-options" required />
            <datalist id="route-options"></datalist>
          </label>
        </div>

        <div class="row youtube-only">
          <label>
            URL
            <input id="url" name="url" placeholder="https://youtu.be/..." />
          </label>
          <label>
            Resolution
            <select id="resolution" name="resolution">
              <option value="480">480</option>
              <option value="720" selected>720</option>
              <option value="1080">1080</option>
              <option value="1440">1440</option>
            </select>
          </label>
        </div>

        <div class="local-only" hidden>
          <label>
            Local file path
            <input id="local_path" name="local_path"
                   placeholder="downloads/Midnight_Lightning_V8.mp4" />
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend><span class="idx">02</span>Capture conditions</legend>
        <div class="row">
          <label>
            Route orientation
            <select id="route_orientation" name="route_orientation">
              <option value="unknown" selected>unknown</option>
              <option value="left">left</option>
              <option value="right">right</option>
              <option value="head-on">head-on</option>
            </select>
          </label>
        </div>

        <div class="row-3">
          <label>
            Camera angle
            <select id="camera_angle" name="camera_angle">
              <option value="unknown" selected>unknown</option>
              <option value="low">low</option>
              <option value="level">level</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            Shadows
            <select id="shadows" name="shadows">
              <option value="unknown" selected>unknown</option>
              <option value="none">none</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            Climber contrast
            <select id="climber_contrast" name="climber_contrast">
              <option value="unknown" selected>unknown</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
        </div>

        <div class="row-3">
          <label>
            Wall contrast
            <select id="wall_contrast" name="wall_contrast">
              <option value="unknown" selected>unknown</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            Motion blur
            <select id="motion_blur" name="motion_blur">
              <option value="unknown" selected>unknown</option>
              <option value="none">none</option>
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
          <label>
            Occlusion
            <select id="occlusion" name="occlusion">
              <option value="unknown" selected>unknown</option>
              <option value="none">none</option>
              <option value="some">some</option>
              <option value="heavy">heavy</option>
            </select>
          </label>
        </div>

        <div class="row">
          <label>
            Camera stability
            <select id="camera_stability" name="camera_stability">
              <option value="unknown" selected>unknown</option>
              <option value="steady">steady</option>
              <option value="some-shake">some-shake</option>
              <option value="moving">moving</option>
            </select>
          </label>
          <label>
            Notes
            <input id="notes" name="notes" />
          </label>
        </div>
      </fieldset>

      <div class="actions">
        <button id="submit-btn" type="submit">Build bundle</button>
        <div id="status"></div>
      </div>
    </form>

    <div class="output">
      <div class="output-head">
        <span>Response</span>
        <span class="dots"><i></i><i></i><i></i></span>
      </div>
      <pre id="result-json">{}</pre>
    </div>
  </div>

  <script>
    const form = document.getElementById('download-form');
    const submitBtn = document.getElementById('submit-btn');
    const status = document.getElementById('status');
    const resultJson = document.getElementById('result-json');
    const routeFolderInput = document.getElementById('route_folder');
    const routeOptions = document.getElementById('route-options');
    const sourceType = document.getElementById('source_type');
    const youtubeOnly = document.querySelector('.youtube-only');
    const localOnly = document.querySelector('.local-only');

    function syncSourceFields() {
      const isLocal = sourceType.value === 'local';
      youtubeOnly.hidden = isLocal;
      localOnly.hidden = !isLocal;
    }

    sourceType.addEventListener('change', syncSourceFields);
    syncSourceFields();

    async function loadRouteOptions() {
      try {
        const response = await fetch('/api/routes');
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Could not load routes');
        }

        routeOptions.innerHTML = '';
        data.routes.forEach((route) => {
          const option = document.createElement('option');
          option.value = route;
          routeOptions.appendChild(option);
        });
      } catch {
        // Keep form usable even if route list cannot be fetched.
      }
    }

    loadRouteOptions();

    form.addEventListener('submit', async (event) => {
      event.preventDefault();

      function setStatus(text, state) {
        status.textContent = text;
        status.classList.toggle('is-error', state === 'error');
        status.classList.toggle('is-ok', state === 'ok');
      }

      const payload = Object.fromEntries(new FormData(form).entries());
      payload.route_folder = String(payload.route_folder || '').trim();

      if (!payload.route_folder) {
        setStatus('Route folder is required.', 'error');
        routeFolderInput.focus();
        return;
      }

      const isLocal = payload.source_type === 'local';
      const endpoint = isLocal ? '/api/import' : '/api/download';

      if (isLocal) {
        payload.local_path = String(payload.local_path || '').trim();
        if (!payload.local_path) {
          setStatus('Local file path is required.', 'error');
          return;
        }
        delete payload.url;
        delete payload.resolution;
      } else {
        payload.url = String(payload.url || '').trim();
        if (!payload.url) {
          setStatus('URL is required.', 'error');
          return;
        }
        payload.resolution = Number(payload.resolution);
        delete payload.local_path;
      }
      delete payload.source_type;

      submitBtn.disabled = true;
      setStatus('Working...', null);

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }

        setStatus('Complete', 'ok');
        resultJson.textContent = JSON.stringify(data, null, 2);
        loadRouteOptions();
      } catch (error) {
        setStatus(`Failed: ${error.message}`, 'error');
        resultJson.textContent = JSON.stringify({ error: error.message }, null, 2);
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def homepage() -> str:
    return render_homepage()


@app.get("/api/routes")
def get_routes() -> dict[str, list[str]]:
  return {"routes": list_route_folders()}


def _analysis_inputs(payload: AnalysisMetadata, **extra: object) -> dict[str, object]:
    return {
        "route_folder": payload.route_folder,
        "route_orientation": payload.route_orientation,
        "camera_angle": payload.camera_angle,
        "shadows": payload.shadows,
        "climber_contrast": payload.climber_contrast,
        "wall_contrast": payload.wall_contrast,
        "motion_blur": payload.motion_blur,
        "occlusion": payload.occlusion,
        "camera_stability": payload.camera_stability,
        "notes": payload.notes,
        **extra,
    }


def _bundle_response(download_result, user_metadata: dict[str, object]) -> dict[str, object]:
    bundle = build_analysis_bundle(download_result, ANALYSIS_DIR, user_metadata)
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
        "analysis_inputs": bundle["metadata"]["analysis_inputs"],
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
            _analysis_inputs(payload, requested_resolution=payload.resolution),
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
        return _bundle_response(
            download_result,
            _analysis_inputs(payload, imported_from=payload.local_path),
        )
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
