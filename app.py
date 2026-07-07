from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from youtube_core import build_analysis_bundle, download_video, generate_timestamp


BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
ANALYSIS_DIR = BASE_DIR / "analysis"

app = FastAPI(title="Climb Video Analyzer")


class DownloadRequest(BaseModel):
    url: str = Field(..., min_length=5)
    resolution: int = Field(default=720, ge=144, le=4320)
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


def list_route_folders() -> list[str]:
    routes: set[str] = set()
    for base_dir in (DOWNLOADS_DIR, ANALYSIS_DIR):
        if not base_dir.exists():
            continue
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.strip():
                routes.add(child.name)
    return sorted(routes)


def render_homepage() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Video Upload</title>
  <style>
    body {
      margin: 16px;
      font-family: sans-serif;
      max-width: 920px;
    }

    h1 {
      margin: 0 0 12px;
      font-size: 1.4rem;
    }

    form {
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }

    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .row-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    label {
      display: grid;
      gap: 4px;
      font-size: 0.9rem;
    }

    input, select, button {
      font: inherit;
      padding: 6px 8px;
    }

    #status {
      margin-bottom: 8px;
      font-size: 0.95rem;
    }

    pre {
      margin: 0;
      border: 1px solid #ccc;
      padding: 10px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.84rem;
    }

    @media (max-width: 720px) {
      .row,
      .row-3 {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <h1>Video Upload</h1>

  <form id="download-form">
    <label>
      URL
      <input id="url" name="url" placeholder="https://youtu.be/..." required />
    </label>

    <div class="row">
      <label>
        Resolution
        <select id="resolution" name="resolution">
          <option value="480">480</option>
          <option value="720" selected>720</option>
          <option value="1080">1080</option>
          <option value="1440">1440</option>
        </select>
      </label>
      <label>
        Route folder
        <input id="route_folder" name="route_folder" list="route-options" required />
        <datalist id="route-options"></datalist>
      </label>
    </div>

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

    <button id="submit-btn" type="submit">Submit</button>
  </form>

  <div id="status"></div>
  <pre id="result-json">{}</pre>

  <script>
    const form = document.getElementById('download-form');
    const submitBtn = document.getElementById('submit-btn');
    const status = document.getElementById('status');
    const resultJson = document.getElementById('result-json');
    const routeFolderInput = document.getElementById('route_folder');
    const routeOptions = document.getElementById('route-options');

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

      const payload = Object.fromEntries(new FormData(form).entries());
      payload.resolution = Number(payload.resolution);
      payload.route_folder = String(payload.route_folder || '').trim();

      if (!payload.route_folder) {
        status.textContent = 'Route folder is required.';
        routeFolderInput.focus();
        return;
      }

      submitBtn.disabled = true;
      status.textContent = 'Working...';

      try {
        const response = await fetch('/api/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Request failed');
        }

        status.textContent = 'Complete';
        resultJson.textContent = JSON.stringify(data, null, 2);
        loadRouteOptions();
      } catch (error) {
        status.textContent = `Failed: ${error.message}`;
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


@app.post("/api/download")
def create_download_bundle(payload: DownloadRequest) -> dict[str, object]:
    try:
        timestamp = generate_timestamp()
        download_result = download_video(
            payload.url,
            DOWNLOADS_DIR,
            payload.resolution,
            route_folder=payload.route_folder,
            timestamp=timestamp,
        )
        bundle = build_analysis_bundle(
            download_result,
            ANALYSIS_DIR,
            {
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
                "requested_resolution": payload.resolution,
            },
        )

        return {
            "timestamp": download_result.timestamp,
            "route_folder": download_result.route_folder,
            "download_path": str(download_result.video_path),
            "analysis_dir": str(bundle["route_dir"]),
            "metadata_path": str(bundle["metadata_path"]),
            "frame_path": str(bundle["frame_path"]),
            "source_title": bundle["metadata"]["source_video"].get("title"),
            "source_video_id": bundle["metadata"]["source_video"].get("video_id"),
            "analysis_inputs": bundle["metadata"]["analysis_inputs"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
