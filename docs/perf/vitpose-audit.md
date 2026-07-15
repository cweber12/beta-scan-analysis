# ViTPose job performance audit

- Date: 2026-07-15
- Scope: the `POST /api/vitpose` job path — `vitpose_job.py` + its wiring in `app.py`.
  The scanner-side polling loop was audited and cleared (2 s cadence, negligible).
- Tracked as issue #17; optimization implemented on `perf/vitpose-optimization`.

## Symptom

beta-scanner's Ground Truth authoring waits minutes between the detection preview and the
ViTPose landmarks appearing. The scanner POSTs the job, gets a 202, and polls the bundle
every 2 s — so the wait is, within 2 s, the job's own compute time in this repo.

## Measured baseline

| Quantity | Value |
| --- | --- |
| Test video (`get-carter/eYPR7JTRRMk_20260711-144114`) | 53 s @ 24 fps, 720p ≈ 1,270 frames |
| Frames requested by the scanner | 53 |
| Observed job wall time (CPU) | ~60–90 s |
| Hardware present | RTX 4060 Laptop (8 GB), driver CUDA 13.1 |
| torch in venv at audit time | `2.13.0+cpu` — CUDA **unavailable** |

## Findings, ranked by impact

### 1. CPU-only torch despite a capable GPU (environment — the dominant cost)

`requirements.txt`'s `torch>=2.2` resolves to the CPU wheel on Windows. Everything —
~1,270 YOLO inferences and ~53 ViTPose forwards per job — runs on CPU. The code already
selects CUDA when available (`TransformersViTPoseBackend._ensure_model`); the environment
simply never provides it. Expected ~5–15x from installing a `+cu130` build.

### 2. The pose pass re-decodes the entire video, sequentially (`TransformersViTPoseBackend.pose`)

To pose ~53 target frames it `cap.read()`s every frame from 0 to **EOF** — it does not even
stop after the last target. This is also the video's *second* full decode; the tracker
already decoded it once. Fix: sort target indices, seek with `CAP_PROP_POS_FRAMES`, stop
after the last target. ~53 reads instead of ~1,270.

### 3. ViTPose forwards are unbatched (`_pose_one`)

One batch-of-1 forward per target frame, each with its own processor preprocess,
`dataset_index` tensor, and post-process call. The HF processor accepts N images + N boxes
per call. Chunked batching (16) amortizes per-call overhead; large win on GPU.

### 4. Full-rate YOLO tracking dominates the CPU fallback

yolov8n@640 on every frame ≈ 40–75 s per minute of video on CPU. Decision (user-confirmed):
keep full-rate tracking when CUDA is available (quality where it's cheap); apply
`vid_stride` on the CPU fallback only. Striding requires the frame timestamp math to
account for the stride (`idx * stride / fps`) and the Climber-association slack to scale
with it, and the pose pass must seek the *decoded* frame number (`history_index * stride`).

### 5. YOLO device is never pinned

`model.track(...)` relies on ultralytics auto-selection. Pin `device=` explicitly (and use
`half=True` on GPU) so behavior is deterministic and fp16 speed is realized.

### 6. `_ensure_model` is not thread-safe

`app.py` warms the models on a boot thread; a job that arrives during warm-up races it and
can load a second copy of each model (wasted tens of seconds + RAM). Needs a lock
(the job path itself is already serialized by `_vitpose_lock`).

### 7. No observability

Neither the status sidecar nor the logs record where time went, so this audit had to
estimate the track/pose split analytically. Adding `timings` (`track_s`, `pose_s`,
`total_s`) and `device` to the `done` sidecar is contract-safe: the scanner's sidecar
reader (`app/api/dev/corpus/vitpose/route.ts`) only inspects `status` and `error`.

## Minor issues noted, not fixed

- `timestamp = idx / fps` assumes constant frame rate; a VFR video would drift. Mitigated
  by nearest-frame matching; worth revisiting only if VFR sources appear.
- `cap.get(CAP_PROP_FRAME_WIDTH) or 1.0` silently yields garbage coordinates when the video
  is unreadable instead of failing loudly.
- `_nearest_frame_index` is an O(frames × requests) linear scan — negligible at this scale.
- The tracker opens the video once just to read fps, then ultralytics decodes it again —
  negligible.

## Expected outcome

| Path | Baseline (53 s video) | After |
| --- | --- | --- |
| GPU (RTX 4060) | n/a (never used) | ~15–25 s (track ~10–15 s, batched pose ~2–4 s, seeks ~2 s) |
| CPU fallback | ~60–90 s | ~30–50 s (stride 2 + seek + batching) |
