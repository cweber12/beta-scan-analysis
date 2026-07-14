# ADR 0003 — the downloader runs ViTPose++ to seed beta-scanner's Ground Truth

- Status: Accepted
- Date: 2026-07-13

## Context

beta-scanner's detection-eval harness authors per-video **Ground Truth** poses by
letting a human drag a scaffold skeleton into place on sampled Detection Frames.
Until now that scaffold came from **MediaPipe** — the same detector being graded —
so the human started from a poor, self-referential seed, and untouched frames were
MediaPipe grading itself (the circularity beta-scanner's `docs/adr/0019` flags).

We want the scaffold seeded by **ViTPose++**, a stronger, independent model.
ViTPose is *only a seed*: the human still corrects it and remains the truth
authority. beta-scanner's pipeline is browser-bound and cannot host a PyTorch
model, so the model has to run **here** — the downloader already owns the video
bytes and the bundle. This is a cross-program contract: beta-scanner's
`HARNESS_API_BASE` points at this service and it polls the bundle for the artifact.

Two things make this a real decision rather than a feature bolt-on:

1. **The downloader has never run a model.** `save_detection_run` only *receives*
   pose/orb JSON computed by the scanner. ViTPose is top-down, so this also pulls
   in a person **detector** and a **tracker** (Climber Identity) upstream of it —
   three models where there were zero.
2. **It breaks the lean-footprint rule.** `CLAUDE.md` mandates
   `numpy`/`pandas`/`opencv-python` only. ViTPose needs `torch` + `transformers` +
   `ultralytics`. We accept this as the repo's one heavyweight exception, quarantined
   to the `POST /api/vitpose` path and kept out of the `analysis_pipeline` import
   graph so `python -m analysis_pipeline` stays lean.

## Decision

Expose `POST /api/vitpose`. It accepts a Climber selection (`climber_point`,
`climber_crop`, `wall_crop`, `panning`) + the video's relative path + an explicit
list of `frames[].timestamp` to pose, returns `202` immediately, and runs the job
on a daemon thread. The job: detect + track people (ultralytics YOLO + ByteTrack),
select the Climber track from the tap, run top-down ViTPose++ (HF transformers) on
that track's box for the nearest decoded frame to each requested timestamp, and
write **`vitpose.json`** into the bundle.

Hard contract points, enforced in `vitpose_job.py`:

- **Timestamps are echoed verbatim** (beta-scanner matches within 1 ms); one output
  frame per requested frame.
- **Coordinates are full-frame-normalized `[0, 1]`**, clamped into range.
- **The 13 COCO core joints** carry their exact names; extra COCO points (eyes/ears)
  are allowed as faint context.
- **Per-keypoint confidence** rides in `score`; joints/frames are never thinned —
  an untracked Climber frame emits `keypoints: []` (seeded `absent`).
- **`vitpose.json` is NOT a scored run.** It is written at the bundle root by a
  dedicated writer, never through `save_detection_run` — no `detections/*_pose.json`
  is produced (the issue-07 invariant beta-scanner relies on).

A `vitpose.status.json` sidecar (`running` / `done` / `error`) is written alongside
so a job that crashes *after* the 202 is observable, rather than leaving
beta-scanner polling a file that will never appear.

## Consequences

- Far less human dragging and the MediaPipe self-reference broken, at the cost of
  making this repo an ML-inference service with a heavy dependency set and GPU-class
  runtime for the ViTPose path.
- The bundle vocabulary gains a first-class file: `vitpose.json` (+ its status
  sidecar). See `CONTEXT.md` → **Bundle**.
- The `analysis_pipeline` correlation path is unchanged and still lean.

## Alternatives considered

- **Make ViTPose the truth** — would measure MediaPipe-vs-ViTPose divergence, not
  detection-vs-reality, and ViTPose is unproven on out-of-distribution climbing
  poses. Rejected; the human stays the authority.
- **Optional dependency extra / separate microservice** — keeps the core lean, but
  the user chose a single `requirements.txt` for operational simplicity; recorded
  here as the exception instead.
- **Official mmpose ViTPose** — most faithful to the paper, but `mmcv` is painful to
  build on Windows. Chose HF `transformers` ViTPose + ultralytics for pip-only
  installability.
