# ADR 0012 — MediaPipe is a second heavyweight dependency, and the harness runs it

- Status: Accepted
- Date: 2026-07-31
- Amends: ADR 0003 (which recorded torch as "the repo's *one* heavyweight exception")

## Context

ADR 0003 accepted `torch` + `transformers` + `ultralytics` against `CLAUDE.md`'s
lean-footprint rule, and named it **the repo's one heavyweight exception**, quarantined
to `POST /api/vitpose` and kept out of the `analysis_pipeline` import graph.

PRD #156 adds a second detection stack. Its reason is not convenience — it is that the
corpus cannot answer the question it exists to answer. Every historical batch varied
scanner build, calibration, truth revision, evaluation schema and video set at once, so
a metric that moved could never be attributed to a cause. Issue #134 then found the
corpus cannot even bound its own noise: applying the constraints a genuine repeat
requires leaves **6 repeat groups** in the whole corpus, and 27 of 33 apparent repeats
were a single detection pass re-exported. The corpus is *observational*, and no amount
of further analysis makes it experimental.

Making it experimental means the harness has to **produce** detection runs rather than
receive them — one factor varied at a time, against frozen truth and a frozen schema,
with repeats as a job parameter so each batch produces its own variance floor. That
requires running the detector here, and the detector the scanner runs is MediaPipe.

So a `requirements.txt` line would silently disprove a sentence in an accepted ADR. The
line is the cheap part; the amendment is the decision.

## Decision

**Add `mediapipe>=1.0` as a second heavyweight exception**, and amend ADR 0003's "one
heavyweight exception" to "the first of two, each recorded".

The quarantine is identical to ViTPose's, because it is the property that matters and
not the count of exceptions:

- The import is **lazy and confined to `MediaPipeDetector`** in `mediapipe_job.py`.
  Nothing else in the repo imports `mediapipe`, at any depth.
- `mediapipe_job.py` is **outside the `analysis_pipeline` import graph**, so
  `python -m analysis_pipeline` runs with mediapipe uninstalled.
- Everything that decides what a run *means* — arm identity, the configuration hash,
  the artifact shape, repeat enumeration, the status sidecar — is pure, and the detector
  sits behind a `Detector` Protocol. `test_mediapipe_job.py` runs its full suite with
  the `mediapipe` import hard-blocked, exactly as `test_vitpose_job.py` runs without
  torch.

**Version floor `>=1.0`, and the Tasks API.** MediaPipe 1.0 removed the legacy
`mp.solutions.pose` API; `PoseLandmarker` plus a downloaded `.task` model bundle is the
only API it ships. That is also the API the browser scanner runs
(`@mediapipe/tasks-vision`), which matters for the parity gate below: with both sides on
the Tasks API, a divergence is attributable to the *runtime* (Python CPU vs browser
WASM) rather than to two different MediaPipe APIs, which would be an uninteresting
confound sitting directly on top of the interesting question.

**The model bundles are downloaded on first use and cached outside the repo**
(`~/.cache/beta-scan-mediapipe`, overridable via `BETA_SCAN_MEDIAPIPE_MODEL_DIR`). Same
pattern as the ViTPose checkpoint and the YOLO weights: `analysis/` is a data record,
not a model store, and the bundles are 6–30 MB binaries.

**The mode → model-bundle mapping is a module-version concern.** `mode` 0/1/2 resolves
to `pose_landmarker_lite`/`full`/`heavy`, so mode alone identifies the weights *given a
module version*. Editing that mapping therefore requires bumping `MODULE_VERSION` —
otherwise two arms built from different weights would share a configuration stamp and
pool as one. This is issue #149's failure mode (a seed hash that omitted model identity
turned a model change into a measured null) stated on the detection side, and it is
recorded here because it is the one factor the hash covers only *indirectly*.

## Consequences

- The repo now carries two independent ML stacks. Install size and cold-start cost grow;
  the `analysis_pipeline` correlation path is unchanged and still lean.
- Installing `mediapipe` moved `opencv-python` from 4.x to 5.0.0 (`mediapipe` depends on
  `opencv-contrib-python`, and pip aligned both). This is a real upgrade to a dependency
  the whole repo uses, not a MediaPipe-local change.
- **A harness run is not automatically a proxy for a scanner run.** The scanner runs
  MediaPipe in the browser over WASM; this runs it in Python. PRD #156 gates every batch
  on a parity check against existing scanner runs, judged against the #134 variance
  floor. This ADR admits the dependency; it does not claim the two runtimes agree. If
  they do not, that is a recorded finding and the experimental design changes.
- The bundle vocabulary gains `mediapipe.status.json`, on the `vitpose.status.json`
  model. See `CONTEXT.md` → **Bundle**.

## Alternatives considered

- **Pin `mediapipe<1.0` for the legacy `solutions.pose` API.** Self-contained (models
  ship in the wheel, no download) and it matches the `model_complexity` vocabulary the
  module's core slice was written against. Rejected: it is a deprecated API on a
  deprecated release line, and it is *not* what the browser runs, so it would put an API
  difference underneath the parity question the whole PRD gates on.
- **Keep detection on the scanner side and vary settings there.** No new dependency, and
  it measures exactly what ships. Rejected for this cycle: every experiment would need a
  cross-repo change and a human-driven browser batch per arm, which is what made the
  existing corpus observational in the first place. The scanner stays untouched (PRD
  #156 out-of-scope); the output of this work is evidence to hand it later.
- **A separate package / optional extra for the experimental module.** Keeps the default
  install lean. Rejected on the same grounds ADR 0003 rejected it: the user chose a
  single `requirements.txt` for operational simplicity, and the exception is recorded
  here instead.
