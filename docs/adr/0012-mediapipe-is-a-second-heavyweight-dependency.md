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

**The model bundles are pinned locally, and nothing is fetched at run time.** MediaPipe
publishes its bundles at a **`latest`** URL. Resolving that during a job would mean the
weights behind an arm could change without anything in the repo moving — and since the
arm's `configHash` is what makes two experimental runs comparable, two arms built from
different weights would share a stamp and pool as one. That is issue #149 verbatim: a
hash that omitted model identity turned a model change into a measured null.

So:

- `models/mediapipe.lock.json` is the **tracked record**, pinning each bundle's sha256,
  size, source URL and pin date. The `.task` binaries are **gitignored** — the rule the
  repo already applies to video binaries, `*.pt` weights and `downloads/`.
- `scripts/fetch_mediapipe_models.py` fetches and verifies against the lock. It
  **refuses** a bundle whose sha256 has moved rather than adopting it; `--update` is the
  deliberate act of re-pinning, and `--check` verifies without writing.
- At run time `mediapipe_job` reads the local file, verifies it against the lock, and
  fails loudly if it is missing or altered. There is no download path in the module at
  all, and a test asserts that (`urlopen` must not appear in it). A batch must not depend
  on a network round trip, and an arm must not depend on what upstream was serving that
  afternoon.
- **The pinned sha256 joins the arm identity** (`DetectionConfig.model_sha`), derived by
  the job from the detector it is about to build rather than supplied by a caller — the
  discipline `vitpose_job.stamp_model_identity` already applies. Adopting new weights
  therefore changes every arm's `configHash` automatically, which correctly makes runs
  from before and after non-comparable instead of silently pooling them.

Drift is detected by CI polling upstream and **notifying**, never by adopting silently.

**The mode → model-bundle mapping remains a module-version concern.** `mode` 0/1/2
resolves to `pose_landmarker_lite`/`full`/`heavy`. The *contents* of each bundle are now
covered by `model_sha`, but editing the mapping itself — pointing a mode at a different
bundle — still requires bumping `MODULE_VERSION`.

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
- A fresh clone needs one `python scripts/fetch_mediapipe_models.py` before the module
  can run — the same shape as needing the video binaries, which are also not in git.
- Re-pinning the models is a **corpus event, not a maintenance chore**. Every arm's
  `configHash` moves, so runs made before and after cannot be pooled. Prefer re-pinning
  between experimental cycles rather than inside one, for the same reason ADR 0009
  freezes the evaluation schema for a cycle.

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
