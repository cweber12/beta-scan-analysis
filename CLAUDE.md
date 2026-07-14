# CLAUDE.md

Guidance for agents working in this repository.

## What this repo is

The **analysis harness** for the Beta Scanner climbing app (the scanner itself is a
separate Next.js repo — see `README.md`). It downloads/imports climbing videos and
pairs them with the scanner's pose/ORB detection diagnostics into self-contained
**analysis bundles**, then correlates video conditions against detection quality.

- `app.py` / `youtube_core.py` — FastAPI service + core logic that builds the bundles.
- `analysis_pipeline/` — reusable correlation pipeline over the bundles.
  Run: `python -m analysis_pipeline analysis -o reports`.
- `analysis/<route>/<video_key>/` — the bundles: `metadata.json`, `setup.json`,
  `final_frame.png`, `detections/<ts>_{pose,orb}.json`. **Video binaries are
  gitignored** (the JSON/PNG record is tracked); `reports/` is gitignored.

## Commit conventions

**Commit after each implementation.** When you finish a self-contained unit of work
(a feature, fix, or refactor), commit it before moving on — don't leave completed work
uncommitted. Group code and its tests together; keep unrelated changes in separate
commits. Use a `feat:` / `fix:` / `chore:` / `refactor:` prefix.

**Commit new analysis data as its own `data:` commit.** Whenever new bundles land under
`analysis/` (a new route folder or a new detection run), commit just those files in a
separate commit prefixed `data:` — e.g. `data: add <route> detection bundle`. Never mix
data bundles with code changes in the same commit. The `.gitignore` already excludes the
video binaries, so `git add analysis/` stages only the queryable JSON/PNG record.

**General git rules** (also in the harness defaults):
- If on the default branch (`main`), create a feature branch before committing.
- End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Only push when explicitly asked.

## Code quality

- Keep the **`analysis_pipeline`** dependency footprint lean: `numpy`, `pandas`,
  `opencv-python` only — no `scipy`/`statsmodels`/`matplotlib` (stats are
  hand-rolled; charts are inline SVG). The one sanctioned exception is the ViTPose
  Ground Truth scaffold (`POST /api/vitpose`), which pulls `torch`/`transformers`/
  `ultralytics`; it is quarantined to `vitpose_job.py` and kept out of the
  `analysis_pipeline` import graph. See `docs/adr/0003`.
- Run the smoke tests after touching the pipeline:
  `python -m analysis_pipeline.tests.test_smoke`. After touching the ViTPose
  scaffold, run `python test_vitpose_job.py` (stub-backed; no torch needed).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`cweber12/beta-scan-analysis`), managed via the `gh`
CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical five-role vocabulary — `needs-triage` / `needs-info` / `ready-for-agent` /
`ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See
`docs/agents/domain.md`.
