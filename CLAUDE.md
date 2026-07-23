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
If there are existing `analysis/` changes in the current worktree while working an
issue, commit them on the current issue branch as part of that issue's work — but only
while that branch is still open and the data fits its scope (see **Branch, PR & sync
flow**). Keep the data in its own `data:` commit, and don't leave it uncommitted when
you push the branch. If the branch's PR has already merged, or the data is a distinct
concern, put it on a fresh branch instead.

**General git rules** (also in the harness defaults):
- If on the default branch (`main`), create a feature branch before committing.
- End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Only push when explicitly asked.

## Branch, PR & sync flow

The agent pushes and opens PRs; **only the human merges**. Follow this lifecycle so
branches and `main` never drift:

1. **Start clean.** Before new work: `git checkout main && git pull`, then branch
   from an up-to-date `main`.
2. **One branch = one PR = one concern.** Don't grow a PR's scope after it's opened
   without flagging it. If unrelated `analysis/` data appears mid-issue, prefer a
   separate branch/PR over appending it to a code PR under review.
3. **A merged branch is frozen.** Never push new commits to a branch whose PR is
   closed/merged — GitHub can't reopen it and the commits get orphaned. New work =
   a fresh branch off updated `main`.
4. **After a merge, sync and clean** by running `python scripts/git_cleanup.py`
   (or the `/cleanup` command): it fast-forwards `main`, deletes every PR-merged
   branch local **and** remote, and prunes. Idempotent and safe — it never touches
   `main` or an unmerged branch (`--dry-run` to preview). Do this every time a PR
   merges, so branches never accumulate.
   After the cleanup, close the related GitHub issue and any dependent PRD / slice
   issues that are now complete, and delete any now-unused local worktrees before
   moving on.
5. **Before pushing, confirm the target branch isn't already merged**
   (`gh pr view <branch>`); if it is, start a fresh branch.

## Code quality

- Prefer a lean **`analysis_pipeline`** footprint: `numpy`, `pandas`,
  `opencv-python`, with stats hand-rolled and charts as inline SVG. This is now a
  *preference, not a hard rule* — the pipeline is local-only, so pulling a well-known
  dependency (`scipy`/`statsmodels`/`matplotlib`) is a judgement call, not forbidden.
  Reach for one only when hand-rolling would be materially worse; keep the default
  lean. The v1 `evaluate` subcommand (`evaluate.py`, issue #6) stays numpy-only *by
  fit* — the PCK math is trivial — not by policy. The ViTPose Ground Truth scaffold
  (`POST /api/vitpose`) remains a deliberately quarantined heavy exception: it pulls
  `torch`/`transformers`/`ultralytics`, lives in `vitpose_job.py`, and is kept out of
  the `analysis_pipeline` import graph. See `docs/adr/0003`.
- Run the smoke tests after touching the pipeline:
  `python -m analysis_pipeline.tests.test_smoke`. After touching the ViTPose
  scaffold, run `python test_vitpose_job.py` (stub-backed; no torch needed).
  After touching the Video Stats core (`video_stats.py`) run
  `python -m tests.test_video_stats`, and after touching `app.py` run
  `python -m tests.test_api` (both dependency-free beyond numpy/cv2).

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
