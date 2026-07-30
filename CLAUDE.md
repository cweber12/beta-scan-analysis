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

## Shell environment

This machine is Windows and two shells are available. They are **not**
interchangeable — pick the tool first, then use that tool's syntax:

- **Bash tool** = Git Bash (POSIX `sh`). Heredocs (`<<'EOF'`) are correct here.
- **PowerShell tool** = `pwsh`. Here-strings (`@'…'@`) are correct here.

Never carry one shell's multi-line syntax into the other tool. PowerShell
here-strings used inside the Bash tool have left stray `@` characters in commit
messages on at least three occasions, each needing a `git commit --amend`. This
is a tool-selection rule, not a ban on multi-line strings.

For commit messages and PR bodies specifically, prefer repeated `-m` flags or
`--body-file` — correct in both shells, and it sidesteps the question entirely.

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

- Before starting work, check the current branch and worktree. If the branch is
  not `main` and it has uncommitted changes or no merged PR, stop and tell the
  human before doing any implementation.
- If on the default branch (`main`), create a feature branch before committing.
- End every commit message with a `Co-Authored-By:` trailer naming the model that
  actually authored it — currently:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
  Name the model you are actually running as rather than copying a hardcoded
  string; this line sat stale at a superseded model long enough to put wrong
  attributions in the history. Include any variant suffix the harness specifies.
- Push and open a PR automatically after completing a self-contained task, unless
  there are loose ends to clarify or the human explicitly asks to keep the work
  local. Before pushing, confirm the target branch is not already merged. This
  applies to Codex agents as well as Claude agents.

## Branch, PR & sync flow

The agent pushes and opens PRs, then presents them to the human for review.
**Merge only after the human has reviewed and explicitly confirmed in-session** —
never merge on your own initiative, and never treat an earlier confirmation as
covering a later PR. Once the human confirms, the agent runs the merge
(`gh pr merge <n> --merge`) and immediately follows with the cleanup in rule 4.
Follow this lifecycle so branches and `main` never drift:

1. **Start clean.** Before new work, run all three: `git status -sb`,
   `git branch -a --no-merged main`, and `gh pr list`. A clean working tree is
   **not** evidence that nothing is in flight — unmerged branches and open PRs
   do not appear in `git status`. Never report "nothing in flight" on the
   strength of `git status` alone. The worst case is a branch that is unmerged
   with **no** open PR: invisible to `gh pr list`, holding work `main` does not
   have. If already on a non-`main` branch, confirm whether that branch has an
   open/merged PR and whether the worktree is clean. If it has uncommitted
   changes or has not been merged, report that to the human and wait for
   direction before implementing. Otherwise `git checkout main && git pull`,
   then branch from an up-to-date `main`.
2. **One branch = one PR = one concern.** Don't grow a PR's scope after it's opened
   without flagging it. If unrelated `analysis/` data appears mid-issue, prefer a
   separate branch/PR over appending it to a code PR under review.
3. **A merged branch is frozen.** Never push new commits to a branch whose PR is
   closed/merged — GitHub can't reopen it and the commits get orphaned. New work =
   a fresh branch off updated `main`.
4. **After a merge, sync and clean** by running `python scripts/git_cleanup.py`
   (or the `/cleanup` command): it fast-forwards `main`, deletes every PR-merged
   branch local **and** remote, and prunes. Idempotent and safe — it never touches
   `main` or an unmerged branch (`--dry-run` to preview). The agent runs this
   itself immediately after every confirmed merge, so branches never accumulate.
   After the cleanup, close the related GitHub issue and any dependent PRD / slice
   issues that are now complete, and delete any now-unused local worktrees before
   moving on.
5. **Before pushing, confirm the target branch isn't already merged**
   (`gh pr view <branch>`); if it is, start a fresh branch.

## The working tree is shared with the human — treat it as live

The human works the corpus through the beta-scanner UI **against the files in this
working tree**, while long GPU jobs write into `analysis/` for hours. A branch
operation rewrites those files underneath both. Every one of these rules exists
because ignoring it cost real work:

- **A checkout while the human is working is a data-loss event.** `git checkout`,
  `git_cleanup.py` (which checks out `main`), and anything else that rewrites
  `analysis/` must not run while the scanner is open or a re-seed is in flight. The
  human accepted Ground Truth against scaffolds that a checkout had swapped mid-session,
  and the acceptance recorded empty truth. **Ask before any branch operation once the
  human is working the corpus**, and say plainly that it will rewrite files under them.
- **Never leave a data PR parked.** An unmerged `data:` branch means the working tree
  and the corpus disagree: any branch created from `main` silently reverts the corpus
  to a pre-data state, and the human then reviews against superseded artifacts. Merge
  data PRs as soon as they are confirmed — before starting the next piece of work, not
  after. If a data PR must stay open, **do not create branches from `main`** until it
  lands.
- **Commit before touching branches.** Uncommitted `analysis/` output is the only copy
  of an expensive GPU run. A throwaway holding commit (`wip: …`) before any checkout is
  cheap and has already saved the corpus twice; replay it onto the proper branch
  afterwards.
- **Verify the tree is what you think before measuring.** After any branch operation,
  confirm the artifacts on disk match the branch you believe is checked out — compare a
  known bundle against `git show <branch>:<path>`. A "regression" was reported to the
  human that was purely a checkout artifact, because measurements were taken against a
  tree that had silently reverted.
- **Long-running jobs pin the branch.** While a re-seed or batch is running, stay on
  one branch. Land code fixes by editing the file in place (Python reads the script at
  start, so an in-flight run is unaffected) and commit them afterwards — do not switch
  branches to make the change.

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
- **Prove data-affecting changes against the real corpus.** Green unit tests do
  not demonstrate that a change did anything. For any filter, gate, predicate or
  threshold change, run it over the real corpus and report the before/after
  count of affected records. If it moves zero real records, say so plainly
  rather than reporting green tests — an inert filter has already shipped here
  once on the strength of a passing suite.

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

### Scanner handoffs

The scanner is a separate repo with its own agents; **never push to it**. Work that
lands on the scanner side travels as a handoff doc in `docs/handoffs/`, committed
`docs:` on its own branch and linked from a `ready-for-human` issue — the issue tracks,
the doc specifies. See `docs/agents/scanner-handoffs.md`.
