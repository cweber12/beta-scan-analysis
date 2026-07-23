# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues in `cweber12/beta-scan-analysis`.
Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## After merge

- Close the merged issue once the work is done; if the work came from a PRD or
	dependent slice issues, close those completed tickets too.
- Run the repository cleanup procedure so merged branches are removed locally and
	remotely.
- Delete no-longer-used local worktrees when the branch they tracked is gone.

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## Idempotent publish workflow (PRD + slices)

When publishing a PRD and issue slices, avoid duplicate tickets by making each step
idempotent:

1. **Preflight lookup before create**
	- Run `gh issue list --state open --search "<exact title> in:title" --json number,title,url`
	  for the PRD title and each planned slice title.
	- If an open issue with the exact title already exists, **reuse it** (edit body/labels
	  if needed) instead of creating a new one.

2. **Use body files for long content**
	- Write markdown to a local temp file and publish with
	  `gh issue create --title "..." --body-file <path>`.
	- This avoids shell truncation/quoting failures that can lead to uncertain outcomes.

3. **After every create call, verify immediately**
	- Capture the returned URL/number.
	- Confirm with `gh issue view <number> --json number,title,state,url` before creating
	  the next issue.

4. **Publish in dependency order, wiring blockers last**
	- Create parent PRD first, then unblocked slices, then dependent slices.
	- If a dependent issue already exists, update its body to reflect current blocker IDs.

5. **If a retry is needed**
	- Re-run preflight lookup first.
	- Never assume failure means nothing was created.

6. **Cleanup and reconciliation**
	- Remove temporary body files after publish.
	- If duplicates were accidentally created, close the redundant issues with a comment
	  pointing to the canonical issue numbers.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
