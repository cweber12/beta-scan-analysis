---
description: Sync main and delete PR-merged branches (local + remote)
---

Run the repository's branch cleanup and report the result:

```
python scripts/git_cleanup.py
```

This is rule 4 of the **Branch, PR & sync flow** in CLAUDE.md. It fast-forwards
`main`, deletes every branch whose PR has merged (local **and** remote), and prunes
stale remote refs. It is safe and idempotent — it never deletes `main` or any
unmerged branch (e.g. `perf/*`).

To preview without changing anything, run `python scripts/git_cleanup.py --dry-run`
first and report what it would delete.
