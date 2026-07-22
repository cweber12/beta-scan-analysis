#!/usr/bin/env python3
"""Sync the default branch and delete branches fully merged into it.

Safe + idempotent: a branch is deleted only when *every* commit on it is already
contained in ``origin/<default>`` (it is an ancestor). That never loses unmerged
work — an in-progress ``perf/*``, or a branch reused after an earlier PR merged,
both keep unique commits and are left alone. Run it after a PR merges, directly or
via the ``/cleanup`` command. This is rule 4 of the "Branch, PR & sync flow" in
CLAUDE.md.

Only ordinary merge-commit merges (this repo's default) leave the branch as an
ancestor. A squash- or cherry-pick-merged branch keeps non-ancestor commits by
design and is intentionally *not* auto-deleted — delete those by hand once you have
confirmed the PR merged.

Usage: python scripts/git_cleanup.py [--default main] [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str], check: bool = True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def git(*args: str, check: bool = True) -> str:
    return _run(["git", *args], check=check).stdout.strip()


def merged_branches(upstream: str, *, remote: bool) -> list[str]:
    """Branch names whose tip is an ancestor of ``upstream`` (fully merged)."""

    args = ["branch", "--merged", upstream] + (["-r"] if remote else [])
    names = []
    for line in git(*args).splitlines():
        name = line.replace("*", "").replace("+", "").strip()
        if not name or " -> " in name:  # skip 'origin/HEAD -> origin/main'
            continue
        names.append(name)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--default", default="main", help="default branch (default: main)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted; change nothing")
    args = ap.parse_args()
    default = args.default
    upstream = f"origin/{default}"

    git("fetch", "--prune", "origin")

    # Real run: stand on the default branch first, so a just-merged branch you are
    # sitting on is no longer checked out and can be deleted. Fast-forward only — a
    # diverged default is a real problem to surface, not paper over.
    if not args.dry_run:
        git("switch", default)
        _run(["git", "pull", "--ff-only"], check=False)

    local = [b for b in merged_branches(upstream, remote=False) if b != default]
    remote = [r for r in merged_branches(upstream, remote=True)
              if not r.endswith(f"/{default}")]

    for b in local:
        if not args.dry_run:
            # -D not -d: the ancestor check already proved it is fully merged, and
            # -d trips over the upstream-tracking comparison for merge-commit merges.
            git("branch", "-D", b)
    for r in remote:
        name = r.split("/", 1)[1] if "/" in r else r
        if not args.dry_run:
            git("push", "origin", "--delete", name)

    if not args.dry_run:
        git("remote", "prune", "origin")

    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} local:  {', '.join(local) or '(none)'}")
    print(f"{verb} remote: {', '.join(r.split('/', 1)[1] for r in remote) or '(none)'}")
    if not args.dry_run:
        print(f"synced {default}; pruned origin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
