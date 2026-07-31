"""Fetch and pin the MediaPipe Pose Landmarker model bundles (PRD #156, ADR 0012).

MediaPipe publishes its bundles at a **`latest`** URL. Resolving that at run time would
mean the weights behind an arm could change without anything in the repo moving — and
because the arm's `configHash` is what makes two experimental runs comparable, two arms
built from different weights would share a stamp and pool as one. That is issue #149
exactly: a hash that omitted model identity turned a model change into a measured null.

So the bundles are **fetched once and pinned**. `models/mediapipe.lock.json` records each
bundle's sha256, and it is the tracked record; the `.task` binaries themselves are
gitignored, the same rule the repo already applies to video binaries, `*.pt` weights and
`downloads/`. At run time `mediapipe_job` reads the local file and verifies it against the
lock — no network, and a swapped or truncated bundle fails loudly instead of quietly
producing runs nobody can attribute.

    python scripts/fetch_mediapipe_models.py            # fetch what the lock pins, verify
    python scripts/fetch_mediapipe_models.py --check    # verify only; never write
    python scripts/fetch_mediapipe_models.py --update   # re-pin to upstream latest

`--update` is the deliberate act of adopting new weights: it rewrites the lock, which
changes every arm's `configHash`, which means runs made before and after are *not*
comparable. That is the point — it makes a model change as visible as a code change.
Drift is detected by CI polling upstream and notifying; it is never adopted silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "models" / "mediapipe.lock.json"
MODEL_DIR = REPO_ROOT / "models" / "mediapipe"

BUNDLES = ("pose_landmarker_lite", "pose_landmarker_full", "pose_landmarker_heavy")
URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "{name}/float16/latest/{name}.task"
)
LOCK_VERSION = 1


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path = LOCK_PATH) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def download(name: str) -> bytes:
    url = URL_TEMPLATE.format(name=name)
    print(f"  fetching {url}")
    with urlopen(url, timeout=600) as response:
        return response.read()


def write_lock(entries: dict[str, dict], path: Path = LOCK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": LOCK_VERSION,
                "note": (
                    "Pinned MediaPipe Pose Landmarker bundles. sha256 joins every "
                    "experimental arm's configHash, so adopting new weights (via "
                    "--update) deliberately makes prior runs non-comparable. See "
                    "docs/adr/0012 and scripts/fetch_mediapipe_models.py."
                ),
                "urlTemplate": URL_TEMPLATE,
                "models": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def fetch(update: bool, check_only: bool) -> int:
    lock = load_lock()
    pinned = lock.get("models") or {}
    if not pinned and not update:
        print("No lock file. Run with --update to pin the current upstream bundles.",
              file=sys.stderr)
        return 2

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    failures: list[str] = []

    for name in BUNDLES:
        target = MODEL_DIR / f"{name}.task"
        expected = (pinned.get(name) or {}).get("sha256")

        if update:
            blob = download(name)
            digest = hashlib.sha256(blob).hexdigest()
            if expected and digest != expected:
                print(f"  {name}: upstream MOVED {expected[:16]} -> {digest[:16]}")
            target.write_bytes(blob)
            entries[name] = {
                "sha256": digest,
                "size": len(blob),
                "url": URL_TEMPLATE.format(name=name),
                "pinnedAt": date.today().isoformat(),
            }
            print(f"  {name}: pinned {digest[:16]} ({len(blob)} bytes)")
            continue

        entries[name] = dict(pinned[name]) if name in pinned else {}
        if target.is_file() and sha256_of(target) == expected:
            print(f"  {name}: ok ({expected[:16]})")
            continue
        if check_only:
            state = "missing" if not target.is_file() else "sha256 mismatch"
            print(f"  {name}: {state}")
            failures.append(name)
            continue

        blob = download(name)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected:
            # Refuse rather than adopt. An unpinned bundle silently entering a batch is
            # the failure this whole file exists to prevent.
            print(
                f"  {name}: REFUSED — upstream sha256 {digest[:16]} does not match the "
                f"pinned {str(expected)[:16]}. Upstream has republished the bundle. "
                f"Adopt it deliberately with --update (this makes prior runs "
                f"non-comparable), or keep the pin.",
                file=sys.stderr,
            )
            failures.append(name)
            continue
        target.write_bytes(blob)
        print(f"  {name}: fetched {digest[:16]} ({len(blob)} bytes)")

    if update:
        write_lock(entries)
        print(f"wrote {LOCK_PATH.relative_to(REPO_ROOT)}")
    if failures:
        print(f"\n{len(failures)} bundle(s) unresolved: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true",
                       help="re-pin to upstream latest, rewriting the lock file")
    group.add_argument("--check", action="store_true",
                       help="verify local bundles against the lock; download nothing")
    args = parser.parse_args(argv)
    return fetch(update=args.update, check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
