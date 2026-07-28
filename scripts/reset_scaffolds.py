"""Re-seed every bundle's ViTPose scaffold under the ADR 0007 contract (issue #101).

The corpus reset's expensive leg. For each bundle it POSTs ``/api/vitpose`` with the
seed inputs read from that bundle's own ``setup.json``, then polls the status sidecar
until the job reaches a terminal state.

**Resumable, and cheap to re-run.** The harness stamps a seed hash into every scaffold
and skips a request whose seed is unchanged (issue #101 idempotence), so re-running this
script after an interruption re-does only what is actually outstanding. That is what
makes a multi-hour reset safe to start: Ctrl-C costs you the job in flight, nothing more.

**Dry run by default.** ``--apply`` is required to send anything.

    python scripts/reset_scaffolds.py                 # plan only
    python scripts/reset_scaffolds.py --apply         # do it
    python scripts/reset_scaffolds.py --apply --force # re-seed even unchanged seeds

The seed inputs, per ADR 0006/0007:

- ``seed_tap``    — ``setup.json.seedTap`` when present, else the setup tap
  (``climberPoint``). The two are separate values now; a bundle that has never been
  re-seeded has only the setup tap, and seeding from it is correct.
- ``seed_region`` — ``setup.json.seedRegion`` **only**. No bundle writes one today, so
  in practice the seed gate is left open and the tap alone anchors identity. Falling
  back to ``climberCrop`` is wrong and was actively breaking seeding: ADR 0006 decoupled
  the seed gate from the Climber Crop because the crop is a Video Stats input, and ADR
  0007 made it worse by letting the seed tap move mid-climb. The crop is drawn around
  the Climber at *setup* time; a re-seed tap is where they are *now*, often well above
  it. Measured: 7 of the 16 re-tapped bundles have a seed tap outside their own Climber
  Crop, and gating on it rejected every candidate (``seedFailureReason: region-gated``).
- the **climb window** is deliberately *not* sent: the service resolves it from the
  bundle's calibration (``resolve_climb_window``), which is the single source of truth
  and avoids this script disagreeing with it.
- ``frames``      — the analysis grid, bounded to the climb window. Frames outside the
  window can contribute no evidence, so posing them is waste; frames off the grid the
  truth is exported on are what fabricated the ``not-sampled`` absences this reset exists
  to remove.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# The scanner's analysis timeline: 100 ms. Truth is exported onto this grid, so the
# scaffold must be sampled on it too — a coarser scaffold is the rate mismatch that
# fabricated 6,189 "absent" frames on the pre-reset corpus.
FRAME_INTERVAL_SEC = 0.1
POLL_INTERVAL_SEC = 3.0
TERMINAL = {"done", "error", "skipped"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _load(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _video_for(bundle: Path) -> Path | None:
    canonical = bundle / f"{bundle.name}.mp4"
    if canonical.is_file():
        return canonical
    for path in sorted(bundle.iterdir()):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _climb_window(setup: dict[str, Any]) -> tuple[float | None, float | None]:
    """Mirrors ``vitpose_job.resolve_climb_window`` — for planning the frame list only.

    The service resolves the window itself; this is here so the script can decide which
    timestamps to request and report the plan honestly.
    """

    start = _num(setup.get("climbStart"))
    if start is None:
        tap = setup.get("climberPoint")
        start = _num(tap.get("t")) if isinstance(tap, dict) else None
    return start, _num(setup.get("climbEnd"))


def _duration_sec(bundle: Path) -> float | None:
    """Video duration from the bundle's recorded source facts (no decode)."""

    metadata = _load(bundle / "metadata.json")
    source = metadata.get("source_video")
    if isinstance(source, dict):
        for key in ("duration_seconds", "duration"):
            value = _num(source.get(key))
            if value:
                return value
    stats = metadata.get("video_stats")
    if isinstance(stats, dict):
        value = _num(stats.get("durationSec"))
        if value:
            return value
    return None


def _frames_for(bundle: Path, setup: dict[str, Any]) -> list[float]:
    start, end = _climb_window(setup)
    if end is None:
        end = _duration_sec(bundle)
    if end is None:
        return []
    begin = start or 0.0
    n = int(round((end - begin) / FRAME_INTERVAL_SEC))
    return [round(begin + i * FRAME_INTERVAL_SEC, 3) for i in range(max(0, n) + 1)]


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        return exc.code, {"error": detail}


def _plan(bundle: Path) -> dict[str, Any] | None:
    setup = _load(bundle / "setup.json")
    if not setup:
        return {"bundle": bundle, "skip": "no setup.json"}
    video = _video_for(bundle)
    if video is None:
        return {"bundle": bundle, "skip": "no video binary"}

    tap = setup.get("seedTap") if isinstance(setup.get("seedTap"), dict) else None
    tap = tap or (setup.get("climberPoint") if isinstance(setup.get("climberPoint"), dict) else None)
    if tap is None:
        return {"bundle": bundle, "skip": "no seed tap or setup tap"}
    # Deliberately NOT falling back to climberCrop — see the module docstring.
    region = setup.get("seedRegion") if isinstance(setup.get("seedRegion"), dict) else None

    frames = _frames_for(bundle, setup)
    if not frames:
        return {"bundle": bundle, "skip": "no climb end and no known duration"}

    start, end = _climb_window(setup)
    payload: dict[str, Any] = {
        "video_path": str(video.relative_to(ANALYSIS_DIR.parent)).replace("\\", "/"),
        "route_folder": bundle.parent.name,
        "video_key": bundle.name,
        "seed_tap": {"x": tap.get("x"), "y": tap.get("y"), "t": tap.get("t")},
        "frames": [{"timestamp": t} for t in frames],
    }
    if region is not None:
        payload["seed_region"] = {k: region.get(k) for k in ("x", "y", "w", "h")}
    if setup.get("setupHash"):
        payload["setup_hash"] = setup["setupHash"]

    scaffold = _load(bundle / "vitpose.json")
    return {
        "bundle": bundle,
        "payload": payload,
        "frames": len(frames),
        "window": (start, end),
        "current_seed_hash": scaffold.get("seedHash"),
        "reseeded_tap": bool(setup.get("seedTap")),
    }


def _await_terminal(bundle: Path, job_id: str, timeout_sec: float,
                    stall_sec: float) -> tuple[str, str]:
    """Poll the status sidecar until **this** job reaches a terminal state.

    Matching on ``jobId`` is the whole point. The service returns 202 and writes
    ``running`` from a background thread a moment later, so a poller that accepts any
    terminal status races that write and reads the *previous* run's ``done`` — reporting
    success instantly while the real job is still posing frames. Ask for this job.

    Returns ``wedged`` when the sidecar stops changing for ``stall_sec``. That is a
    *fatal* condition for the whole run, not a per-bundle one: the service runs scaffold
    jobs under a single process-wide lock, so a thread stuck inside one holds the lock
    forever and every later job queues behind it. Observed on the real corpus — a
    38-second video sat 21 minutes with the GPU at 0% while the service itself stayed
    responsive. Without this the run would have burned the full per-job timeout on each
    of the remaining bundles and finished nothing.
    """

    sidecar = bundle / "vitpose.status.json"
    deadline = time.monotonic() + timeout_sec
    last_seen = time.monotonic()
    last_mtime = sidecar.stat().st_mtime if sidecar.exists() else 0.0
    last = ""
    while time.monotonic() < deadline:
        status = _load(sidecar)
        state = str(status.get("status") or "")
        if status.get("jobId") == job_id:
            if state in TERMINAL:
                return state, str(status.get("error") or status.get("skipReason") or "")
            last = state
        mtime = sidecar.stat().st_mtime if sidecar.exists() else 0.0
        if mtime != last_mtime:
            last_mtime, last_seen = mtime, time.monotonic()
        elif time.monotonic() - last_seen > stall_sec:
            return "wedged", (f"no sidecar activity for {stall_sec/60:.0f} min "
                              f"(last status {last or 'unwritten'!r})")
        time.sleep(POLL_INTERVAL_SEC)
    return "timeout", f"last status {last!r} for job {job_id[:8]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="actually POST (default is a dry run that sends nothing)")
    parser.add_argument("--force", action="store_true",
                        help="re-seed even when the seed is unchanged")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--job-timeout", type=float, default=900.0,
                        help="hard ceiling for one scaffold job (default 900). A job "
                             "costs roughly its climb-window length, so this is already "
                             "generous")
    parser.add_argument("--stall-timeout", type=float, default=300.0,
                        help="abort the run when a job's status sidecar stops changing "
                             "for this long (default 300). The service serializes jobs "
                             "under one lock, so a wedged job stops everything")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="person-detector inference resolution. The lever for a "
                             "Climber too small to detect: at the default 640 a distant "
                             "Climber is ~35 px tall and goes undetected. Try 1920 on "
                             "bundles that fail to seed. Costs inference time")
    parser.add_argument("--conf", type=float, default=None,
                        help="person-detector confidence floor (default: backend's own)")
    parser.add_argument("--skip", default="",
                        help="comma-separated video_keys to leave alone (e.g. one that "
                             "wedged the service last run)")
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N bundles (0 = all); useful for a trial run")
    parser.add_argument("--retry-empty", action="store_true",
                        help="only bundles whose current scaffold posed no frames, and "
                             "force them (implies --force); the repair pass after a run "
                             "whose seeding failed")
    args = parser.parse_args(argv)

    skip = {k.strip() for k in args.skip.split(",") if k.strip()}
    bundles = [p.parent for p in sorted(ANALYSIS_DIR.glob("*/*/metadata.json"))
               if p.parent.name not in skip]
    if skip:
        print(f"skipping {len(skip)} bundle(s) by request: {', '.join(sorted(skip))}")
    plans = [_plan(b) for b in bundles]
    todo = [p for p in plans if p and "payload" in p]
    skipped = [p for p in plans if p and "skip" in p]

    if args.retry_empty:
        args.force = True
        empty = []
        for plan in todo:
            scaffold = _load(plan["bundle"] / "vitpose.json")
            frames = scaffold.get("frames") or []
            if frames and not any(f.get("keypoints") for f in frames):
                empty.append(plan)
        print(f"--retry-empty: {len(empty)} of {len(todo)} scaffold(s) pose nothing today")
        todo = empty

    already = [p for p in todo if p["current_seed_hash"]]
    total_frames = sum(p["frames"] for p in todo)
    print(f"bundles: {len(bundles)} | seedable: {len(todo)} | unseedable: {len(skipped)}")
    print(f"already carry a seedHash: {len(already)} "
          f"(the service will skip these unless --force)")
    print(f"frames to pose across the corpus: {total_frames:,}")
    for p in skipped:
        print(f"  SKIP {p['bundle'].name[:44]:44s} {p['skip']}")

    if not args.apply:
        print("\nDry run — nothing sent. Re-run with --apply to start.")
        for p in todo[:5]:
            start, end = p["window"]
            print(f"  would seed {p['bundle'].name[:40]:40s} "
                  f"window=({start}, {end}) frames={p['frames']}")
        if len(todo) > 5:
            print(f"  ... and {len(todo) - 5} more")
        return 0

    if args.limit:
        todo = todo[: args.limit]

    url = args.base_url.rstrip("/") + "/api/vitpose"
    counts = {"done": 0, "skipped": 0, "error": 0, "timeout": 0,
              "rejected": 0, "wedged": 0}
    started = time.monotonic()
    for i, plan in enumerate(todo, 1):
        bundle = plan["bundle"]
        payload = dict(plan["payload"])
        if args.force:
            payload["force"] = True
        if args.imgsz is not None:
            payload["detector_imgsz"] = args.imgsz
        if args.conf is not None:
            payload["detector_conf"] = args.conf
        label = f"[{i}/{len(todo)}] {bundle.parent.name}/{bundle.name}"
        code, body = _post(url, payload, timeout=60.0)

        if code == 200 and body.get("status") == "skipped":
            counts["skipped"] += 1
            print(f"{label}: seed unchanged, skipped")
            continue
        job_id = str(body.get("jobId") or "")
        if code != 202 or not job_id:
            counts["rejected"] += 1
            print(f"{label}: HTTP {code} {str(body)[:160]}")
            continue

        state, detail = _await_terminal(bundle, job_id, args.job_timeout,
                                        args.stall_timeout)
        counts[state] = counts.get(state, 0) + 1
        if state == "wedged":
            print(f"{label}: WEDGED — {detail}")
            print(
                "\nAborting: scaffold jobs run under one process-wide lock, so a "
                "wedged job blocks every bundle after it.\n"
                "  1. restart the harness service (kills the stuck thread)\n"
                "  2. re-run this script - it resumes, skipping what is already done\n"
                f"  3. if it wedges here again: --skip {bundle.name}")
            break
        elapsed = time.monotonic() - started
        rate = elapsed / i
        # Posed coverage is how you tell the reset actually did something. A bundle that
        # was truthless before and poses frames now is a recovered bundle (issue #101
        # stories 9-12); one that still poses nothing is a genuinely hard video.
        coverage = ""
        if state == "done":
            scaffold = _load(bundle / "vitpose.json")
            frames = scaffold.get("frames") or []
            posed = sum(1 for f in frames if f.get("keypoints"))
            if frames:
                coverage = f" posed {posed}/{len(frames)} ({posed / len(frames):.0%})"
                if posed == 0:
                    counts["posed_none"] = counts.get("posed_none", 0) + 1
        print(f"{label}: {state}{coverage}{(' — ' + detail) if detail else ''} "
              f"[{elapsed/60:.1f} min elapsed, ~{rate*(len(todo)-i)/60:.0f} min left]")

    print(f"\n{json.dumps(counts)}")
    if counts.get("posed_none"):
        print(f"{counts['posed_none']} scaffold(s) posed no frames at all — check their "
              "seedDebug.seedFailureReason: 'no-detections'/'no-candidates' means a "
              "genuinely hard video, 'region-gated'/'no-frames-in-window' means a "
              "repairable seed.")
    print("Re-run this script to retry anything that errored — unchanged seeds skip.")
    return 1 if any(counts[k] for k in
                    ("error", "timeout", "rejected", "wedged")) else 0


if __name__ == "__main__":
    sys.exit(main())
