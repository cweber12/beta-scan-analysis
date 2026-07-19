"""One-off sweep: compute Video Stats (issue #23) for every existing bundle.

For each ``analysis/<route>/<video_key>/`` bundle with a video binary:

- **Phase 1** — whole-frame source stats into ``metadata.json["video_stats"]``
  (skipped when the block already exists, unless ``--force``).
- **Phase 2** — region stats into ``video-stats.json`` stamped with the current
  ``setup.json`` ``setupHash`` (requires a ``setup.json`` with a wall crop;
  skipped when the artifact already matches that hash, unless ``--force``).

One decode pass per bundle feeds both phases. Existing hand labels are untouched
and keep human provenance; a pre-existing ``cameraAngle`` block is preserved.

Run:  python -m scripts.backfill_video_stats [analysis_root] [--force]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_stats  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_video(bundle: Path, video_key: str) -> Path | None:
    canonical = bundle / f"{video_key}.mp4"
    if canonical.is_file():
        return canonical
    for path in sorted(bundle.iterdir()):
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            return path
    return None


def backfill(analysis_root: Path, force: bool = False) -> dict[str, int]:
    stats = {"phase1": 0, "phase2": 0, "no_video": 0, "no_setup": 0,
             "already": 0, "failed": 0}

    for metadata_path in sorted(analysis_root.glob("*/*/metadata.json")):
        bundle = metadata_path.parent
        name = f"{bundle.parent.name}/{bundle.name}"
        metadata = _load(metadata_path)
        video_path = _find_video(bundle, metadata.get("video_key", bundle.name))
        if video_path is None:
            stats["no_video"] += 1
            print(f"skip {name}: no video binary")
            continue

        setup_path = bundle / "setup.json"
        setup = _load(setup_path) if setup_path.exists() else {}
        setup_hash = setup.get("setupHash")
        wall_crop = setup.get("wallCrop")

        artifact_path = bundle / video_stats.VIDEO_STATS_NAME
        artifact = _load(artifact_path) if artifact_path.exists() else {}
        need_phase1 = force or "video_stats" not in metadata
        need_phase2 = bool(wall_crop) and (
            force or artifact.get("regionStats") is None
            or artifact.get("setupHash") != setup_hash
        )
        if not need_phase1 and not need_phase2:
            stats["already"] += 1
            continue

        try:
            frames, timestamps = video_stats.sample_video_frames(video_path)

            if need_phase1:
                block = video_stats.build_source_stats_block(
                    video_path, metadata.get("source_video"), frames, timestamps
                )
                video_stats.write_source_stats(bundle, block)
                stats["phase1"] += 1

            if need_phase2:
                region = video_stats.compute_region_stats(
                    frames,
                    timestamps,
                    wall_crop,
                    climber_crop=setup.get("climberCrop"),
                    climber_point_t=(setup.get("climberPoint") or {}).get("t"),
                    panning=bool(setup.get("panning")),
                )
                source_stats = _load(metadata_path).get("video_stats")
                suggestions = video_stats.suggest_labels(source_stats, region)
                video_stats.write_region_stats(
                    bundle, region, suggestions, setup_hash, source="backfill"
                )
                stats["phase2"] += 1
            elif not wall_crop:
                stats["no_setup"] += 1

            done = [p for p, ok in (("p1", need_phase1), ("p2", need_phase2)) if ok]
            print(f"backfilled {name}: {'+'.join(done)} ({len(frames)} samples)")
        except Exception as exc:  # noqa: BLE001 — sweep on; report at the end
            stats["failed"] += 1
            print(f"FAILED {name}: {type(exc).__name__}: {exc}")

    return stats


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    force = "--force" in argv
    paths = [a for a in argv if not a.startswith("--")]
    root = Path(paths[0]) if paths else Path("analysis")
    stats = backfill(root.resolve(), force=force)
    print(
        f"done: {stats['phase1']} phase-1 blocks, {stats['phase2']} phase-2 artifacts, "
        f"{stats['already']} already current, {stats['no_video']} without video, "
        f"{stats['no_setup']} without calibration (phase 1 only), {stats['failed']} failed"
    )


if __name__ == "__main__":
    main()
