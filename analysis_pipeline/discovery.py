"""Walk the analysis/ tree, pair pose+orb detection files, and dedup re-runs.

Mirrors the bundle layout written by ``youtube_core.build_analysis_bundle`` and the
pose/orb pairing stem from ``youtube_core._paired_detection_paths``:

    analysis/<route>/<video_key>/
        <video_key>.mp4
        metadata.json
        setup.json
        detections/<stem>_pose.json
        detections/<stem>_orb.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detector_attempts import detector_attempt_evidence, parse_detector_attempts


@dataclass
class RunRecord:
    """One detection run, resolved to everything the pipeline needs downstream."""

    route_folder: str
    video_key: str
    run_ts: str
    written_at: str
    video_dir: Path
    video_path: Path | None
    metadata: dict[str, Any]
    setup: dict[str, Any]
    pose: dict[str, Any]  # the inner ``data`` blob of the pose envelope
    orb: dict[str, Any]  # the inner ``data`` blob of the orb envelope
    detector_attempts: list[dict[str, Any]] | None = None
    detector_attempt_evidence: str = "unknown"
    # per-bundle video-stats.json (issue #23); {} when the artifact is absent
    video_stats: dict[str, Any] = field(default_factory=dict)
    # dedup identity
    video_hash: str = ""
    setup_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    @property
    def dedup_key(self) -> tuple[str, str, str]:
        return (self.video_hash, self.setup_hash, self.config_hash)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _unwrap(envelope: dict[str, Any]) -> dict[str, Any]:
    """Detection files are ``{video_key, run_ts, ..., data}`` envelopes."""

    return envelope.get("data", {}) if isinstance(envelope, dict) else {}


def _iter_video_dirs(analysis_root: Path):
    """Yield every ``<route>/<video_key>`` dir that carries a metadata.json."""

    for metadata_path in sorted(analysis_root.glob("*/*/metadata.json")):
        yield metadata_path.parent


def _pair_stems(detections_dir: Path) -> dict[str, dict[str, Path]]:
    """Group detection files by shared run stem into ``{stem: {pose, orb}}``."""

    pairs: dict[str, dict[str, Path]] = {}
    for path in sorted(detections_dir.glob("*.json")):
        name = path.stem  # e.g. "20260710-191812_pose"
        for kind in ("pose", "orb"):
            suffix = f"_{kind}"
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                pairs.setdefault(stem, {})[kind] = path
                break
    return pairs


def discover_runs(analysis_root: Path) -> list[RunRecord]:
    """Load and dedup every detection run under ``analysis_root``.

    Deduplication collapses byte-identical re-runs: the newest ``written_at`` wins
    per ``(video_hash, setup_hash, config_hash)``. Distinct configs / crops on the
    same video survive as separate observations.
    """

    records: list[RunRecord] = []

    for video_dir in _iter_video_dirs(analysis_root):
        metadata = _load_json(video_dir / "metadata.json")
        setup_path = video_dir / "setup.json"
        setup = _load_json(setup_path) if setup_path.exists() else {}
        stats_path = video_dir / "video-stats.json"
        try:
            video_stats = _load_json(stats_path) if stats_path.exists() else {}
        except ValueError:
            video_stats = {}

        route_folder = metadata.get("route_folder", video_dir.parent.name)
        video_key = metadata.get("video_key", video_dir.name)

        video_path = video_dir / f"{video_key}.mp4"
        if not video_path.exists():
            # Fall back to any single video binary in the folder.
            candidates = [
                p
                for p in video_dir.iterdir()
                if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
            ]
            video_path = candidates[0] if candidates else None

        detections_dir = video_dir / "detections"
        if not detections_dir.is_dir():
            continue

        for stem, kinds in _pair_stems(detections_dir).items():
            if "pose" not in kinds:
                continue  # per-frame + pose outcomes require the pose file
            pose_env = _load_json(kinds["pose"])
            orb_env = _load_json(kinds["orb"]) if "orb" in kinds else {}
            pose = _unwrap(pose_env)
            orb = _unwrap(orb_env)

            diagnostics = pose.get("diagnostics", {})
            config = diagnostics.get("config", {})
            detector_attempts = parse_detector_attempts(pose)

            records.append(
                RunRecord(
                    route_folder=route_folder,
                    video_key=video_key,
                    run_ts=pose_env.get("run_ts", stem),
                    written_at=pose_env.get("written_at", ""),
                    video_dir=video_dir,
                    video_path=video_path,
                    metadata=metadata,
                    setup=setup,
                    pose=pose,
                    orb=orb,
                    detector_attempts=detector_attempts,
                    detector_attempt_evidence=detector_attempt_evidence(detector_attempts),
                    video_stats=video_stats,
                    video_hash=diagnostics.get("videoHash", ""),
                    setup_hash=pose.get("setupHash", setup.get("setupHash", "")),
                    config=config,
                    config_hash=_config_hash(config),
                )
            )

    return _dedup(records)


def _dedup(records: list[RunRecord]) -> list[RunRecord]:
    """Keep the newest ``written_at`` per dedup key."""

    latest: dict[tuple[str, str, str], RunRecord] = {}
    for rec in records:
        key = rec.dedup_key
        current = latest.get(key)
        if current is None or rec.written_at > current.written_at:
            latest[key] = rec
    # Stable order: by route, then video, then run timestamp.
    return sorted(
        latest.values(),
        key=lambda r: (r.route_folder, r.video_key, r.run_ts),
    )
