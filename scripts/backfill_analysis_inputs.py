"""One-off migration: move condition labels from metadata.json into setup.json.

The harness no longer collects analysis-input labels at upload — the scanner
writes them into ``setup.json.analysisInputs`` at calibration, and the pipeline
reads them there ([runs.py]). This backfills the existing corpus so those bundles
keep their labels under the new location.

For every ``analysis/<route>/<video_key>/`` bundle that has both a
``metadata.json`` with an ``analysis_inputs`` block and a ``setup.json``, copy the
label subset (+ notes) into ``setup.json.analysisInputs`` using snake_case keys
(matching the pipeline's ``LABEL_KEYS``). Idempotent, and never overwrites an
``analysisInputs`` the scanner already wrote.

Run:  python -m scripts.backfill_analysis_inputs [analysis_root]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The label keys the pipeline reads (runs.LABEL_KEYS) plus free-text notes.
LABEL_KEYS = (
    "route_orientation",
    "camera_angle",
    "shadows",
    "climber_contrast",
    "wall_contrast",
    "motion_blur",
    "occlusion",
    "camera_stability",
    "notes",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def backfill(analysis_root: Path) -> dict[str, int]:
    stats = {"migrated": 0, "already": 0, "no_setup": 0, "no_labels": 0}

    for metadata_path in sorted(analysis_root.glob("*/*/metadata.json")):
        bundle = metadata_path.parent
        metadata = _load(metadata_path)
        inputs = metadata.get("analysis_inputs") or {}
        labels = {k: inputs[k] for k in LABEL_KEYS if k in inputs}
        if not labels:
            stats["no_labels"] += 1
            continue

        setup_path = bundle / "setup.json"
        if not setup_path.exists():
            # No calibration -> no detections -> not part of the analysis; skip.
            stats["no_setup"] += 1
            continue

        setup = _load(setup_path)
        if setup.get("analysisInputs"):
            stats["already"] += 1
            continue

        setup["analysisInputs"] = labels
        setup_path.write_text(
            json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        stats["migrated"] += 1
        print(f"migrated {bundle}")

    return stats


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0]) if argv else Path("analysis")
    stats = backfill(root.resolve())
    print(
        f"done: {stats['migrated']} migrated, {stats['already']} already had "
        f"analysisInputs, {stats['no_setup']} without setup.json (skipped), "
        f"{stats['no_labels']} without labels"
    )


if __name__ == "__main__":
    main()
