"""Orchestrate discovery → tables → stats → report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import report, stats
from .discovery import discover_runs
from .frames import build_frame_table
from .runs import build_run_table


def _display_run_table(run_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in run_df.columns if "_flag_" not in c and not c.startswith("orb_ref_")]
    slim = run_df[cols].copy()
    return slim.round(3)


def run(analysis_root: Path, out_dir: Path, decode: bool = True) -> dict[str, Path]:
    records = discover_runs(analysis_root)
    if not records:
        raise SystemExit(f"No detection runs found under {analysis_root}")

    n_pose_files = len(list(analysis_root.glob("*/*/detections/*_pose.json")))

    run_df = build_run_table(records)
    frame_df = build_frame_table(records, decode=decode)

    kept_labels, dropped_labels = stats.prune_labels(run_df)

    frame_corr = stats.within_run_correlations(frame_df)
    run_predictors = stats.run_numeric_predictor_cols(run_df)
    run_corr = stats.pooled_run_correlations(run_df, predictors=run_predictors)
    cat_effects = stats.categorical_effects(run_df, kept_labels)
    orb_corr = stats.orb_correlations(run_df)

    print(f"discovered {len(records)} runs "
          f"({n_pose_files - len(records)} re-runs collapsed) across "
          f"{run_df['video_key'].nunique()} videos; {len(frame_df)} per-frame samples")
    if dropped_labels:
        print("pruned labels: " + ", ".join(f"{c.replace('label_','')} ({r})" for c, r in dropped_labels))

    ctx = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_root": str(analysis_root),
        "n_runs": len(records),
        "n_videos": int(run_df["video_key"].nunique()),
        "n_collapsed": n_pose_files - len(records),
        "n_frame_rows": len(frame_df),
        "dropped_labels": dropped_labels,
        "frame_corr": frame_corr,
        "frame_df": frame_df,
        "run_corr": run_corr,
        "cat_effects": cat_effects,
        "orb_corr": orb_corr,
        "run_table_display": _display_run_table(run_df),
    }

    outputs = report.write_outputs(out_dir, run_df, frame_df, ctx)
    print(f"wrote {outputs['html']}")
    print(f"wrote {outputs['run_csv']} and {outputs['frame_csv']}")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="analysis_pipeline",
        description="Correlate video conditions with beta-scanner pose/ORB detection quality.",
    )
    parser.add_argument("analysis_root", nargs="?", default="analysis", type=Path,
                        help="root of the analysis/ bundle tree (default: analysis)")
    parser.add_argument("-o", "--out", default="reports", type=Path,
                        help="output directory for report + CSVs (default: reports)")
    parser.add_argument("--no-decode", action="store_true",
                        help="skip cv2 video decoding (pose-derived predictors only)")
    args = parser.parse_args(argv)
    run(args.analysis_root.resolve(), args.out.resolve(), decode=not args.no_decode)


if __name__ == "__main__":
    main()
