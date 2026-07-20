"""Orchestrate discovery → tables → stats → report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import crossmatch, report, stats, trends
from .discovery import discover_runs
from .evaluate import evaluate
from .frames import build_frame_table
from .runs import build_run_table


def _display_run_table(run_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in run_df.columns if "_flag_" not in c and not c.startswith("orb_ref_")]
    slim = run_df[cols].copy()
    return slim.round(3)


def run(analysis_root: Path, out_dir: Path, decode: bool = True,
        matrix: Path | None = None) -> dict[str, Path]:
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

    # ORB cross-match matrix (produced by the scanner repo; absent -> graceful).
    matrix_path = matrix or (out_dir / "orb_match_matrix.json")
    match_df = crossmatch.load_match_matrix(matrix_path)
    orb_matrix = crossmatch.ordered_matrix(match_df)
    orb_separation = crossmatch.separation_stats(match_df)
    orb_threshold = crossmatch.best_threshold(match_df)
    final_frames = {rec.video_key: rec.video_dir / "final_frame.png" for rec in records}

    print(f"discovered {len(records)} runs "
          f"({n_pose_files - len(records)} re-runs collapsed) across "
          f"{run_df['video_key'].nunique()} videos; {len(frame_df)} per-frame samples")
    if dropped_labels:
        print("pruned labels: " + ", ".join(f"{c.replace('label_','')} ({r})" for c, r in dropped_labels))
    if orb_matrix.get("available"):
        print(f"loaded ORB cross-match matrix: {len(match_df)} pairs from {matrix_path}")
    else:
        print(f"no ORB cross-match matrix at {matrix_path} (ORB cross-match section will be empty)")

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
        "run_df": run_df,
        "final_frames": final_frames,
        "run_corr": run_corr,
        "cat_effects": cat_effects,
        "orb_corr": orb_corr,
        "orb_matrix": orb_matrix,
        "orb_separation": orb_separation,
        "orb_threshold": orb_threshold,
        "run_table_display": _display_run_table(run_df),
    }

    trend_ctx = trends.build_trend_context(analysis_root)
    ctx.update(trend_ctx)

    outputs = report.write_outputs(out_dir, run_df, frame_df, ctx)
    outputs.update(trends.write_trend_tables(out_dir, trend_ctx))
    print(f"wrote {outputs['html']}")
    print(f"wrote {outputs['run_csv']} and {outputs['frame_csv']}")
    if trend_ctx.get("eval_count", 0):
        print(f"loaded {trend_ctx['eval_count']} evaluation record(s)")
        print(f"verified truth frames: {trend_ctx.get('verified_frames_total', 0)}")
    return outputs


def run_evaluate(analysis_root: Path, prune: bool = False) -> None:
    """Pair pose Runs with bundle truth, write eval records, print a summary."""

    summary = evaluate(analysis_root, prune=prune)
    print(f"wrote {len(summary.written)} evaluation record(s) "
          f"from {analysis_root}")
    for p in summary.written:
        print(f"  {p.route_folder}/{p.video_key} {p.run_ts} "
              f"vs {p.truth_source} -> {p.record_path.name}")
    if summary.skipped:
        print(f"skipped {len(summary.skipped)} pair(s):")
        for p in summary.skipped:
            print(f"  {p.route_folder}/{p.video_key} {p.run_ts}: {p.reason}")
    if summary.orphans:
        verb = "pruned" if prune else "would prune"
        print(f"{verb} {len(summary.orphans)} stale-run orphan record(s)"
              + ("" if prune else " (dry run; pass --prune to remove)") + ":")
        for o in summary.orphans:
            print(f"  {o.route_folder}/{o.video_key} -> {o.record_path.name}")
    if summary.truthless_videos:
        print(f"no truth for {len(summary.truthless_videos)} bundle(s) "
              "(no ground-truth.json or vitpose.json)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="analysis_pipeline",
        description="Correlate video conditions with beta-scanner pose/ORB detection quality.",
    )
    sub = parser.add_subparsers(dest="command")

    p_an = sub.add_parser("analysis", help="build the correlation report (default)")
    p_an.add_argument("analysis_root", nargs="?", default="analysis", type=Path,
                      help="root of the analysis/ bundle tree (default: analysis)")
    p_an.add_argument("-o", "--out", default="reports", type=Path,
                      help="output directory for report + CSVs (default: reports)")
    p_an.add_argument("--no-decode", action="store_true",
                      help="skip cv2 video decoding (pose-derived predictors only)")
    p_an.add_argument("--matrix", default=None, type=Path,
                      help="path to the scanner's orb_match_matrix.json "
                           "(default: <out>/orb_match_matrix.json)")

    p_ev = sub.add_parser(
        "evaluate", help="write detection-vs-truth evaluation records into the bundles")
    p_ev.add_argument("analysis_root", nargs="?", default="analysis", type=Path,
                      help="root of the analysis/ bundle tree (default: analysis)")
    p_ev.add_argument("--prune", action="store_true",
                      help="delete stale-run orphan evaluation records (records whose "
                           "run no longer pairs and whose truth hash is no longer "
                           "current); without it, orphans are only reported (dry run)")

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        run_evaluate(args.analysis_root.resolve(), prune=args.prune)
        return

    # Default (no subcommand) and the explicit "analysis" subcommand both build the
    # report, preserving `python -m analysis_pipeline analysis -o reports`.
    root = getattr(args, "analysis_root", Path("analysis"))
    out = getattr(args, "out", Path("reports"))
    matrix = getattr(args, "matrix", None)
    matrix = matrix.resolve() if matrix else None
    run(root.resolve(), out.resolve(),
        decode=not getattr(args, "no_decode", False), matrix=matrix)


if __name__ == "__main__":
    main()
