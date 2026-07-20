"""Correlation analysis pipeline for beta-scanner detection bundles.

Reads the ``analysis/`` tree written by ``youtube_core.build_analysis_bundle`` /
``save_detection_run`` and produces a two-level (per-frame + per-run) feature and
outcome table plus a self-contained HTML correlation report.

The corpus is deliberately treated as *exploratory*: the run is the independent
unit, and everything is reported effect-size first. See the plan / report banner.
"""

__all__ = ["discovery", "frames", "runs", "stats", "report", "trends"]
