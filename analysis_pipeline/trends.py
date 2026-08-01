"""Evaluation-record trend summaries for the analysis report.

The analysis command reads committed evaluation records under each bundle's
``evaluations/`` folder and derives trend sections for issue #9:

- per-joint failure ranking with bootstrap CIs (frame/joint unit),
- within-video condition trends (size, speed, edge proximity) vs joint error,
- cross-video descriptive splits (resolution, panning, source type) with CIs,
- coverage/shame accounting (truthless bundles, stale setup runs),
- scanner appVersion run-over-run regression tracking (issue #10): consecutive
  versions delta'd per joint over a truth-hash-matched video pool.

This module never writes evaluation records and never calls the evaluate
subcommand; it only consumes existing artifacts in the bundle tree.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from . import floors
from .detector_attempts import parse_detector_attempts
from .discovery import _iter_video_dirs, _load_json, _pair_stems, _unwrap
from .detector_attempts import (
    DETECTOR_ATTEMPT_STATUS_ORDER,
    DETECTOR_ATTEMPT_STATUS_UNKNOWN,
    DETECTOR_ATTEMPT_STATUSES,
    _slug,
    condition_flags as _condition_flags,
    region_metric as _region_metric,
)
from .runs import _detector_attempt_summary
from .evaluate import (
    ABSENCE_CONFIRMED,
    scaffold_truth_drift,
    RATE_MISMATCH_MIN_RATIO,
    ABSENCE_REASONS,
    ABSENCE_UNKNOWN,
    BASELINE_CYCLE_SCHEMA,
    COCO_CORE_JOINTS,
    EVIDENCE_ATTEMPTS,
    EVIDENCE_GENERATIONS,
    MISS_CAUSES,
    NONCONFORMANCE_CAUSES,
    NONCONFORMANCE_TRAJECTORY_DIVERGENCE,
    ATTRIBUTION_TRUTH_IDENTITY,
    record_attribution,
    SCHEMA_VERSION,
    _dist,
    _iter_pose_runs,
    _nearest_within,
    _pose_frame_joints,
    _scanner_frame_interval,
    load_truth,
    record_conforms,
    record_evidence_generation,
    record_nonconformance_cause,
    record_schema_version,
    record_trusted,
    torso_length,
)

N_BOOT = 300
BOOT_SEED = 42


@dataclass
class EvalRecord:
    path: Path
    route_folder: str
    video_key: str
    run_ts: str
    truth_hash: str
    data: dict[str, Any]


def _pct_ci(samples: list[float], alpha: float = 0.05) -> tuple[float, float]:
    if not samples:
        return (math.nan, math.nan)
    s = sorted(samples)
    lo_i = max(0, int((alpha / 2) * (len(s) - 1)))
    hi_i = min(len(s) - 1, int((1 - alpha / 2) * (len(s) - 1)))
    return (s[lo_i], s[hi_i])


def _pool_total(df: pd.DataFrame, col: str) -> int:
    """Sum a count column over a pooled frame, tolerating a column that isn't there.

    An absent column is a genuine case, not a caller error: a corpus with no reacquire
    evidence has no reacquire columns at all, and it must total zero rather than raise —
    ``DataFrame.get`` on a missing key returns a bare ``nan`` that has no ``.fillna``."""

    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _pool_rates(df: pd.DataFrame, col: str) -> pd.Series:
    """The non-null values of a rate column, tolerating a column that isn't there.

    Unlike ``_pool_total`` a missing rate is dropped rather than zeroed: a run that never
    attempted a reacquire has no success *rate*, and counting it as 0.0 would report a
    failure the run never had."""

    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _p90(values: pd.Series) -> float | None:
    return float(np.quantile(values.to_numpy(dtype=float), 0.9)) if len(values) else None


def _bootstrap_rate(values: list[int], n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    if not values:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(values)
    mean = sum(values) / n
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    lo, hi = _pct_ci(draws)
    return (mean, lo, hi)


# The three columns that identify a Run. The Run is the unit of inference (CONTEXT.md),
# so anything that puts a CI on a per-frame outcome resamples these, not the frames.
_RUN_KEY_COLS = ("route_folder", "video_key", "run_ts")


def _cluster_bootstrap_rate(sums: list[float], counts: list[int],
                            n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    """Percentile bootstrap of a pooled rate that resamples **runs**, not frames (#70).

    ``sums[i]`` / ``counts[i]`` are run *i*'s outcome total and row count. Each draw
    takes ``len(counts)`` runs with replacement and recomputes the pooled rate over the
    drawn runs, so the interval's width tracks how many independent runs the estimate
    rests on rather than how many correlated frames they happened to contribute. The
    point estimate is untouched — it is still the pooled rate over every frame.
    """
    total = sum(counts)
    if not counts or total == 0:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(counts)
    pooled = sum(sums) / total
    draws: list[float] = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            s += sums[i]
            c += counts[i]
        if c:
            draws.append(s / c)
    if not draws:
        return None
    lo, hi = _pct_ci(draws)
    return (pooled, lo, hi)


def _run_unit_rate(df: pd.DataFrame, outcome: str) -> dict[str, Any] | None:
    """Pooled ``outcome`` rate with a run-unit CI and the per-run dispersion beside it.

    Frames inside a run are massively pseudo-replicated — a band can hold 100k frames
    drawn from a few dozen runs — so an iid bootstrap over those frames reports a CI far
    tighter than the design supports and makes marginal condition effects read as
    significant (#70). The rate stays pooled (that is what the corpus shows); the
    interval comes from the cluster bootstrap, and ``run_rate_median`` / ``run_rate_p90``
    expose the spread across runs that a single pooled number hides.

    Returns ``None`` when the frame carries no run identity — better to drop the band
    than to publish a frame-pooled interval that looks like a run-unit one.
    """
    key_cols = [c for c in _RUN_KEY_COLS if c in df.columns]
    if not key_cols or df.empty:
        return None
    grouped = df.groupby(key_cols, dropna=False)[outcome]
    sums = [float(v) for v in grouped.sum()]
    counts = [int(v) for v in grouped.size()]
    boot = _cluster_bootstrap_rate(sums, counts)
    if boot is None:
        return None
    run_rates = pd.Series([s / c for s, c in zip(sums, counts) if c], dtype=float)
    return {
        "n": int(sum(counts)),
        "n_runs": len(counts),
        "rate": boot[0],
        "ci_low": boot[1],
        "ci_high": boot[2],
        "run_rate_median": float(run_rates.median()),
        "run_rate_p90": _p90(run_rates),
    }


def _iter_eval_records(analysis_root: Path) -> list[EvalRecord]:
    latest_by_run: dict[tuple[str, str, str], EvalRecord] = {}
    for video_dir in _iter_video_dirs(analysis_root):
        eval_dir = video_dir / "evaluations"
        if not eval_dir.is_dir():
            continue
        for path in sorted(eval_dir.glob("*.json")):
            try:
                data = _load_json(path)
            except Exception:
                continue
            route = str(data.get("routeFolder") or video_dir.parent.name)
            key = str(data.get("videoKey") or video_dir.name)
            run_ts = str(data.get("runTs") or "")
            if not run_ts:
                continue
            rec = EvalRecord(
                path=path,
                route_folder=route,
                video_key=key,
                run_ts=run_ts,
                truth_hash=str(data.get("truthHash") or ""),
                data=data,
            )
            dedup = (route, key, run_ts)
            cur = latest_by_run.get(dedup)
            if cur is None or path.stat().st_mtime > cur.path.stat().st_mtime:
                latest_by_run[dedup] = rec
    return sorted(latest_by_run.values(), key=lambda r: (r.route_folder, r.video_key, r.run_ts))


def _dedup_evidence_generations(
    recs: list[EvalRecord],
    origins: dict[RunKey, str] | None = None,
) -> tuple[list[EvalRecord], list[dict[str, Any]]]:
    """Keep one **evidence generation** per video+truth pairing **per origin** (issue #89).

    .. note:: **Origin joins the key (issue #160).** Measured on the first harness runs:
       all three were found and scored by ``evaluate`` and then silently superseded behind
       the bundle's attempt-backed scanner records, because harness runs carry no
       ``detectorAttempts[]`` — a scanner-owned concept the harness has no equivalent of —
       and so read as ``legacy-frames``. They vanished from every pooled number *before*
       any origin segregation downstream could see them.

       A harness run and a scanner run are not two generations of one evidence stream; they
       are two different producers. Superseding across that line is never right.

    A video re-scanned after the scanner started exporting ``detectorAttempts[]`` carries
    two records for the same ``(route, video, truthHash)`` pairing: the attempt-backed one
    and the legacy-frames one it superseded. Pooling both counts that pairing twice, and
    blends two generations of evidence into one number — the legacy record's frame-derived
    quality answers a question the attempt stream answers directly, and its appVersion
    differs, so a generation change would read as a scanner change.

    So: when a pairing has any attempt-backed record, only its attempt-backed records
    pool. Everything else in that pairing is *superseded* — returned as rows for the
    report's accounting, never deleted. Records stay on disk and readable exactly as
    written; only the aggregation drops them.

    A pairing with no attempt-backed record is untouched, so a legacy-only corpus
    aggregates exactly as it did before this gate existed. Truth revision is part of the
    pairing key on purpose: an attempt-backed run under a *different* truth supersedes
    nothing, because the two records were never measuring the same thing.
    """

    origins = origins or {}
    by_pairing: dict[tuple[str, str, str, str], list[EvalRecord]] = {}
    for rec in recs:
        by_pairing.setdefault(
            (rec.route_folder, rec.video_key, rec.truth_hash, record_origin(rec, origins)),
            [],
        ).append(rec)

    kept: list[EvalRecord] = []
    superseded: list[dict[str, Any]] = []
    for group in by_pairing.values():
        attempt_backed = [
            r for r in group
            if record_evidence_generation(r.data) == EVIDENCE_ATTEMPTS
        ]
        if not attempt_backed:
            kept.extend(group)
            continue
        kept.extend(attempt_backed)
        superseded_by = ", ".join(sorted(r.run_ts for r in attempt_backed))
        for rec in group:
            if record_evidence_generation(rec.data) == EVIDENCE_ATTEMPTS:
                continue
            superseded.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "truth_hash": rec.truth_hash,
                "evidence_generation": record_evidence_generation(rec.data),
                "superseded_by": superseded_by,
            })

    return (
        sorted(kept, key=lambda r: (r.route_folder, r.video_key, r.run_ts)),
        sorted(superseded, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"])),
    )


def _evidence_generation_summary(recs: list[EvalRecord], pool: str) -> dict[str, Any]:
    """What evidence generation(s) one pooled set of records is made of (issue #89).

    Every pooled section reports this, so a mixed pool is never something a reader has to
    infer. Dedup removes the *superseded* mixture (same pairing, two generations); a pool
    can still legitimately span generations across different videos — a corpus mid-
    migration — and that is exactly the case worth naming rather than silently averaging.
    """

    counts = {g: 0 for g in EVIDENCE_GENERATIONS}
    for rec in recs:
        counts[record_evidence_generation(rec.data)] += 1
    present = [g for g in EVIDENCE_GENERATIONS if counts[g]]
    return {
        "pool": pool,
        "n_records": len(recs),
        "counts": counts,
        "generations": present,
        "mixed": len(present) > 1,
        "label": " + ".join(present) if present else "none",
    }


SCHEMA_UNKNOWN = "unknown"

# Who produced a run (issue #160, PRD #156). ``scanner`` is the default and the entire
# historical corpus: a run with no experiment stamp predates the harness module. A pooled
# number must never blend the two — whether a browser-WASM run and a Python run agree is the
# open question #162 asks, and pooling them before it is answered assumes the answer.
ORIGIN_SCANNER = "scanner"
ORIGIN_HARNESS = "harness-mediapipe"


def _measurement_basis(
    recs: list[EvalRecord],
    pool: str,
    build_ids: dict[tuple[str, str, str], BuildId] | None = None,
) -> dict[str, Any]:
    """The measurement basis one pooled set of records rests on (issue #131).

    Two things make a pooled number comparable to another pooled number: the **schema** it
    was scored under and the **build set** it was collected from. The records stamp
    ``schemaVersion``, so the basis was always *recorded* — but nothing stated it next to
    the numbers, so nothing stopped two sections resting on different bases from being read
    against each other. This states it.

    Printed per pool rather than once at the top, for the same reason
    ``_evidence_generation_summary`` is (#89): a number read out of the middle of the report
    has to carry its own provenance.

    Mixture is *flagged, not refused*. A corpus mid-migration legitimately spans bases, and
    refusing to report would destroy the accounting that shows what the mixture is. This
    follows the fail-open-and-name-it convention the rest of the module uses — the failure
    being prevented is a silent blend, not a blend.

    ``build_ids`` is optional so a caller with no pose cache still gets the schema half;
    an absent build set reports as unknown rather than as empty, which would read as
    "no builds" instead of "not established".
    """

    schema_counts: dict[str, int] = {}
    for rec in recs:
        version = record_schema_version(rec.data)
        key = SCHEMA_UNKNOWN if version is None else str(version)
        schema_counts[key] = schema_counts.get(key, 0) + 1
    # Numeric ascending, with unknown last: it is not a version and must not sort among
    # them, but it is the most important cell to see when present.
    present = sorted(
        (k for k in schema_counts if k != SCHEMA_UNKNOWN), key=int
    ) + ([SCHEMA_UNKNOWN] if SCHEMA_UNKNOWN in schema_counts else [])
    on_basis = schema_counts.get(str(BASELINE_CYCLE_SCHEMA), 0)

    builds: list[dict[str, Any]] = []
    if build_ids is not None:
        by_identity: dict[str, set[BuildId]] = {}
        counts_by_identity: dict[str, int] = {}
        for rec in recs:
            build = build_ids.get((rec.route_folder, rec.video_key, rec.run_ts))
            if build is None:
                continue
            identity = _build_identity(build)
            by_identity.setdefault(identity, set()).add(build)
            counts_by_identity[identity] = counts_by_identity.get(identity, 0) + 1
        builds = [
            {
                "identity": identity,
                "label": _build_label(identity, by_identity[identity]),
                "n_records": counts_by_identity[identity],
            }
            for identity in sorted(by_identity, key=lambda i: (-counts_by_identity[i], i))
        ]

    return {
        "pool": pool,
        "n_records": len(recs),
        "schema_counts": schema_counts,
        "schema_versions": present,
        "schema_mixed": len(present) > 1,
        "schema_label": " + ".join(f"v{v}" if v != SCHEMA_UNKNOWN else v for v in present)
                        or "none",
        # The frozen basis for this cycle, and how much of the pool actually sits on it.
        "frozen_schema": BASELINE_CYCLE_SCHEMA,
        "on_basis": on_basis,
        "off_basis": len(recs) - on_basis,
        # A writer/freeze disagreement is the mid-cycle bump: what forces a full re-score.
        "writer_schema": SCHEMA_VERSION,
        "cycle_broken": SCHEMA_VERSION != BASELINE_CYCLE_SCHEMA,
        "builds": builds,
        "build_set_known": build_ids is not None,
        "n_builds": len(builds),
        "build_mixed": len(builds) > 1,
        "build_label": " + ".join(b["label"] for b in builds) if builds else SCHEMA_UNKNOWN,
    }


class PoseRun(NamedTuple):
    """One bundle run's build identity and pose evidence.

    ``detector_code_hash`` is the scanner's per-run build identity (#130): a digest of
    the detector source that actually executed, which — unlike ``app_version``, resolved
    once at dev-server start — cannot survive a hot reload. Empty means the run predates
    the field or the scanner's derivation failed. That is *unknown* provenance, never a
    conflict; every consumer here fails open on it.
    """

    app_version: str
    frames: list[dict[str, Any]]
    attempts: list[dict[str, Any]] | None
    detector_code_hash: str = ""
    # Experiment provenance (issue #160, PRD #156). ``origin`` says *who produced the run*:
    # the scanner posting through the API, or this harness running MediaPipe itself. A run
    # with no stamp is scanner-origin, which is the entire historical corpus.
    #
    # ``config_hash`` is the **arm** — the experimental condition — and ``pass_index`` which
    # repeat of it this run is. Both are empty for scanner runs, which have no arm.
    #
    # ``config`` is the block the hash was taken over, kept alongside it because a hash is
    # a grouping key and not a description: a reader comparing two arms needs to see *which
    # factor differs*, and reconstructing that from a 16-hex digest is impossible. It is
    # read for display only — grouping is always on the hash, never on the block (issue
    # #164).
    origin: str = ORIGIN_SCANNER
    config_hash: str = ""
    pass_index: int | None = None
    config: dict[str, Any] | None = None
    sampled_frames: int | None = None


# The corpus-wide cache of them, keyed ``(route_folder, video_key)``. Four derivations
# below need the same pose files; one cache threaded through them all is what keeps a
# trend build from re-reading every detection file once per derivation.
PoseRunCache = dict[tuple[str, str], dict[str, PoseRun]]

# A record whose run has no pose file at all: no appVersion, no frames, and — the part
# that matters — *unknown* rather than empty detector attempts.
_NO_POSE_RUN = PoseRun("", [], None)

# A run's build identity: ``(appVersion, detectorCodeHash)``. Either half may be empty.
BuildId = tuple[str, str]

# A run key, unique corpus-wide.
RunKey = tuple[str, str, str]     # (route_folder, video_key, run_ts)


def _pose_run(
    analysis_root: Path,
    rec: EvalRecord,
    cache: PoseRunCache,
) -> PoseRun | None:
    """One record's pose run, loading (and caching) its bundle on first use.

    ``None`` means the bundle has no pose file for that ``run_ts`` — distinct from a run
    whose pose file holds zero frames, which callers must be able to tell apart."""

    vid = (rec.route_folder, rec.video_key)
    if vid not in cache:
        cache[vid] = _load_pose_runs(analysis_root / rec.route_folder / rec.video_key)
    return cache[vid].get(rec.run_ts)


def _load_pose_runs(video_dir: Path) -> dict[str, PoseRun]:
    """Map ``run_ts -> PoseRun`` for one bundle.

    Both halves of the build identity live only in the pose envelope's diagnostics —
    evaluation records don't carry either — so version tracking resolves them from the
    detection files at trend time.
    """

    out: dict[str, PoseRun] = {}
    detections_dir = video_dir / "detections"
    if not detections_dir.is_dir():
        return out
    for stem, kinds in _pair_stems(detections_dir).items():
        if "pose" not in kinds:
            continue
        try:
            env = _load_json(kinds["pose"])
        except Exception:
            continue
        data = _unwrap(env)
        run_ts = str(env.get("run_ts", stem))
        diagnostics = data.get("diagnostics") or {}
        attempts = parse_detector_attempts(data)
        # The experiment stamp rides in the same diagnostics block as build identity, which
        # is exactly why PRD #156 put it there — grouping arms reuses the machinery that
        # already groups builds instead of adding a second axis nobody's readers know about.
        experiment = diagnostics.get("experiment") or {}
        pass_index = experiment.get("passIndex")
        # How many frames this run actually scored. Runs sample ``12·√n`` of the Bundle's
        # truth grid, and a run written before the sweep landed carries no count — which
        # matters, because two runs of one arm over *different* frame sets are not repeats
        # and must not be read as a variance floor.
        sampled = experiment.get("frameCount")
        out[run_ts] = PoseRun(
            app_version=str(diagnostics.get("appVersion") or ""),
            frames=data.get("frames", []) or [],
            attempts=attempts,
            # ``null`` is the scanner's documented "derivation failed" value, and a record
            # predating the field has no key at all. Both land here as "".
            detector_code_hash=str(diagnostics.get("detectorCodeHash") or ""),
            # Absent stamp means scanner: the whole historical corpus, and the default that
            # keeps every existing number unchanged.
            origin=str(diagnostics.get("origin") or ORIGIN_SCANNER),
            config_hash=str(experiment.get("configHash") or ""),
            pass_index=pass_index if isinstance(pass_index, int) else None,
            config=experiment.get("config") if isinstance(
                experiment.get("config"), dict) else None,
            sampled_frames=sampled if isinstance(sampled, int) else None,
        )
    return out


def _origin_index(pose_cache: PoseRunCache) -> dict[RunKey, str]:
    """``(route, video, run_ts) -> origin`` for every cached pose run.

    Origin lives only in the pose envelope, exactly as both halves of build identity do, so
    it is resolved from the detection files at trend time rather than stamped into
    evaluation records. That keeps the v15 schema freeze (ADR 0009) intact — this adds no
    field to any record — and follows the precedent ``_load_pose_runs`` already set.
    """

    return {
        (route, key, run_ts): run.origin
        for (route, key), runs in pose_cache.items()
        for run_ts, run in runs.items()
    }


def record_origin(rec: EvalRecord, origins: dict[RunKey, str]) -> str:
    """One record's origin, defaulting to scanner when its pose run is unreadable."""

    return origins.get((rec.route_folder, rec.video_key, rec.run_ts), ORIGIN_SCANNER)


def _origin_populations(
    recs: list[EvalRecord], origins: dict[RunKey, str]
) -> pd.DataFrame:
    """How many records each origin contributes, and to which pool.

    The accounting that makes segregation checkable rather than asserted: a reader can see
    that the scanner population is the one every historical section pools, and that the
    harness population is reported separately and never added to it.
    """

    rows: list[dict[str, Any]] = []
    by_origin: dict[str, list[EvalRecord]] = {}
    for rec in recs:
        by_origin.setdefault(record_origin(rec, origins), []).append(rec)
    for origin in sorted(by_origin):
        group = by_origin[origin]
        rows.append({
            "origin": origin,
            "records": len(group),
            "bundles": len({(r.route_folder, r.video_key) for r in group}),
            "trusted": sum(1 for r in group if record_trusted(r.data)),
            "pool": ("historical pooled sections" if origin == ORIGIN_SCANNER
                     else "experiment arms (never pooled with scanner)"),
        })
    return pd.DataFrame(rows)


def _arm_groups(
    recs: list[EvalRecord],
    origins: dict[RunKey, str],
    pose_cache: PoseRunCache,
) -> pd.DataFrame:
    """Experimental runs grouped by **arm** (``configHash``), one row per arm per origin.

    The arm is the experimental condition, and two runs differing in *any* factor — mode,
    preprocessing, crop policy, crop trajectory, model weights, module version — carry
    different hashes and therefore land in different groups. That is the property PRD #156
    rests on: without it, two arms pool as one and the experiment silently degrades back
    into the observational corpus it exists to escape (issue #149's failure mode).

    ``passIndex`` rides along so a repeat set is visible as repeats rather than as arms.

    Each row also carries the run's **outcomes** (issue #164): agreement PCK first, because
    that is the primary outcome an arm comparison is read on, plus coverage, the frame-class
    shares the harness can produce, and the funnel/crop metrics it structurally cannot.
    The absent ones are carried as ``None`` rather than 0 — a detector that emits no
    Detector Attempt stream has not rejected nothing, it has not been asked.
    """

    rows: list[dict[str, Any]] = []
    for rec in recs:
        run = _pose_run_cached(rec, pose_cache)
        if run is None:
            continue
        row = {
            "origin": record_origin(rec, origins),
            "config_hash": run.config_hash,
            "arm": _arm_factor_label(run.config),
            # How many preprocessing steps the arm applies. Carried as data rather than
            # inferred from the rendered label, because the baseline arm is chosen on it
            # and a selection that depended on prose would move whenever the prose did.
            "preprocess_steps": _preprocess_step_count(run.config),
            "app_version": run.app_version,
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "pass_index": run.pass_index,
            "sampled_frames": run.sampled_frames,
            "truth_hash": rec.truth_hash,
            "conforms": record_conforms(rec.data),
            "trusted": record_trusted(rec.data),
        }
        row.update(_arm_outcomes(rec.data))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_ARM_COLUMNS)
    return pd.DataFrame(rows).sort_values(
        ["origin", "config_hash", "route_folder", "video_key", "pass_index"]
    ).reset_index(drop=True)


# Every column ``_arm_groups`` emits, so an empty corpus produces an empty frame of the
# right shape rather than one downstream consumers have to special-case.
_ARM_COLUMNS = [
    "origin", "config_hash", "arm", "preprocess_steps", "app_version", "route_folder",
    "video_key", "run_ts", "pass_index", "sampled_frames", "truth_hash",
    "conforms", "trusted",
    "pck", "coverage_rate", "norm_dist_median", "scoreable_frames",
    "class_ok_share", "class_hallucination_fp_share", "class_flipped_rotated_share",
    "class_wrong_subject_share", "class_distorted_share",
] + list(floors.FUNNEL_FLOOR_KEYS)


def _arm_factor_label(config: dict[str, Any] | None) -> str:
    """The arm's factors as prose — ``mode 1 · contrast(1.5) · crop:tracked``.

    A ``configHash`` is a grouping key, not a description. Two arms that differ are
    guaranteed to carry different digests, but no reader can see *which factor* differs
    from ``26f1333d`` versus ``e1ddb710``, and "which settings work for which videos" is
    precisely the question this reporting exists to answer. So the block the hash was taken
    over is rendered beside it.

    Deliberately built from whatever keys the block holds rather than a fixed list: a
    factor added later (a new preprocessing step, a delegate, a resolution) shows up in the
    label the moment it shows up in the stamp, instead of being silently invisible until
    someone remembers to extend this. Unknown keys print as ``key=value``.
    """

    if not isinstance(config, dict) or not config:
        return ""
    parts: list[str] = []
    mode = config.get("mode")
    if mode is not None:
        parts.append(f"mode {mode}")
    steps = config.get("preprocess")
    if isinstance(steps, list):
        if not steps:
            parts.append("no preprocessing")
        for step in steps:
            if not isinstance(step, dict):
                continue
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            args = ", ".join(f"{k}={_trim_num(v)}" for k, v in sorted(params.items()))
            parts.append(f"{step.get('name') or 'step'}({args})" if args
                         else str(step.get("name") or "step"))
    crop = config.get("crop")
    if crop is not None:
        parts.append(f"crop:{crop}")
    for key in sorted(config):
        # Identity of the *build*, not of the condition: the module version, the model pin
        # and the crop trajectory all belong in the hash (a change to any of them changes
        # the output) but they are constant across a comparison and would drown the label.
        if key in {"mode", "preprocess", "crop", "moduleVersion", "origin",
                   "modelSha", "cropTrackHash"}:
            continue
        parts.append(f"{key}={config[key]}")
    return " · ".join(parts)


def _preprocess_step_count(config: dict[str, Any] | None) -> int:
    """How many pixel filters the arm applies. An unstamped arm counts as 0."""

    steps = (config or {}).get("preprocess")
    return len(steps) if isinstance(steps, list) else 0


def _trim_num(v: Any) -> str:
    """``1.5`` not ``1.5000000000000002``; ``-20`` not ``-20.0``."""

    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _share(counts: Any, key: str) -> float | None:
    """One count as a share of its group's total, or ``None`` when nothing was counted.

    ``None`` and 0.0 are different answers and the difference decides how a row reads: a
    funnel with no attempts at all has no accepted *share*, and printing 0.0 would claim
    the detector accepted nothing.
    """

    if not isinstance(counts, dict):
        return None
    total = sum(v for v in counts.values() if isinstance(v, (int, float)))
    value = counts.get(key)
    if not total or not isinstance(value, (int, float)):
        return None
    return float(value) / float(total)


def _arm_outcomes(data: dict[str, Any]) -> dict[str, Any]:
    """One run's outcome metrics, keyed to match ``floors.SCANNER_FLOORS``.

    Keys line up with the floor registry on purpose, so the report looks a floor up by
    column name instead of matching prose — the coupling that stops a metric from being
    displayed without the noise it should be read against.
    """

    agg = ((data.get("agreement") or {}).get("aggregate") or {})
    frames = ((data.get("agreement") or {}).get("frames") or {})
    fq = data.get("frameQuality") or {}
    rejection = fq.get("rejectionCorrectness") or {}
    crop = data.get("cropQuality") or {}
    classes = fq.get("classCounts") or {}
    statuses = fq.get("detectorAttemptStatusCounts") or {}
    misses = crop.get("missCauseCounts") or {}
    return {
        "pck": (agg.get("pck") or {}).get("value"),
        "coverage_rate": (agg.get("coverage") or {}).get("rate"),
        "norm_dist_median": (agg.get("normDist") or {}).get("median"),
        "scoreable_frames": frames.get("scoreable"),
        "class_ok_share": _share(classes, "ok"),
        "class_hallucination_fp_share": _share(classes, "hallucination-fp"),
        "class_flipped_rotated_share": _share(classes, "flipped-rotated"),
        "class_wrong_subject_share": _share(classes, "wrong-subject"),
        "class_distorted_share": _share(classes, "distorted"),
        # Funnel-derived. Structurally absent on a harness arm — the harness emits no
        # Detector Attempt stream — and ``_share`` returns None rather than 0 for that.
        "funnel_accepted_share": _share(statuses, "accepted"),
        "funnel_missing_share": _share(statuses, "missing"),
        "funnel_flip_rejected_share": _share(statuses, "flipRejected"),
        "over_rejection_rate": rejection.get("overRejectionRate"),
        "over_rejection_rate_truth_present": rejection.get("overRejectionRateTruthPresent"),
        "crop_contained_rate": (crop.get("cropContainedTruth") or {}).get("rate"),
        "crop_iou_median": (crop.get("detectionRegionIou") or {}).get("median"),
        "miss_no_candidates_share": _share(misses, "no-candidates"),
        "miss_identity_gated_share": _share(misses, "identity-gated"),
        "miss_climber_absent_share": _share(misses, "climber-absent"),
    }


def _pose_run_cached(rec: EvalRecord, pose_cache: PoseRunCache) -> PoseRun | None:
    """The cached pose run for a record, without touching disk. ``None`` if absent."""

    return pose_cache.get((rec.route_folder, rec.video_key), {}).get(rec.run_ts)


# --------------------------------------------------------------------------- #
# Arm-versus-arm reporting (issue #164)
#
# The corpus this PRD exists to escape produced pooled numbers that could not be
# attributed, and a pooled mean is exactly the shape that failure takes. Every derivation
# below is therefore **per Bundle first**, with the pooled figure printed beside the spread
# rather than instead of it: tracked-crop detection ranged 59–100% across six Bundles, so a
# pooled median of 81% describes none of them.
# --------------------------------------------------------------------------- #

def _arm_bundle_pck(arms: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, Bundle): the PCK for that condition on that video.

    An arm may hold several runs on one Bundle. Two legitimate reasons and one illegitimate
    one, and they must not be confused:

    - **repeats** (``passIndex`` 0,1,2…) — which on this detector are byte-identical, so
      their PCK range is 0 and collapsing them loses nothing;
    - **the same arm re-run over a different frame set** — a full-grid proof run and a
      ``12·√n`` sampled batch carry the same stamp because the arm identity deliberately
      does not name the frame set. Those are *not* repeats, and the spread column is what
      exposes it.

    So the runs are collapsed to a median with the range kept beside it, and the range is
    the tripwire: on a bit-deterministic detector it must be exactly 0.
    """

    if arms.empty:
        return pd.DataFrame(columns=[
            "origin", "config_hash", "arm", "preprocess_steps", "route_folder",
            "video_key", "truth_hash", "runs", "pck", "pck_range", "sampled_frames",
            "frame_sets", "conforms", "trusted"])
    rows: list[dict[str, Any]] = []
    group_cols = ["origin", "config_hash", "route_folder", "video_key", "truth_hash"]
    for keys, g in arms.groupby(group_cols, dropna=False):
        vals = [float(v) for v in g["pck"] if isinstance(v, (int, float)) and pd.notna(v)]
        samples = {int(v) for v in g["sampled_frames"]
                   if isinstance(v, (int, float)) and pd.notna(v)}
        rows.append({
            **dict(zip(group_cols, keys)),
            "arm": str(g["arm"].iloc[0]),
            "preprocess_steps": int(g["preprocess_steps"].iloc[0]),
            "runs": int(len(g)),
            "pck": float(np.median(vals)) if vals else None,
            "pck_range": (max(vals) - min(vals)) if len(vals) > 1 else (0.0 if vals else None),
            "sampled_frames": (sorted(samples)[0] if len(samples) == 1 else None),
            "frame_sets": int(len(samples)),
            "conforms": bool(g["conforms"].all()),
            "trusted": bool(g["trusted"].all()),
        })
    return pd.DataFrame(rows).sort_values(
        ["origin", "config_hash", "route_folder", "video_key"]).reset_index(drop=True)


def _arm_overview(per_bundle: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: how many Bundles it ran, and the **spread** of PCK across them.

    The pooled median is printed, but never alone. "Which settings work for which videos"
    cannot be answered by a central value, and the min/max columns are what stop a reader
    from taking one.
    """

    cols = ["origin", "config_hash", "arm", "bundles", "runs", "pck_median",
            "pck_min", "pck_max", "pck_spread", "conforming_bundles"]
    if per_bundle.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for (origin, cfg), g in per_bundle.groupby(["origin", "config_hash"], dropna=False):
        vals = [float(v) for v in g["pck"] if isinstance(v, (int, float)) and pd.notna(v)]
        rows.append({
            "origin": origin,
            "config_hash": cfg,
            "arm": str(g["arm"].iloc[0]),
            "bundles": int(len(g)),
            "runs": int(g["runs"].sum()),
            "pck_median": float(np.median(vals)) if vals else None,
            "pck_min": min(vals) if vals else None,
            "pck_max": max(vals) if vals else None,
            "pck_spread": (max(vals) - min(vals)) if vals else None,
            "conforming_bundles": int(g["conforms"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["origin", "pck_median"], ascending=[True, False]).reset_index(drop=True)


def _arm_baseline(per_bundle: pd.DataFrame, origin: str) -> str:
    """Which arm the deltas are measured *against*, within one origin.

    The reference is the arm applying the **fewest preprocessing steps** — the untouched
    condition every other arm was built from — then the one that ran on the most Bundles,
    so the comparison has the widest shared set. Ties break on the hash: arbitrary, but
    stable across runs of the report, which matters more than which arbitrary answer.

    Selected on the step *count* rather than on the rendered label: a baseline that
    depended on prose would move whenever the prose did.

    Returned rather than assumed, and named in the output, because "improved by 0.04"
    means nothing without saying over what.
    """

    scope = per_bundle[per_bundle["origin"] == origin]
    if scope.empty:
        return ""
    ranked = sorted(
        scope.groupby("config_hash"),
        key=lambda kv: (int(kv[1]["preprocess_steps"].iloc[0]), -len(kv[1]), str(kv[0])),
    )
    return str(ranked[0][0])


def _arm_deltas(per_bundle: pd.DataFrame) -> pd.DataFrame:
    """Every arm against its origin's baseline arm, **paired on shared Bundles**.

    Paired, not a difference of pooled means. Bundles differ from each other far more than
    arms differ on one Bundle — the 59–100% spread again — so a pooled difference over
    non-identical Bundle sets measures which videos each arm happened to run on. Only
    Bundles *both* arms ran contribute, and a pair with no shared Bundle yields no row at
    all rather than a number that looks like a comparison.

    The delta's uncertainty is not the absolute's. Both arms scored the same ``12·√n``
    frames of the same Bundle against the same truth, so the sampling error is a shared
    offset that largely cancels; what remains is compared against its p90 anyway, as the
    conservative bar for "could this be nothing?".
    """

    cols = ["origin", "config_hash", "arm", "baseline_hash", "baseline_arm",
            "route_folder", "video_key", "pck", "baseline_pck", "delta_pck",
            "below_sampling_error", "conforms"]
    if per_bundle.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for origin, scope in per_bundle.groupby("origin", dropna=False):
        base_hash = _arm_baseline(scope, str(origin))
        base = scope[scope["config_hash"] == base_hash]
        base_pck = {(r.route_folder, r.video_key): r.pck for r in base.itertuples()}
        base_arm = str(base["arm"].iloc[0]) if not base.empty else ""
        for cfg, g in scope.groupby("config_hash", dropna=False):
            if cfg == base_hash:
                continue
            for r in g.itertuples():
                bundle = (r.route_folder, r.video_key)
                if bundle not in base_pck:
                    continue                      # not a comparison; say nothing
                bp, ap = base_pck[bundle], r.pck
                delta = (float(ap) - float(bp)) if (
                    isinstance(ap, (int, float)) and isinstance(bp, (int, float))
                    and pd.notna(ap) and pd.notna(bp)) else None
                rows.append({
                    "origin": origin,
                    "config_hash": cfg,
                    "arm": str(g["arm"].iloc[0]),
                    "baseline_hash": base_hash,
                    "baseline_arm": base_arm,
                    "route_folder": r.route_folder,
                    "video_key": r.video_key,
                    "pck": ap,
                    "baseline_pck": bp,
                    "delta_pck": delta,
                    "below_sampling_error": floors.below_sampling_error(delta),
                    "conforms": bool(r.conforms),
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(
        ["origin", "config_hash", "route_folder", "video_key"]).reset_index(drop=True)


def _arm_delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    """Each arm's delta pooled over its shared Bundles — with the per-video spread.

    Pooled *last* and never alone. An arm that helps one video by 0.10 and hurts another by
    0.10 has a pooled delta of zero and is not a null result, and the min/max columns are
    the only thing that distinguishes the two.
    """

    cols = ["origin", "config_hash", "arm", "baseline_hash", "baseline_arm",
            "shared_bundles", "delta_median", "delta_min", "delta_max",
            "bundles_improved", "bundles_regressed", "bundles_below_sampling_error",
            "all_below_sampling_error"]
    if deltas.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for (origin, cfg), g in deltas.groupby(["origin", "config_hash"], dropna=False):
        vals = [float(v) for v in g["delta_pck"]
                if isinstance(v, (int, float)) and pd.notna(v)]
        below = [bool(v) for v in g["below_sampling_error"] if isinstance(v, (bool, np.bool_))]
        rows.append({
            "origin": origin,
            "config_hash": cfg,
            "arm": str(g["arm"].iloc[0]),
            "baseline_hash": str(g["baseline_hash"].iloc[0]),
            "baseline_arm": str(g["baseline_arm"].iloc[0]),
            "shared_bundles": int(len(g)),
            "delta_median": float(np.median(vals)) if vals else None,
            "delta_min": min(vals) if vals else None,
            "delta_max": max(vals) if vals else None,
            "bundles_improved": sum(1 for v in vals if v > 0),
            "bundles_regressed": sum(1 for v in vals if v < 0),
            "bundles_below_sampling_error": sum(1 for v in below if v),
            "all_below_sampling_error": bool(below) and all(below),
        })
    return pd.DataFrame(rows).sort_values(
        ["origin", "delta_median"], ascending=[True, False]).reset_index(drop=True)


def _arm_comparison_reach(
    per_bundle: pd.DataFrame,
    overview: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> dict[str, Any]:
    """What the comparison can and cannot support — computed, not asserted in prose.

    Two ways an arm table lies while every cell in it is correct, and both are properties
    of the *sweep design* rather than of any number:

    - **Between-Bundle variation swamps the arm effect.** If the baseline arm's own PCK
      ranges more across Bundles than any arm moves it, then "arm A beats arm B" is a
      statement about which Bundle each ran on. The comparison is still valid — it is
      paired — but a reader must not generalise it, and the honest way to say so is to put
      the two magnitudes side by side.
    - **An arm was measured on one Bundle.** A single-Bundle delta is an anecdote with a
      decimal point. It is reported, because refusing to report it teaches nothing, but it
      is named as resting on one video.

    Neither is a defect to hide; both are what a three-Bundle sweep can honestly claim.
    """

    out: dict[str, Any] = {
        "baseline_hash": "", "baseline_spread": None, "baseline_arm": "",
        "baseline_bundles": 0, "max_abs_delta": None, "max_abs_delta_arm": "",
        "deltas_under_spread": 0, "delta_arms": 0, "single_bundle_arms": 0,
        "uncomparable_arms": [],
    }
    if per_bundle.empty:
        return out
    if not delta_summary.empty:
        base_hash = str(delta_summary["baseline_hash"].iloc[0])
        out["baseline_arm"] = str(delta_summary["baseline_arm"].iloc[0])
    else:
        base_hash = _arm_baseline(per_bundle, str(per_bundle["origin"].iloc[0]))
        row = overview[overview["config_hash"] == base_hash]
        out["baseline_arm"] = str(row["arm"].iloc[0]) if not row.empty else ""
    out["baseline_hash"] = base_hash
    base = overview[overview["config_hash"] == base_hash]
    if not base.empty:
        spread = base["pck_spread"].iloc[0]
        out["baseline_spread"] = (float(spread) if isinstance(spread, (int, float))
                                  and pd.notna(spread) else None)
        out["baseline_bundles"] = int(base["bundles"].iloc[0])

    if not delta_summary.empty:
        mags = [abs(float(v)) for v in delta_summary["delta_median"]
                if isinstance(v, (int, float)) and pd.notna(v)]
        out["max_abs_delta"] = max(mags) if mags else None
        if mags:
            out["max_abs_delta_arm"] = str(
                delta_summary.loc[delta_summary["delta_median"].abs().idxmax(), "arm"])
        out["delta_arms"] = int(len(delta_summary))
        spread = out["baseline_spread"]
        if spread is not None:
            out["deltas_under_spread"] = sum(1 for m in mags if m < spread)
        out["single_bundle_arms"] = int((delta_summary["shared_bundles"] == 1).sum())

    # Arms the baseline never met on a Bundle. Not a failure of the arm — a gap in the
    # sweep — and naming it is the difference between "no effect" and "not measured".
    compared = set(delta_summary["config_hash"]) if not delta_summary.empty else set()
    for cfg, g in per_bundle.groupby("config_hash", dropna=False):
        if cfg == base_hash or cfg in compared:
            continue
        out["uncomparable_arms"].append({
            "config_hash": str(cfg),
            "arm": str(g["arm"].iloc[0]),
            "bundles": int(len(g)),
        })
    return out


def _arm_repeat_checks(per_bundle: pd.DataFrame) -> list[dict[str, Any]]:
    """(arm, Bundle) groups whose several runs are not the repeats they look like.

    This check exists because the harness floor is *exactly* 0. On a bit-deterministic
    detector, two runs of one arm on one Bundle must agree to the last digit, so any
    nonzero range is not noise — it is evidence that the two runs are not the same
    measurement. Two ways that happens, both real:

    - the arm identity does not name the frame set (by design — that is what makes sampling
      error common-mode), so a full-grid run and a sampled run collide under one stamp;
    - the detector stopped being deterministic, which is what the Cycle canary (#168) is
      for and what this would catch between canaries.

    Reported as a named list rather than folded into a spread column, because a silent
    nonzero range would be read as ordinary scatter — the exact misreading the zero floor
    exists to prevent.
    """

    if per_bundle.empty:
        return []
    out: list[dict[str, Any]] = []
    for r in per_bundle.itertuples():
        if int(r.runs) < 2:
            continue
        rng = r.pck_range
        if not isinstance(rng, (int, float)) or pd.isna(rng) or rng == 0.0:
            continue
        out.append({
            "origin": r.origin,
            "config_hash": r.config_hash,
            "arm": r.arm,
            "route_folder": r.route_folder,
            "video_key": r.video_key,
            "runs": int(r.runs),
            "pck_range": float(rng),
            "frame_sets": int(r.frame_sets),
            "cause": _repeat_flag_cause(int(r.frame_sets)),
        })
    return sorted(out, key=lambda d: -d["pck_range"])


def _repeat_flag_cause(frame_sets: int) -> str:
    """Why an (arm, Bundle) group's runs disagree, as far as the stamps can establish it.

    Three answers, and the third is the honest one most of the time: runs written before
    the sweep recorded ``frameCount`` carry no frame count at all, so "these ran over
    different frames" and "the detector moved" are indistinguishable from the stamp alone.
    Saying so beats picking whichever is more flattering.
    """

    if frame_sets > 1:
        return ("differing frame counts — the arm stamp does not name the sampled frames "
                "by design, so these runs are not repeats")
    if frame_sets == 1:
        return ("same frame count, differing result — either the same-sized sample covered "
                "different frames, or the detector is no longer deterministic")
    return ("no frame count stamped on either run, so the stamps cannot say whether these "
            "ran over the same frames; re-run under the current module to find out")


# Video Stats source stats (issue #23) worth carrying into an arm comparison. Phase-1
# whole-frame statistics out of ``metadata.json``, stamped at download and therefore
# **never stale** — unlike the phase-2 region stats, which carry a ``setupHash`` and go
# stale exactly as Ground Truth does when recalibration mints a new one.
_ARM_SOURCE_STAT_PATHS = {
    "luma_mean": ("luma", "mean"),
    "rms_contrast": ("rmsContrast",),
    "sharpness_mean": ("sharpness", "mean"),
    "frame_diff_mean": ("frameDiff", "mean"),
}


def _arm_video_stats(analysis_root: Path, per_bundle: pd.DataFrame) -> pd.DataFrame:
    """The Video Stats condition Predictors for every Bundle an arm ran on.

    Without this an arm result is a property of the four videos that happened to be in the
    sweep. With it, a finding can be *stated as a condition* — "contrast preprocessing helps
    on low-contrast walls" generalises; "contrast preprocessing helps on planet-x" does not.

    Phase-1 source stats and phase-2 region stats are both carried, and the region stats
    carry their staleness with them: a ``setupHash`` that no longer matches the Bundle's
    setup means those columns describe a crop that has since moved.
    """

    cols = (["route_folder", "video_key", "arms", "vs_stale"]
            + list(_ARM_SOURCE_STAT_PATHS) + list(_VS_CONDITION_PATHS))
    if per_bundle.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for (route, key), g in per_bundle.groupby(["route_folder", "video_key"], dropna=False):
        video_dir = analysis_root / str(route) / str(key)
        metadata, setup = _bundle_meta(video_dir)
        source = metadata.get("video_stats") if isinstance(metadata, dict) else None
        row: dict[str, Any] = {
            "route_folder": route,
            "video_key": key,
            "arms": int(g["config_hash"].nunique()),
        }
        for name, path in _ARM_SOURCE_STAT_PATHS.items():
            cur: Any = source
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            row[name] = float(cur) if isinstance(cur, (int, float)) else None
        row.update(_video_stats_conditions(video_dir))
        stats_doc = video_dir / "video-stats.json"
        if stats_doc.exists():
            try:
                doc = _load_json(stats_doc)
            except Exception:
                doc = {}
            row["vs_stale"] = bool(
                (doc.get("setupHash") or "") != (setup.get("setupHash") or ""))
        else:
            row["vs_stale"] = None
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols].sort_values(["route_folder", "video_key"]).reset_index(drop=True)


def _bundle_meta(video_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(video_dir / "metadata.json")
    setup_path = video_dir / "setup.json"
    setup = _load_json(setup_path) if setup_path.exists() else {}
    return metadata, setup


def _resolution_bucket(metadata: dict[str, Any]) -> str:
    src = metadata.get("source_video", {}) if isinstance(metadata, dict) else {}
    h = src.get("height")
    if isinstance(h, (int, float)) and h > 0:
        return f"{int(h)}p"
    return "unknown"


def _frame_bbox_metrics(joints: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    if not joints:
        return None
    xs = [v[0] for v in joints.values()]
    ys = [v[1] for v in joints.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox_h = max(0.0, ymax - ymin)
    edge_dist = max(0.0, min(xmin, 1 - xmax, ymin, 1 - ymax))
    return bbox_h, edge_dist


def _build_frame_joint_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: PoseRunCache,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in recs:
        video_dir = analysis_root / rec.route_folder / rec.video_key
        if not video_dir.exists():
            continue
        truth = load_truth(video_dir)
        if truth is None:
            continue
        if rec.truth_hash and truth.truth_hash and rec.truth_hash != truth.truth_hash:
            # Keep trend analysis anchored to the same truth revision as the record.
            continue

        pose_run = _pose_run(analysis_root, rec, pose_cache)
        if pose_run is None:
            continue  # no pose file for this run_ts — nothing to score against truth
        app_version = pose_run.app_version
        detector_code_hash = pose_run.detector_code_hash
        pose_frames = pose_run.frames

        metadata, setup = _bundle_meta(video_dir)
        source_type = str(metadata.get("source_type") or "unknown")
        resolution = _resolution_bucket(metadata)
        panning = setup.get("panning")
        panning_label = "panning" if panning is True else "static" if panning is False else "unknown"

        scanner_ts = sorted(float(f.get("timestamp", 0.0)) for f in pose_frames)
        if not scanner_ts:
            continue
        by_ts = {float(f.get("timestamp", 0.0)): f for f in pose_frames}
        interval = _scanner_frame_interval(scanner_ts)
        tol = interval / 2

        scored_frames: list[dict[str, Any]] = []
        for tf in truth.frames:
            if tf.excluded:
                continue  # known-bad seed or deprecated manual flag (ADR 0005)
            if not tf.present:
                continue
            torso = torso_length(tf.joints)
            if torso is None:
                continue
            bm = _frame_bbox_metrics(tf.joints)
            if bm is None:
                continue
            idx = _nearest_within(scanner_ts, tf.timestamp, tol)
            scanner = _pose_frame_joints(by_ts[scanner_ts[idx]]) if idx is not None else {}
            cx = sum(j[0] for j in tf.joints.values()) / len(tf.joints)
            cy = sum(j[1] for j in tf.joints.values()) / len(tf.joints)
            scored_frames.append({
                "timestamp": tf.timestamp,
                "verified": bool(tf.verified),
                "torso": torso,
                "bbox_h": bm[0],
                "edge_dist": bm[1],
                "cx": cx,
                "cy": cy,
                "truth_joints": tf.joints,
                "scanner": scanner,
            })

        scored_frames.sort(key=lambda r: r["timestamp"])
        prev_center: tuple[float, float] | None = None
        for sf in scored_frames:
            center = (sf["cx"], sf["cy"])
            speed = None
            if prev_center is not None:
                speed = _dist(center, prev_center)
            prev_center = center

            for joint in COCO_CORE_JOINTS:
                truth_pt = sf["truth_joints"].get(joint)
                if truth_pt is None:
                    continue
                pred = sf["scanner"].get(joint)
                norm_dist = None
                correct = 0
                if pred is not None:
                    norm_dist = _dist(pred, truth_pt) / sf["torso"]
                    correct = 1 if norm_dist <= 0.5 else 0
                base = {
                    "route_folder": rec.route_folder,
                    "video_key": rec.video_key,
                    "run_ts": rec.run_ts,
                    "app_version": app_version,
                    "detector_code_hash": detector_code_hash,
                    "truth_hash": truth.truth_hash,
                    "source_type": source_type,
                    "resolution": resolution,
                    "panning": panning_label,
                    "joint": joint,
                    "correct": correct,
                    "failure": 1 - correct,
                    "norm_dist": norm_dist,
                    "size_frac": sf["bbox_h"],
                    "speed": speed,
                    "edge_dist": sf["edge_dist"],
                }
                rows.append({**base, "tier": "agreement"})
                if sf["verified"]:
                    rows.append({**base, "tier": "accuracy"})

    return pd.DataFrame(rows)


def _joint_ranking(frame_joint_df: pd.DataFrame) -> pd.DataFrame:
    if frame_joint_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (tier, joint), g in frame_joint_df.groupby(["tier", "joint"]):
        vals = g["correct"].astype(int).tolist()
        boot = _bootstrap_rate(vals)
        if boot is None:
            continue
        rows.append({
            "tier": tier,
            "joint": joint,
            "n": len(vals),
            "pck": boot[0],
            "ci_low": boot[1],
            "ci_high": boot[2],
            "failure_rate": 1 - boot[0],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["tier", "pck", "joint"], ascending=[True, True, True])


def _condition_bands(frame_joint_df: pd.DataFrame, col: str, bins: int = 3) -> pd.DataFrame:
    """Failure rate per quantile band of a geometric condition, CI'd at the run unit.

    A frame/joint row is not an independent observation — one run contributes thousands
    of them — so the band's interval comes from ``_run_unit_rate``'s cluster bootstrap
    and the per-run median/p90 travel with it (#70)."""

    if frame_joint_df.empty or col not in frame_joint_df.columns:
        return pd.DataFrame()
    key_cols = [c for c in _RUN_KEY_COLS if c in frame_joint_df.columns]
    rows: list[dict[str, Any]] = []
    for tier, tg in frame_joint_df.groupby("tier"):
        d = tg[[*key_cols, col, "failure"]].dropna(subset=[col, "failure"])
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            stats = _run_unit_rate(bg.assign(failure=bg["failure"].astype(int)), "failure")
            if stats is None:
                continue
            rows.append({
                "tier": tier,
                "condition": col,
                "band": int(band) + 1,
                "n": stats["n"],
                "n_runs": stats["n_runs"],
                "failure_rate": stats["rate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "run_rate_median": stats["run_rate_median"],
                "run_rate_p90": stats["run_rate_p90"],
                "band_min": float(bg[col].min()),
                "band_max": float(bg[col].max()),
            })
    return pd.DataFrame(rows)


def _cross_video_splits(recs: list[EvalRecord], analysis_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in recs:
        video_dir = analysis_root / rec.route_folder / rec.video_key
        if not video_dir.exists():
            continue
        metadata, setup = _bundle_meta(video_dir)
        row_base = {
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "source_type": str(metadata.get("source_type") or "unknown"),
            "resolution": _resolution_bucket(metadata),
            "panning": "panning" if setup.get("panning") is True else "static" if setup.get("panning") is False else "unknown",
        }
        for tier in ("agreement", "accuracy"):
            agg = ((rec.data.get(tier) or {}).get("aggregate") or {})
            pck = ((agg.get("pck") or {}).get("value"))
            cov = ((agg.get("coverage") or {}).get("rate"))
            if pck is None and cov is None:
                continue
            rows.append({
                **row_base,
                "tier": tier,
                "pck": pck,
                "coverage": cov,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    out_rows: list[dict[str, Any]] = []
    split_cols = ["resolution", "panning", "source_type"]
    for split_col in split_cols:
        for (tier, split_val), g in df.groupby(["tier", split_col]):
            for metric in ("pck", "coverage"):
                vals = [float(v) for v in g[metric].dropna().tolist()]
                if len(vals) < 2:
                    continue
                boot = _bootstrap_rate([1 if v >= 0.5 else 0 for v in vals])
                rng = random.Random(BOOT_SEED)
                draws = []
                n = len(vals)
                for _ in range(N_BOOT):
                    s = [vals[rng.randrange(n)] for _ in range(n)]
                    draws.append(sum(s) / n)
                lo, hi = _pct_ci(draws)
                out_rows.append({
                    "tier": tier,
                    "split": split_col,
                    "value": str(split_val),
                    "metric": metric,
                    "n_runs": n,
                    "mean": sum(vals) / n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "share_ge_0_5": boot[0] if boot is not None else None,
                })
    return pd.DataFrame(out_rows)


def _bootstrap_rate_delta(a: list[int], b: list[int],
                          n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Delta of means ``b - a`` for 0/1 outcomes with a percentile bootstrap CI.

    Resampling n iid 0/1 values and taking the mean is Binomial(n, p̂)/n, so the
    bootstrap draws come straight from the binomial (vectorised, deterministic).
    """

    rng = np.random.default_rng(BOOT_SEED)
    na, nb = len(a), len(b)
    pa, pb = sum(a) / na, sum(b) / nb
    draws = rng.binomial(nb, pb, n_boot) / nb - rng.binomial(na, pa, n_boot) / na
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return (pb - pa, float(lo), float(hi))


def _bootstrap_median_delta(a: list[float], b: list[float],
                            n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Delta of medians ``b - a`` with a percentile bootstrap CI."""

    rng = np.random.default_rng(BOOT_SEED)

    def boot_medians(vals: list[float]) -> np.ndarray:
        v = np.asarray(vals, dtype=float)
        n = len(v)
        out = np.empty(n_boot)
        batch = max(1, 20_000_000 // n)  # cap the index matrix at ~20M cells
        i = 0
        while i < n_boot:
            j = min(n_boot, i + batch)
            out[i:j] = np.median(v[rng.integers(0, n, size=(j - i, n))], axis=1)
            i = j
        return out

    draws = boot_medians(b) - boot_medians(a)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    delta = float(np.median(np.asarray(b)) - np.median(np.asarray(a)))
    return (delta, float(lo), float(hi))


_ALL_JOINTS = "(all joints)"


def _build_identity(build: BuildId) -> str:
    """The behavioural grouping key for a run.

    The ``detectorCodeHash`` when the scanner emitted one, because that is the thing
    that actually determines detection behaviour. Two *different* commits sharing a hash
    did not touch detection, so they are one behavioural group and pooling them is a
    gain in n rather than a loss of resolution — the benefit of this field that runs
    opposite to conflict detection.

    Falls back to the commit stamp when there is no hash, which is every run predating
    #130. An unhashed group and a hashed group are never merged: they might be the same
    code, but nothing on disk says so, and guessing is what this field exists to stop.
    """

    app_version, detector_code_hash = build
    return detector_code_hash or app_version


def _build_label(identity: str, builds: set[BuildId]) -> str:
    """Display name for a behavioural group.

    Unhashed groups keep the bare appVersion, so every row that existed before #130
    renders exactly as it did. Hashed groups name both halves, and a group spanning more
    than one commit shows all of them — that is the pooling gain made visible.
    """

    app_versions = sorted({av for av, _ in builds if av})
    if not any(h for _, h in builds):
        return " / ".join(app_versions)
    if not app_versions:
        return identity  # hashed but unstamped — name it by the hash, not "·hash"
    return f"{'+'.join(app_versions)}·{identity}"


def _build_identity_conflicts(pose_cache: PoseRunCache) -> tuple[pd.DataFrame, list[str]]:
    """appVersions that stamp more than one ``detectorCodeHash`` (#130).

    This is the ``c305954`` signature: one build stamp covering runs that executed
    *different* detector code, which a Next dev hot reload produces without moving
    ``appVersion``.

    Scanned over **every** pose run in the cache, not just runs an evaluation record
    scored. A hot reload during an unscored batch contaminates exactly as much, and the
    contamination is most useful to know about *before* the scoring pass, not after.

    An empty hash never participates in a conflict. The scanner emits ``null`` when its
    derivation fails and records predating the field have no key at all; both mean
    unknown provenance, and unknown-versus-known is not a contradiction.
    """

    runs_by_version: dict[str, dict[str, list[str]]] = {}
    for runs in pose_cache.values():
        for run_ts, run in runs.items():
            if not run.app_version or not run.detector_code_hash:
                continue
            runs_by_version.setdefault(run.app_version, {}).setdefault(
                run.detector_code_hash, []).append(run_ts)

    rows: list[dict[str, Any]] = []
    flags: list[str] = []
    for app_version in sorted(runs_by_version):
        by_hash = runs_by_version[app_version]
        if len(by_hash) < 2:
            continue
        for detector_code_hash in sorted(by_hash, key=lambda h: min(by_hash[h])):
            run_tss = sorted(by_hash[detector_code_hash])
            rows.append({
                "app_version": app_version,
                "detector_code_hash": detector_code_hash,
                "n_runs": len(run_tss),
                "first_run_ts": run_tss[0],
                "last_run_ts": run_tss[-1],
            })
        flags.append(
            f"{app_version}: {len(by_hash)} distinct detectorCodeHash values across "
            f"{sum(len(v) for v in by_hash.values())} run(s) "
            f"({', '.join(sorted(by_hash))}) — one build stamp, more than one detector "
            "build. Runs under this appVersion are not a single behavioural group and "
            "are never pooled as one.")
    return pd.DataFrame(rows), flags


def _version_regression(
    recs: list[EvalRecord],
    frame_joint_df: pd.DataFrame,
    build_ids: dict[tuple[str, str, str], BuildId],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Group eval records by build identity and delta consecutive builds.

    The grouping key is the *pair* ``(appVersion, detectorCodeHash)``, not the stamp
    alone (#130). A build that hot-reloaded mid-batch therefore splits into its real
    behavioural groups instead of averaging them, and the inverse case pays off too: two
    appVersions sharing one hash are a commit that did not touch detection, so their
    runs stay comparable rather than reading as a version boundary.

    Builds are ordered by first-seen run timestamp. For each consecutive pair the
    comparison pool is restricted to ``(video, truthHash)`` combos with records on
    *both* sides — a truth revision must never masquerade as a scanner change — and
    per-joint PCK / median-error deltas carry bootstrap CIs so noise at small n reads as
    noise. Videos where both builds ran but never under the same truth are flagged as
    mixed-truth and excluded.

    A missing hash groups on the appVersion alone rather than dropping out: fail-open is
    the contract on both sides, and 495 of the 499 runs on disk when this landed predate
    the field.
    """

    flags: list[str] = []
    by_build: dict[str, list[EvalRecord]] = {}
    builds_of: dict[str, set[BuildId]] = {}
    unknown = 0
    for rec in recs:
        build = build_ids.get((rec.route_folder, rec.video_key, rec.run_ts), ("", ""))
        if not build[0] and not build[1]:
            unknown += 1
            continue
        identity = _build_identity(build)
        by_build.setdefault(identity, []).append(rec)
        builds_of.setdefault(identity, set()).add(build)
    if unknown:
        flags.append(
            f"{unknown} evaluation record(s) with no build identity at all "
            "(neither appVersion nor detectorCodeHash in the pose diagnostics) "
            "excluded from version tracking")

    ordered = sorted(by_build, key=lambda b: min(r.run_ts for r in by_build[b]))
    labels = {b: _build_label(b, builds_of[b]) for b in ordered}

    overview = pd.DataFrame([{
        "app_version": labels[b],
        "detector_code_hash": b if any(h for _, h in builds_of[b]) else "",
        "first_run_ts": min(r.run_ts for r in by_build[b]),
        "last_run_ts": max(r.run_ts for r in by_build[b]),
        "n_records": len(by_build[b]),
        "n_videos": len({(r.route_folder, r.video_key) for r in by_build[b]}),
    } for b in ordered])

    if frame_joint_df.empty:
        pool_key = pd.Series(dtype=object)
        fj_identity = pd.Series(dtype=object)
    else:
        pool_key = pd.Series(
            list(zip(frame_joint_df["route_folder"], frame_joint_df["video_key"],
                     frame_joint_df["truth_hash"])),
            index=frame_joint_df.index)
        # Same hash-first rule as _build_identity, vectorised over the frame rows.
        fj_identity = frame_joint_df["detector_code_hash"].where(
            frame_joint_df["detector_code_hash"].astype(bool),
            frame_joint_df["app_version"])

    delta_rows: list[dict[str, Any]] = []
    for build_a, build_b in zip(ordered, ordered[1:]):
        va, vb = labels[build_a], labels[build_b]
        truths: list[dict[tuple[str, str], set[str]]] = []
        for build in (build_a, build_b):
            per_video: dict[tuple[str, str], set[str]] = {}
            for r in by_build[build]:
                if r.truth_hash:
                    per_video.setdefault((r.route_folder, r.video_key), set()).add(r.truth_hash)
            truths.append(per_video)
        truths_a, truths_b = truths

        comparable: set[tuple[str, str, str]] = set()
        for vid in sorted(set(truths_a) & set(truths_b)):
            shared = truths_a[vid] & truths_b[vid]
            if shared:
                comparable.update((vid[0], vid[1], th) for th in shared)
            else:
                flags.append(
                    f"{va} → {vb}: {vid[0]}/{vid[1]} has runs from both versions "
                    "but never under the same truth revision — excluded (mixed truth)")
        if not comparable:
            flags.append(f"{va} → {vb}: no videos with both versions under a "
                         "shared truth revision — no deltas computed")
            continue
        if frame_joint_df.empty:
            continue

        n_videos = len({(r, k) for r, k, _ in comparable})
        in_pool = pool_key.isin(comparable)
        # Select on the behavioural identity, not the label — a group can span more than
        # one commit, and the frame rows carry both halves raw.
        sub_a = frame_joint_df[(fj_identity == build_a) & in_pool]
        sub_b = frame_joint_df[(fj_identity == build_b) & in_pool]
        for tier in ("agreement", "accuracy"):
            ta = sub_a[sub_a["tier"] == tier]
            tb = sub_b[sub_b["tier"] == tier]
            if ta.empty or tb.empty:
                continue
            for joint in [_ALL_JOINTS, *COCO_CORE_JOINTS]:
                ja = ta if joint == _ALL_JOINTS else ta[ta["joint"] == joint]
                jb = tb if joint == _ALL_JOINTS else tb[tb["joint"] == joint]
                a_correct = ja["correct"].astype(int).tolist()
                b_correct = jb["correct"].astype(int).tolist()
                if not a_correct or not b_correct:
                    continue
                pck_delta, pck_lo, pck_hi = _bootstrap_rate_delta(a_correct, b_correct)
                a_dist = ja["norm_dist"].dropna().tolist()
                b_dist = jb["norm_dist"].dropna().tolist()
                if a_dist and b_dist:
                    med_a = float(np.median(a_dist))
                    med_b = float(np.median(b_dist))
                    med_delta, med_lo, med_hi = _bootstrap_median_delta(a_dist, b_dist)
                else:
                    med_a = med_b = med_delta = med_lo = med_hi = math.nan
                delta_rows.append({
                    "from_version": va,
                    "to_version": vb,
                    "tier": tier,
                    "joint": joint,
                    "n_videos": n_videos,
                    "n_from": len(a_correct),
                    "n_to": len(b_correct),
                    "pck_from": sum(a_correct) / len(a_correct),
                    "pck_to": sum(b_correct) / len(b_correct),
                    "pck_delta": pck_delta,
                    "pck_ci_low": pck_lo,
                    "pck_ci_high": pck_hi,
                    "med_from": med_a,
                    "med_to": med_b,
                    "med_delta": med_delta,
                    "med_ci_low": med_lo,
                    "med_ci_high": med_hi,
                })

    return overview, pd.DataFrame(delta_rows), flags


def _stale_truth_worklist(analysis_root: Path) -> list[dict[str, Any]]:
    """Bundles whose Ground Truth has fallen behind the scaffold it was authored from.

    Worst shortfall first — this is a re-review queue, and the Bundle whose truth records
    zero present frames against a fully-posed scaffold is the one contributing the most
    phantom absences.
    """

    rows: list[dict[str, Any]] = []
    for video_dir in _iter_video_dirs(analysis_root):
        drift = scaffold_truth_drift(video_dir)
        if not drift or not drift["drifted"]:
            continue
        metadata = _load_json(video_dir / "metadata.json")
        rows.append({
            "route_folder": str(metadata.get("route_folder") or video_dir.parent.name),
            "video_key": str(metadata.get("video_key") or video_dir.name),
            "truth_present": drift["truthPresent"],
            "scaffold_posed": drift["scaffoldPosed"],
            "shortfall": drift["shortfall"],
            "ratio": drift["ratio"],
            "scaffold_seed_hash": drift["scaffoldSeedHash"],
        })
    return sorted(rows, key=lambda r: -r["shortfall"])


def _shame_lists(analysis_root: Path) -> tuple[list[str], list[str]]:
    no_truth: list[str] = []
    stale_runs: list[str] = []
    for video_dir in _iter_video_dirs(analysis_root):
        metadata = _load_json(video_dir / "metadata.json")
        route = str(metadata.get("route_folder") or video_dir.parent.name)
        key = str(metadata.get("video_key") or video_dir.name)
        truth = load_truth(video_dir)
        if truth is None:
            no_truth.append(f"{route}/{key}")
            continue
        setup = _load_json(video_dir / "setup.json") if (video_dir / "setup.json").exists() else {}
        effective_setup_hash = truth.setup_hash or setup.get("setupHash", "")
        for run_ts, pose_setup_hash, _, _ in _iter_pose_runs(video_dir / "detections"):
            if pose_setup_hash != effective_setup_hash:
                stale_runs.append(
                    f"{route}/{key} {run_ts} (run {pose_setup_hash[:8] or '∅'} vs truth {effective_setup_hash[:8] or '∅'})"
                )
    return no_truth, stale_runs


# Worklist rows to surface in the report (the truth re-review queue is long; the
# CSV keeps the full list, the HTML shows the worst K).
LOW_CONF_WORKLIST_TOP_K = 40


def _visible_histogram(recs: list[EvalRecord]) -> list[int]:
    """Corpus visible-joint histogram, index ``i`` == matched-present frames whose
    truth carried ``i`` non-occluded core joints, pooled across records from each
    agreement tier's ``visibleJoints``. This is the measure-first fit input for
    ``evaluate.MIN_VISIBLE_JOINTS`` — the exact population the gate would act on
    (matched-present frames). Records predating schema v3 simply contribute nothing.
    """

    hist = [0] * (len(COCO_CORE_JOINTS) + 1)
    for rec in recs:
        vj = (rec.data.get("agreement") or {}).get("visibleJoints") or []
        if not isinstance(vj, list):
            continue  # pre-v3 records carried no positional histogram
        for i, v in enumerate(vj):
            if 0 <= i < len(hist):
                hist[i] += int(v)
    return hist


def _low_confidence_worklist(analysis_root: Path) -> pd.DataFrame:
    """Present truth frames ranked by fewest visible (non-occluded) core joints —
    the re-seed / re-review queue for low-confidence truth.

    Truth-side and per-bundle (independent of scanner runs), so a bundle's frames
    are listed once regardless of how many pose runs it has. Excluded frames
    (flagged-wrong / deprecated manual-absent) are skipped. A frame's occluded
    joints are the core joints ``load_truth`` dropped as occluded (ADR 0004),
    i.e. the ones ViTPose was not confident about.
    """

    rows: list[dict[str, Any]] = []
    for video_dir in _iter_video_dirs(analysis_root):
        truth = load_truth(video_dir)
        if truth is None:
            continue
        metadata = _load_json(video_dir / "metadata.json")
        route = str(metadata.get("route_folder") or video_dir.parent.name)
        key = str(metadata.get("video_key") or video_dir.name)
        for tf in truth.frames:
            if tf.excluded or not tf.present:
                continue
            occluded = [j for j in COCO_CORE_JOINTS if j not in tf.joints]
            rows.append({
                "route_folder": route,
                "video_key": key,
                "timestamp": tf.timestamp,
                "visible": len(tf.joints),
                "occluded_joints": ", ".join(occluded),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["visible", "route_folder", "video_key", "timestamp"],
        ascending=True,
    ).reset_index(drop=True)


# Per-frame quality worklist rows to surface in the HTML (the CSV keeps the full list).
FRAME_QUALITY_WORKLIST_TOP_K = 40

# The auto classes that count as a detection-quality *failure* (issue #44 deliverable 1);
# ``ok`` is the only non-failure. ``frozen-stale`` is a cross-cutting flag, not a class.
_FQ_FLAGGED = frozenset({"wrong-subject", "hallucination-fp", "flipped-rotated", "distorted"})

# Worst-first severity order for the worklist.
_FQ_SEVERITY = {"hallucination-fp": 0, "wrong-subject": 1, "flipped-rotated": 2,
                "distorted": 3, "ok": 4}

# A small set of numeric Video Stats conditions (issue #23) to band the per-frame class
# rate against (issue #44 deliverable 3). Nested key paths into ``video-stats.json``.
_VS_CONDITION_PATHS = {
    "wall_luma_mean": ("regionStats", "wall", "luma", "mean"),
    "wall_rms_contrast": ("regionStats", "wall", "rmsContrast"),
    "climber_wall_deltaE": ("regionStats", "climberWall", "deltaE"),
    "shadow_fraction": ("regionStats", "shadow", "fraction", "mean"),
}

_ATTEMPT_CONDITION_KEYS = {
    "mean": "luma_mean",
    "stdDev": "luma_stdDev",
    "sharpness": "sharpness",
}
_ATTEMPT_REGION_METRICS = ("area", "cx", "cy", "edge_distance")


def _attempts_by_timestamp(
    attempts: list[dict[str, Any]] | None,
) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for attempt in attempts or []:
        out[round(float(attempt.get("timestamp", 0.0)), 1)] = attempt
    return out


def _attempt_frame_context(
    attempt: dict[str, Any] | None,
    evidence: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "detector_attempt_evidence": evidence,
        "detector_attempt_status": None,
        "reacquire_attempted": None,
        "reacquire_succeeded": None,
        "reacquire_failed": None,
        "candidate_count": None,
        "rejected_candidate_count": None,
        "selection_method": None,
    }
    for prefix in ("initial_search_region", "detection_region"):
        for metric in _ATTEMPT_REGION_METRICS:
            out[f"{prefix}_{metric}"] = None
    for prefix in ("search", "reacquire"):
        for suffix in _ATTEMPT_CONDITION_KEYS.values():
            out[f"{prefix}_{suffix}"] = None

    if not isinstance(attempt, dict):
        return out

    attempted = bool(attempt.get("reacquireAttempted"))
    reacquired = bool(attempt.get("reacquired"))
    out.update({
        "detector_attempt_status": attempt.get("status"),
        "reacquire_attempted": attempted,
        "reacquire_succeeded": attempted and reacquired,
        "reacquire_failed": attempted and not reacquired,
        "candidate_count": attempt.get("candidateCount"),
        "rejected_candidate_count": attempt.get("rejectedCandidateCount"),
        "selection_method": attempt.get("selectionMethod"),
    })
    for prefix, source_key in (
        ("initial_search_region", "initialSearchRegion"),
        ("detection_region", "detectionRegion"),
    ):
        for metric in _ATTEMPT_REGION_METRICS:
            out[f"{prefix}_{metric}"] = _region_metric(attempt.get(source_key), metric)
    for prefix, source_key in (
        ("search", "searchConditions"),
        ("reacquire", "reacquireConditions"),
    ):
        conditions = attempt.get(source_key)
        for src_key, suffix in _ATTEMPT_CONDITION_KEYS.items():
            value = conditions.get(src_key) if isinstance(conditions, dict) else None
            out[f"{prefix}_{suffix}"] = float(value) if isinstance(value, (int, float)) else None
    for name, value in _condition_flags(attempt.get("searchConditions")).items():
        out[f"search_flag_{name}"] = value
    return out


def _video_stats_conditions(video_dir: Path) -> dict[str, float]:
    """Numeric Video Stats condition values for one bundle (issue #23 → #44), or {}."""

    path = video_dir / "video-stats.json"
    if not path.exists():
        return {}
    try:
        doc = _load_json(path)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for name, keys in _VS_CONDITION_PATHS.items():
        cur: Any = doc
        for k in keys:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        if isinstance(cur, (int, float)):
            out[name] = float(cur)
    return out


def _frame_quality_rows(analysis_root: Path, recs: list[EvalRecord],
                        pose_cache: PoseRunCache | None = None) -> pd.DataFrame:
    """Pool every record's ``frameQuality`` frames into one long table (issue #44).

    Pooled across **all** records — including #15-quarantined and #44-loose ones —
    because the frames most worth fixing live in exactly those bundles; the trusted
    metric pool (conforming, setupHash-matched only) is an independent pool. Each row
    carries the bundle's Video Stats conditions so the class rate can be banded against
    them. Records predating schema v6 carry no ``frameQuality`` and contribute nothing."""

    rows: list[dict[str, Any]] = []
    vs_cache: dict[tuple[str, str], dict[str, float]] = {}
    pose_cache = {} if pose_cache is None else pose_cache
    for rec in recs:
        fq = rec.data.get("frameQuality")
        if not isinstance(fq, dict):
            continue
        vid = (rec.route_folder, rec.video_key)
        if vid not in vs_cache:
            vs_cache[vid] = _video_stats_conditions(
                analysis_root / rec.route_folder / rec.video_key)
        conds = vs_cache[vid]
        attempts = (_pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN).attempts
        attempt_index = _attempts_by_timestamp(attempts)
        attempt_evidence = "unknown" if attempts is None else "attempts"
        loose = bool(rec.data.get("loosePaired"))
        conforming = record_conforms(rec.data)
        cause = record_nonconformance_cause(rec.data)
        for e in fq.get("frames") or []:
            cls = str(e.get("class") or "ok")
            t = e.get("t")
            attempt = (
                attempt_index.get(round(float(t), 1))
                if isinstance(t, (int, float))
                else None
            )
            rows.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": t,
                "class": cls,
                "auto_class": e.get("autoClass"),
                "failure_class": e.get("failureClass"),
                # Tri-state on purpose (issue #69): True / False / None, where None is a
                # pre-schema-v12 frame that never recorded presence. Never coerce the
                # missing case to False — that would count unknown frames as real
                # false positives.
                "truth_present": e.get("truthPresent"),
                # Why an absent frame is absent (issue #101). ``None`` on a present
                # frame; a pre-v14 record carries no field and reads as ``unknown``
                # downstream — never as a confirmed absence.
                "absence_reason": e.get("absenceReason"),
                "source": e.get("source"),
                "distractor": e.get("distractor"),
                "annotation_setup_hash": e.get("annotationSetupHash"),
                "flagged": int(cls in _FQ_FLAGGED),
                "held_pose": int(bool(e.get("heldPose"))),
                "frozen_stale": int(bool(e.get("frozenStale"))),
                "centroid_dist": e.get("centroidDist"),
                "residual": e.get("residual"),
                # Rejection correctness (issue #85), read from the record rather than
                # re-derived. None on non-rejection frames and pre-v9 records.
                "rejection_verdict": e.get("rejectionVerdict"),
                "rejection_reason": e.get("rejectionReason"),
                "rejection_centroid_dist": e.get("rejectionCentroidDist"),
                "rejection_joint_agreement": e.get("rejectionJointAgreement"),
                "rejection_raw_class": e.get("rejectionRawClass"),
                "crop": e.get("crop"),
                "loose": loose,
                "conforming": conforming,
                "nonconformance_cause": cause,
                **_attempt_frame_context(attempt, attempt_evidence),
                **{f"vs_{k}": v for k, v in conds.items()},
            })
    return pd.DataFrame(rows)


def _truth_presence_counts(g: pd.DataFrame) -> dict[str, Any]:
    """Split one class's pooled frames by truth presence (issue #69, #101).

    Counts plus the two shares *within the class*, taken over the frames whose presence
    is actually known: a pre-schema-v12 record carries no ``truthPresent``, and folding
    those into the denominator would report a split the records never measured. When
    nothing is known the shares are ``None``, not 0.0.

    From v14 an absence additionally has to be **confirmed** to count (issue #101). An
    absence that is out of scope, never sampled or a tracking loss is reported as
    ``truth_absent_unconfirmed`` and kept out of both the numerator and the
    denominator — those frames are the 44% of the pooled absent population that made
    the old truth-absent share unsafe to act on."""

    col = g["truth_present"] if "truth_present" in g.columns else pd.Series(dtype=object)
    reasons = (g["absence_reason"] if "absence_reason" in g.columns
               else pd.Series([None] * len(g), index=g.index, dtype=object))
    known = col.notna()
    vals = col[known].astype(bool)
    present = int(vals.sum())
    absent_idx = vals[~vals].index
    confirmed_mask = reasons.reindex(absent_idx) == ABSENCE_CONFIRMED
    absent = int(confirmed_mask.sum())
    unconfirmed = int(len(absent_idx) - absent)
    n_known = present + absent
    return {
        "truth_present": present,
        "truth_absent": absent,
        "truth_absent_unconfirmed": unconfirmed,
        "truth_unknown": int(len(g) - present - len(absent_idx)),
        "truth_present_share": present / n_known if n_known else None,
        "truth_absent_share": absent / n_known if n_known else None,
    }


def _absence_reason_counts(fq_df: pd.DataFrame) -> pd.DataFrame:
    """How the pooled truth-*absent* frames split by reason (issue #101).

    The table that says how much of "the Climber was not there" actually means that.
    Every reason is keyed even at zero, so "no scaffold gaps this batch" is a readable
    result rather than something inferred from an absent row."""

    if fq_df.empty or "truth_present" not in fq_df.columns:
        return pd.DataFrame()
    absent = fq_df[fq_df["truth_present"] == False]  # noqa: E712 — object column
    if absent.empty:
        return pd.DataFrame()
    reasons = (absent["absence_reason"] if "absence_reason" in absent.columns
               else pd.Series([None] * len(absent), index=absent.index, dtype=object))
    reasons = reasons.fillna(ABSENCE_UNKNOWN)
    total = len(absent)
    rows = [{
        "reason": reason,
        "n": int((reasons == reason).sum()),
        "share": float((reasons == reason).sum()) / total,
        "counts_as_absent": reason == ABSENCE_CONFIRMED,
    } for reason in ABSENCE_REASONS]
    return pd.DataFrame(rows)


def _frame_quality_classes(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Failure-class frequency table over the pooled per-frame quality rows.

    Each class is additionally split by truth presence (issue #69). The split matters
    most for ``hallucination-fp``, where truth-absent is a real false positive
    (presence gating) and truth-present is a tracking miss (tracking robustness), but
    it is carried for every class because the axis is per-frame, not per-class."""

    if fq_df.empty:
        return pd.DataFrame()
    total = len(fq_df)
    rows: list[dict[str, Any]] = []
    for cls, g in fq_df.groupby("class"):
        rows.append({
            "class": str(cls),
            "n": int(len(g)),
            "share": len(g) / total,
            "held_pose": int(g["held_pose"].sum()),
            "frozen_stale": int(g["frozen_stale"].sum()),
            **_truth_presence_counts(g),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "class"], ascending=[False, True]).reset_index(drop=True)


def _hallucination_split_totals(fq_df: pd.DataFrame) -> dict[str, Any]:
    """Pooled truth-presence split of the ``hallucination-fp`` frames (issue #69).

    The headline the class table's extra columns are there to support: of every frame
    pooled as a hallucination, how many were emitted where no Climber was (a real false
    positive, fixed by presence gating) versus where one was (a tracking miss)."""

    empty = fq_df.empty or "class" not in fq_df.columns
    sub = pd.DataFrame() if empty else fq_df[fq_df["class"] == "hallucination-fp"]
    if sub.empty:
        return {"total": 0, "truth_present": 0, "truth_absent": 0,
                "truth_absent_unconfirmed": 0, "truth_unknown": 0,
                "truth_present_share": None, "truth_absent_share": None}
    return {"total": int(len(sub)), **_truth_presence_counts(sub)}


def _frame_quality_distractors(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Human distractor frequency table over annotated per-frame quality rows."""

    if fq_df.empty or "distractor" not in fq_df.columns:
        return pd.DataFrame()
    sub = fq_df[fq_df["distractor"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    total = len(sub)
    rows: list[dict[str, Any]] = []
    for distractor, g in sub.groupby("distractor"):
        rows.append({
            "distractor": str(distractor),
            "n": int(len(g)),
            "share": len(g) / total,
            "held_pose": int(g["held_pose"].sum()),
            "frozen_stale": int(g["frozen_stale"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "distractor"], ascending=[False, True]).reset_index(drop=True)


def _frame_quality_worklist(fq_df: pd.DataFrame) -> pd.DataFrame:
    """Flagged + frozen frames, worst-first — the per-frame re-review / crop queue."""

    if fq_df.empty:
        return pd.DataFrame()
    sub = fq_df[(fq_df["flagged"] == 1) | (fq_df["frozen_stale"] == 1)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_sev"] = sub["class"].map(lambda c: _FQ_SEVERITY.get(c, 4))
    sub = sub.sort_values(
        ["_sev", "centroid_dist"], ascending=[True, False], na_position="last")
    cols = ["route_folder", "video_key", "run_ts", "t", "class", "truth_present",
            "source", "held_pose", "frozen_stale", "centroid_dist", "residual",
            "rejection_verdict", "rejection_centroid_dist", "rejection_joint_agreement",
            "detector_attempt_evidence", "detector_attempt_status",
            "reacquire_attempted", "reacquire_succeeded", "reacquire_failed",
            "search_luma_mean", "search_luma_stdDev", "search_sharpness",
            "initial_search_region_area", "detection_region_area", "crop"]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def _frame_quality_condition_bands(fq_df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """Flagged-frame rate per Video Stats condition tercile (issue #44 deliverable 3).

    Reuses the condition-band machinery (``pd.qcut`` + ``_run_unit_rate``) from the
    within-video trends, but the outcome is the auto ``flagged`` flag and the predictor
    is a per-bundle Video Stats condition rather than a per-frame geometric one.

    A Video Stats condition is constant *within* a bundle, so a band here is really a
    handful of videos' worth of frames — the pseudo-replication is even starker than in
    the geometric bands, and the CI is likewise a run-unit cluster bootstrap (#70)."""

    if fq_df.empty:
        return pd.DataFrame()
    cond_cols = [c for c in fq_df.columns if c.startswith("vs_")]
    key_cols = [c for c in _RUN_KEY_COLS if c in fq_df.columns]
    rows: list[dict[str, Any]] = []
    for col in cond_cols:
        d = fq_df[[*key_cols, col, "flagged"]].dropna(subset=[col, "flagged"])
        if len(d) < bins * 10:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[col], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            stats = _run_unit_rate(bg.assign(flagged=bg["flagged"].astype(int)), "flagged")
            if stats is None:
                continue
            rows.append({
                "condition": col[len("vs_"):],
                "band": int(band) + 1,
                "n": stats["n"],
                "n_runs": stats["n_runs"],
                "flagged_rate": stats["rate"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "run_rate_median": stats["run_rate_median"],
                "run_rate_p90": stats["run_rate_p90"],
                "band_min": float(bg[col].min()),
                "band_max": float(bg[col].max()),
            })
    return pd.DataFrame(rows)


def _bootstrap_mean(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float] | None:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return None
    rng = random.Random(BOOT_SEED)
    n = len(vals)
    mean = sum(vals) / n
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    lo, hi = _pct_ci(draws)
    return mean, lo, hi


_ATTEMPT_ERROR_PREDICTORS = [
    "attempt_search_luma_mean_mean",
    "attempt_search_luma_stdDev_mean",
    "attempt_search_sharpness_mean",
    "attempt_initial_search_region_area_mean",
    "attempt_initial_search_region_edge_distance_mean",
    "attempt_detection_region_area_mean",
    "attempt_detection_region_edge_distance_mean",
    "attempt_reacquire_attempt_rate",
    "attempt_reacquire_success_rate",
    "attempt_full_frame_reacquire_success_rate",
]


def _crop_quality_rows(recs: list[EvalRecord]) -> pd.DataFrame:
    """Pool every record's ``cropQuality`` attempts into one long table (issue #86).

    Pooled across **all** records — quarantined and loose included — for the same reason
    the per-frame quality pool is: the runs whose crops wander are exactly the ones worth
    inspecting. Records predating schema v10 carry no ``cropQuality`` and contribute
    nothing."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        cq = rec.data.get("cropQuality")
        if not isinstance(cq, dict):
            continue
        loose = bool(rec.data.get("loosePaired"))
        conforming = record_conforms(rec.data)
        cause = record_nonconformance_cause(rec.data)
        for e in cq.get("frames") or []:
            bbox = e.get("truthBbox") or {}
            rows.append({
                "route_folder": rec.route_folder,
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": e.get("t"),
                "status": e.get("status"),
                "truth_present": e.get("truthPresent"),
                "miss_cause": e.get("missCause"),
                "miss_reason": e.get("missReason"),
                "best_unselected_candidate_score": e.get("bestUnselectedCandidateScore"),
                "initial_search_region_iou": e.get("initialSearchRegionIou"),
                "detection_region_iou": e.get("detectionRegionIou"),
                "initial_crop_containment": e.get("initialCropContainment"),
                "crop_contained_truth": e.get("cropContainedTruth"),
                "search_flags_fired": e.get("searchFlagsFired"),
                "fired_search_flags": ", ".join(e.get("firedSearchFlags") or []),
                "reacquire_attempted": e.get("reacquireAttempted"),
                "truth_bbox_area": (
                    bbox.get("w") * bbox.get("h")
                    if isinstance(bbox.get("w"), (int, float))
                    and isinstance(bbox.get("h"), (int, float)) else None),
                "loose": loose,
                "conforming": conforming,
                "nonconformance_cause": cause,
            })
    return pd.DataFrame(rows)


def _miss_cause_table(crop_df: pd.DataFrame) -> pd.DataFrame:
    """Miss-cause frequency over the pooled attempts, with the crop-placement evidence
    beside each cause.

    ``crop_missed_truth`` is carried per cause on purpose: on a corpus where full-frame
    reacquire always runs, no miss is *caused* by the crop, yet the crop can still have
    excluded the Climber on most of them. Showing both stops the reader inferring either
    fact from the other."""

    if crop_df.empty or "miss_cause" not in crop_df.columns:
        return pd.DataFrame()
    sub = crop_df[crop_df["miss_cause"].notna()]
    if sub.empty:
        return pd.DataFrame()
    total = len(sub)
    rows: list[dict[str, Any]] = []
    for cause, g in sub.groupby("miss_cause"):
        contained = g["crop_contained_truth"]
        scored = int(contained.notna().sum())
        best = g.get("best_unselected_candidate_score")
        rows.append({
            "miss_cause": str(cause),
            "n": int(len(g)),
            "share": len(g) / total,
            "crop_missed_truth": int((contained == False).sum()),  # noqa: E712
            "crop_containment_scored": scored,
            "median_initial_crop_containment": (
                float(g["initial_crop_containment"].median())
                if g["initial_crop_containment"].notna().any() else None),
            "flags_fired": int((g["search_flags_fired"] == True).sum()),  # noqa: E712
            # The gate-tuning number (scanner issues 03-04): on identity-gated misses,
            # how confident the best candidate the gate rejected was.
            "median_best_unselected_candidate_score": (
                float(best.median()) if best is not None and best.notna().any() else None),
        })
    return pd.DataFrame(rows).sort_values(
        ["n", "miss_cause"], ascending=[False, True]).reset_index(drop=True)


def _crop_run_columns(cq: Any) -> dict[str, Any]:
    """Per-run crop-quality columns read off a record's ``cropQuality`` (issue #86).

    Pre-v10 and legacy frames-only records carry no block, so counts are zero and every
    rate/median is ``None`` — an unmeasured Run must not read as a Run with perfect
    crops."""

    cq = cq if isinstance(cq, dict) else {}
    causes = cq.get("missCauseCounts") if isinstance(cq.get("missCauseCounts"), dict) else {}
    contained = cq.get("cropContainedTruth") if isinstance(
        cq.get("cropContainedTruth"), dict) else {}
    initial = cq.get("initialSearchRegionIou") if isinstance(
        cq.get("initialSearchRegionIou"), dict) else {}
    missing = int(cq.get("missingAttempts") or 0)
    out: dict[str, Any] = {
        "crop_matched_attempts": int(cq.get("matchedAttempts") or 0),
        "missing_attempts": missing,
        "crop_contained_truth_rate": contained.get("rate"),
        "initial_search_region_iou_median": initial.get("median"),
    }
    for cause in MISS_CAUSES:
        count = int(causes.get(cause) or 0)
        out[f"miss_{_slug(cause)}_count"] = count
        out[f"miss_{_slug(cause)}_share"] = (count / missing) if missing else None
    return out


def _rejection_run_columns(fq: dict[str, Any]) -> dict[str, Any]:
    """Per-run rejection-correctness columns read off a record's ``frameQuality``
    (issue #85). Pre-v9 and legacy frames-only records carry no
    ``rejectionCorrectness`` block, so every count is zero and the rates are ``None`` —
    an unmeasured run must not read as a zero over-rejection rate.

    ``over_rejection_rate`` is the pooled rate across both rejection gates;
    ``flip_over_rejection_rate`` isolates the flip gate, which is the one the corpus
    baseline and the scanner-side flip-gate work are about. The ``*_truth_present``
    variants drop Climber-absent rejections from the denominator — see
    ``evaluate._rejection_rate_block`` for why both denominators are reported."""

    rc = fq.get("rejectionCorrectness")
    rc = rc if isinstance(rc, dict) else {}
    counts = rc.get("verdictCounts") if isinstance(rc.get("verdictCounts"), dict) else {}
    flip = rc.get("byStatus", {}).get("flipRejected") if isinstance(
        rc.get("byStatus"), dict) else None
    flip = flip if isinstance(flip, dict) else {}
    return {
        "rejected_attempts": int(rc.get("rejected") or 0),
        "good_pose_rejected": int(counts.get("goodPoseRejected") or 0),
        "bad_pose_rejected": int(counts.get("badPoseRejected") or 0),
        "rejection_truth_absent": int(rc.get("truthAbsent") or 0),
        "rejection_truth_unknown": int(counts.get("truthUnknown") or 0),
        "rejection_truth_checkable": int(rc.get("truthCheckable") or 0),
        "rejection_truth_present_checkable": int(rc.get("truthPresentCheckable") or 0),
        "over_rejection_rate": rc.get("overRejectionRate"),
        "over_rejection_rate_truth_present": rc.get("overRejectionRateTruthPresent"),
        "flip_rejected_attempts": int(flip.get("rejected") or 0),
        "flip_rejection_truth_checkable": int(flip.get("truthCheckable") or 0),
        "flip_over_rejection_rate": flip.get("overRejectionRate"),
        "flip_over_rejection_rate_truth_present": flip.get(
            "overRejectionRateTruthPresent"),
    }


def _detection_error_attempt_run_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: PoseRunCache | None = None,
) -> pd.DataFrame:
    """One row per evaluation record, joining Detection Errors to attempt summaries.

    The outcome is the record's frameQuality flagged rate; predictors are aggregated
    over that Run's Detector Attempts. This preserves the Run as the independent unit.
    Rejection correctness (issue #85) rides along as per-run columns so flip-gate
    changes are comparable batch-over-batch at the Run unit.
    """

    rows: list[dict[str, Any]] = []
    pose_cache = {} if pose_cache is None else pose_cache
    for rec in recs:
        fq = rec.data.get("frameQuality")
        if not isinstance(fq, dict):
            continue
        frames = fq.get("frames") or []
        detected = len(frames)
        flagged = sum(
            1 for e in frames
            if str((e or {}).get("class") or "ok") in _FQ_FLAGGED
        )
        attempts = (_pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN).attempts
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "loose": bool(rec.data.get("loosePaired")),
            "conforming": record_conforms(rec.data),
            "nonconformance_cause": record_nonconformance_cause(rec.data),
            "detected_frames": detected,
            "flagged_frames": flagged,
            "flagged_rate": flagged / detected if detected else None,
            "frozen_stale_frames": int(fq.get("frozenStaleCount") or 0),
            **_rejection_run_columns(fq),
            **_crop_run_columns(rec.data.get("cropQuality")),
            **_detector_attempt_summary(attempts),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["route_folder", "video_key", "run_ts"]).reset_index(drop=True)


def _detection_error_attempt_bands(run_df: pd.DataFrame, bins: int = 3) -> pd.DataFrame:
    """Group Detection Error rates against attempt evidence at the Run unit."""

    if run_df.empty or "flagged_rate" not in run_df.columns:
        return pd.DataFrame()
    predictors = [
        c for c in _ATTEMPT_ERROR_PREDICTORS
        if c in run_df.columns and pd.api.types.is_numeric_dtype(run_df[c])
    ]
    predictors += [
        c for c in run_df.columns
        if c.startswith("attempt_search_flag_") and c.endswith("_rate")
        and pd.api.types.is_numeric_dtype(run_df[c])
    ]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        d = run_df[[predictor, "flagged_rate"]].dropna()
        if len(d) < max(3, bins):
            continue
        if d[predictor].nunique() < 2:
            continue
        try:
            d = d.assign(_bin=pd.qcut(d[predictor], q=bins, labels=False, duplicates="drop"))
        except ValueError:
            continue
        for band, bg in d.groupby("_bin"):
            vals = bg["flagged_rate"].astype(float).tolist()
            boot = _bootstrap_mean(vals)
            if boot is None:
                continue
            rows.append({
                "predictor": predictor,
                "band": int(band) + 1,
                "n_runs": len(vals),
                "flagged_rate_mean": boot[0],
                "ci_low": boot[1],
                "ci_high": boot[2],
                "band_min": float(bg[predictor].min()),
                "band_max": float(bg[predictor].max()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Detector Attempt funnel (issue #87)
#
# What the detector *did*, before Ground Truth is consulted: how the attempt stream
# splits across accepted / missing / flipRejected / qualityRejected, how often reacquire
# ran and worked, and which search conditions were flagged under each status.
#
# The Run is the unit of inference (CONTEXT.md), so every pooled share here is reported
# beside its run-unit distribution — median, p90, and the count of runs the status
# dominates. A corpus where one very long run misses everything and a corpus where every
# run misses a quarter of its frames produce the same pooled missing share; only the
# run-unit columns tell them apart. Deliberately no pooled-frame CIs: attempts within a
# run are correlated, so a CI over pooled attempts would claim precision the design
# cannot support (#70).
# --------------------------------------------------------------------------- #

# A run is a *tail* run for a status when that status takes more than this share of its
# attempts. 0.5 is the corpus baseline's "runs > 50% missing" line — a run that misses
# most of what it looked at is a different failure from one that misses some of it.
ATTEMPT_FUNNEL_TAIL_SHARE = 0.5

# Run-unit distributions worth reporting outside the status mix (the per-status shares
# already carry their own median/p90 in the status table).
_FUNNEL_RUN_METRICS = [
    ("attempt_count", "attempts per run"),
    ("attempt_reacquire_attempt_rate", "reacquire attempted / attempts"),
    ("attempt_reacquire_success_rate", "reacquire succeeded / reacquires attempted"),
    ("attempt_full_frame_reacquire_success_rate",
     "full-frame reacquire succeeded / attempts"),
]


def _status_columns(status: str) -> tuple[str, str]:
    slug = _slug(status)
    return f"attempt_status_{slug}_count", f"attempt_status_{slug}_rate"


def _attempt_funnel_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run funnel rows — the attempt-backed subset of the Detection Error run table.

    Derived from that table rather than re-walking the attempt streams, so the funnel and
    the Detection Error section can never disagree about a run's status mix: there is one
    ``_detector_attempt_summary`` per run and both read it. Legacy runs are dropped by the
    same rule that makes them legacy — no attempt stream, so no funnel to report."""

    if run_df.empty or "attempt_evidence" not in run_df.columns:
        return pd.DataFrame()
    sub = run_df[run_df["attempt_evidence"].astype("string") == EVIDENCE_ATTEMPTS]
    if sub.empty:
        return pd.DataFrame()
    cols = ["route_folder", "video_key", "run_ts", "conforming", "nonconformance_cause",
            "loose", "attempt_count"]
    for status in DETECTOR_ATTEMPT_STATUS_ORDER:
        cols.extend(_status_columns(status))
    cols += [
        "attempt_reacquire_attempted_count", "attempt_reacquire_succeeded_count",
        "attempt_reacquire_failed_count", "attempt_reacquire_attempt_rate",
        "attempt_reacquire_success_rate",
        "attempt_full_frame_reacquire_success_count",
        "attempt_full_frame_reacquire_success_rate",
    ]
    cols += sorted(c for c in sub.columns if c.startswith("attempt_search_flag_"))
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)


def _attempt_funnel_status_table(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """The status mix, pooled over attempts *and* distributed over runs.

    Every status is a row even at zero: "nothing was quality-rejected this batch" is a
    result, and leaving the row out would leave it to be inferred from an absence."""

    if funnel_df.empty:
        return pd.DataFrame()
    counts_total = pd.to_numeric(funnel_df["attempt_count"], errors="coerce").fillna(0)
    total = int(counts_total.sum())
    rows: list[dict[str, Any]] = []
    for status in DETECTOR_ATTEMPT_STATUS_ORDER:
        count_col, rate_col = _status_columns(status)
        if count_col not in funnel_df.columns:
            continue
        counts = pd.to_numeric(funnel_df[count_col], errors="coerce").fillna(0)
        shares = _pool_rates(funnel_df, rate_col)
        n = int(counts.sum())
        rows.append({
            "status": status,
            "attempts": n,
            "share": (n / total) if total else None,
            "runs_with_any": int((counts > 0).sum()),
            "run_share_median": float(shares.median()) if len(shares) else None,
            "run_share_p90": _p90(shares),
            "run_share_max": float(shares.max()) if len(shares) else None,
            "tail_runs": int((shares > ATTEMPT_FUNNEL_TAIL_SHARE).sum()),
        })
    return pd.DataFrame(rows)


def _attempt_funnel_run_stats(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """Run-unit distribution of the funnel measures that are not per-status shares."""

    if funnel_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for metric, label in _FUNNEL_RUN_METRICS:
        if metric not in funnel_df.columns:
            continue
        vals = pd.to_numeric(funnel_df[metric], errors="coerce").dropna()
        if vals.empty:
            continue  # e.g. no run ever attempted a reacquire — an unmeasured metric
        rows.append({
            "metric": metric,
            "meaning": label,
            "n_runs": int(len(vals)),
            "median": float(vals.median()),
            "p90": _p90(vals),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    return pd.DataFrame(rows)


def _attempt_funnel_flag_rows(
    analysis_root: Path,
    recs: list[EvalRecord],
    pose_cache: PoseRunCache | None = None,
) -> pd.DataFrame:
    """Condition-flag rate per attempt status — the one funnel table the per-run summary
    cannot supply, because it needs each attempt's flags *and* its status together.

    The denominator is attempts of that status **whose conditions carry the flag**: a
    scanner build that never emitted ``underexposed`` must not read as one that emitted it
    and found nothing. Pooled rates come with the per-run distribution beside them, since
    flags cluster hard within a run (one dark video floods the pool). The p90 is there
    because the median is usually zero — most runs never fire a given flag, so the median
    alone would report "no signal" for a flag a handful of runs fire on constantly."""

    pose_cache = {} if pose_cache is None else pose_cache
    pooled: dict[tuple[str, str], list[int]] = {}          # (flag, status) -> [scored, fired]
    per_run: dict[tuple[str, str], list[float]] = {}       # (flag, status) -> run rates
    for rec in recs:
        if record_evidence_generation(rec.data) != EVIDENCE_ATTEMPTS:
            continue
        attempts = (_pose_run(analysis_root, rec, pose_cache) or _NO_POSE_RUN).attempts
        run_tally: dict[tuple[str, str], list[int]] = {}
        for attempt in attempts or []:
            raw_status = attempt.get("status")
            status = (raw_status if raw_status in DETECTOR_ATTEMPT_STATUSES
                      else DETECTOR_ATTEMPT_STATUS_UNKNOWN)
            for flag, fired in _condition_flags(attempt.get("searchConditions")).items():
                for tally in (pooled, run_tally):
                    slot = tally.setdefault((flag, status), [0, 0])
                    slot[0] += 1
                    slot[1] += int(bool(fired))
        for key, (scored, fired) in run_tally.items():
            if scored:
                per_run.setdefault(key, []).append(fired / scored)

    order = {status: i for i, status in enumerate(DETECTOR_ATTEMPT_STATUS_ORDER)}
    rows: list[dict[str, Any]] = []
    for (flag, status), (scored, fired) in pooled.items():
        run_rates = per_run.get((flag, status), [])
        rows.append({
            "flag": flag,
            "status": status,
            "attempts_scored": scored,
            "flag_fired": fired,
            "rate": (fired / scored) if scored else None,
            "n_runs": len(run_rates),
            "run_rate_median": (float(np.median(run_rates)) if run_rates else None),
            "run_rate_p90": (float(np.quantile(run_rates, 0.9)) if run_rates else None),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_order"] = df["status"].map(lambda s: order.get(s, len(order)))
    return df.sort_values(["flag", "_order"]).drop(columns="_order").reset_index(drop=True)


def _attempt_funnel_totals(funnel_df: pd.DataFrame,
                           status_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide funnel headline, summed off the per-run rows so the tiles and the CSV
    can never disagree."""

    if funnel_df.empty:
        return {"runs": 0, "attempts": 0, "status_shares": {},
                "reacquire_attempted": 0, "reacquire_succeeded": 0,
                "reacquire_success_rate": None, "reacquire_success_rate_run_median": None,
                "missing_share_run_median": None, "tail_runs_missing": 0}

    def total(name: str) -> int:
        return _pool_total(funnel_df, name)

    by_status = status_df.set_index("status") if not status_df.empty else pd.DataFrame()
    attempted = total("attempt_reacquire_attempted_count")
    succeeded = total("attempt_reacquire_succeeded_count")
    run_success = _pool_rates(funnel_df, "attempt_reacquire_success_rate")
    return {
        "runs": int(len(funnel_df)),
        "attempts": total("attempt_count"),
        "status_shares": ({s: by_status.loc[s, "share"] for s in by_status.index}
                          if not by_status.empty else {}),
        "reacquire_attempted": attempted,
        "reacquire_succeeded": succeeded,
        "reacquire_success_rate": (succeeded / attempted) if attempted else None,
        "reacquire_success_rate_run_median": (
            float(run_success.median()) if len(run_success) else None),
        "missing_share_run_median": (
            by_status.loc["missing", "run_share_median"]
            if "missing" in getattr(by_status, "index", []) else None),
        "tail_runs_missing": (
            int(by_status.loc["missing", "tail_runs"])
            if "missing" in getattr(by_status, "index", []) else 0),
    }


def _rejection_totals(run_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide rejection-correctness headline (issue #85), summed over the per-run
    rows so the counts and the CSV can never disagree.

    Counts pool, but the *rate* is reported three ways on purpose. Two are denominator
    choices carried up from the record (``over_rejection_rate`` over every truth-checkable
    rejection, ``..._truth_present`` dropping the Climber-absent ones — see
    ``evaluate._rejection_rate_block``). The third, ``over_rejection_rate_run_mean``,
    averages the per-run rates: the Run is the unit of inference, so a corpus where one
    long run dominates the pooled frames is visibly different from one where every run
    over-rejects."""

    def col(name: str) -> pd.Series:
        return pd.to_numeric(run_df.get(name, pd.Series(dtype=float)), errors="coerce")

    def total(name: str) -> int:
        return 0 if run_df.empty else int(col(name).fillna(0).sum())

    rates = pd.Series(dtype=float) if run_df.empty else col("over_rejection_rate").dropna()
    runs_with = int(len(rates))
    good, bad, absent = (total("good_pose_rejected"), total("bad_pose_rejected"),
                         total("rejection_truth_absent"))
    checkable = good + bad
    present_checkable = checkable - absent
    return {
        "rejected_attempts": total("rejected_attempts"),
        "good_pose_rejected": good,
        "bad_pose_rejected": bad,
        "truth_absent": absent,
        "truth_unknown": total("rejection_truth_unknown"),
        "truth_checkable": checkable,
        "truth_present_checkable": present_checkable,
        "over_rejection_rate": (good / checkable) if checkable else None,
        "over_rejection_rate_truth_present": (
            (good / present_checkable) if present_checkable else None),
        "over_rejection_rate_run_mean": float(rates.mean()) if runs_with else None,
        "runs_with_checkable_rejections": runs_with,
    }


def _crop_totals(crop_df: pd.DataFrame) -> dict[str, Any]:
    """Corpus-wide crop-placement headline (issue #86).

    ``crop_missed_truth_rate`` is over truth-present attempts with a scorable crop, and is
    reported independently of the miss-cause mix: it is the crop-placement defect, not a
    causal claim about misses."""

    if crop_df.empty:
        return {"matched_attempts": 0, "missing_attempts": 0,
                "crop_containment_scored": 0, "crop_missed_truth": 0,
                "crop_missed_truth_rate": None,
                "median_initial_crop_containment": None,
                "median_initial_search_region_iou": None,
                "miss_cause_counts": {c: 0 for c in MISS_CAUSES}}

    contained = crop_df["crop_contained_truth"]
    scored = int(contained.notna().sum())
    missed = int((contained == False).sum())  # noqa: E712
    causes = crop_df["miss_cause"].dropna()
    return {
        "matched_attempts": int(len(crop_df)),
        "missing_attempts": int(len(causes)),
        "crop_containment_scored": scored,
        "crop_missed_truth": missed,
        "crop_missed_truth_rate": (missed / scored) if scored else None,
        "median_initial_crop_containment": (
            float(crop_df["initial_crop_containment"].median())
            if crop_df["initial_crop_containment"].notna().any() else None),
        "median_initial_search_region_iou": (
            float(crop_df["initial_search_region_iou"].median())
            if crop_df["initial_search_region_iou"].notna().any() else None),
        "miss_cause_counts": {
            c: int((causes == c).sum()) for c in MISS_CAUSES},
    }


# --------------------------------------------------------------------------- #
# Conformance as a covariate, not a filter (issue #132)
#
# The #15 gate exists to keep runs whose truth does not fit out of *truth-fit* metrics —
# accuracy, agreement, PCK, normDist — where a bad fit makes the number meaningless. On
# *failure-mode* metrics (the attempt funnel, miss causes, rejection correctness, crop
# placement) the same gate does active harm: it selects on the very failure being
# measured. The `sparse-match` cause is the clearest case — by its own definition it is
# the detector supplying too little to fit, so gating on conformance removes the worst
# detector failures from the pool by construction.
#
# So the failure-mode pools stay over **all** runs (they already did), and conformance is
# reported here as a *dimension* instead. Two columns carry the argument:
# ``share_of_attempts`` (how much of the corpus a population is) against the population's
# own failure rate — a population holding half the attempts and nearly all the misses is
# the gate's selectivity made visible. Without this breakout the pooled number is still
# right but silently re-weights every batch, which is why cross-batch numbers have not
# lined up.
#
# Every breakout is built by running the section's *existing* totals function once per
# population, so the ``all`` row is by construction the same arithmetic as the section's
# headline tiles and the two can never disagree.
# --------------------------------------------------------------------------- #

CONFORMANCE_POOL_ALL = "all"
CONFORMANCE_POOL_CONFORMING = "conforming"
CONFORMANCE_POOL_NONCONFORMING = "non-conforming"

# ``pool`` rows partition the corpus at the gate; ``cause`` rows partition the
# non-conforming pool by *why* it failed. Kept as a column so a reader (and the report's
# renderer) can tell a total from a breakdown of a total without parsing the label.
CONFORMANCE_ROW_POOL = "pool"
CONFORMANCE_ROW_CAUSE = "cause"


def _conformance_pools(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """``(population, kind, subset)`` for every conformance population worth reporting.

    Always leads with ``all`` — the pooled population is the headline, and the splits are
    there to explain it, not to replace it. The per-cause rows appear only once something
    is actually quarantined, but then *every* cause is emitted even at zero: "no
    suspected mis-tracks tripped the gate this batch" is a result, and an omitted row
    would leave it to be inferred from an absence.

    A frame with no ``conforming`` column (a pool built before the covariate existed)
    yields the ``all`` row alone rather than a fabricated split."""

    if df.empty or "conforming" not in df.columns:
        return [(CONFORMANCE_POOL_ALL, CONFORMANCE_ROW_POOL, df)]
    conforming = df["conforming"].fillna(True).astype(bool)
    pools = [
        (CONFORMANCE_POOL_ALL, CONFORMANCE_ROW_POOL, df),
        (CONFORMANCE_POOL_CONFORMING, CONFORMANCE_ROW_POOL, df[conforming]),
        (CONFORMANCE_POOL_NONCONFORMING, CONFORMANCE_ROW_POOL, df[~conforming]),
    ]
    if bool((~conforming).any()) and "nonconformance_cause" in df.columns:
        causes = df["nonconformance_cause"].astype("string")
        for cause in NONCONFORMANCE_CAUSES:
            pools.append((cause, CONFORMANCE_ROW_CAUSE, df[causes == cause]))
    return pools


def _attempt_funnel_conformance(funnel_df: pd.DataFrame) -> pd.DataFrame:
    """The attempt funnel broken out by conformance population (issue #132).

    ``share_of_attempts`` and ``share_of_missing`` are the pair to read together: they
    are what shows a population carrying a small slice of the corpus and a large slice of
    its failures. The per-status shares are *within* the population, so each row is a
    self-contained funnel, and the run-unit missing columns ride along because the Run is
    the unit of inference — a population's pooled missing share can be one collapsed run
    or every run missing a quarter of its attempts."""

    if funnel_df.empty:
        return pd.DataFrame()
    missing_count_col, missing_rate_col = _status_columns("missing")
    corpus_attempts = _pool_total(funnel_df, "attempt_count")
    corpus_missing = _pool_total(funnel_df, missing_count_col)
    rows: list[dict[str, Any]] = []
    for population, kind, sub in _conformance_pools(funnel_df):
        attempts = _pool_total(sub, "attempt_count")
        missing = _pool_total(sub, missing_count_col)
        rates = _pool_rates(sub, missing_rate_col)
        row: dict[str, Any] = {
            "population": population,
            "kind": kind,
            "runs": int(len(sub)),
            "attempts": attempts,
            "share_of_attempts": (attempts / corpus_attempts) if corpus_attempts else None,
            "missing_attempts": missing,
            "share_of_missing": (missing / corpus_missing) if corpus_missing else None,
        }
        for status in DETECTOR_ATTEMPT_STATUS_ORDER:
            count_col, _ = _status_columns(status)
            row[f"{_slug(status)}_share"] = (
                (_pool_total(sub, count_col) / attempts) if attempts else None)
        row["missing_share_run_median"] = float(rates.median()) if len(rates) else None
        row["missing_share_run_p90"] = _p90(rates)
        row["tail_runs_missing"] = (
            int((rates > ATTEMPT_FUNNEL_TAIL_SHARE).sum()) if len(rates) else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _miss_cause_conformance(crop_df: pd.DataFrame) -> pd.DataFrame:
    """The miss-cause mix broken out by conformance population (issue #132).

    This is the distortion the issue was opened on: the cause mix read off the conforming
    pool is not the corpus's cause mix, because the quarantined runs hold most of the
    misses. Shares are within the population so the mix reads as a mix; ``share_of_misses``
    says how much of the corpus's misses that mix speaks for."""

    if crop_df.empty or "miss_cause" not in crop_df.columns:
        return pd.DataFrame()
    scored = crop_df[crop_df["miss_cause"].notna()]
    if scored.empty:
        return pd.DataFrame()
    corpus_misses = int(len(scored))
    rows: list[dict[str, Any]] = []
    for population, kind, sub in _conformance_pools(scored):
        n = int(len(sub))
        causes = sub["miss_cause"].astype("string") if n else pd.Series(dtype="string")
        runs = (int(sub.groupby(["route_folder", "video_key", "run_ts"]).ngroups)
                if n else 0)
        row: dict[str, Any] = {
            "population": population,
            "kind": kind,
            "runs": runs,
            "misses": n,
            "share_of_misses": (n / corpus_misses) if corpus_misses else None,
        }
        for cause in MISS_CAUSES:
            row[f"{_slug(cause)}_share"] = (int((causes == cause).sum()) / n) if n else None
        rows.append(row)
    return pd.DataFrame(rows)


def _rejection_conformance(run_df: pd.DataFrame) -> pd.DataFrame:
    """Rejection correctness broken out by conformance population (issue #132).

    Built by running ``_rejection_totals`` per population rather than re-deriving the
    arithmetic, so the ``all`` row is the section's headline by construction."""

    if run_df.empty:
        return pd.DataFrame()
    keep = ("rejected_attempts", "good_pose_rejected", "bad_pose_rejected",
            "truth_absent", "truth_checkable", "truth_present_checkable",
            "over_rejection_rate", "over_rejection_rate_truth_present",
            "over_rejection_rate_run_mean", "runs_with_checkable_rejections")
    rows: list[dict[str, Any]] = []
    for population, kind, sub in _conformance_pools(run_df):
        totals = _rejection_totals(sub)
        rows.append({"population": population, "kind": kind, "runs": int(len(sub)),
                     **{k: totals.get(k) for k in keep}})
    return pd.DataFrame(rows)


def _crop_conformance(crop_df: pd.DataFrame) -> pd.DataFrame:
    """Crop placement broken out by conformance population (issue #132).

    Same construction as the rejection breakout: ``_crop_totals`` per population. The
    nested ``miss_cause_counts`` is dropped here — the cause mix has its own breakout, and
    duplicating it would give two places for the same number to drift."""

    if crop_df.empty:
        return pd.DataFrame()
    keep = ("matched_attempts", "missing_attempts", "crop_containment_scored",
            "crop_missed_truth", "crop_missed_truth_rate",
            "median_initial_crop_containment", "median_initial_search_region_iou")
    rows: list[dict[str, Any]] = []
    for population, kind, sub in _conformance_pools(crop_df):
        totals = _crop_totals(sub)
        rows.append({"population": population, "kind": kind,
                     **{k: totals.get(k) for k in keep}})
    return pd.DataFrame(rows)


def _quarantined_rows(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Non-conforming records (issue #15 gate), flattened for the report's shame
    accounting: which bundle/run tripped the gate, why, and the offending fit.

    Each row carries the issue #88 ``cause`` and the evidence behind it, so the section
    can be read cause-first: a sparse-match row is a detector problem that happens to trip
    a truth gate. It also carries the issue #147 ``attribution`` — the cause says what the
    fit found, the attribution says whether anyone can name a side. **Only a row attributed
    ``truth-identity`` is established as a truth problem**; a ``trajectory-divergence``
    cause on its own is not, which is what #34's worklist got wrong."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if record_conforms(rec.data):
            continue
        conf = rec.data.get("conformance") or {}
        evidence = conf.get("causeEvidence") or {}
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "cause": record_nonconformance_cause(rec.data),
            "attribution": record_attribution(rec.data),
            "flagged_wrong_frames": (
                conf.get("attributionEvidence") or {}).get("flaggedWrongFrames"),
            "reasons": ", ".join(conf.get("reasons") or []),
            "n": conf.get("n"),
            "fit_frames": evidence.get("fitFrames"),
            "accepted_share": evidence.get("acceptedShare"),
            "slope_x": (conf.get("x") or {}).get("slope"),
            "r2_x": (conf.get("x") or {}).get("r2"),
            "slope_y": (conf.get("y") or {}).get("slope"),
            "r2_y": (conf.get("y") or {}).get("r2"),
        })
    return sorted(rows, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"]))


def _quarantine_cause_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Non-conforming records per cause. Every cause is keyed even at zero, so a report
    reading "0 suspected mis-tracks" is distinguishable from a report that never split."""

    counts = {c: 0 for c in NONCONFORMANCE_CAUSES}
    for row in rows:
        cause = row.get("cause")
        if cause in counts:
            counts[cause] += 1
    return counts


def _rate_mismatch_records(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Records whose scaffold sampled coarser than the truth grid (issue #101).

    Reported independently of the conformance gate on purpose. ``rate-mismatch`` is a
    *non-conformance cause*, so it only speaks when a record also fails — and a Bundle
    can under-sample its truth grid tenfold while still fitting cleanly on the frames it
    did sample. Those Bundles fabricate absences by the thousand, and absence provenance
    now keeps that out of the numbers; this list is what stops the underlying data defect
    from staying invisible, because the fix (regenerate the scaffold) is the same either
    way."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        conf = rec.data.get("conformance") or {}
        evidence = conf.get("causeEvidence") or {}
        ratio = evidence.get("samplingRatio")
        if not isinstance(ratio, (int, float)) or ratio < RATE_MISMATCH_MIN_RATIO:
            continue
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "scaffold_step_sec": evidence.get("scaffoldStepSec"),
            "truth_step_sec": evidence.get("truthStepSec"),
            "sampling_ratio": ratio,
            "conforms": bool(conf.get("conforms")),
        })
    return sorted(rows, key=lambda r: (-r["sampling_ratio"], r["route_folder"],
                                       r["video_key"], r["run_ts"]))


def _truth_repair_worklist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of quarantined records worth re-seeding truth for (issues #21/#34).

    Issue #88 scoped this to the divergence cause, on the reasoning that a sparse-match
    record fails the gate because the detector found almost nothing. That was right about
    sparse-match and wrong about the rest: **the cause cannot identify a side**, so the
    worklist filled with bundles whose truth was sound. #34 was built from this list and
    named 12 bundles, of which one had a truth defect; regenerating the other eleven
    would have burned GPU hours rewriting good scaffolds.

    From v15 the worklist requires *positive* truth-side evidence — a human-attested
    wrong-person stretch — not merely a cause that used to imply one (issue #147). A
    divergent record nobody can attribute is not a truth-repair candidate; it is an open
    question, and it stays visible in the quarantine table where it can be read as one."""

    return [r for r in rows
            if r.get("cause") == NONCONFORMANCE_TRAJECTORY_DIVERGENCE
            and r.get("attribution") == ATTRIBUTION_TRUTH_IDENTITY]


def _loose_rows(recs: list[EvalRecord]) -> list[dict[str, Any]]:
    """Best-overlap loose pairings (issue #44), flattened for the report's shame
    accounting: which bundle/run fell back, and why. Held out of the trusted pool but
    kept for the per-frame quality worklist + crops."""

    rows: list[dict[str, Any]] = []
    for rec in recs:
        if not rec.data.get("loosePaired"):
            continue
        rows.append({
            "route_folder": rec.route_folder,
            "video_key": rec.video_key,
            "run_ts": rec.run_ts,
            "reason": str(rec.data.get("loosePairReason") or ""),
        })
    return sorted(rows, key=lambda r: (r["route_folder"], r["video_key"], r["run_ts"]))


def build_trend_context(analysis_root: Path) -> dict[str, Any]:
    # Issue #89: dedup evidence generations *before* anything pools or is accounted for.
    # A superseded legacy record is not a quarantined record and not a loose pairing — it
    # is the same pairing measured twice — so it must not appear in either shame list.
    on_disk = _iter_eval_records(analysis_root)
    # One pose-file read per bundle for the whole trend build: the frame/joint rows, the
    # per-frame quality pool, the Detection Error run table and the attempt funnel all draw
    # from this cache. Built *before* dedup because origin lives in the pose envelope and
    # dedup now keys on it (issue #160).
    pose_cache: PoseRunCache = {}
    for rec in on_disk:
        _pose_run(analysis_root, rec, pose_cache)
    origins = _origin_index(pose_cache)

    all_on_disk, superseded = _dedup_evidence_generations(on_disk, origins)

    # Origin segregation (issue #160). Everything below this line pools **scanner** runs
    # only, so every historical number is byte-identical to what it was before the harness
    # module existed. Harness runs are a separate population with their own derivations —
    # never summed into a scanner number, because whether the two agree is exactly the open
    # question #162 asks, and pooling them would assume the answer.
    harness_recs = [r for r in all_on_disk if record_origin(r, origins) != ORIGIN_SCANNER]
    all_recs = [r for r in all_on_disk if record_origin(r, origins) == ORIGIN_SCANNER]
    experiment_arms = _arm_groups(harness_recs, origins, pose_cache)
    origin_populations = _origin_populations(all_on_disk, origins)
    # Arm-versus-arm reporting (issue #164), built per Bundle first and pooled only after,
    # so the spread is always available to print beside a central value.
    arm_bundles = _arm_bundle_pck(experiment_arms)
    arm_overview = _arm_overview(arm_bundles)
    arm_deltas = _arm_deltas(arm_bundles)
    arm_delta_summary = _arm_delta_summary(arm_deltas)
    arm_repeat_flags = _arm_repeat_checks(arm_bundles)
    arm_video_stats = _arm_video_stats(analysis_root, arm_bundles)
    arm_reach = _arm_comparison_reach(arm_bundles, arm_overview, arm_delta_summary)
    # Issue #15 gate: quarantine non-conforming bundles (truth mis-tracking) from
    # every *pooled* derivation below. Issue #44: best-overlap loose pairings are
    # likewise held out of the trusted pool (their setupHash never matched the truth).
    # Both classes stay on disk and inspectable; only the aggregation drops them, and
    # the report accounts for each by name.
    quarantined = _quarantined_rows(all_recs)
    quarantine_causes = _quarantine_cause_counts(quarantined)
    truth_repair = _truth_repair_worklist(quarantined)
    loose_records = _loose_rows(all_recs)
    recs = [r for r in all_recs if record_trusted(r.data)]
    evidence_trusted = _evidence_generation_summary(recs, "trusted pooled metrics")
    evidence_frames = _evidence_generation_summary(
        all_recs, "per-frame / attempt pools (all records)")
    build_ids = {
        (route, key, run_ts): (run.app_version, run.detector_code_hash)
        for (route, key), runs in pose_cache.items()
        for run_ts, run in runs.items()
    }
    # Conflicts are derived from every cached pose run, including runs no evaluation
    # record scored — see _build_identity_conflicts. They lead the version flags because
    # a conflict invalidates the grouping the rest of the section rests on.
    build_conflicts, conflict_flags = _build_identity_conflicts(pose_cache)
    # Issue #131: state the basis — schema + build set — behind each pooled population.
    # Derived after ``build_ids`` because the build half needs the pose cache, and named
    # per pool so the three sections that pool different populations each carry their own.
    basis_trusted = _measurement_basis(recs, "trusted pooled metrics", build_ids)
    basis_frames = _measurement_basis(
        all_recs, "per-frame / attempt pools (all records)", build_ids)
    frame_joint_df = _build_frame_joint_rows(analysis_root, recs, pose_cache)
    joint_rank = _joint_ranking(frame_joint_df)
    version_overview, version_deltas, version_flags = _version_regression(
        recs, frame_joint_df, build_ids)
    version_flags = conflict_flags + version_flags
    cond_df = pd.concat(
        [
            _condition_bands(frame_joint_df, "size_frac"),
            _condition_bands(frame_joint_df, "speed"),
            _condition_bands(frame_joint_df, "edge_dist"),
        ],
        ignore_index=True,
    ) if not frame_joint_df.empty else pd.DataFrame()
    split_df = _cross_video_splits(recs, analysis_root)
    no_truth, stale_runs = _shame_lists(analysis_root)
    stale_truth = _stale_truth_worklist(analysis_root)
    visible_hist = _visible_histogram(recs)
    low_conf_worklist = _low_confidence_worklist(analysis_root)

    # Per-frame detection quality (issue #44): pooled across ALL records — quarantined
    # and loose included — because those bundles hold the frames most worth fixing. This
    # is an independent pool from the trusted metrics above (conforming-only).
    fq_df = _frame_quality_rows(analysis_root, all_recs, pose_cache)
    fq_classes = _frame_quality_classes(fq_df)
    fq_hallucination = _hallucination_split_totals(fq_df)
    fq_absence_reasons = _absence_reason_counts(fq_df)
    rate_mismatches = _rate_mismatch_records(all_recs)
    fq_distractors = _frame_quality_distractors(fq_df)
    fq_worklist = _frame_quality_worklist(fq_df)
    fq_condition_bands = _frame_quality_condition_bands(fq_df)
    fq_attempt_runs = _detection_error_attempt_run_rows(analysis_root, all_recs, pose_cache)
    fq_attempt_bands = _detection_error_attempt_bands(fq_attempt_runs)
    rejection_totals = _rejection_totals(fq_attempt_runs)
    rejection_conformance = _rejection_conformance(fq_attempt_runs)

    # Detector Attempt funnel (issue #87): scanner behavior before truth is consulted,
    # over the attempt-backed records only — a legacy run has no attempt stream to funnel.
    # Quarantined and loose records stay in: a run that fails the #15 gate is often
    # exactly the run whose funnel collapsed, and dropping it would hide the failure the
    # section exists to show.
    funnel_recs = [r for r in all_recs
                   if record_evidence_generation(r.data) == EVIDENCE_ATTEMPTS]
    funnel_runs = _attempt_funnel_runs(fq_attempt_runs)
    funnel_status = _attempt_funnel_status_table(funnel_runs)
    funnel_run_stats = _attempt_funnel_run_stats(funnel_runs)
    funnel_flags = _attempt_funnel_flag_rows(analysis_root, funnel_recs, pose_cache)
    funnel_totals = _attempt_funnel_totals(funnel_runs, funnel_status)
    funnel_conformance = _attempt_funnel_conformance(funnel_runs)
    evidence_funnel = _evidence_generation_summary(funnel_recs, "attempt funnel")
    basis_funnel = _measurement_basis(funnel_recs, "attempt funnel", build_ids)

    # Crop placement + miss causes (issue #86), pooled over the same all-records set.
    crop_df = _crop_quality_rows(all_recs)
    miss_causes = _miss_cause_table(crop_df)
    crop_totals = _crop_totals(crop_df)
    crop_conformance = _crop_conformance(crop_df)
    miss_cause_conformance = _miss_cause_conformance(crop_df)

    verified_total = 0
    verified_records = 0
    for rec in recs:
        counts = rec.data.get("counts") or {}
        vf = int(counts.get("truthFramesVerified") or 0)
        verified_total += vf
        if vf > 0:
            verified_records += 1

    return {
        "eval_records": recs,
        "eval_count": len(recs),
        "eval_count_total": len(all_recs),
        "eval_count_on_disk": len(on_disk),
        "superseded_records": superseded,
        "superseded_count": len(superseded),
        # Issue #160: origin segregation, made checkable rather than asserted. Every pooled
        # section above draws on the scanner population only; the arm table is the harness
        # population, reported beside it and never summed into it.
        "origin_populations": origin_populations,
        "experiment_arms": experiment_arms,
        "experiment_arm_count": int(experiment_arms["config_hash"].nunique())
                                if not experiment_arms.empty else 0,
        "experiment_run_count": int(len(experiment_arms)),
        # Issue #164: the arm comparison itself. Per-Bundle first, pooled after, with the
        # uncertainty each number should be read against travelling beside it.
        "arm_bundles": arm_bundles,
        "arm_overview": arm_overview,
        "arm_deltas": arm_deltas,
        "arm_delta_summary": arm_delta_summary,
        "arm_repeat_flags": arm_repeat_flags,
        "arm_video_stats": arm_video_stats,
        "arm_bundle_count": int(len(arm_bundles)),
        "arm_reach": arm_reach,
        "evidence_generation_trusted": evidence_trusted,
        "evidence_generation_frames": evidence_frames,
        "measurement_basis_trusted": basis_trusted,
        "measurement_basis_frames": basis_frames,
        "quarantined_bundles": quarantined,
        "quarantined_count": len(quarantined),
        "quarantine_cause_counts": quarantine_causes,
        "truth_repair_worklist": truth_repair,
        "truth_repair_count": len(truth_repair),
        "loose_bundles": loose_records,
        "loose_count": len(loose_records),
        "frame_joint_df": frame_joint_df,
        "joint_rank": joint_rank,
        "condition_bands": cond_df,
        "cross_video_splits": split_df,
        "version_overview": version_overview,
        "version_deltas": version_deltas,
        "version_flags": version_flags,
        "build_conflicts": build_conflicts,
        "truthless_bundles": no_truth,
        "stale_truth_bundles": stale_truth,
        "stale_truth_count": len(stale_truth),
        "stale_runs": stale_runs,
        "visible_histogram": visible_hist,
        "low_conf_worklist": low_conf_worklist,
        "frame_quality_classes": fq_classes,
        "frame_quality_hallucination": fq_hallucination,
        "frame_quality_absence_reasons": fq_absence_reasons,
        "rate_mismatch_records": rate_mismatches,
        "rate_mismatch_count": len(rate_mismatches),
        "frame_quality_distractors": fq_distractors,
        "frame_quality_worklist": fq_worklist,
        "frame_quality_condition_bands": fq_condition_bands,
        "detection_error_attempt_runs": fq_attempt_runs,
        "detection_error_attempt_bands": fq_attempt_bands,
        "rejection_correctness": rejection_totals,
        # Issue #132: conformance as a reported dimension on every failure-mode section.
        # These pool over all runs; the gate stays a quarantine only on truth-fit metrics.
        "rejection_conformance": rejection_conformance,
        "attempt_funnel_conformance": funnel_conformance,
        "crop_quality_conformance": crop_conformance,
        "crop_quality_miss_cause_conformance": miss_cause_conformance,
        "attempt_funnel_runs": funnel_runs,
        "attempt_funnel_status": funnel_status,
        "attempt_funnel_run_stats": funnel_run_stats,
        "attempt_funnel_flags": funnel_flags,
        "attempt_funnel": funnel_totals,
        "evidence_generation_funnel": evidence_funnel,
        "measurement_basis_funnel": basis_funnel,
        "crop_quality_attempts": crop_df,
        "crop_quality_miss_causes": miss_causes,
        "crop_quality": crop_totals,
        "frame_quality_detected": int(len(fq_df)),
        "frame_quality_flagged": int(fq_df["flagged"].sum()) if not fq_df.empty else 0,
        "frame_quality_held": int(fq_df["held_pose"].sum()) if not fq_df.empty else 0,
        "frame_quality_frozen": int(fq_df["frozen_stale"].sum()) if not fq_df.empty else 0,
        "verified_frames_total": verified_total,
        "verified_records": verified_records,
        "confound_caveat": (
            "Cross-video splits are descriptive only: route and videographer are "
            "confounded with source/resolution/panning in this corpus."
        ),
    }


def write_trend_tables(out_dir: Path, ctx: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    quarantined = ctx.get("quarantined_bundles") or []
    quarantined_df = pd.DataFrame(quarantined) if quarantined else pd.DataFrame()
    truth_repair = ctx.get("truth_repair_worklist") or []
    truth_repair_df = pd.DataFrame(truth_repair) if truth_repair else pd.DataFrame()
    superseded = ctx.get("superseded_records") or []
    superseded_df = pd.DataFrame(superseded) if superseded else pd.DataFrame()
    stale_truth = ctx.get("stale_truth_bundles") or []
    stale_truth_df = pd.DataFrame(stale_truth) if stale_truth else pd.DataFrame()
    rate_mismatch = ctx.get("rate_mismatch_records") or []
    rate_mismatch_df = pd.DataFrame(rate_mismatch) if rate_mismatch else pd.DataFrame()
    tables = {
        "eval_joint_ranking.csv": ctx.get("joint_rank"),
        "eval_condition_bands.csv": ctx.get("condition_bands"),
        "eval_cross_video_splits.csv": ctx.get("cross_video_splits"),
        "eval_version_overview.csv": ctx.get("version_overview"),
        "eval_version_deltas.csv": ctx.get("version_deltas"),
        "eval_build_identity_conflicts.csv": ctx.get("build_conflicts"),
        "eval_low_confidence_worklist.csv": ctx.get("low_conf_worklist"),
        "eval_quarantined_bundles.csv": quarantined_df,
        "eval_truth_repair_worklist.csv": truth_repair_df,
        "eval_superseded_records.csv": superseded_df,
        # Issue #160: the experiment arms, and the origin accounting that shows which
        # population each pooled section drew on.
        "eval_experiment_arms.csv": ctx.get("experiment_arms"),
        "eval_origin_populations.csv": ctx.get("origin_populations"),
        # Issue #164: the arm comparison, exported per Bundle as well as pooled — the
        # per-video files are the ones that answer "which settings work for which videos".
        "eval_arm_bundles.csv": ctx.get("arm_bundles"),
        "eval_arm_overview.csv": ctx.get("arm_overview"),
        "eval_arm_deltas.csv": ctx.get("arm_deltas"),
        "eval_arm_delta_summary.csv": ctx.get("arm_delta_summary"),
        "eval_arm_video_stats.csv": ctx.get("arm_video_stats"),
        "eval_frame_quality_classes.csv": ctx.get("frame_quality_classes"),
        "eval_frame_quality_absence_reasons.csv": ctx.get("frame_quality_absence_reasons"),
        "eval_stale_truth_worklist.csv": stale_truth_df,
        "eval_rate_mismatch_records.csv": rate_mismatch_df,
        "eval_frame_quality_distractors.csv": ctx.get("frame_quality_distractors"),
        "eval_frame_quality_worklist.csv": ctx.get("frame_quality_worklist"),
        "eval_frame_quality_condition_bands.csv": ctx.get("frame_quality_condition_bands"),
        "eval_detection_error_attempt_runs.csv": ctx.get("detection_error_attempt_runs"),
        "eval_detection_error_attempt_bands.csv": ctx.get("detection_error_attempt_bands"),
        "eval_attempt_funnel_status.csv": ctx.get("attempt_funnel_status"),
        "eval_attempt_funnel_conformance.csv": ctx.get("attempt_funnel_conformance"),
        "eval_rejection_conformance.csv": ctx.get("rejection_conformance"),
        "eval_crop_quality_conformance.csv": ctx.get("crop_quality_conformance"),
        "eval_crop_quality_miss_cause_conformance.csv": ctx.get(
            "crop_quality_miss_cause_conformance"),
        "eval_attempt_funnel_runs.csv": ctx.get("attempt_funnel_runs"),
        "eval_attempt_funnel_run_stats.csv": ctx.get("attempt_funnel_run_stats"),
        "eval_attempt_funnel_flags.csv": ctx.get("attempt_funnel_flags"),
        "eval_crop_quality_attempts.csv": ctx.get("crop_quality_attempts"),
        "eval_crop_quality_miss_causes.csv": ctx.get("crop_quality_miss_causes"),
    }
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            p = out_dir / name
            table.to_csv(p, index=False)
            outputs[name] = p
    return outputs
