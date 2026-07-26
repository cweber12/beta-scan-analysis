"""Per-frame table: join per-frame image-quality predictors to per-frame pose
outcomes.

Three sources of per-frame conditions/evidence, in priority order:

1. **Detector Attempts (preferred).** When the pose export carries
   ``detectorAttempts[]``, expose that scanner-owned attempt stream directly and
   keep raw/accepted keypoints distinct. This is detector evidence, not a dense
   playback-frame proxy.
2. **Scanner frame export (legacy).** When the pose export carries per-frame
   ``source`` provenance and ``climber``/``wall`` region stats (the Phase 2 data
   contract), use them directly — no video decode, so the committed record is
   self-sufficient — and expose ``raw_detected`` (source == "raw") as a *real*
   outcome instead of the proxy.
3. **cv2 decode (fallback).** For older bundles without exported per-frame stats,
   decode the video at sampled timestamps and compute the crop stats here. In this
   path ``kp_count`` / ``mean_score`` are an explicit quality *proxy* (the exported
   frames are already post-processed and filled), not raw detector output.

Issue #101 runtime contract: superseded legacy Runs are dropped *before* the table
is built (``dedup_run_evidence_generations``), decode is sequential rather than
seek-per-timestamp (``SequentialFrameReader``), and crop stats for a given video,
timestamp and crop boxes are computed once, however many Runs sample them.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .detector_attempts import DETECTOR_ATTEMPT_EVIDENCE_UNKNOWN
from .discovery import RunRecord

try:  # optional at import time so discovery/stats work without cv2 installed
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only when cv2 is absent
    cv2 = None
    np = None

# torso keypoints used for the centroid-velocity predictor
_TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def dedup_run_evidence_generations(
    records: list[RunRecord],
) -> tuple[list[RunRecord], list[RunRecord]]:
    """Split Runs into ``(kept, superseded)`` by evidence generation (issue #101).

    The run-level counterpart of the evaluation-record dedup (issue #89): when a
    video carries any attempt-backed Run, its legacy frames-only Runs are
    superseded — the attempt stream answers directly what the legacy Run's
    frame-derived proxy only approximates, and (measured on the 2026-07 corpus)
    8,568 of the frame table's 8,569 video-decode operations served Runs the
    evidence dedup was about to discard anyway. A video with no attempt-backed
    Run is untouched, so a legacy-only corpus builds exactly as before.

    Scoped per ``(route_folder, video_key)``: unlike the evaluation dedup there is
    no truth revision in play — the frame table joins conditions to outcomes, and
    for that pairing the newer evidence generation supersedes the old one outright.
    """

    by_video: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        by_video.setdefault((rec.route_folder, rec.video_key), []).append(rec)

    kept: list[RunRecord] = []
    superseded: list[RunRecord] = []
    for group in by_video.values():
        attempt_backed = [r for r in group if r.detector_attempts is not None]
        if not attempt_backed:
            kept.extend(group)
            continue
        kept.extend(attempt_backed)
        superseded.extend(r for r in group if r.detector_attempts is None)

    sort_key = lambda r: (r.route_folder, r.video_key, r.run_ts)  # noqa: E731
    return sorted(kept, key=sort_key), sorted(superseded, key=sort_key)


class SequentialFrameReader:
    """cv2-backed grayscale frame reader that reads forward instead of seeking.

    A positioned seek (``CAP_PROP_POS_MSEC``) costs a keyframe seek plus decode —
    ~54 ms per frame measured — while a sequential grab costs ~7 ms, and sampling
    is dense enough that reading forward and discarding un-sampled frames is
    almost always cheaper. Seeks survive only as the fallback for a backwards
    request or a video with no usable fps.
    """

    def __init__(self) -> None:
        # path -> [cap, fps, next_frame_index] (None marks a video that failed to open)
        self._caps: dict[str, list[Any] | None] = {}

    def _state(self, video_path: Any) -> list[Any] | None:
        key = str(video_path)
        if key not in self._caps:
            if cv2 is None:
                self._caps[key] = None
            else:
                cap = cv2.VideoCapture(key)
                if not cap.isOpened():
                    self._caps[key] = None
                else:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    self._caps[key] = [cap, fps, 0]
        return self._caps[key]

    def read_gray(self, video_path: Any, t: float):
        """The frame nearest ``t`` seconds as a grayscale array, or ``None``."""

        state = self._state(video_path)
        if state is None:
            return None
        cap, fps, next_idx = state
        if fps <= 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            state[2] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not ok or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        target = int(round(t * fps))
        if target < next_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            next_idx = target
        while next_idx < target:
            if not cap.grab():
                state[2] = next_idx
                return None
            next_idx += 1
        ok, frame = cap.read()
        state[2] = target + 1
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self) -> None:
        for state in self._caps.values():
            if state is not None:
                state[0].release()
        self._caps.clear()


def _sample_interval_sec(config: dict[str, Any]) -> float:
    frame_step = config.get("frameStep", 1) or 1
    frame_interval_ms = config.get("frameIntervalMs", 100) or 100
    return frame_step * frame_interval_ms / 1000.0


def _frames_by_timestamp(pose: dict[str, Any]) -> dict[float, dict[str, Any]]:
    """Index exported frames (whole dict) by timestamp rounded to the 0.1s grid."""

    out: dict[float, dict[str, Any]] = {}
    for fr in pose.get("frames", []) or []:
        ts = round(float(fr.get("timestamp", 0.0)), 1)
        out[ts] = fr
    return out


def _attempts_by_timestamp(rec: RunRecord) -> dict[float, dict[str, Any]]:
    out: dict[float, dict[str, Any]] = {}
    for attempt in rec.detector_attempts or []:
        ts = round(float(attempt.get("timestamp", 0.0)), 1)
        out[ts] = attempt
    return out


# Map the scanner's per-frame region stat keys to the decode-path column suffixes
# (so exported stats and cv2-computed stats land in the same columns).
_REGION_STAT_KEYS = {"mean": "luma_mean", "stdDev": "luma_std", "sharpness": "sharpness"}
_ATTEMPT_STAT_KEYS = {
    "mean": "luma_mean",
    "stdDev": "luma_stdDev",
    "sharpness": "sharpness",
}
_REGION_GEOM_KEYS = ("x", "y", "w", "h", "area", "cx", "cy", "edge_distance")


def _exported_region_stats(frame: dict[str, Any]) -> dict[str, Any]:
    """Per-frame climber/wall stats from the scanner export, keyed to the
    decode-path column names. Returns {} when the frame carries none."""

    out: dict[str, Any] = {}
    present = False
    for region in ("climber", "wall"):
        block = frame.get(region) if isinstance(frame, dict) else None
        for src_key, col_suffix in _REGION_STAT_KEYS.items():
            val = block.get(src_key) if isinstance(block, dict) else None
            if isinstance(val, (int, float)):
                present = True
                out[f"{region}_{col_suffix}"] = float(val)
            else:
                out[f"{region}_{col_suffix}"] = None
    return out if present else {}


def _attempt_region_stats(attempt: dict[str, Any]) -> dict[str, Any]:
    """Attempt-level search conditions mapped onto the legacy frame stat columns."""

    out: dict[str, Any] = {}
    conditions = attempt.get("searchConditions")
    if not isinstance(conditions, dict):
        return out
    for src_key, col_suffix in _REGION_STAT_KEYS.items():
        val = conditions.get(src_key)
        out[f"climber_{col_suffix}"] = float(val) if isinstance(val, (int, float)) else None
    return out


def _slug(name: str) -> str:
    out = []
    prev_underscore = False
    for ch in name:
        if ch.isalnum():
            if ch.isupper() and out and not prev_underscore:
                out.append("_")
            out.append(ch.lower())
            prev_underscore = False
        elif not prev_underscore:
            out.append("_")
            prev_underscore = True
    return "".join(out).strip("_")


def _attempt_condition_columns(
    conditions: Any,
    prefix: str,
) -> dict[str, Any]:
    out = {f"{prefix}_{suffix}": None for suffix in _ATTEMPT_STAT_KEYS.values()}
    if not isinstance(conditions, dict):
        return out
    for src_key, col_suffix in _ATTEMPT_STAT_KEYS.items():
        val = conditions.get(src_key)
        out[f"{prefix}_{col_suffix}"] = float(val) if isinstance(val, (int, float)) else None

    flags = conditions.get("flags")
    if isinstance(flags, dict):
        for key, value in flags.items():
            out[f"{prefix}_flag_{_slug(str(key))}"] = bool(value)
    for key, value in conditions.items():
        if key in _ATTEMPT_STAT_KEYS or key == "flags":
            continue
        if isinstance(value, bool):
            out[f"{prefix}_flag_{_slug(str(key))}"] = value
    return out


def _region_geometry(region: Any, prefix: str) -> dict[str, Any]:
    out = {f"{prefix}_{key}": None for key in _REGION_GEOM_KEYS}
    if not isinstance(region, dict):
        return out
    x = region.get("x")
    y = region.get("y")
    w = region.get("w")
    h = region.get("h")
    if isinstance(x, (int, float)):
        out[f"{prefix}_x"] = float(x)
    if isinstance(y, (int, float)):
        out[f"{prefix}_y"] = float(y)
    if isinstance(w, (int, float)):
        out[f"{prefix}_w"] = float(w)
    if isinstance(h, (int, float)):
        out[f"{prefix}_h"] = float(h)
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        out[f"{prefix}_area"] = max(0.0, float(w) * float(h))
    if isinstance(x, (int, float)) and isinstance(w, (int, float)):
        out[f"{prefix}_cx"] = float(x) + float(w) / 2
    if isinstance(y, (int, float)) and isinstance(h, (int, float)):
        out[f"{prefix}_cy"] = float(y) + float(h) / 2
    if all(isinstance(v, (int, float)) for v in (x, y, w, h)):
        out[f"{prefix}_edge_distance"] = max(
            0.0,
            min(float(x), float(y), 1.0 - (float(x) + float(w)), 1.0 - (float(y) + float(h))),
        )
    return out


def _attempt_source(status: str | None) -> str | None:
    if status == "accepted":
        return "raw"
    if status in {"missing", "flipRejected", "qualityRejected"}:
        return status
    return None


def _pose_export_flags(pose: dict[str, Any]) -> tuple[bool, bool]:
    """(has_frame_stats, has_provenance) for a pose export.

    ``has_frame_stats`` gates skipping the video decode; ``has_provenance`` lets a
    sampled timestamp absent from ``frames[]`` be read as an undetected frame.
    """

    has_stats = False
    has_provenance = False
    for fr in pose.get("frames", []) or []:
        if not has_provenance and fr.get("source"):
            has_provenance = True
        if not has_stats:
            for region in ("climber", "wall"):
                block = fr.get(region)
                if isinstance(block, dict) and any(
                    isinstance(block.get(k), (int, float)) for k in ("mean", "stdDev", "sharpness")
                ):
                    has_stats = True
                    break
        if has_stats and has_provenance:
            break
    return has_stats, has_provenance


def _proxy_and_kinematics(keypoints: list[dict[str, Any]]) -> dict[str, Any]:
    if not keypoints:
        return {"kp_count": 0, "mean_score": None, "coverage": None, "_cx": None, "_cy": None}
    scores = [kp.get("score", 0.0) for kp in keypoints]
    xs = [kp.get("x") for kp in keypoints if kp.get("x") is not None]
    ys = [kp.get("y") for kp in keypoints if kp.get("y") is not None]
    coverage = None
    if xs and ys:
        coverage = (max(xs) - min(xs)) * (max(ys) - min(ys))

    by_name = {kp.get("name"): kp for kp in keypoints}
    torso = [by_name[n] for n in _TORSO if n in by_name]
    src = torso if torso else keypoints
    cx = sum(kp.get("x", 0.0) for kp in src) / len(src)
    cy = sum(kp.get("y", 0.0) for kp in src) / len(src)

    return {
        "kp_count": len(keypoints),
        "mean_score": sum(scores) / len(scores),
        "coverage": coverage,
        "_cx": cx,
        "_cy": cy,
    }


def _box_key(box: dict[str, Any]) -> tuple:
    """A hashable identity for a crop box, for the per-(video, t, boxes) memo."""

    if not isinstance(box, dict):
        return ()
    return tuple(sorted((k, v) for k, v in box.items() if isinstance(v, (int, float))))


def _crop_stats(gray, box: dict[str, Any], h: int, w: int, prefix: str) -> dict[str, Any]:
    """Laplacian-variance sharpness + luma mean/std for a normalized crop box."""

    keys = [f"{prefix}_sharpness", f"{prefix}_luma_mean", f"{prefix}_luma_std"]
    if not box:
        return {k: None for k in keys}
    x0 = max(0, min(w - 1, int(round(box.get("x", 0) * w))))
    y0 = max(0, min(h - 1, int(round(box.get("y", 0) * h))))
    x1 = max(x0 + 1, min(w, int(round((box.get("x", 0) + box.get("w", 0)) * w))))
    y1 = max(y0 + 1, min(h, int(round((box.get("y", 0) + box.get("h", 0)) * h))))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return {k: None for k in keys}
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    return {
        keys[0]: sharpness,
        keys[1]: float(crop.mean()),
        keys[2]: float(crop.std()),
    }


def build_frame_table(records: list[RunRecord], decode: bool = True,
                      frame_reader: Any = None) -> pd.DataFrame:
    """Build the per-frame table over the current evidence generation of each video.

    ``frame_reader`` is the decode collaborator — any object with
    ``read_gray(video_path, t)`` and ``close()``. It defaults to the cv2-backed
    ``SequentialFrameReader``; tests inject a fake to assert the access pattern
    without a video file or codec. Superseded legacy Runs are dropped up front
    (issue #101) so their decode never happens; crop stats are memoised per
    (video, timestamp, crop boxes) so re-runs over the same video read each
    frame once.
    """

    records, _ = dedup_run_evidence_generations(records)
    rows: list[dict[str, Any]] = []
    reader = frame_reader
    owns_reader = False
    # (video path, t, climber box, wall box) -> computed crop stats ({} on a failed read)
    crop_memo: dict[tuple, dict[str, Any]] = {}

    for rec in records:
        config = rec.config
        interval = _sample_interval_sec(config)
        duration = (
            rec.pose.get("diagnostics", {}).get("input", {}).get("video", {}).get("durationSec")
            or 0.0
        )
        frame_index = _frames_by_timestamp(rec.pose)
        attempt_index = _attempts_by_timestamp(rec)
        has_attempts = rec.detector_attempts is not None
        has_frame_stats, has_provenance = _pose_export_flags(rec.pose)

        # Exported per-frame stats make the video decode unnecessary (and let the
        # committed record be analysed without the git-ignored binary).
        can_decode = (
            decode
            and rec.video_path is not None
            and not has_frame_stats
            and (frame_reader is not None or cv2 is not None)
        )
        if can_decode and reader is None:
            reader = SequentialFrameReader()
            owns_reader = True

        climber_box = rec.setup.get("climberCrop", {})
        wall_box = rec.setup.get("wallCrop", {})

        prev_c: tuple[float, float] | None = None
        n_samples = int(math.floor(duration / interval)) + 1 if interval > 0 else 0

        for k in range(n_samples):
            t = round(k * interval, 1)
            fr = frame_index.get(t)
            attempt = attempt_index.get(t) if has_attempts else None
            if isinstance(attempt, dict):
                keypoints = attempt.get("acceptedKeypoints") or []
            elif has_attempts:
                keypoints = []
            else:
                keypoints = (fr.get("keypoints") if isinstance(fr, dict) else None) or []
            kin = _proxy_and_kinematics(keypoints)

            velocity = None
            cx, cy = kin.pop("_cx"), kin.pop("_cy")
            if cx is not None and prev_c is not None:
                velocity = math.hypot(cx - prev_c[0], cy - prev_c[1])
            if cx is not None:
                prev_c = (cx, cy)

            if isinstance(attempt, dict):
                attempt_status = attempt.get("status")
                attempt_source = _attempt_source(attempt_status)
                source = attempt_source
                raw_detected = 1.0 if attempt_status == "accepted" else 0.0
                evidence = rec.detector_attempt_evidence
            else:
                attempt_status = None
                attempt_source = None
                evidence = (
                    rec.detector_attempt_evidence
                    if has_attempts
                    else DETECTOR_ATTEMPT_EVIDENCE_UNKNOWN
                )
                if has_attempts:
                    source = None
                    raw_detected = None
                else:
                    # Provenance -> real per-frame outcome. A sampled timestamp with no
                    # exported frame is an undetected frame *only* when this run carries
                    # provenance at all (else it's an old bundle and we can't tell).
                    source = fr.get("source") if isinstance(fr, dict) else None
                    if source is None and has_provenance:
                        source = "missing"
                    raw_detected = None if source is None else (1.0 if source == "raw" else 0.0)

            row: dict[str, Any] = {
                "video_key": rec.video_key,
                "run_ts": rec.run_ts,
                "t": t,
                "velocity": velocity,
                "source": source,
                "raw_detected": raw_detected,
                "detector_attempt_evidence": evidence,
                "detector_attempt_status": attempt_status,
                "detector_attempt_source": attempt_source,
                "initial_search_region": (
                    attempt.get("initialSearchRegion") if isinstance(attempt, dict) else None
                ),
                "detection_region": (
                    attempt.get("detectionRegion") if isinstance(attempt, dict) else None
                ),
                "reacquire_attempted": (
                    attempt.get("reacquireAttempted") if isinstance(attempt, dict) else None
                ),
                "reacquired": (
                    attempt.get("reacquired") if isinstance(attempt, dict) else None
                ),
                "reacquire_succeeded": (
                    bool(attempt.get("reacquireAttempted")) and bool(attempt.get("reacquired"))
                    if isinstance(attempt, dict) else None
                ),
                "reacquire_failed": (
                    bool(attempt.get("reacquireAttempted")) and not bool(attempt.get("reacquired"))
                    if isinstance(attempt, dict) else None
                ),
                "raw_keypoints": (
                    attempt.get("rawKeypoints") if isinstance(attempt, dict) else None
                ),
                "accepted_keypoints": (
                    attempt.get("acceptedKeypoints") if isinstance(attempt, dict) else None
                ),
                "search_conditions": (
                    attempt.get("searchConditions") if isinstance(attempt, dict) else None
                ),
                "reacquire_conditions": (
                    attempt.get("reacquireConditions") if isinstance(attempt, dict) else None
                ),
                "candidate_count": (
                    attempt.get("candidateCount") if isinstance(attempt, dict) else None
                ),
                "rejected_candidate_count": (
                    attempt.get("rejectedCandidateCount") if isinstance(attempt, dict) else None
                ),
                "selection_method": (
                    attempt.get("selectionMethod") if isinstance(attempt, dict) else None
                ),
                **kin,
            }

            if isinstance(attempt, dict):
                row.update(_region_geometry(attempt.get("initialSearchRegion"), "initial_search_region"))
                row.update(_region_geometry(attempt.get("detectionRegion"), "detection_region"))
                row.update(_attempt_condition_columns(attempt.get("searchConditions"), "search"))
                row.update(_attempt_condition_columns(attempt.get("reacquireConditions"), "reacquire"))
                row.update(_attempt_region_stats(attempt))
            elif has_frame_stats and isinstance(fr, dict):
                row.update(_exported_region_stats(fr))
            elif can_decode:
                memo_key = (str(rec.video_path), t,
                            _box_key(climber_box), _box_key(wall_box))
                if memo_key not in crop_memo:
                    gray = reader.read_gray(rec.video_path, t)
                    if gray is not None:
                        vh, vw = gray.shape[:2]
                        crop_memo[memo_key] = {
                            **_crop_stats(gray, climber_box, vh, vw, "climber"),
                            **_crop_stats(gray, wall_box, vh, vw, "wall"),
                        }
                    else:
                        crop_memo[memo_key] = {}
                row.update(crop_memo[memo_key])

            rows.append(row)

    if owns_reader and reader is not None:
        reader.close()

    return pd.DataFrame(rows)
