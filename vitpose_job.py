"""ViTPose++ Ground Truth scaffold job.

beta-scanner's detection-eval harness seeds its human-authored Ground Truth poses
from a *stronger, independent* model instead of MediaPipe (the detector it grades).
That model — ViTPose++ — is top-down, so it cannot find people itself; this module
runs a person detector + tracker, selects the **Climber** track from the calibration
tap, poses the Climber's box on exactly the requested frames, and writes a
``vitpose.json`` artifact into the video's bundle. See beta-scanner ``docs/adr/0019``
and this repo's ``docs/adr/0003`` for the rationale and the cross-program contract.

The heavy ML stack (torch / transformers / ultralytics) is imported lazily inside the
default backend factories, so importing this module — and unit-testing the geometry
(Climber selection, timestamp echo, artifact shape, path safety) with stub
collaborators — needs none of it. The two seams:

- ``Tracker.track(video_path)``  -> per-frame track boxes (detect + track).
- ``PoseBackend.pose(video_path, targets)`` -> keypoints for the Climber's box on
  specific frames.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence

import video_stats


# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

# Frame timestamps are echoed verbatim; beta-scanner matches the seed to its
# Detection Frames within 1 ms, so we never re-derive them from the decoder clock.
ARTIFACT_NAME = "vitpose.json"
STATUS_NAME = "vitpose.status.json"
SETUP_NAME = "setup.json"
ARTIFACT_VERSION = 1

# COCO-17 keypoint order emitted by ViTPose. The first 13 are the joints beta-scanner
# scores (it must see these exact names); eyes/ears are drawn faintly as context and
# ignored by scoring — we include them because the contract permits extra points.
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


# --------------------------------------------------------------------------- #
# Value types (all coordinates video-normalized to the FULL frame, [0, 1])
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    t: float | None = None


@dataclass(frozen=True)
class Box:
    """Person box, normalized to the full frame: (x, y) is the top-left corner."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def contains(self, p: Point) -> bool:
        return self.x <= p.x <= self.x + self.w and self.y <= p.y <= self.y + self.h


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class Appearance:
    """Clothing-color signature of one person box: L1-normalized HSV hue-sat
    histograms (16x8 bins, flattened) of two box sub-regions — ``shirt`` (center
    50% width, 20–55% height) and ``pants`` (60–90% height). Either may be empty
    when its region was degenerate (box clipped at the frame edge).
    """

    shirt: tuple[float, ...] = ()
    pants: tuple[float, ...] = ()


@dataclass(frozen=True)
class FrameTracks:
    """Tracks present on one decoded frame, keyed by track id -> full-frame box.

    ``frame_number`` is the frame's position in the *source* video. When the tracker
    decodes with a stride it differs from the entry's index in the history list;
    ``None`` (stub histories, legacy callers) means the two coincide.
    ``features`` (optional) carries per-track-id clothing-color signatures; when
    absent the stitcher degrades to motion-only association.
    """

    timestamp: float
    boxes: dict[int, Box]
    frame_number: int | None = None
    features: dict[int, Appearance] | None = None


@dataclass(frozen=True)
class PoseTarget:
    """A frame the Climber must be posed on.

    ``frame_index`` keys the pose result back into the track history;
    ``frame_number`` is the source-video frame the backend must decode (they differ
    when the tracker ran with a stride; ``None`` means they coincide).
    """

    frame_index: int
    box: Box
    frame_number: int | None = None

    @property
    def source_frame(self) -> int:
        return self.frame_number if self.frame_number is not None else self.frame_index


@dataclass(frozen=True)
class VitPoseRequest:
    video_path: str
    route_folder: str
    video_key: str
    frames: tuple[float, ...]              # requested timestamps (seconds), echoed verbatim
    # Seed contract of record (scanner branch feat/harness-vitpose-seed-region): the
    # ``seed_tap`` point anchors Climber identity and ``seed_region`` gates the seed —
    # decoupled from the Climber Crop, which the harness no longer treats as the seed
    # gate. The API layer resolves legacy ``climber_point``/``climber_crop`` aliases
    # into these fields (preferring the new names when both are present).
    seed_tap: Point | None = None
    seed_region: Box | None = None
    panning: bool = False
    # Hash of the setup.json (Climber selection) this job runs under. Stamped into the
    # artifact as the provenance anchor: downstream pairing treats a pose run and a
    # truth file as comparable only when their setupHashes match. When the request
    # omits it, the job falls back to the bundle's current setup.json (see
    # run_vitpose_job) so new artifacts always carry the field.
    setup_hash: str | None = None


# --------------------------------------------------------------------------- #
# Backend seams
# --------------------------------------------------------------------------- #

class Tracker(Protocol):
    def track(self, video_path: Path) -> list[FrameTracks]:
        """Decode the video, detect people, and track them across frames."""
        ...


class PoseBackend(Protocol):
    def pose(self, video_path: Path, targets: Sequence[PoseTarget]) -> dict[int, list[Keypoint]]:
        """Run top-down ViTPose on each target's box; key results by frame_index."""
        ...


# --------------------------------------------------------------------------- #
# Path resolution (mirrors save_detection_run's traversal guard)
# --------------------------------------------------------------------------- #

def resolve_video_path(analysis_root: Path, video_path: str) -> Path:
    """Resolve a request's video path under the analysis root, rejecting traversal.

    ``video_path`` may be repo-relative (``analysis/<route>/<key>/<file>``) or
    already rooted at the analysis dir; either way the result must stay inside it.
    """
    analysis_root = analysis_root.resolve()
    raw = Path(video_path)

    # Strip a leading "analysis/" so both "analysis/route/key/x.mp4" and
    # "route/key/x.mp4" resolve against the analysis root the same way.
    if not raw.is_absolute() and raw.parts and raw.parts[0] == analysis_root.name:
        raw = Path(*raw.parts[1:])

    candidate = raw if raw.is_absolute() else (analysis_root / raw)
    resolved = candidate.resolve()

    if analysis_root != resolved and analysis_root not in resolved.parents:
        raise ValueError("Resolved video path escapes the analysis root.")
    return resolved


def bundle_dir_for(analysis_root: Path, route_folder: str, video_key: str) -> Path:
    analysis_root = analysis_root.resolve()
    resolved = (analysis_root / route_folder / video_key).resolve()
    if analysis_root not in resolved.parents:
        raise ValueError("Resolved bundle path escapes the analysis root.")
    return resolved


# --------------------------------------------------------------------------- #
# Climber Identity — a per-frame box trajectory stitched across ByteTrack ids
# --------------------------------------------------------------------------- #
#
# ByteTrack fragments the climber into several track ids over a boulder problem
# (scale/pose changes and occlusion break the track), and no single id is both
# near the base tap and present for the whole ascent. So we don't trust one id:
# seed from the tap, then follow the climber frame-to-frame in both directions,
# stitching across id switches. Association is *scored*, not just gated: the
# motion window is a generous candidate pre-filter (traverses, down-climbs and
# pans are legitimate), and clothing-color appearance against a rolling reference
# picks among candidates — measured on the planet-x pair, appearance separates
# the reappearing climber (0.35) from base bystanders (0.76–0.84) at the exact
# gap where nearest-box association latched onto the wrong person (issue #19).

# Max normalized center displacement to associate the same climber between frames,
# with a per-elapsed-frame slack so the trajectory survives short detection gaps.
# The slack counts *source-video* frames (FrameTracks.frame_number), so a strided
# tracker history gets proportionally more room per history step.
_ASSOC_BASE = 0.08
_ASSOC_PER_FRAME = 0.04
_ASSOC_MAX = 0.18
# Hard per-step area-consistency gate for association. The seed box can be tight
# while a foreground passer fills a much larger fraction of frame; when centers
# are nearby, motion-only ranking can otherwise latch onto the passer in one hop.
_ASSOC_AREA_MIN_RATIO = 1.0 / 3.0
_ASSOC_AREA_MAX_RATIO = 3.0
_REACQUIRE_AREA_MIN_RATIO = 1.0 / 3.0
_REACQUIRE_AREA_MAX_RATIO = 3.0

_SEED_WINDOW_S = 0.75
_CROP_GATE_EXPAND = 0.10

# Appearance scoring. Distances are Bhattacharyya in [0, 1] against a rolling
# (EMA) reference — never a frozen seed-time snapshot, which decays over an
# ascent as lighting/scale change. Appearance is comparative, not an absolute
# veto: a candidate with no features scores the neutral 0.5, so histories
# without features (stubs, legacy) reduce to pure motion ranking.
_APP_WEIGHT = 1.0        # weight of appearance vs gate-normalized motion distance
_APP_CONFIDENT = 0.45    # at/below: accept AND fold into the rolling reference
_APP_MISMATCH = 0.65     # at/above: accept but count toward the mismatch streak
_MISMATCH_STREAK = 5     # consecutive mismatch accepts that trigger a backtrack
_APP_EMA_ALPHA = 0.2     # rolling-reference update rate on confident accepts

# Soft area-consistency weight in the association score. A climber's box area
# changes slowly frame-to-frame; detectors sometimes emit a sloppy oversized box
# (person + crash pads) next to the tight one, and pure motion+appearance can
# latch onto the sloppy chain (its regions then pollute the rolling reference
# with pad/rock colors). |ln(area ratio)| is 0 for consistent boxes and ~1.8 for
# a 6x jump, steering the walk to the size-consistent candidate.
_AREA_WEIGHT = 0.3

# seedDebug event thresholds/caps.
_JUMP_EVENT_DIST = 0.08
_EVENT_CAP = 50


def _source_frame_no(history: Sequence[FrameTracks], i: int) -> int:
    fn = history[i].frame_number
    return fn if fn is not None else i


def _center_dist(a: Box, b: Box) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _point_to_box_dist(point: Point, box: Box) -> float:
    return ((box.cx - point.x) ** 2 + (box.cy - point.y) ** 2) ** 0.5


def _seed_box_passes_region_gate(seed_box: Box, seed_region: Box | None) -> bool:
    if seed_region is None:
        return True
    pad_x = seed_region.w * _CROP_GATE_EXPAND
    pad_y = seed_region.h * _CROP_GATE_EXPAND
    min_x = seed_region.x - pad_x
    max_x = seed_region.x + seed_region.w + pad_x
    min_y = seed_region.y - pad_y
    max_y = seed_region.y + seed_region.h + pad_y
    return min_x <= seed_box.cx <= max_x and min_y <= seed_box.cy <= max_y


def _bhattacharyya(p: Sequence[float], q: Sequence[float]) -> float:
    """Bhattacharyya distance between two L1-normalized histograms, in [0, 1]."""
    bc = 0.0
    for a, b in zip(p, q):
        if a > 0.0 and b > 0.0:
            bc += (a * b) ** 0.5
    return max(0.0, 1.0 - min(bc, 1.0)) ** 0.5


def _appearance_dist(ref: Appearance | None, cand: Appearance | None) -> float | None:
    """Mean shirt/pants distance over the regions both sides have; None if none."""
    if ref is None or cand is None:
        return None
    dists = []
    if ref.shirt and cand.shirt:
        dists.append(_bhattacharyya(ref.shirt, cand.shirt))
    if ref.pants and cand.pants:
        dists.append(_bhattacharyya(ref.pants, cand.pants))
    return sum(dists) / len(dists) if dists else None


def _ema_appearance(ref: Appearance | None, new: Appearance | None) -> Appearance | None:
    if ref is None:
        return new
    if new is None:
        return ref

    def mix(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
        if not a:
            return b
        if not b:
            return a
        mixed = [(1.0 - _APP_EMA_ALPHA) * x + _APP_EMA_ALPHA * y for x, y in zip(a, b)]
        total = sum(mixed) or 1.0
        return tuple(v / total for v in mixed)

    return Appearance(shirt=mix(ref.shirt, new.shirt), pants=mix(ref.pants, new.pants))


def _features_for(frame: FrameTracks, track_id: int | None) -> Appearance | None:
    if track_id is None or frame.features is None:
        return None
    return frame.features.get(track_id)


def _track_id_at(history: Sequence[FrameTracks], idx: int, box: Box) -> int | None:
    for tid, b in history[idx].boxes.items():
        if b == box:
            return tid
    return None


def _seed_climber(
    history: Sequence[FrameTracks],
    seed_tap: Point | None,
    seed_region: Box | None,
) -> tuple[int | None, Box | None]:
    """Find the (frame_index, box) to start stitching the Climber trajectory from.

    With a tap timestamp: search only near that time, prefer containing boxes, then
    nearest centers in-window; no global fallback. Without a tap timestamp: prefer
    the earliest containing frame, then nearest center over the clip. Seed candidates
    must pass the expanded seed-region gate when a seed region is present.

    Note: this is a seed-time gate only. Per-frame identity tracking uses motion,
    appearance, and area-consistency gating in ``_best_candidate``.

    Without a tap: seed from the first box of the most prominent persistent track
    inside the seed region.
    """
    if seed_tap is not None:
        frame_indices = list(range(len(history)))
        anchored = seed_tap.t is not None
        if anchored:
            frame_indices = [
                i for i, frame in enumerate(history)
                if abs(frame.timestamp - seed_tap.t) <= _SEED_WINDOW_S
            ]
            if not frame_indices:
                return None, None

        containing: list[tuple[int, float, int, Box]] = []
        non_containing: list[tuple[float, int, Box]] = []
        for i in frame_indices:
            for box in history[i].boxes.values():
                d = _point_to_box_dist(seed_tap, box)
                if box.contains(seed_tap):
                    frame_key = d if anchored else float(i)
                    tie = i if anchored else d
                    containing.append((frame_key, tie, i, box))
                else:
                    non_containing.append((d, i, box))

        containing.sort(key=lambda c: (c[0], c[1]))
        non_containing.sort(key=lambda c: (c[0], c[1]))

        for _, _, idx, box in containing:
            if _seed_box_passes_region_gate(box, seed_region):
                return idx, box
        for _, idx, box in non_containing:
            if _seed_box_passes_region_gate(box, seed_region):
                return idx, box
        return None, None

    track_id = _largest_track(history, seed_region)
    if track_id is None:
        return None, None
    for i, frame in enumerate(history):
        if track_id in frame.boxes:
            return i, frame.boxes[track_id]
    return None, None


@dataclass
class StitchResult:
    """Output of ``stitch_climber_track``: the trajectory plus its provenance."""

    trajectory: dict[int, Box]
    track_ids: dict[int, int]        # frame_index -> ByteTrack id the box came from
    reseeds: list[dict]              # backtrack events (wrong-person recoveries)
    seed_index: int | None = None
    seed_box: Box | None = None


def _best_candidate(
    frame: FrameTracks,
    prev: Box,
    threshold: float,
    ref: Appearance | None,
    long_gap: bool,
) -> tuple[int, Box, float | None] | None:
    """Best (track_id, box, appearance_dist) within the motion gate, or None.

    Score = gate-normalized motion distance + weighted appearance distance; a
    candidate with no appearance scores the neutral 0.5, so feature-less
    histories reduce to pure nearest-box ranking.
    """
    best: tuple[float, int, Box, float | None] | None = None
    for tid, box in frame.boxes.items():
        d = _center_dist(box, prev)
        if d > threshold:
            continue
        area_pen = 0.0
        if prev.area > 0.0 and box.area > 0.0:
            ratio = box.area / prev.area
            if not (_ASSOC_AREA_MIN_RATIO <= ratio <= _ASSOC_AREA_MAX_RATIO):
                continue
            if long_gap and not (
                _REACQUIRE_AREA_MIN_RATIO <= ratio <= _REACQUIRE_AREA_MAX_RATIO
            ):
                continue
            area_pen = abs(math.log(ratio))
        app_d = _appearance_dist(ref, _features_for(frame, tid))
        score = (
            d / threshold
            + _APP_WEIGHT * (app_d if app_d is not None else 0.5)
            + _AREA_WEIGHT * area_pen
        )
        if best is None or score < best[0]:
            best = (score, tid, box, app_d)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _confident_match(
    frame: FrameTracks,
    ref: Appearance | None,
    exclude_tid: int | None,
    ref_area: float,
) -> tuple[int, Box, float] | None:
    """The best box in the frame whose appearance confidently matches ``ref``.

    Position-free: used both as the *evidence* that the right person is visible
    elsewhere (so a mismatch streak may fire) and as recovery's far
    reacquisition target. Gated only by the area ratio against the climber's
    last confident box — appearance strictness is the safety here.
    """
    best: tuple[float, int, Box] | None = None
    for tid, box in frame.boxes.items():
        if tid == exclude_tid:
            continue
        if ref_area > 0.0 and box.area > 0.0:
            ratio = box.area / ref_area
            if not (_REACQUIRE_AREA_MIN_RATIO <= ratio <= _REACQUIRE_AREA_MAX_RATIO):
                continue
        app_d = _appearance_dist(ref, _features_for(frame, tid))
        if app_d is not None and app_d <= _APP_CONFIDENT:
            if best is None or app_d < best[0]:
                best = (app_d, tid, box)
    if best is None:
        return None
    return best[1], best[2], best[0]


def _walk_direction(
    history: Sequence[FrameTracks],
    seed_idx: int,
    seed_box: Box,
    seed_app: Appearance | None,
    direction: int,
    trajectory: dict[int, Box],
    track_ids: dict[int, int],
    reseeds: list[dict],
) -> None:
    """Stitch one direction from the seed, with wrong-person backtrack recovery.

    A run of ``_MISMATCH_STREAK`` accepted frames whose appearance sits far from
    the rolling reference — on a foreign ByteTrack id, *while a confidently
    matching person is visible elsewhere in the frame* — means the walk latched
    onto someone else (the reference only updates on confident accepts, so it
    still describes the climber). The whole contiguous run on the offending ids
    back to the last confident accept is discarded, the walk rewinds, and
    *recovery mode* re-associates from the last confident climber box: gated
    candidates that mismatch are refused, and the confident match may be taken
    from anywhere in the frame — appearance strictness plus the area gate
    replace the motion gate, so the climber high on the wall is reacquirable
    from a base-level anchor.

    The visible-alternative requirement is what makes the detector safe on
    single-climber videos and same-person appearance lurches (mantling the top
    lip, exposure shifts): without positive evidence of the right person
    elsewhere, a mismatch is never treated as a switch.
    """
    prev, last = seed_box, seed_idx
    ref = seed_app
    # Last accept that matched the reference: box, index, and its ByteTrack id.
    conf_box, conf_idx, conf_tid = seed_box, seed_idx, track_ids.get(seed_idx)
    streak: list[int] = []                    # consecutive mismatch-accept indices
    recovery = False
    backtracked: set[int] = set()             # streak starts already rewound to once
    i = seed_idx + direction
    while 0 <= i < len(history):
        gap = abs(_source_frame_no(history, i) - _source_frame_no(history, last))
        threshold = min(_ASSOC_BASE + _ASSOC_PER_FRAME * gap, _ASSOC_MAX)
        cand = _best_candidate(history[i], prev, threshold, ref, long_gap=abs(i - last) > 1)
        if recovery and (
            cand is None or (cand[2] is not None and cand[2] >= _APP_MISMATCH)
        ):
            # The gate offers nothing trustworthy; reacquire on appearance alone.
            far = _confident_match(history[i], ref, None, conf_box.area)
            if far is None:
                i += direction
                continue
            cand = far
        if cand is None:
            i += direction
            continue
        tid, box, app_d = cand
        trajectory[i] = box
        track_ids[i] = tid
        prev, last = box, i
        if app_d is None or app_d <= _APP_CONFIDENT:
            ref = _ema_appearance(ref, _features_for(history[i], tid))
            conf_box, conf_idx, conf_tid = box, i, tid
            streak = []
            recovery = False
        elif (
            app_d >= _APP_MISMATCH
            and (conf_tid is None or tid != conf_tid)
            and _confident_match(history[i], ref, tid, conf_box.area) is not None
        ):
            streak.append(i)
            if len(streak) >= _MISMATCH_STREAK and streak[0] not in backtracked:
                # Discard the whole contiguous run on the offending ids back to
                # the last confident accept, not just the streak frames — the
                # switch may predate the point where the evidence appeared.
                streak_tids = {track_ids[j] for j in streak if j in track_ids}
                span = range(conf_idx + direction, i + direction, direction)
                discard_idx = [
                    j for j in span
                    if j in trajectory and track_ids.get(j) in streak_tids
                ]
                discarded = [(j, track_ids.get(j), trajectory[j]) for j in discard_idx]
                for j in discard_idx:
                    trajectory.pop(j, None)
                    track_ids.pop(j, None)
                backtracked.add(streak[0])
                first = discard_idx[0] if discard_idx else streak[0]
                reseeds.append({
                    "frameIndex": first,
                    "sourceFrame": _source_frame_no(history, first),
                    "timestamp": history[first].timestamp,
                    "reason": "appearance-mismatch-streak",
                    "_discarded": discarded,
                    "_direction": direction,
                })
                prev, last = conf_box, conf_idx
                streak = []
                recovery = True
                i = first
                continue
        else:
            # Between the thresholds, mismatch on the climber's own id, or no
            # better-matching person visible: plausible enough to keep, not
            # enough to fold into the reference.
            streak = []
            recovery = False
        i += direction


def stitch_climber_track(
    history: Sequence[FrameTracks],
    seed_tap: Point | None,
    seed_region: Box | None,
) -> StitchResult:
    """Stitch the per-frame Climber trajectory (with provenance) from the seed.

    Empty when no person was tracked or no seed was found. Frames where the
    climber can't be associated are simply absent from the map (they'll be
    seeded ``absent`` downstream).
    """
    result = StitchResult(trajectory={}, track_ids={}, reseeds=[])
    if not any(f.boxes for f in history):
        return result

    seed_idx, seed_box = _seed_climber(history, seed_tap, seed_region)
    if seed_idx is None or seed_box is None:
        return result

    result.seed_index, result.seed_box = seed_idx, seed_box
    result.trajectory[seed_idx] = seed_box
    seed_tid = _track_id_at(history, seed_idx, seed_box)
    if seed_tid is not None:
        result.track_ids[seed_idx] = seed_tid
    seed_app = _features_for(history[seed_idx], seed_tid)

    for direction in (1, -1):
        _walk_direction(
            history, seed_idx, seed_box, seed_app, direction,
            result.trajectory, result.track_ids, result.reseeds,
        )

    # Finalize reseed events. A discard is a *false alarm* when the walk ended up
    # continuing on the very id it discarded (the wrong-person hypothesis produced
    # no alternative — e.g. the climber re-emerging with a fresh ByteTrack id and
    # lurched appearance after mantling the top lip); restore those frames.
    # A genuine switch shows the post-streak trajectory on a different id.
    for event in result.reseeds:
        discarded = event.pop("_discarded")
        direction = event.pop("_direction")
        indices = [j for j, _, _ in discarded]
        edge = max(indices) if direction > 0 else min(indices)
        following = [
            j for j in result.trajectory
            if (j > edge if direction > 0 else j < edge)
        ]
        next_idx = None
        if following:
            next_idx = min(following) if direction > 0 else max(following)
        next_tid = result.track_ids.get(next_idx) if next_idx is not None else None
        streak_tids = {tid for _, tid, _ in discarded if tid is not None}
        recovered = sum(1 for j in indices if j in result.trajectory)
        restored = 0
        if next_tid is not None and next_tid in streak_tids:
            for j, tid, box in discarded:
                if j not in result.trajectory:
                    result.trajectory[j] = box
                    if tid is not None:
                        result.track_ids[j] = tid
                    restored += 1
            # Recovery mode also rejected this id's frames between the streak and
            # where the walk resumed; the id is trusted now, so backfill them.
            span = indices + [next_idx]
            for j in range(min(span), max(span)):
                if j not in result.trajectory and next_tid in history[j].boxes:
                    result.trajectory[j] = history[j].boxes[next_tid]
                    result.track_ids[j] = next_tid
                    restored += 1
        event["discarded"] = len(discarded)
        event["recovered"] = recovered
        event["restored"] = restored
    return result


def build_climber_track(
    history: Sequence[FrameTracks],
    seed_tap: Point | None,
    seed_region: Box | None,
) -> dict[int, Box]:
    """Per-frame Climber box (frame_index -> box), stitched across id switches."""
    return stitch_climber_track(history, seed_tap, seed_region).trajectory


def _largest_track(history: Sequence[FrameTracks], seed_region: Box | None) -> int | None:
    """The track with the greatest summed box area over the clip (inside ``seed_region``).

    Summed — not average — area rewards both size and persistence, so a brief
    close-up of a passerby can't outrank the climber who is smaller but on screen
    for most of the send.
    """
    areas: dict[int, float] = {}
    for frame in history:
        for track_id, box in frame.boxes.items():
            if seed_region is not None and not seed_region.contains(Point(box.cx, box.cy)):
                continue
            areas[track_id] = areas.get(track_id, 0.0) + box.area

    if not areas:
        return None
    return max(areas, key=lambda tid: areas[tid])


def _nearest_frame_index(history: Sequence[FrameTracks], timestamp: float) -> int | None:
    if not history:
        return None
    best_i, best_dist = 0, float("inf")
    for i, frame in enumerate(history):
        dist = abs(frame.timestamp - timestamp)
        if dist < best_dist:
            best_dist, best_i = dist, i
    return best_i


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_artifact(
    request: VitPoseRequest,
    history: Sequence[FrameTracks],
    pose_backend: PoseBackend,
    video_path: Path,
    stitch: StitchResult | None = None,
) -> dict:
    """Assemble the vitpose.json payload: one echoed frame per requested timestamp.

    A requested frame gets non-empty ``keypoints`` only when the stitched Climber
    trajectory has a box on the nearest decoded frame; otherwise ``keypoints: []``
    (seeded ``absent``). Pass ``stitch`` to reuse an already-computed trajectory.
    """
    if stitch is None:
        stitch = stitch_climber_track(history, request.seed_tap, request.seed_region)
    trajectory = stitch.trajectory

    # Map each requested timestamp to the nearest decoded frame that has the Climber,
    # collecting the unique (frame_index, box) targets to pose.
    ts_to_index: dict[float, int] = {}
    targets: dict[int, PoseTarget] = {}
    for ts in request.frames:
        idx = _nearest_frame_index(history, ts)
        if idx is None:
            continue
        box = trajectory.get(idx)
        if box is None:
            continue
        ts_to_index[ts] = idx
        targets.setdefault(
            idx, PoseTarget(frame_index=idx, box=box, frame_number=history[idx].frame_number)
        )

    posed: dict[int, list[Keypoint]] = (
        pose_backend.pose(video_path, list(targets.values())) if targets else {}
    )

    frames_out = []
    for ts in request.frames:
        keypoints: list[dict] = []
        idx = ts_to_index.get(ts)
        if idx is not None:
            for kp in posed.get(idx, []):
                keypoints.append({
                    "name": kp.name,
                    "x": _clamp01(float(kp.x)),
                    "y": _clamp01(float(kp.y)),
                    "score": _clamp01(float(kp.score)),
                })
        # Echo the requested timestamp verbatim — never the decoder's frame time.
        frames_out.append({"timestamp": ts, "keypoints": keypoints})

    artifact: dict = {"version": ARTIFACT_VERSION}
    # Stamp the provenance anchor when known. Omitted (not null) when absent, so
    # consumers' `artifact.get("setupHash", <fallback>)` reaches the legacy fallback.
    if request.setup_hash:
        artifact["setupHash"] = request.setup_hash
    artifact["frames"] = frames_out
    return artifact


def _read_setup_hash(bundle_dir: Path) -> str | None:
    """Return the bundle's current ``setup.json`` setupHash, or ``None`` if unreadable.

    The writer's fallback for a request that omits ``setup_hash``: stamp the setup the
    bundle is currently calibrated under. Mirrors the consumer-side fallback documented
    in the artifact contract, so a freshly written artifact never lacks the field.
    """
    setup_path = bundle_dir / SETUP_NAME
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = setup.get("setupHash")
    return value if isinstance(value, str) and value else None


def _log(message: str) -> None:
    # The job runs on a background thread that is otherwise silent; a line to stderr
    # (which uvicorn surfaces) lets a manual run see it start and finish.
    print(f"[vitpose] {message}", file=sys.stderr, flush=True)


def _write_status(
    bundle_dir: Path,
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    timings: dict[str, float] | None = None,
    device: str | None = None,
    warnings: list[str] | None = None,
    seed_debug: dict | None = None,
) -> None:
    # Extra keys are contract-safe: beta-scanner's sidecar reader only inspects
    # `status` and `error` (app/api/dev/corpus/vitpose/route.ts).
    payload: dict = {
        "jobId": job_id,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if error is not None:
        payload["error"] = error
    if timings is not None:
        payload["timings"] = timings
    if device is not None:
        payload["device"] = device
    if warnings:
        payload["warnings"] = warnings
    if seed_debug is not None:
        payload["seedDebug"] = seed_debug
    (bundle_dir / STATUS_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _seed_mode(request: VitPoseRequest) -> str:
    if request.seed_tap is None:
        return "largest_track"
    if request.seed_tap.t is not None:
        return "tap_time_window"
    return "tap_legacy"


def _stitch_events(
    history: Sequence[FrameTracks], stitch: StitchResult
) -> tuple[list[dict], list[dict]]:
    """(idSwitches, jumps) between consecutive stitched frames, capped for size."""
    id_switches: list[dict] = []
    jumps: list[dict] = []
    indices = sorted(stitch.trajectory)
    for prev_i, i in zip(indices, indices[1:]):
        prev_tid = stitch.track_ids.get(prev_i)
        tid = stitch.track_ids.get(i)
        if prev_tid is not None and tid is not None and prev_tid != tid:
            if len(id_switches) < _EVENT_CAP:
                id_switches.append({
                    "sourceFrame": _source_frame_no(history, i),
                    "from": prev_tid,
                    "to": tid,
                })
        dist = _center_dist(stitch.trajectory[prev_i], stitch.trajectory[i])
        if dist > _JUMP_EVENT_DIST and len(jumps) < _EVENT_CAP:
            jumps.append({
                "sourceFrame": _source_frame_no(history, i),
                "dist": round(dist, 4),
            })
    return id_switches, jumps


def _seed_debug(
    request: VitPoseRequest,
    history: Sequence[FrameTracks],
    stitch: StitchResult,
) -> dict[str, object]:
    seed_idx, seed_box = stitch.seed_index, stitch.seed_box
    debug: dict[str, object] = {
        "mode": _seed_mode(request),
        "historyFrames": len(history),
    }
    # Output keys stay `tap`/`crop` — cross-program seedDebug shape the scanner reads.
    if request.seed_tap is not None:
        debug["tap"] = {
            "x": request.seed_tap.x,
            "y": request.seed_tap.y,
            "t": request.seed_tap.t,
        }
    if request.seed_region is not None:
        debug["crop"] = {
            "x": request.seed_region.x,
            "y": request.seed_region.y,
            "w": request.seed_region.w,
            "h": request.seed_region.h,
        }
    if seed_idx is None or seed_box is None:
        debug["seedFound"] = False
        return debug

    debug["seedFound"] = True
    debug["seed"] = {
        "frameIndex": seed_idx,
        "timestamp": history[seed_idx].timestamp,
        "sourceFrame": _source_frame_no(history, seed_idx),
        "box": {
            "x": seed_box.x,
            "y": seed_box.y,
            "w": seed_box.w,
            "h": seed_box.h,
            "cx": seed_box.cx,
            "cy": seed_box.cy,
        },
    }
    id_switches, jumps = _stitch_events(history, stitch)
    debug["stitch"] = {
        "stitchedFrames": len(stitch.trajectory),
        "idSwitches": id_switches,
        "jumps": jumps,
        "reseeds": stitch.reseeds,
    }
    return debug


def _request_warnings(request: VitPoseRequest, history: Sequence[FrameTracks]) -> list[str]:
    warnings: list[str] = []
    point = request.seed_tap
    if point is None:
        return warnings

    if point.t is None:
        warnings.append(
            "seed_tap.t is missing; using legacy global tap seeding. "
            "Recalibrate in beta-scanner so tap timestamp is sent."
        )
    if point.t is not None and abs(point.t) <= 1e-9:
        near_start = [frame for frame in history if frame.timestamp <= _SEED_WINDOW_S]
        if any(len(frame.boxes) > 1 for frame in near_start):
            warnings.append(
                "seed_tap.t is 0 while multiple people appear near clip start; "
                "seeding may anchor to the wrong subject. Re-tap on the intended climber frame."
            )

    return warnings


def run_vitpose_job(
    analysis_root: Path,
    request: VitPoseRequest,
    tracker: Tracker,
    pose_backend: PoseBackend,
    job_id: str | None = None,
) -> Path:
    """Run the full job and write ``vitpose.json`` (+ a status sidecar) to the bundle.

    Returns the artifact path. On any failure the status sidecar records ``error`` so
    beta-scanner — which only polls the filesystem — can distinguish a crashed job
    from one still running instead of polling forever.
    """
    job_id = job_id or uuid.uuid4().hex
    bundle_dir = bundle_dir_for(analysis_root, request.route_folder, request.video_key)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"No bundle at route={request.route_folder!r} video_key={request.video_key!r}."
        )

    _write_status(bundle_dir, job_id, "running")
    _log(
        f"job {job_id[:8]} started: {request.route_folder}/{request.video_key} "
        f"({len(request.frames)} frames requested)"
    )
    try:
        video_path = resolve_video_path(analysis_root, request.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Stamp the setup the job ran under. Prefer the hash the request carries; fall
        # back to the bundle's current setup.json so the artifact always names a setup.
        if not request.setup_hash:
            request = replace(request, setup_hash=_read_setup_hash(bundle_dir))

        started = time.perf_counter()
        history = tracker.track(video_path)
        track_s = time.perf_counter() - started
        warnings = _request_warnings(request, history)
        stitch = stitch_climber_track(history, request.seed_tap, request.seed_region)
        seed_debug = _seed_debug(request, history, stitch)

        posed_at = time.perf_counter()
        artifact = build_artifact(request, history, pose_backend, video_path, stitch=stitch)
        pose_s = time.perf_counter() - posed_at

        artifact_path = bundle_dir / ARTIFACT_NAME
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Camera viewing-angle estimate from the posed keypoints (issue #23): Video
        # Stats, not Ground Truth — vitpose.json stays pure keypoint scaffold, and
        # the estimate lands in video-stats.json with its own provenance. Best-effort:
        # a failure here must not fail the scaffold job.
        try:
            angle = video_stats.estimate_camera_angle(artifact["frames"])
            if angle is not None:
                angle["source"] = "vitpose"
                if request.setup_hash:
                    angle["setupHash"] = request.setup_hash
                video_stats.write_camera_angle(bundle_dir, angle)
        except Exception as exc:  # noqa: BLE001 — additive, never job-fatal
            _log(f"camera-angle estimate failed (ignored): {exc}")
        timings = {
            "track_s": round(track_s, 2),
            "pose_s": round(pose_s, 2),
            "total_s": round(time.perf_counter() - started, 2),
        }
        device = getattr(tracker, "device", None) or getattr(pose_backend, "device", None)
        _write_status(
            bundle_dir,
            job_id,
            "done",
            timings=timings,
            device=device,
            warnings=warnings,
            seed_debug=seed_debug,
        )
        posed = sum(1 for f in artifact["frames"] if f["keypoints"])
        warning_note = f", {len(warnings)} warning(s)" if warnings else ""
        _log(
            f"job {job_id[:8]} done: {posed}/{len(artifact['frames'])} frames posed "
            f"in {timings['total_s']}s (track {timings['track_s']}s, "
            f"pose {timings['pose_s']}s, device {device or 'unknown'}{warning_note}) -> {artifact_path}"
        )
        return artifact_path
    except Exception as exc:  # noqa: BLE001 — surface every failure via the sidecar
        _write_status(
            bundle_dir, job_id, "error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        _log(f"job {job_id[:8]} FAILED: {type(exc).__name__}: {exc}")
        raise


# --------------------------------------------------------------------------- #
# Default (real) backends — heavy imports are lazy so this module stays importable
# and testable without torch/transformers/ultralytics installed.
# --------------------------------------------------------------------------- #

def _box_appearance(frame_bgr, box: Box) -> Appearance | None:
    """Clothing-color signature of one person box on a decoded BGR frame.

    Shirt = center 50% width, 20–55% box height; pants = 60–90% height. Each is
    an L1-normalized 16x8 HSV hue-sat histogram. Returns None when both regions
    are degenerate (box clipped to nothing at the frame edge).
    """
    import cv2  # lazy — only the real tracker path calls this

    height, width = frame_bgr.shape[:2]

    def region_hist(y0f: float, y1f: float) -> tuple[float, ...]:
        x0 = max(0, int((box.x + 0.25 * box.w) * width))
        x1 = min(width, int((box.x + 0.75 * box.w) * width))
        y0 = max(0, int((box.y + y0f * box.h) * height))
        y1 = min(height, int((box.y + y1f * box.h) * height))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return ()
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        total = float(hist.sum())
        if total <= 0.0:
            return ()
        return tuple(round(float(v) / total, 5) for v in hist.flatten())

    shirt = region_hist(0.20, 0.55)
    pants = region_hist(0.60, 0.90)
    if not shirt and not pants:
        return None
    return Appearance(shirt=shirt, pants=pants)


class UltralyticsTracker:
    """Person detect + ByteTrack via ultralytics YOLO (lazy-loaded).

    On CUDA the video is tracked at full frame rate; on CPU — where YOLO dominates
    the job's wall time — every ``CPU_VID_STRIDE``-th frame is tracked instead, and
    each history entry carries its source ``frame_number`` so downstream timestamp
    matching, association slack, and pose-frame seeking stay exact.
    """

    CPU_VID_STRIDE = 2

    def __init__(self, model_name: str = "yolov8n.pt", vid_stride: int | None = None) -> None:
        self._model_name = model_name
        self._vid_stride = vid_stride  # explicit override; None = pick by device
        self._model = None
        self._load_lock = threading.Lock()
        self.device: str | None = None  # "cuda" / "cpu", known after the model loads

    def _ensure_model(self):
        # Locked: the boot-time warm thread and an early first job otherwise race
        # and load a second copy of the model.
        with self._load_lock:
            if self._model is None:
                import torch  # lazy
                from ultralytics import YOLO  # lazy

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = YOLO(self._model_name)
        return self._model

    def warm(self) -> None:
        """Load the model now so the first real request doesn't pay for it."""
        self._ensure_model()

    def track(self, video_path: Path) -> list[FrameTracks]:
        import cv2  # lazy

        model = self._ensure_model()
        stride = self._vid_stride or (1 if self.device == "cuda" else self.CPU_VID_STRIDE)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        history: list[FrameTracks] = []
        # classes=[0] -> person only; persist=True keeps ByteTrack ids across frames.
        # half (fp16) only on CUDA — passing the kwarg at all on CPU trips an
        # ultralytics deprecation warning.
        fp16 = {"half": True} if self.device == "cuda" else {}
        results = model.track(
            source=str(video_path), classes=[0], persist=True,
            tracker="bytetrack.yaml", stream=True, verbose=False,
            device=0 if self.device == "cuda" else "cpu",
            vid_stride=stride, **fp16,
        )
        for idx, result in enumerate(results):
            boxes: dict[int, Box] = {}
            features: dict[int, Appearance] = {}
            b = getattr(result, "boxes", None)
            if b is not None and b.id is not None:
                ids = b.id.int().tolist()
                # xywhn is center-based, normalized; convert to top-left corner.
                xywhn = b.xywhn.tolist()
                frame_bgr = getattr(result, "orig_img", None)
                for track_id, (cx, cy, w, h) in zip(ids, xywhn):
                    box = Box(x=cx - w / 2.0, y=cy - h / 2.0, w=w, h=h)
                    boxes[int(track_id)] = box
                    if frame_bgr is not None:
                        app = _box_appearance(frame_bgr, box)
                        if app is not None:
                            features[int(track_id)] = app
            frame_no = idx * stride
            history.append(
                FrameTracks(
                    timestamp=frame_no / fps, boxes=boxes,
                    frame_number=frame_no, features=features or None,
                )
            )
        return history


def _require_scipy(scipy_available: bool) -> None:
    """Fail fast, with guidance, when scipy is missing.

    The transformers ViTPose image processor warps each person box (and refines the
    output heatmaps) via ``scipy.ndimage``; with scipy absent it dies deep inside the
    library with a cryptic ``NameError: name 'inv' is not defined`` — impossible to
    act on from the job's error sidecar. scipy is declared in ``requirements.txt`` for
    exactly this; surface an actionable message if the venv hasn't been synced.
    """
    if not scipy_available:
        raise ImportError(
            "ViTPose's image processor requires scipy (it warps person boxes and "
            "refines heatmaps via scipy.ndimage). Install it: "
            "pip install -r requirements.txt"
        )


class TransformersViTPoseBackend:
    """Top-down ViTPose++ via HuggingFace transformers (lazy-loaded).

    Seeks directly to each target's source frame (instead of decoding the whole
    video a second time) and runs the forwards in batches of ``BATCH_SIZE``.
    """

    # vitpose-plus-* is a mixture-of-experts model: its forward needs a dataset_index
    # to pick an expert head. 0 = the COCO expert, which emits the 17 COCO joints.
    COCO_EXPERT_INDEX = 0
    BATCH_SIZE = 16

    def __init__(self, model_name: str = "usyd-community/vitpose-plus-base") -> None:
        self._model_name = model_name
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self.device = "cpu"

    def _ensure_model(self):
        # Locked: the boot-time warm thread and an early first job otherwise race
        # and load a second copy of the model.
        with self._load_lock:
            if self._model is None:
                import torch  # lazy
                from transformers import AutoProcessor, VitPoseForPoseEstimation  # lazy
                from transformers.utils import is_scipy_available  # lazy

                _require_scipy(is_scipy_available())

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._processor = AutoProcessor.from_pretrained(self._model_name)
                model = VitPoseForPoseEstimation.from_pretrained(self._model_name)
                self._model = model.to(self.device).eval()
        return self._processor, self._model

    def warm(self) -> None:
        """Load processor + model now so the first real request doesn't pay for it."""
        self._ensure_model()

    def pose(self, video_path: Path, targets: Sequence[PoseTarget]) -> dict[int, list[Keypoint]]:
        import cv2  # lazy
        import torch  # lazy

        if not targets:
            return {}
        processor, model = self._ensure_model()

        cap = cv2.VideoCapture(str(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0

        # Grab exactly the target frames, seeking in source order so every seek is
        # forward-only. A frame the decoder can't produce is simply skipped (its
        # requested timestamps stay `keypoints: []`, same as an untracked Climber).
        grabbed: list[tuple[PoseTarget, object]] = []
        for target in sorted(targets, key=lambda t: t.source_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target.source_frame)
            ok, frame = cap.read()
            if ok:
                grabbed.append((target, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        out: dict[int, list[Keypoint]] = {}
        for start in range(0, len(grabbed), self.BATCH_SIZE):
            self._pose_batch(
                processor, model, torch,
                grabbed[start:start + self.BATCH_SIZE], width, height, out,
            )
        return out

    def _pose_batch(self, processor, model, torch, batch, width: float, height: float, out):
        """Pose one chunk: N frames, one Climber box each, one forward pass."""
        images = [rgb for _, rgb in batch]
        # ViTPose processor wants COCO boxes in absolute pixels: [x, y, w, h],
        # one list of person boxes per image (always exactly one — the Climber).
        boxes = [
            [[t.box.x * width, t.box.y * height, t.box.w * width, t.box.h * height]]
            for t, _ in batch
        ]
        inputs = processor(images, boxes=boxes, return_tensors="pt").to(self.device)
        # One box per image -> one expert selection per image.
        dataset_index = torch.tensor([self.COCO_EXPERT_INDEX] * len(batch), device=self.device)
        with torch.no_grad():
            outputs = model(**inputs, dataset_index=dataset_index)
        results = processor.post_process_pose_estimation(outputs, boxes=boxes)

        for (target, _), image_results in zip(batch, results):
            person = image_results[0]
            keypoints: list[Keypoint] = []
            for (x_px, y_px), score, label in zip(
                person["keypoints"].tolist(),
                person["scores"].tolist(),
                person["labels"].tolist(),
            ):
                label = int(label)
                if label >= len(COCO_KEYPOINT_NAMES):
                    continue
                keypoints.append(Keypoint(
                    name=COCO_KEYPOINT_NAMES[label],
                    x=x_px / width, y=y_px / height, score=float(score),
                ))
            out[target.frame_index] = keypoints


# Cached singletons: the models are expensive to load and re-init on CUDA, so we
# keep one resident instance each and reuse it across every job (this server runs
# locally for a single user — jobs are serialized by the caller's lock).
_TRACKER: Tracker | None = None
_POSE_BACKEND: PoseBackend | None = None


def default_tracker() -> Tracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = UltralyticsTracker()
    return _TRACKER


def default_pose_backend() -> PoseBackend:
    global _POSE_BACKEND
    if _POSE_BACKEND is None:
        _POSE_BACKEND = TransformersViTPoseBackend()
    return _POSE_BACKEND


def warm_backends() -> None:
    """Pre-load both models (call at startup so the first calibration is fast)."""
    b = default_tracker()
    p = default_pose_backend()
    if hasattr(b, "warm"):
        b.warm()
    if hasattr(p, "warm"):
        p.warm()
