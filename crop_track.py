"""Per-Bundle crop trajectory for harness detection runs (issue #169, PRD #156).

**Why this exists.** Full-frame MediaPipe cannot see the climber on much of this corpus.
Probing three present truth frames per Bundle: 24% of Bundles (21 of 86) detect *nothing*,
and only about half of all probes detect at all — while the scanner, which crops, reaches a
median 86.9% on those same Bundles with the climber occupying a median 5.1% of frame. The
gap is the crop. Until a crop exists, every arm comparison is measured in a regime where the
detector is blind on a quarter of the corpus.

**Two stages.** A tracking pass (stage A) walks the video and writes the trajectory this
module models; the sparse experimental runs (stage B) read it and crop at the nearest
recorded position. That split is what lets a run sample ``12·√n`` scattered frames without
needing frame-to-frame continuity to find the climber.

**Where the seed comes from, and where it must not.** Tracking is seeded from
``setup.json.climberPoint`` — the **setup tap**, human calibration, frozen by ADR 0007
precisely so it stays a stable anchor. It is *not* Ground Truth. Seeding from
``vitpose.json`` or ``ground-truth.json`` would hand the detector the answer: detection rate
would approach 100% and mean nothing. That path is closed deliberately, and tested for.

**Re-acquisition is the whole problem.** A prototype without it reached 59–100% (median 81%)
across six Bundles where full-frame gets 0%, and every one of the missing frames went the
same way: the crop lost the climber and then sat on a stale location forever. A finer step
does not fix it — 0.25 s scored *worse* than 0.5 s, because the climber was never moving fast
enough for step to be the binding constraint. So this module spends its complexity on
recovery: velocity extrapolation, an expanding search ladder, and an appearance check that
stops the ladder from latching onto a belayer or a bystander on the way back.

This module is **pure**. It takes a ``probe`` callable and knows nothing about MediaPipe or
video decoding, which is what lets the whole state machine be tested with a stub — the same
property ``test_vitpose_job.py`` has with torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ARTIFACT_NAME = "crop-track.json"
ARTIFACT_VERSION = 1

# How a trajectory entry was arrived at. Kept distinct because they mean different things to
# a reader: a ``detected`` crop is where the climber *was*, a ``predicted`` one is where we
# think they went, and stage B cropping at a predicted position is a materially weaker
# proposition than cropping at a detected one.
SRC_DETECTED = "detected"
SRC_PREDICTED = "predicted"
SRC_LOST = "lost"


@dataclass(frozen=True)
class CropTrackConfig:
    """Everything about *how* a trajectory was produced that could change it.

    Joins the arm identity, because two runs cropped by different trajectories are not
    comparable and must not share a stamp. Same reasoning as ``DetectionConfig`` — anything
    that can move the output and is not recorded here is a silent confound.
    """

    # Candidate half-widths of the square crop, as a fraction of frame. Stage A tries each
    # and keeps whichever tracked best.
    #
    # **This is a policy, and the policy is what the arm identity stamps — never the value
    # that won.** Crop size has no global optimum: measured across 12 Bundles where
    # full-frame detects nothing, 0.15 and 0.20 tie on the median and win in opposite
    # directions on individual videos (klkrpdk7zbo 76% vs 6%; DEDBeWcqxK8 32% vs 96%). So a
    # single global size leaves about ten points on the table.
    #
    # Selecting per Bundle is legitimate because stage A is *not what is being measured* —
    # the trajectory is held fixed across every arm compared against it, so a per-Bundle
    # choice is a nuisance parameter, not a factor. But it must stay out of the arm hash:
    # ``crop_track_hash`` feeds ``DetectionConfig``, so stamping the winning value would make
    # every Bundle its own arm and pooling would collapse entirely.
    half_candidates: tuple[float, ...] = (0.15, 0.20)
    # Seconds between tracking probes. 0.5 measured no worse than 0.25 — the climber does not
    # move fast enough for step to bind — so the default buys nothing by going finer.
    step: float = 0.5
    # Search ladder: multipliers applied to ``half`` on consecutive misses. Widening trades
    # the size floor (a bigger crop makes the climber smaller in frame) against the chance of
    # covering wherever they actually went, so it widens only after a miss and collapses back
    # on the first hit.
    ladder: tuple[float, ...] = (1.0, 1.6, 2.4)
    # Velocity carried into the next predicted centre, damped per step. Climbers move
    # steadily, so extrapolation is worth something; they also stop, so it decays.
    velocity_decay: float = 0.6
    # Consecutive misses after which the track is declared lost and stops predicting. Beyond
    # this the prediction is fantasy and an honest ``lost`` is worth more than a stale box.
    lost_after: int = 8
    # Appearance distance at/below which a detection is folded into the rolling reference,
    # and at/above which it is treated as probably-somebody-else. Mirrors vitpose_job's
    # thresholds, which were measured on the planet-x pair: the reappearing climber sat at
    # 0.35 while base bystanders sat at 0.76–0.84.
    appearance_confident: float = 0.45
    appearance_mismatch: float = 0.65
    appearance_ema: float = 0.2

    def identity(self) -> dict[str, Any]:
        return {
            "halfCandidates": list(self.half_candidates),
            "step": self.step,
            "ladder": list(self.ladder),
            "velocityDecay": self.velocity_decay,
            "lostAfter": self.lost_after,
            "appearanceConfident": self.appearance_confident,
            "appearanceMismatch": self.appearance_mismatch,
            "appearanceEma": self.appearance_ema,
        }


def track_hash(config: CropTrackConfig) -> str:
    blob = json.dumps(config.identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CropBox:
    """A square crop, normalized to the full frame and centred on ``(cx, cy)``."""

    timestamp: float
    cx: float
    cy: float
    half: float
    source: str
    appearance_dist: float | None = None

    def rect(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` clamped into frame."""
        return (
            max(0.0, self.cx - self.half), max(0.0, self.cy - self.half),
            min(1.0, self.cx + self.half), min(1.0, self.cy + self.half),
        )


@dataclass
class CropTrack:
    """One Bundle's crop trajectory, and the config that produced it."""

    config: CropTrackConfig
    boxes: list[CropBox] = field(default_factory=list)
    seed: dict[str, Any] | None = None
    setup_hash: str | None = None
    # Which candidate half-width actually produced this trajectory. Recorded in the artifact
    # for audit, deliberately absent from the arm hash — see CropTrackConfig.half_candidates.
    selected_half: float | None = None

    @property
    def detected(self) -> int:
        return sum(1 for b in self.boxes if b.source == SRC_DETECTED)

    def rate(self) -> float:
        return self.detected / len(self.boxes) if self.boxes else 0.0

    def nearest(self, timestamp: float) -> CropBox | None:
        """The recorded crop closest in time to ``timestamp``.

        Stage B samples a sparse, arbitrary grid that will not line up with the tracking
        step, so every experimental timestamp maps onto its nearest tracked position. No
        tolerance: a distant match is still the best information available, and refusing it
        would silently reintroduce the full-frame blindness this module exists to remove.
        The distance is recorded per run instead, so a reader can judge it.
        """

        if not self.boxes:
            return None
        return min(self.boxes, key=lambda b: abs(b.timestamp - timestamp))

    def to_artifact(self) -> dict[str, Any]:
        return {
            "version": ARTIFACT_VERSION,
            "trackerConfig": self.config.identity(),
            "trackerHash": track_hash(self.config),
            "setupHash": self.setup_hash,
            "selectedHalf": self.selected_half,
            "seed": self.seed,
            "stats": {
                "frames": len(self.boxes),
                "detected": self.detected,
                "rate": round(self.rate(), 4),
                "predicted": sum(1 for b in self.boxes if b.source == SRC_PREDICTED),
                "lost": sum(1 for b in self.boxes if b.source == SRC_LOST),
            },
            "frames": [
                {
                    "timestamp": b.timestamp, "cx": round(b.cx, 6), "cy": round(b.cy, 6),
                    "half": round(b.half, 6), "source": b.source,
                    **({"appearanceDist": round(b.appearance_dist, 4)}
                       if b.appearance_dist is not None else {}),
                }
                for b in self.boxes
            ],
        }

    @classmethod
    def from_artifact(cls, doc: dict[str, Any]) -> "CropTrack":
        cfg = doc.get("trackerConfig") or {}
        config = CropTrackConfig(
            half_candidates=tuple(
                float(v) for v in (cfg.get("halfCandidates") or (0.15, 0.20))),
            step=float(cfg.get("step", 0.5)),
            ladder=tuple(float(v) for v in (cfg.get("ladder") or (1.0, 1.6, 2.4))),
            velocity_decay=float(cfg.get("velocityDecay", 0.6)),
            lost_after=int(cfg.get("lostAfter", 8)),
            appearance_confident=float(cfg.get("appearanceConfident", 0.45)),
            appearance_mismatch=float(cfg.get("appearanceMismatch", 0.65)),
            appearance_ema=float(cfg.get("appearanceEma", 0.2)),
        )
        boxes = [
            CropBox(
                timestamp=float(f.get("timestamp", 0.0)), cx=float(f.get("cx", 0.5)),
                cy=float(f.get("cy", 0.5)),
                half=float(f.get("half", config.half_candidates[0])),
                source=str(f.get("source", SRC_DETECTED)),
                appearance_dist=f.get("appearanceDist"),
            )
            for f in (doc.get("frames") or []) if isinstance(f, dict)
        ]
        return cls(config=config, boxes=boxes, seed=doc.get("seed"),
                   setup_hash=doc.get("setupHash"),
                   selected_half=doc.get("selectedHalf"))


def load_crop_track(bundle_dir: Path) -> CropTrack | None:
    try:
        doc = json.loads((bundle_dir / ARTIFACT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return CropTrack.from_artifact(doc) if isinstance(doc, dict) else None


def write_crop_track(bundle_dir: Path, track: CropTrack) -> Path:
    path = bundle_dir / ARTIFACT_NAME
    path.write_text(json.dumps(track.to_artifact(), indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Appearance — the guard that stops recovery latching onto the wrong person
# --------------------------------------------------------------------------- #

def bhattacharyya(p: Sequence[float], q: Sequence[float]) -> float:
    """Distance between two L1-normalized histograms, in [0, 1].

    Same measure ``vitpose_job`` uses for Climber Identity; reimplemented here rather than
    imported because that module's version is bound to its own ``Appearance`` type and this
    one must stay importable without it. Kept deliberately identical so the two sides of the
    harness agree on what "looks like the same person" means.
    """

    bc = 0.0
    for a, b in zip(p, q):
        if a > 0.0 and b > 0.0:
            bc += (a * b) ** 0.5
    return max(0.0, 1.0 - min(bc, 1.0)) ** 0.5


def _blend(ref: tuple[float, ...], new: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    """Rolling reference update. Never a frozen seed-time snapshot — appearance decays over
    an ascent as lighting and scale change, and a frozen reference would slowly declare the
    climber to be a stranger."""

    if not ref:
        return new
    if not new:
        return ref
    mixed = [(1.0 - alpha) * a + alpha * b for a, b in zip(ref, new)]
    total = sum(mixed) or 1.0
    return tuple(v / total for v in mixed)


# --------------------------------------------------------------------------- #
# The tracker
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Probe:
    """What a probe reports when it finds someone inside the crop it was given."""

    cx: float                       # full-frame normalized centre of the found pose
    cy: float
    appearance: tuple[float, ...] = ()   # optional histogram; () means "no signature"


# ``(timestamp, frame, cx, cy, half) -> Probe | None`` — run the detector on the crop
# ``(cx, cy, half)`` of ``frame`` and report where the pose landed, in full-frame
# coordinates. ``None`` means nothing was found there.
ProbeFn = Callable[[float, Any, float, float, float], "Probe | None"]


def track(
    frames: Iterable[tuple[float, Any]],
    probe: ProbeFn,
    config: CropTrackConfig,
    seed_x: float,
    seed_y: float,
    half: float | None = None,
) -> CropTrack:
    """Walk the video, following the climber, and return the crop trajectory.

    ``frames`` yields ``(timestamp, frame)`` in ascending time; ``probe`` runs the detector
    on a crop. The state machine is the whole point of this module, so it is spelled out
    rather than tuned:

    - **Predict** the next centre by carrying the last observed velocity, damped. Climbers
      move steadily enough for this to be worth something and stop often enough that it
      must decay.
    - **Ladder** the crop wider on each consecutive miss, and collapse back to the base size
      on the first hit. Widening is not free — a bigger crop makes the climber smaller,
      which is the exact size floor that makes full-frame fail — so it is a response to
      failure, never a default.
    - **Gate on appearance** so the widened search cannot quietly adopt a belayer. A
      candidate far from the rolling reference is recorded and used, but never folded into
      the reference, so one bad frame cannot drag the reference onto the wrong person.
    - **Give up honestly** after ``lost_after`` consecutive misses: emit ``lost`` rather
      than a confident-looking stale box. Stage B can then see that those timestamps were
      cropped on a guess.
    """

    half = config.half_candidates[0] if half is None else half
    track_out = CropTrack(config=config, selected_half=half)
    cx, cy = seed_x, seed_y
    vx = vy = 0.0
    misses = 0
    ref: tuple[float, ...] = ()
    last_t: float | None = None

    for timestamp, frame in frames:
        dt = (timestamp - last_t) if last_t is not None else 0.0
        # Predicted centre: where the climber would be if they kept going.
        px = min(1.0, max(0.0, cx + vx * dt))
        py = min(1.0, max(0.0, cy + vy * dt))

        hit: Probe | None = None
        used_half = half
        for rung in config.ladder[:max(1, min(len(config.ladder), misses + 1))]:
            used_half = half * rung
            hit = probe(timestamp, frame, px, py, used_half)
            if hit is not None:
                break

        if hit is None:
            misses += 1
            source = SRC_LOST if misses >= config.lost_after else SRC_PREDICTED
            # Keep coasting on decayed velocity while predicting; freeze once lost, since
            # extrapolating past the give-up point invents a trajectory nobody observed.
            if source == SRC_PREDICTED:
                cx, cy = px, py
                vx *= config.velocity_decay
                vy *= config.velocity_decay
            else:
                vx = vy = 0.0
            track_out.boxes.append(CropBox(timestamp, cx, cy, used_half, source))
            last_t = timestamp
            continue

        dist = bhattacharyya(ref, hit.appearance) if (ref and hit.appearance) else None
        if dt > 0.0:
            vx, vy = (hit.cx - cx) / dt, (hit.cy - cy) / dt
        cx, cy = hit.cx, hit.cy
        misses = 0
        # Fold into the reference only on a confident match, so a frame that may be someone
        # else cannot become the definition of who we are following.
        if hit.appearance and (dist is None or dist <= config.appearance_confident):
            ref = _blend(ref, hit.appearance, config.appearance_ema) if ref else hit.appearance
        track_out.boxes.append(
            CropBox(timestamp, cx, cy, half, SRC_DETECTED, appearance_dist=dist))
        last_t = timestamp

    return track_out
