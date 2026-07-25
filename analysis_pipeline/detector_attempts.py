"""Parser for scanner detector-attempt evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DETECTOR_ATTEMPT_STATUSES = frozenset({
    "accepted",
    "missing",
    "flipRejected",
    "qualityRejected",
})

DETECTOR_ATTEMPT_EVIDENCE_ATTEMPTS = "attempts"
DETECTOR_ATTEMPT_EVIDENCE_UNKNOWN = "unknown"

_REGION_KEYS = ("x", "y", "w", "h")


def _region(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value[key] for key in _REGION_KEYS if key in value}


def _list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def parse_detector_attempts(pose_data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return normalized Detector Attempts, or ``None`` when the stream is absent.

    ``None`` is the important compatibility state: legacy frame-only runs have
    unknown detector-attempt evidence and must not be read as raw detector success.
    Field values that are already normalized by the scanner are copied without
    clamping or synthesis so full-frame rectangles stay explicit and ``null`` stays
    unknown/not applicable.
    """

    attempts = pose_data.get("detectorAttempts") if isinstance(pose_data, dict) else None
    if attempts is None:
        return None
    if not isinstance(attempts, list):
        return []

    parsed: list[dict[str, Any]] = []
    for raw in attempts:
        if not isinstance(raw, dict):
            continue
        parsed.append({
            "timestamp": float(raw.get("timestamp", 0.0)),
            "status": raw.get("status"),
            "initialSearchRegion": _region(raw.get("initialSearchRegion")),
            "detectionRegion": _region(raw.get("detectionRegion")),
            "reacquireAttempted": bool(raw.get("reacquireAttempted", False)),
            "reacquired": bool(raw.get("reacquired", False)),
            "rawKeypoints": _list(raw.get("rawKeypoints")),
            "acceptedKeypoints": _list(raw.get("acceptedKeypoints")),
            "searchConditions": deepcopy(raw.get("searchConditions")),
            "reacquireConditions": deepcopy(raw.get("reacquireConditions")),
            "candidateCount": raw.get("candidateCount"),
            "rejectedCandidateCount": raw.get("rejectedCandidateCount"),
            "selectionMethod": raw.get("selectionMethod"),
            "statusKnown": raw.get("status") in DETECTOR_ATTEMPT_STATUSES,
        })
    return parsed


def detector_attempt_evidence(attempts: list[dict[str, Any]] | None) -> str:
    return (
        DETECTOR_ATTEMPT_EVIDENCE_UNKNOWN
        if attempts is None
        else DETECTOR_ATTEMPT_EVIDENCE_ATTEMPTS
    )


# --------------------------------------------------------------------------- #
# Attempt-evidence primitives
#
# These read one attempt's regions and scanner-computed conditions. They live here,
# beside the parser, because every consumer needs the *same* reading of the evidence —
# the run table (``runs``), the pooled trend tables (``trends``) and evaluation scoring
# (``evaluate``). ``evaluate`` is deliberately stdlib-only, so it cannot reach into
# ``runs`` (pandas) for them; one implementation here is what keeps the three consumers
# from drifting.
# --------------------------------------------------------------------------- #

# Scanner-computed pixel stats on an attempt's conditions block → column suffixes.
ATTEMPT_STAT_KEYS = {
    "mean": "luma_mean",
    "stdDev": "luma_stdDev",
    "sharpness": "sharpness",
}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _slug(name: str) -> str:
    out: list[str] = []
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


def region_metric(region: Any, metric: str) -> float | None:
    if not isinstance(region, dict):
        return None
    x = _num(region.get("x"))
    y = _num(region.get("y"))
    w = _num(region.get("w"))
    h = _num(region.get("h"))
    if metric == "area" and w is not None and h is not None:
        return max(0.0, w * h)
    if metric == "cx" and x is not None and w is not None:
        return x + w / 2
    if metric == "cy" and y is not None and h is not None:
        return y + h / 2
    if metric == "edge_distance" and all(v is not None for v in (x, y, w, h)):
        assert x is not None and y is not None and w is not None and h is not None
        return max(0.0, min(x, y, 1.0 - (x + w), 1.0 - (y + h)))
    return None


def is_full_frame(region: Any) -> bool:
    if not isinstance(region, dict):
        return False
    x = _num(region.get("x"))
    y = _num(region.get("y"))
    w = _num(region.get("w"))
    h = _num(region.get("h"))
    return (
        x is not None and y is not None and w is not None and h is not None
        and x <= 0.001 and y <= 0.001 and w >= 0.999 and h >= 0.999
    )


def condition_flags(conditions: Any) -> dict[str, bool]:
    """The boolean condition flags on one attempt's conditions block, slug-keyed.

    Reads both the nested ``flags`` dict and any top-level boolean, so a scanner that
    promotes a flag out of ``flags`` doesn't silently drop it from the analysis."""

    if not isinstance(conditions, dict):
        return {}
    out: dict[str, bool] = {}
    flags = conditions.get("flags")
    if isinstance(flags, dict):
        for key, value in flags.items():
            out[_slug(str(key))] = bool(value)
    for key, value in conditions.items():
        if key in ATTEMPT_STAT_KEYS or key == "flags":
            continue
        if isinstance(value, bool):
            out[_slug(str(key))] = value
    return out


def region_rect(region: Any) -> tuple[float, float, float, float] | None:
    """One region as ``(x0, y0, x1, y1)`` in normalized frame coords, or ``None`` when
    any component is missing. Regions are already normalized by the scanner, so nothing
    is clamped here — a rect that overhangs the frame stays explicit."""

    if not isinstance(region, dict):
        return None
    x, y = _num(region.get("x")), _num(region.get("y"))
    w, h = _num(region.get("w")), _num(region.get("h"))
    if x is None or y is None or w is None or h is None:
        return None
    return (x, y, x + w, y + h)


def rect_area(rect: tuple[float, float, float, float] | None) -> float:
    if rect is None:
        return 0.0
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def rect_intersection_area(a: tuple[float, float, float, float] | None,
                           b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def rect_iou(a: tuple[float, float, float, float] | None,
             b: tuple[float, float, float, float] | None) -> float | None:
    """Intersection-over-union of two rects, or ``None`` when either is absent or the
    union is degenerate (a zero-area rect has no meaningful overlap ratio)."""

    inter = rect_intersection_area(a, b)
    union = rect_area(a) + rect_area(b) - inter
    if union <= 0:
        return None
    return inter / union


def rect_containment(inner: tuple[float, float, float, float] | None,
                     outer: tuple[float, float, float, float] | None) -> float | None:
    """The share of ``inner``'s area that falls inside ``outer``.

    Directional on purpose, unlike IoU: "did the searched region cover the Climber" is
    a containment question, and IoU would penalise a correctly-placed crop merely for
    being larger than the Climber."""

    inner_area = rect_area(inner)
    if inner_area <= 0 or outer is None:
        return None
    return rect_intersection_area(inner, outer) / inner_area
