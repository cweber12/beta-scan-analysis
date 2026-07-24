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
