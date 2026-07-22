"""Localized crop export for flagged detection-quality frames (issue #44 deliverable 2).

Writes small PNG thumbnails of each flagged (and, up to a per-Run cap, ``ok``) scanner
bounding box into a *gitignored* per-bundle ``crops/`` dir, and stamps the crop's relative
path back into the matching ``frameQuality`` entry (issue #44 deliverable 1). Reuses the
same cv2 decode as ``frames.py`` — best-effort: when cv2 or the (gitignored) video binary
is absent, or a frame can't be read, it writes nothing and the entry's ``crop`` stays
``None``, so the committed JSON record is never blocked on a binary that isn't in git.

Crops are prioritised worst-first: every flagged frame, then ``ok`` frames to fill the cap
— a Run with many failures spends its budget on the failures. Both negatives (flagged) and
positives (``ok``) are exported so a reviewer can calibrate the auto classes against real
image content. The ``crops/`` dir mirrors the video-binary rule in ``.gitignore``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:  # optional at import time — the JSON record path must work without cv2
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only when cv2 is absent
    cv2 = None

_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

# Per-Run crop budget and geometry. Provisional, like the #44 class thresholds.
CROP_EXPORT_CAP = 40      # max crops per Run (flagged first, then ok) — bounds disk
CROP_MARGIN = 0.15        # bbox padding as a fraction of bbox size
CROP_MAX_WIDTH = 160      # thumbnail max width in px (downscale only)

# Frame reader: timestamp (sec) -> full-frame grayscale ndarray, or None when unreadable.
FrameReader = Callable[[float], Any]


def find_video_path(video_dir: Path, video_key: str) -> Path | None:
    """The bundle's video binary (``<video_key>.mp4``, else any single video file)."""

    primary = video_dir / f"{video_key}.mp4"
    if primary.exists():
        return primary
    candidates = [p for p in video_dir.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES]
    return candidates[0] if candidates else None


def _bbox_from_keypoints(frame: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    """Padded, clamped normalized bbox ``(x0, y0, x1, y1)`` over a scanner frame's
    keypoints, or ``None`` when the frame carries no usable point."""

    kps = (frame or {}).get("keypoints") or []
    xs = [kp["x"] for kp in kps if kp.get("x") is not None]
    ys = [kp["y"] for kp in kps if kp.get("y") is not None]
    if not xs or not ys:
        return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mw = (x1 - x0) * CROP_MARGIN + 1e-3
    mh = (y1 - y0) * CROP_MARGIN + 1e-3
    x0, y0 = max(0.0, x0 - mw), max(0.0, y0 - mh)
    x1, y1 = min(1.0, x1 + mw), min(1.0, y1 + mh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _select_for_crop(entries: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """Flagged frames first (worst-first budget), then ``ok`` frames to fill the cap."""

    if cap <= 0:
        return []
    flagged = [e for e in entries if e.get("class") != "ok"]
    ok = [e for e in entries if e.get("class") == "ok"]
    selected = flagged[:cap]
    return selected + ok[: max(0, cap - len(selected))]


def _write_crop(gray: Any, box: tuple[float, float, float, float], path: Path) -> bool:
    """Crop the normalized box from a grayscale frame, downscale, and write a PNG.
    Returns False (writing nothing) when cv2 is absent or the crop is empty."""

    if cv2 is None or gray is None or getattr(gray, "size", 0) == 0:
        return False
    h, w = gray.shape[:2]
    x0 = max(0, min(w - 1, int(round(box[0] * w))))
    y0 = max(0, min(h - 1, int(round(box[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(box[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(box[3] * h))))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return False
    cw = crop.shape[1]
    if cw > CROP_MAX_WIDTH:
        scale = CROP_MAX_WIDTH / cw
        crop = cv2.resize(crop, (CROP_MAX_WIDTH, max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(path), crop))


class _Cv2FrameReader:
    """Grayscale frame reader over a video binary via cv2 (opened once, seeked per t)."""

    def __init__(self, video_path: Path):
        self._cap = cv2.VideoCapture(str(video_path)) if cv2 is not None else None
        if self._cap is not None and not self._cap.isOpened():
            self._cap = None

    def __call__(self, t: float):
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def export_run_crops(video_dir: Path, run_ts: str, pose_frames: list[dict[str, Any]],
                     frame_quality: dict[str, Any], *, decode: bool = True,
                     cap: int = CROP_EXPORT_CAP, frame_reader: FrameReader | None = None,
                     video_path: Path | None = None) -> int:
    """Export crops for a Run's flagged/ok frames; mutate their ``crop`` paths in place.

    Returns the number of crops written. Best-effort and side-effect-light: with
    ``decode=False`` (or no cv2 / no video / no readable frame) nothing is written and
    every ``crop`` stays ``None``. ``frame_reader`` is injectable for tests so the write
    path can be exercised without a real video binary."""

    entries = frame_quality.get("frames") or []
    selected = _select_for_crop(entries, cap)
    if not selected or not decode:
        return 0

    reader = frame_reader
    owns_reader = False
    if reader is None:
        vp = video_path or find_video_path(video_dir, video_dir.name)
        if vp is None or cv2 is None:
            return 0
        reader = _Cv2FrameReader(vp)
        owns_reader = True

    by_ts = {round(float(f.get("timestamp", 0.0)), 3): f for f in pose_frames}
    crops_dir = video_dir / "crops"
    written = 0
    try:
        for e in selected:
            t = float(e.get("t"))
            frame = by_ts.get(round(t, 3))
            box = _bbox_from_keypoints(frame)
            if box is None:
                continue
            gray = reader(t)
            if gray is None:
                continue
            crops_dir.mkdir(exist_ok=True)
            name = f"{run_ts}_{t:.3f}_{e.get('class', 'ok')}.png"
            if _write_crop(gray, box, crops_dir / name):
                e["crop"] = f"crops/{name}"
                written += 1
    finally:
        if owns_reader and isinstance(reader, _Cv2FrameReader):
            reader.close()
    return written
