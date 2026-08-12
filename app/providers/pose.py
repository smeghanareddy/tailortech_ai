"""Body measurement estimation from a single front-view photo.

Two paths, chosen at runtime:

  1. MediaPipe Pose — 33 skeletal landmarks, giving real joint positions for
     shoulder width, sleeve length and dress length.
  2. Silhouette fallback — if MediaPipe is unavailable, widths are sampled
     from the rembg segmentation mask at anthropometric fractions of stature.

## What this can and cannot know

A single uncalibrated photo carries no absolute scale. Pass `height_hint`
(the person's real height in cm) and everything is scaled from it, which is
accurate to within a few cm. Without it the service assumes a default stature,
which makes every output proportional-but-not-absolute — reported as low
confidence, and the frontend asks the user to review before saving.

Depth is also unobservable from one view, so girths (bust/waist/hip) are
derived from measured width via a per-part depth ratio and Ramanujan's
ellipse-perimeter approximation. These are estimates, not tape measurements.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from ..errors import ProviderUnavailable
from . import local_cv

logger = logging.getLogger(__name__)

name = "pose"

# Stature assumed when the caller gives no height_hint. Only affects the
# absolute scale; all ratios hold regardless.
DEFAULT_STATURE_CM = 163.0

# Height fractions measured up from the soles, used to sample silhouette
# widths. Standard anthropometric landmark heights.
LEVELS = {
    "shoulder": 0.818,
    "bust": 0.720,
    "waist": 0.620,
    "hip": 0.520,
}

# Body depth as a fraction of body width at the same level. Front-view photos
# cannot observe this, so these are population averages.
DEPTH_RATIOS = {
    "bust": 0.72,
    "waist": 0.70,
    "hip": 0.75,
    "neck": 1.00,  # the neck is close to circular
}


def available() -> bool:
    return local_cv.available()


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


# ── geometry helpers ─────────────────────────────────────────────


def ellipse_circumference(width: float, depth: float) -> float:
    """Ramanujan's approximation for the perimeter of an ellipse."""
    a, b = width / 2.0, depth / 2.0
    if a <= 0 or b <= 0:
        return 0.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def girth_from_width(width_cm: float, part: str) -> float:
    return ellipse_circumference(width_cm, width_cm * DEPTH_RATIOS[part])


def _row_width_px(mask: np.ndarray, row: int) -> float:
    """Width of the silhouette on one scanline, ignoring gaps (e.g. arms)."""
    row = int(np.clip(row, 0, mask.shape[0] - 1))
    xs = np.flatnonzero(mask[row] > 0)
    if xs.size == 0:
        return 0.0
    return float(xs[-1] - xs[0] + 1)


def _torso_width_px(mask: np.ndarray, row: int, band: int = 5) -> float:
    """Median width over a small band — robust to a single ragged scanline."""
    rows = range(max(0, row - band), min(mask.shape[0], row + band + 1))
    widths = [w for w in (_row_width_px(mask, r) for r in rows) if w > 0]
    return float(np.median(widths)) if widths else 0.0


# ── MediaPipe path ───────────────────────────────────────────────


def _mediapipe_landmarks(image_rgb: np.ndarray) -> dict | None:
    import mediapipe as mp  # noqa: PLC0415

    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=2, enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return None

    height, width = image_rgb.shape[:2]
    lm = mp.solutions.pose.PoseLandmark
    points = {}
    for landmark in lm:
        point = result.pose_landmarks.landmark[landmark.value]
        points[landmark.name.lower()] = {
            "x": point.x * width,
            "y": point.y * height,
            "visibility": round(float(point.visibility), 3),
        }
    return points


def _distance(points: dict, a: str, b: str) -> float:
    pa, pb = points.get(a), points.get(b)
    if not pa or not pb:
        return 0.0
    return math.dist((pa["x"], pa["y"]), (pb["x"], pb["y"]))


# ── public entry point ───────────────────────────────────────────


def torso_points_from_silhouette(mask: np.ndarray) -> dict:
    """Approximate shoulder and hip points from the segmentation mask alone.

    Used when pose detection finds nothing — a garment placed from the
    silhouette is far more useful than no preview at all.
    """
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return {}

    top, bottom = int(ys[0]), int(ys[-1])
    stature = float(bottom - top + 1)

    def bounds(level: str):
        row = int(np.clip(bottom - LEVELS[level] * stature, 0, mask.shape[0] - 1))
        xs = np.flatnonzero(mask[row] > 0)
        return (float(xs[0]), float(xs[-1]), float(row)) if xs.size else None

    shoulder, hip = bounds("shoulder"), bounds("hip")
    if shoulder is None or hip is None:
        return {}

    # The silhouette at shoulder height includes the arms, so narrow it to
    # roughly the bony shoulder span.
    centre = (shoulder[0] + shoulder[1]) / 2
    half = (shoulder[1] - shoulder[0]) / 2 * 0.62

    return {
        # Mirrors MediaPipe's convention: "left" is the subject's left.
        "right_shoulder": {"x": centre - half, "y": shoulder[2], "visibility": 0.3},
        "left_shoulder": {"x": centre + half, "y": shoulder[2], "visibility": 0.3},
        "right_hip": {"x": hip[0], "y": hip[2], "visibility": 0.3},
        "left_hip": {"x": hip[1], "y": hip[2], "visibility": 0.3},
    }


def estimate(photo: bytes, height_hint: float | None = None) -> dict:
    """Return {measurements, confidence, landmarks, notes, provider}."""
    if not local_cv.available():
        raise ProviderUnavailable(
            "Measurement estimation needs the local CV stack (rembg/opencv)"
        )

    source, _cutout, mask = local_cv.segment_person(photo)
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        raise ProviderUnavailable(
            "No person could be detected in the photo",
            detail="try a full-body, front-facing shot against a plain background",
        )

    top, bottom = int(ys[0]), int(ys[-1])
    stature_px = float(bottom - top + 1)
    if stature_px < 80:
        raise ProviderUnavailable(
            "The detected person is too small in the frame to measure",
        )

    notes: list[str] = []
    if height_hint and 60 <= float(height_hint) <= 250:
        stature_cm = float(height_hint)
        confidence = 0.72
    else:
        stature_cm = DEFAULT_STATURE_CM
        confidence = 0.38
        notes.append(
            "No height was supplied, so an average stature was assumed — "
            "proportions are reliable but absolute sizes are not."
        )

    cm_per_px = stature_cm / stature_px

    landmarks: dict = {}
    measurements: dict[str, float] = {"height": stature_cm}

    # Girths come from the silhouette in both paths — landmarks are skeletal
    # and carry no body width.
    for part in ("bust", "waist", "hip"):
        row = bottom - LEVELS[part] * stature_px
        width_cm = _torso_width_px(mask, int(row)) * cm_per_px
        if width_cm > 0:
            measurements[part] = girth_from_width(width_cm, part)

    points = None
    if mediapipe_available():
        # MediaPipe expects RGB, and reads the *original* photo — the cutout
        # can clip limbs and its transparency flattens to black.
        rgb = np.array(source.convert("RGB"))
        try:
            points = _mediapipe_landmarks(rgb)
        except Exception as exc:  # noqa: BLE001 - mediapipe raises broadly
            logger.warning("mediapipe failed, falling back to silhouette: %s", exc)
            notes.append("Pose detection failed; measurements came from the silhouette.")
        else:
            if points is None:
                notes.append(
                    "No pose could be detected, so measurements came from the "
                    "silhouette alone — a clear full-body front view improves this."
                )
    else:
        notes.append("MediaPipe is not installed; measurements came from the silhouette.")

    if points:
        landmarks = points
        confidence = min(0.85, confidence + 0.1)

        shoulder_px = _distance(points, "left_shoulder", "right_shoulder")
        if shoulder_px:
            measurements["shoulder_width"] = shoulder_px * cm_per_px

        # Shoulder → elbow → wrist, following the arm rather than cutting
        # straight across it.
        for side in ("left", "right"):
            upper = _distance(points, f"{side}_shoulder", f"{side}_elbow")
            fore = _distance(points, f"{side}_elbow", f"{side}_wrist")
            if upper and fore:
                measurements["sleeve_length"] = (upper + fore) * cm_per_px
                break

        # Shoulder → ankle, the usual full-length dress measure.
        for side in ("left", "right"):
            shoulder, ankle = points.get(f"{side}_shoulder"), points.get(f"{side}_ankle")
            if shoulder and ankle:
                measurements["dress_length"] = abs(ankle["y"] - shoulder["y"]) * cm_per_px
                break
    else:
        shoulder_row = bottom - LEVELS["shoulder"] * stature_px
        shoulder_cm = _torso_width_px(mask, int(shoulder_row)) * cm_per_px
        if shoulder_cm > 0:
            # The silhouette at shoulder height includes the arms, so the
            # bony shoulder span is a fraction of the measured span.
            measurements["shoulder_width"] = shoulder_cm * 0.78
        measurements["sleeve_length"] = stature_cm * 0.32
        measurements["dress_length"] = stature_cm * 0.78
        # Silhouette-only results carry no joint positions — sleeve and dress
        # length are pure ratios here, so the score has to reflect that.
        confidence *= 0.8

    if measurements.get("shoulder_width"):
        neck_width = measurements["shoulder_width"] * 0.32
        measurements["neck_size"] = girth_from_width(neck_width, "neck")

    rounded = {k: round(float(v), 1) for k, v in measurements.items() if v and v > 0}

    # Pose ran on the downscaled analysis image, so any consumer warping onto
    # the original photo needs to know what frame these coordinates are in.
    if not landmarks:
        landmarks = torso_points_from_silhouette(mask)

    return {
        "measurements": rounded,
        "confidence": round(confidence, 2),
        "landmarks": landmarks,
        "frame": {"width": source.width, "height": source.height},
        "notes": notes,
        "provider": name if points else f"{name}:silhouette",
    }
