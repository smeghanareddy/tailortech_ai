"""Local layer compositor — the "Layer Merge → Final Preview" step.

Takes the ordered layer manifest the Rails LayerComposer produces and flattens
it into one PNG:

    Body → Dress Base → Neckline → Sleeve → Fabric → Pattern →
    Embroidery → Lace → Belt → Buttons → Shadow

Everything here is Pillow and numpy — no model, no API key, no per-call cost,
and the output is deterministic: it shows exactly what the customer selected,
which a diffusion model cannot guarantee.

Colour is applied as a multiply against the layer's own art, so shading and
folds in an uploaded asset survive being recoloured.
"""

from __future__ import annotations

import io
import logging
import math
from functools import lru_cache

import httpx
import numpy as np
from PIL import Image

from ..errors import InvalidInput

logger = logging.getLogger(__name__)

name = "local_compose"

CANVAS = (900, 1200)  # 3:4, matching the studio preview tiles

# From the spec: when stripes overlap, this decides which paints on top.
DIRECTION_PRIORITY = {"curve": 4, "left_cross": 3, "right_cross": 2,
                      "straight": 1, "triangle": 0}

# Band angle in degrees for each print direction.
DIRECTION_ANGLE = {"straight": 90.0, "left_cross": 45.0, "right_cross": -45.0,
                   "triangle": 0.0, "curve": None}  # curve is radial


def available() -> bool:
    return True


# ── fetching ─────────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _fetch(url: str) -> bytes:
    """Layer art by URL, memoised for the life of the process.

    follow_redirects matters: Active Storage serves blobs via a redirect.
    """
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def _load(url: str) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(_fetch(url))).convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - a bad layer must not kill the render
        logger.warning("layer image unavailable (%s): %s", url, exc)
        return None


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to fit inside the canvas, centred, on transparency."""
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


# ── colour ───────────────────────────────────────────────────────


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = str(value).lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        raise InvalidInput(f"{value!r} is not a hex colour")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _stripe_field(size: tuple[int, int], stripes: list[dict]) -> np.ndarray:
    """An RGB image of coloured bands, one band per stripe entry.

    Stripes are sorted by the spec's direction priority so the highest-priority
    entry owns the first band, and its direction sets the band angle.
    """
    width, height = size
    ordered = sorted(
        stripes, key=lambda s: -DIRECTION_PRIORITY.get(str(s.get("direction", "straight")), 1)
    )
    colours = [_hex_to_rgb(s.get("color", "#cccccc")) for s in ordered]
    direction = str(ordered[0].get("direction", "straight"))

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    if direction == "curve":
        # Concentric bands radiating from the top centre.
        t = np.sqrt((xs - width / 2) ** 2 + (ys * 0.6) ** 2)
        t = t / (t.max() or 1)
    else:
        angle = math.radians(DIRECTION_ANGLE.get(direction, 90.0) or 90.0)
        t = xs * math.cos(angle) + ys * math.sin(angle)
        t = (t - t.min()) / ((t.max() - t.min()) or 1)

    index = np.clip((t * len(colours)).astype(np.int32), 0, len(colours) - 1)
    field = np.zeros((height, width, 3), dtype=np.float32)
    for i, rgb in enumerate(colours):
        field[index == i] = rgb
    return field


def _zone_split(layer: Image.Image, bodice, skirt, waist: float = 0.46) -> Image.Image:
    """Colour one garment layer in two zones, split at the waist.

    A fixed dress (Anarkali, frock, kurti) is a single asset covering bodice
    and skirt, so a single colour would silently discard whatever the customer
    chose for the other zone. The split is taken across the layer's own
    content, not the canvas, so it lands on the garment's waist wherever the
    art sits.
    """
    box = layer.getbbox()
    if box is None:
        return _apply_colour(layer, bodice)

    top, bottom = box[1], box[3]
    cut = int(top + (bottom - top) * waist)

    upper = _apply_colour(layer, bodice)
    lower = _apply_colour(layer, skirt)

    out = upper.copy()
    out.paste(lower.crop((0, cut, layer.width, layer.height)), (0, cut))
    return out


def _apply_colour(layer: Image.Image, colour) -> Image.Image:
    """Multiply the layer's art by a solid colour or a stripe field.

    Multiply (rather than a flat fill) is what keeps shading, seams and folds
    visible in an uploaded photographic asset after recolouring.
    """
    if not colour:
        return layer

    array = np.asarray(layer).astype(np.float32)
    rgb, alpha = array[..., :3], array[..., 3:]

    if isinstance(colour, dict):
        stripes = colour.get("stripes") or []
        if stripes:
            tint = _stripe_field(layer.size, stripes)
        elif colour.get("base"):
            tint = np.array(_hex_to_rgb(colour["base"]), dtype=np.float32)
        else:
            return layer
    else:
        tint = np.array(_hex_to_rgb(colour), dtype=np.float32)

    # Where the art is near-white, multiply gives the flat colour; where it is
    # shaded, the shading is preserved.
    tinted = rgb * (tint / 255.0)
    return Image.fromarray(
        np.concatenate([np.clip(tinted, 0, 255), alpha], axis=-1).astype(np.uint8), "RGBA"
    )


# ── blending ─────────────────────────────────────────────────────


def _blend(base: Image.Image, top: Image.Image, mode: str | None, opacity: float) -> Image.Image:
    """Composite `top` over `base`, honouring blend mode and opacity."""
    if opacity <= 0:
        return base

    if opacity < 1:
        alpha = top.getchannel("A").point(lambda a: int(a * opacity))
        top = top.copy()
        top.putalpha(alpha)

    if mode in (None, "", "normal"):
        return Image.alpha_composite(base, top)

    b = np.asarray(base).astype(np.float32) / 255.0
    t = np.asarray(top).astype(np.float32) / 255.0
    ta = t[..., 3:]

    if mode == "multiply":
        blended = b[..., :3] * t[..., :3]
    elif mode == "overlay":
        blended = np.where(
            b[..., :3] <= 0.5,
            2 * b[..., :3] * t[..., :3],
            1 - 2 * (1 - b[..., :3]) * (1 - t[..., :3]),
        )
    elif mode == "screen":
        blended = 1 - (1 - b[..., :3]) * (1 - t[..., :3])
    else:
        return Image.alpha_composite(base, top)

    # multiply/overlay/screen are adjustment layers: a fabric swatch or print
    # modulates the garment it sits on and must never extend past it. Masking
    # by the accumulated alpha keeps a full-bleed texture inside the silhouette
    # instead of tinting the whole frame.
    mask = ta * b[..., 3:]
    rgb = b[..., :3] * (1 - mask) + blended * mask
    out_alpha = b[..., 3:]  # coverage is unchanged by an adjustment layer
    return Image.fromarray(
        (np.concatenate([rgb, out_alpha], axis=-1) * 255).astype(np.uint8), "RGBA"
    )


# ── public API ───────────────────────────────────────────────────


def compose(layers: list[dict], width: int | None = None, height: int | None = None,
            background: str | None = None) -> tuple[Image.Image, dict]:
    """Flatten the manifest. Returns (image, report)."""
    size = (int(width or CANVAS[0]), int(height or CANVAS[1]))

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    if background:
        canvas = Image.new("RGBA", size, (*_hex_to_rgb(background), 255))

    painted, skipped = [], []

    for layer in sorted(layers, key=lambda item: item.get("z_index", 0)):
        url = layer.get("url")
        if not url:
            # The body/mannequin slot carries no asset — nothing to paint.
            skipped.append({"slot": layer.get("slot"), "reason": "no image"})
            continue

        art = _load(url)
        if art is None:
            skipped.append({"slot": layer.get("slot"), "reason": "fetch failed"})
            continue

        art = _fit(art, size)

        # A full-dress layer carries both zones; anything else takes one colour.
        secondary = layer.get("secondary_color")
        if layer.get("slot") == "dress_base" and secondary:
            art = _zone_split(art, layer.get("color"), secondary,
                              float(layer.get("waist_ratio") or 0.46))
        else:
            art = _apply_colour(art, layer.get("color"))
        canvas = _blend(canvas, art, layer.get("blend_mode"),
                        float(layer.get("opacity", 1.0) or 1.0))
        painted.append(layer.get("slot"))

    return canvas, {"painted": painted, "skipped": skipped,
                    "layers_in": len(layers), "layers_painted": len(painted)}


def _similarity_transform(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """2x3 matrix taking the segment src_a→src_b onto dst_a→dst_b.

    Two corresponding points give scale, rotation and translation — enough to
    seat a garment on a body without the shear a bounding-box fit introduces.
    """
    sx, sy = src_b[0] - src_a[0], src_b[1] - src_a[1]
    dx, dy = dst_b[0] - dst_a[0], dst_b[1] - dst_a[1]

    src_len = math.hypot(sx, sy) or 1.0
    scale = (math.hypot(dx, dy) or 1.0) / src_len
    theta = math.atan2(dy, dx) - math.atan2(sy, sx)

    cos_t, sin_t = math.cos(theta) * scale, math.sin(theta) * scale
    tx = dst_a[0] - (cos_t * src_a[0] - sin_t * src_a[1])
    ty = dst_a[1] - (sin_t * src_a[0] + cos_t * src_a[1])
    return np.float32([[cos_t, -sin_t, tx], [sin_t, cos_t, ty]])


def _forearm_mask(shape, points, thickness) -> np.ndarray:
    """Where the wearer's forearms and hands should sit in front of the cloth.

    Only elbow→wrist: the upper arm stays behind so sleeves still read as
    sleeves rather than being punched through.
    """
    import cv2  # noqa: PLC0415

    mask = np.zeros(shape[:2], np.uint8)
    for side in ("left", "right"):
        elbow, wrist = points.get(f"{side}_elbow"), points.get(f"{side}_wrist")
        if not elbow or not wrist:
            continue
        cv2.line(mask, (int(elbow["x"]), int(elbow["y"])), (int(wrist["x"]), int(wrist["y"])),
                 255, max(6, int(thickness)))
        cv2.circle(mask, (int(wrist["x"]), int(wrist["y"])), max(8, int(thickness * 0.8)), 255, -1)

        for finger in (f"{side}_index", f"{side}_pinky", f"{side}_thumb"):
            tip = points.get(finger)
            if tip:
                cv2.circle(mask, (int(tip["x"]), int(tip["y"])), max(7, int(thickness * 0.6)), 255, -1)

    return cv2.GaussianBlur(mask, (9, 9), 0)


def _match_exposure(garment_rgb: np.ndarray, alpha: np.ndarray,
                    photo_rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Nudge the garment's brightness toward the light it is being placed in.

    A studio-lit product shot dropped into a sunlit photo reads as a sticker.
    Luminance only — hue is left alone so the dress keeps its actual colour.
    """
    import cv2  # noqa: PLC0415

    if not region.any() or not (alpha > 20).any():
        return garment_rgb

    target = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2LAB)[..., 0][region > 0].mean()
    source_l = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2LAB)[..., 0]
    source = source_l[alpha > 20].mean()
    if source <= 1:
        return garment_rgb

    # Clamp: enough to sit in the scene, not enough to blow out the fabric.
    gain = float(np.clip(target / source, 0.75, 1.35))
    return np.clip(garment_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


TPS_POINTS = (
    ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
     "left_elbow", "right_elbow"),
)[0]


def _tps_fit(canvas: np.ndarray, garment_pts: list, wearer_pts: list) -> np.ndarray | None:
    """Bend the garment onto the wearer with a thin-plate spline.

    A similarity transform can only scale, rotate and shift — the dress stays
    rigid, so it never follows a body that is turned, leaning, or a different
    shape from the model it was photographed on. A spline driven by matching
    landmarks deforms it to fit: hips move to the wearer's hips, the torso
    stretches to their length.

    Image corners are pinned as extra correspondences, otherwise the spline is
    free to swing the whole frame around and the result smears.
    """
    import cv2  # noqa: PLC0415

    if len(garment_pts) < 3:
        return None

    h, w = canvas.shape[:2]
    anchors = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
               (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]

    src = np.array(garment_pts + anchors, np.float32).reshape(1, -1, 2)
    dst = np.array(wearer_pts + anchors, np.float32).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(src.shape[1])]

    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(dst, src, matches)
        return tps.warpImage(canvas)
    except (cv2.error, AttributeError) as exc:
        # createThinPlateSplineShapeTransformer needs opencv-contrib.
        logger.info("thin-plate spline unavailable, keeping the rigid fit: %s", exc)
        return None


def _transfer_shading(rgb: np.ndarray, alpha: np.ndarray, photo_rgb: np.ndarray,
                      strength: float = 0.55) -> np.ndarray:
    """Push the wearer's own light and shade onto the flat garment.

    A product shot is evenly lit, so laid over a photo it reads as a panel
    rather than cloth on a body. The wearer's torso already carries the
    shading their shape produces — where the body turns away, where an arm
    casts a shadow. Sampling that as a low-frequency multiplier gives the
    garment the same falloff, which is what makes it sit *on* someone.

    Low frequency only: the original clothing's pattern must not print
    through onto the new garment.
    """
    import cv2  # noqa: PLC0415

    region = alpha > 20
    if not region.any():
        return rgb

    luma = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    # Blur hard enough that texture is gone and only form remains.
    form = cv2.GaussianBlur(luma, (0, 0), sigmaX=max(9, photo_rgb.shape[1] * 0.035))

    mean = form[region].mean()
    if mean <= 1:
        return rgb

    shading = np.clip(form / mean, 0.72, 1.28)
    shading = 1.0 + (shading - 1.0) * strength

    out = rgb.astype(np.float32) * shading[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_on_photo(photo: bytes, garment: Image.Image, landmarks: dict,
                     frame: dict | None = None, garment_landmarks: dict | None = None,
                     blend: str = "seamless", person_mask: np.ndarray | None = None) -> Image.Image:
    """Warp the composed garment onto a photographed torso.

    Uses the shoulder and hip landmarks to build a destination quad and applies
    a perspective transform. This gives correct scale and placement on the real
    body — it is a fit preview, not a photorealistic render: no fabric drape,
    no lighting match, no occlusion by the arms.
    """
    import cv2  # noqa: PLC0415

    base = Image.open(io.BytesIO(photo)).convert("RGBA")

    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(landmarks.get(k) for k in required):
        raise InvalidInput("Pose landmarks are missing the torso points")

    # Landmarks are measured on the downscaled analysis frame; the photo here
    # is full size. Without this the garment lands in the wrong place on any
    # image the analyser resized.
    sx = sy = 1.0
    if frame and frame.get("width") and frame.get("height"):
        sx = base.width / float(frame["width"])
        sy = base.height / float(frame["height"])

    def point(key):
        p = landmarks[key]
        return {"x": p["x"] * sx, "y": p["y"] * sy}

    ls, rs, lh, rh = (point(k) for k in
                      ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))

    # MediaPipe's "left" is the subject's left, i.e. on the viewer's right.
    shoulder_span = math.dist((ls["x"], ls["y"]), (rs["x"], rs["y"])) or 1.0
    torso_drop = abs(((lh["y"] + rh["y"]) / 2) - ((ls["y"] + rs["y"]) / 2)) or shoulder_span

    # Widen past the shoulders for sleeves, and run the hem well below the hips.
    pad = shoulder_span * 0.32
    hem = torso_drop * 1.7

    def unit_normal():
        dx, dy = ls["x"] - rs["x"], ls["y"] - rs["y"]
        length = math.hypot(dx, dy) or 1.0
        return -dy / length, dx / length

    nx, ny = unit_normal()
    ux, uy = (ls["x"] - rs["x"]) / shoulder_span, (ls["y"] - rs["y"]) / shoulder_span

    top_left = (rs["x"] - ux * pad - nx * pad * 0.5, rs["y"] - uy * pad - ny * pad * 0.5)
    top_right = (ls["x"] + ux * pad - nx * pad * 0.5, ls["y"] + uy * pad - ny * pad * 0.5)
    bottom_right = (top_right[0] + nx * hem, top_right[1] + ny * hem)
    bottom_left = (top_left[0] + nx * hem, top_left[1] + ny * hem)

    garment_array = np.array(garment)

    # ── Placement ──
    # With the garment's own shoulder points we can align shoulder-to-shoulder,
    # which sets scale and tilt from the body itself. Falling back to fitting
    # the bounding box into a quad shears the dress to whatever the box was.
    g_ls = (garment_landmarks or {}).get("left_shoulder")
    g_rs = (garment_landmarks or {}).get("right_shoulder")

    if g_ls and g_rs:
        affine = _similarity_transform(
            (g_rs["x"], g_rs["y"]), (g_ls["x"], g_ls["y"]),
            (rs["x"], rs["y"]), (ls["x"], ls["y"]),
        )
        warped = cv2.warpAffine(
            garment_array, affine, (base.width, base.height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )

        # The rigid fit lands the shoulders; the spline then bends everything
        # below them onto the wearer's own hips and arms.
        src_pts, dst_pts = [], []
        for key in TPS_POINTS:
            g = (garment_landmarks or {}).get(key)
            if not g or not landmarks.get(key):
                continue
            moved = affine @ np.array([g["x"], g["y"], 1.0], np.float32)
            target = point(key)
            src_pts.append((float(moved[0]), float(moved[1])))
            dst_pts.append((float(target["x"]), float(target["y"])))

        if len(src_pts) >= 4:
            bent = _tps_fit(warped, src_pts, dst_pts)
            if bent is not None:
                warped = bent
    else:
        content = garment.getbbox() or (0, 0, garment.width, garment.height)
        cx0, cy0, cx1, cy1 = content
        src = np.float32([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]])
        dst = np.float32([top_left, top_right, bottom_right, bottom_left])
        warped = cv2.warpPerspective(
            garment_array, cv2.getPerspectiveTransform(src, dst),
            (base.width, base.height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )

    rgb = warped[..., :3].copy()
    alpha = warped[..., 3].copy()

    # ── Occlusion ── the wearer's forearms and hands belong in front.
    forearms = _forearm_mask(warped.shape, {k: point(k) for k in landmarks if landmarks.get(k)},
                             shoulder_span * 0.16)
    alpha = np.where(forearms > 0, (alpha * (1.0 - forearms / 255.0)).astype(np.uint8), alpha)

    # ── Bound it to the body ── a garment floating over the background is the
    # clearest sign it was pasted on. Dilated, so a flared skirt can still
    # stand away from the figure without escaping into the scene.
    if person_mask is not None and person_mask.any():
        body = person_mask
        if body.shape != alpha.shape:
            body = cv2.resize(body, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
        grow = max(3, int(shoulder_span * 0.18)) | 1
        body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        body = cv2.GaussianBlur(body, (grow, grow), 0)
        alpha = (alpha.astype(np.float32) * (body.astype(np.float32) / 255.0)).astype(np.uint8)

    # ── Lighting ──
    photo_rgb = np.array(base.convert("RGB"))
    torso = np.zeros(alpha.shape, np.uint8)
    cv2.fillConvexPoly(torso, np.int32([top_left, top_right, bottom_right, bottom_left]), 255)
    rgb = _match_exposure(rgb, alpha, photo_rgb, torso)
    # Give the flat garment the wearer's own form shading.
    rgb = _transfer_shading(rgb, alpha, photo_rgb)

    # ── Edge feather ── a hard cut is the loudest sticker tell.
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    if blend == "seamless" and (alpha > 200).sum() > 500:
        try:
            solid = ((alpha > 200) * 255).astype(np.uint8)
            xs, ys = np.nonzero(solid.any(axis=0))[0], np.nonzero(solid.any(axis=1))[0]
            centre = (int((xs[0] + xs[-1]) / 2), int((ys[0] + ys[-1]) / 2))
            cloned = cv2.seamlessClone(rgb, photo_rgb, solid, centre, cv2.NORMAL_CLONE)
            # Poisson blending only inside the solid core; the feathered rim is
            # alpha-composited so the boundary stays soft.
            rim = (alpha.astype(np.float32) / 255.0)[..., None]
            out = photo_rgb * (1 - rim) + cloned * rim
            return Image.fromarray(out.astype(np.uint8), "RGB").convert("RGBA")
        except cv2.error as exc:
            logger.warning("seamless blend failed, using alpha composite: %s", exc)

    out = np.dstack([rgb, alpha])
    return Image.alpha_composite(base, Image.fromarray(out, "RGBA"))
