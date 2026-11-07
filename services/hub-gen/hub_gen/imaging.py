"""Turning any picture into the clean object image that image-to-3D models expect:
background removed, object cropped, centred and padded on a square canvas."""

from __future__ import annotations

import logging
from functools import cache

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

#: rembg model for background removal; BiRefNet is MIT-licensed and sharp on edges.
REMBG_MODEL = "birefnet-general"


def has_useful_alpha(image: Image.Image) -> bool:
    """True when the picture already has a cut-out: some fully clear and some
    fully opaque pixels (at least 1% of each)."""
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel("A"))
    clear = np.count_nonzero(alpha < 13) / alpha.size
    solid = np.count_nonzero(alpha > 242) / alpha.size
    return clear > 0.01 and solid > 0.01


@cache
def _rembg_session():
    from rembg import new_session  # optional dependency

    return new_session(REMBG_MODEL)


def remove_background(image: Image.Image) -> Image.Image:
    """Cuts the object out, returning RGBA. Pictures with a cut-out pass through."""
    image = image.convert("RGBA") if image.mode in ("LA", "P", "RGBA") else image.convert("RGB")
    if has_useful_alpha(image):
        return image
    try:
        from rembg import remove
    except ImportError:
        log.warning("rembg is not installed; using a plain-background key instead")
        return _key_plain_background(image.convert("RGB"))
    return remove(image.convert("RGB"), session=_rembg_session()).convert("RGBA")


def _key_plain_background(image: Image.Image, tolerance: int = 18) -> Image.Image:
    """Fallback cut-out for pictures on a flat background: pixels close to the
    corner colour become transparent."""
    rgb = np.asarray(image).astype(np.int16)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    background = np.median(corners, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    alpha = np.where(distance > tolerance, 255, 0).astype(np.uint8)
    out = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def frame_object(image: Image.Image, size: int = 1024, padding: float = 0.1) -> Image.Image:
    """Crops an RGBA image to its object and centres it on a transparent square,
    with ``padding`` (a fraction of the object size) on each side."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    rows = np.flatnonzero(alpha.max(axis=1) > 8)
    cols = np.flatnonzero(alpha.max(axis=0) > 8)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no object found in the image: it is empty after removing the background")
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    crop = rgba.crop(box)
    side = round(max(crop.size) * (1 + 2 * padding))
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2), crop)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def on_white(image: Image.Image) -> Image.Image:
    """Composites an RGBA image over white, as TripoSG expects."""
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, image.convert("RGBA")).convert("RGB")


#: The order a turnaround sheet shows its views in, left to right. "left" is
#: Hunyuan3D-2mv's name for the view facing left (the figure's right side).
SHEET_VIEWS = ("front", "left", "back")


def split_sheet(image: Image.Image) -> list[Image.Image]:
    """Splits a cut-out turnaround sheet into its figures, left to right.

    Figures are runs of columns holding the object, separated by clear gaps;
    specks under a sixth of the largest figure's area are dropped.
    """
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A")) > 8
    filled = alpha.any(axis=0)
    width = alpha.shape[1]
    gap = max(2, width // 200)
    runs: list[list[int]] = []
    start, empty = None, 0
    for x, on in enumerate(filled):
        if on:
            if start is None:
                start = x
            empty = 0
        elif start is not None:
            empty += 1
            if empty >= gap:
                runs.append([start, x - empty + 1])
                start, empty = None, 0
    if start is not None:
        runs.append([start, width - empty])
    if not runs:
        return []
    mass = [int(alpha[:, a:b].sum()) for a, b in runs]
    keep = [r for r, m in zip(runs, mass, strict=True) if m >= max(mass) / 6]
    return [rgba.crop((a, 0, b, rgba.height)) for a, b in keep]
