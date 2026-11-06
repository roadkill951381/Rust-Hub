"""Colour correction for painted textures.

Hunyuan3D-Paint takes the lighting out of its views before baking, and the
texture it bakes comes out paler and greyer than the picture it was made
from. ``match_colours`` moves the texture's colours to the picture's: the
mean and spread of each channel in CIE Lab, measured over the model's
surface and over the picture's subject, as in Reinhard et al., "Color
Transfer between Images" (2001).
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

#: Limits on how far the spread of a channel may be stretched or squeezed,
#: so a picture with a very different subject can't wreck the texture.
SPREAD_LIMITS = (0.6, 1.8)


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_RGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in 0..1, shape (..., 3), to CIE Lab (D65)."""
    xyz = _to_linear(rgb) @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    return _to_srgb(xyz @ np.linalg.inv(_RGB_TO_XYZ).T)


def texture_of(mesh: trimesh.Trimesh) -> Image.Image | None:
    material = getattr(mesh.visual, "material", None)
    for name in ("baseColorTexture", "image"):
        image = getattr(material, name, None)
        if isinstance(image, Image.Image):
            return image
    return None


def _set_texture(mesh: trimesh.Trimesh, image: Image.Image) -> None:
    material = mesh.visual.material
    if isinstance(getattr(material, "baseColorTexture", None), Image.Image):
        material.baseColorTexture = image
    else:
        material.image = image


def surface_colours(mesh: trimesh.Trimesh, texture: Image.Image, count: int = 20000) -> np.ndarray:
    """Texture colours (0..1) at points spread evenly over the surface."""
    points, faces = trimesh.sample.sample_surface(mesh, count, seed=7)
    corners = mesh.triangles[faces]
    bary = trimesh.triangles.points_to_barycentric(corners, points)
    uv = np.einsum("ij,ijk->ik", bary, mesh.visual.uv[mesh.faces[faces]])
    rgb = np.asarray(texture.convert("RGB"), dtype=np.float64) / 255
    h, w = rgb.shape[:2]
    x = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).round().astype(int)
    y = np.clip((1 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).round().astype(int)
    return rgb[y, x]


def subject_colours(pictures: list[Image.Image]) -> np.ndarray:
    """Colours (0..1) of the subject in cut-out pictures (opaque pixels)."""
    found = []
    for picture in pictures:
        rgba = np.asarray(picture.convert("RGBA"), dtype=np.float64) / 255
        found.append(rgba[rgba[..., 3] > 0.5][:, :3])
    return np.concatenate(found) if found else np.empty((0, 3))


def match_colours(mesh: trimesh.Trimesh, pictures: list[Image.Image]) -> trimesh.Trimesh:
    """The mesh with its texture's colours moved to those of ``pictures``'
    subject. Returns the mesh unchanged when it has no texture or UVs."""
    texture = texture_of(mesh)
    uv = getattr(mesh.visual, "uv", None)
    target = subject_colours(pictures)
    if texture is None or uv is None or len(uv) != len(mesh.vertices) or len(target) < 100:
        return mesh
    source = rgb_to_lab(surface_colours(mesh, texture))
    target = rgb_to_lab(target)
    mean_s, mean_t = source.mean(0), target.mean(0)
    scale = np.clip(target.std(0) / np.maximum(source.std(0), 1e-6), *SPREAD_LIMITS)
    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float64) / 255
    lab = (rgb_to_lab(rgba[..., :3]) - mean_s) * scale + mean_t
    rgb = lab_to_rgb(lab)
    out = np.concatenate([rgb, rgba[..., 3:]], axis=-1)
    image = Image.fromarray((out * 255).round().astype(np.uint8), "RGBA")
    if texture.mode != "RGBA":
        image = image.convert(texture.mode)
    corrected = mesh.copy()
    _set_texture(corrected, image)
    return corrected
