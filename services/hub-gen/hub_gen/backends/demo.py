"""Procedural stand-ins for the AI models, so the whole pipeline (HTTP API,
job queue, hub client, library) can be tested on any machine. Not AI.

``DemoImage`` draws a seeded vase-like silhouette for a prompt; ``DemoShape``
spins any object image's silhouette around the vertical axis (a lathe), so
the shape really follows the picture; ``DemoTexture`` projects the picture on.
"""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from hub_gen.config import Settings
from hub_gen.progress import Report


class DemoImage:
    def __init__(self, settings: Settings) -> None:
        del settings

    def generate(self, prompt: str, *, seed: int, report: Report) -> Image.Image:
        digest = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        size = 512
        image = Image.new("RGB", (size, size), (255, 255, 255))
        # A smooth profile: a few random bulges along the height.
        ys = np.linspace(0.15, 0.85, 64)
        bulges = rng.uniform(0.08, 0.2, 4)
        centres = np.linspace(0.2, 0.8, 4) + rng.uniform(-0.05, 0.05, 4)
        widths = 0.06 + sum(
            b * np.exp(-((ys - c) ** 2) / 0.01) for b, c in zip(bulges, centres, strict=True)
        )
        left = [(size * (0.5 - w), size * y) for w, y in zip(widths, ys, strict=True)]
        right = [(size * (0.5 + w), size * y) for w, y in zip(widths, ys, strict=True)]
        colour = tuple(int(c) for c in rng.integers(40, 200, 3))
        ImageDraw.Draw(image).polygon(left + right[::-1], fill=colour)
        report(1.0, "drew a demo silhouette")
        return image


class DemoShape:
    def __init__(self, settings: Settings) -> None:
        del settings

    def generate(
        self, image: Image.Image, *, seed: int, steps: int | None, report: Report
    ) -> trimesh.Trimesh:
        del seed, steps
        alpha = np.asarray(image.convert("RGBA").getchannel("A")) > 127
        rows = np.flatnonzero(alpha.any(axis=1))
        if rows.size < 2:
            raise ValueError("the image has no object to build a shape from")
        report(0.3, "tracing the silhouette")
        samples = np.linspace(rows[0], rows[-1], 48).astype(int)
        radii = []
        for r in samples:
            cols = np.flatnonzero(alpha[r])
            radii.append((cols[-1] - cols[0] + 1) / 2 if cols.size else 0.0)
        heights = (rows[-1] - samples).astype(float)  # pixels above the bottom row
        # Fit the object into [-1, 1], keeping its proportions.
        scale = 2.0 / max(heights[0], 2 * max(radii), 1.0)
        profile = [(0.0, heights[-1])]
        profile += [(max(r, 0.5), h) for r, h in zip(radii, heights, strict=True)][::-1]
        profile.append((0.0, heights[0]))
        line = np.array(profile) * scale
        line[:, 1] -= (line[:, 1].max() + line[:, 1].min()) / 2
        mesh = trimesh.creation.revolve(line, sections=48)
        # revolve spins about Z; the hub's models are Y-up.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        report(1.0, "lathed")
        return mesh


class DemoTexture:
    def __init__(self, settings: Settings) -> None:
        del settings

    def paint(
        self, mesh: trimesh.Trimesh, image: Image.Image, *, report: Report
    ) -> trimesh.Trimesh:
        v = mesh.vertices
        lo, hi = v.min(axis=0), v.max(axis=0)
        u = (np.arctan2(v[:, 2], v[:, 0]) / (2 * np.pi)) % 1.0
        w = (v[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-9)
        textured = mesh.copy()
        textured.visual = trimesh.visual.TextureVisuals(
            uv=np.column_stack([u, w]), image=image.convert("RGB")
        )
        report(1.0, "projected the picture")
        return textured
