"""One generation, stage by stage: prompt to picture, picture to cut-out,
cut-out to shape, clean-up, optional texture, GLB."""

from __future__ import annotations

import base64
import binascii
import io
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from hub_gen.backends.base import Kind, Models
from hub_gen.config import Settings
from hub_gen.imaging import frame_object, remove_background
from hub_gen.meshops import clean
from hub_gen.progress import Report, span
from hub_gen.schema import GenerateRequest, JobResult

log = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 25 * 1024 * 1024


class BadRequest(ValueError):
    """A request the service cannot run, reported to the client as HTTP 400."""


def decode_image(data: str) -> Image.Image:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BadRequest("image_b64 is not valid base64") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise BadRequest("the image is larger than 25 MB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise BadRequest(f"the image could not be read: {e}") from e
    return image


class Pipeline:
    def __init__(self, settings: Settings, models: Models) -> None:
        self.settings = settings
        self.models = models

    def validate(self, request: GenerateRequest) -> None:
        """Fails fast, before queueing, on names and inputs that cannot work."""
        try:
            self.models.spec(request.model or self.settings.default_model, "shape")
            if not request.image_b64:
                self.models.spec(request.image_model or self.settings.image_model, "image")
            if request.texture:
                self.models.spec(self.settings.texture_model, "texture")
        except ValueError as e:
            raise BadRequest(str(e)) from e
        if request.image_b64:
            decode_image(request.image_b64)

    def _load(self, name: str, kind: Kind, report: Report, at: float) -> Any:
        if name not in self.models.loaded():
            report(at, f"loading {name} (the first run downloads it)")
        return self.models.get(name, kind)

    def run(self, request: GenerateRequest, out: Path, report: Report) -> JobResult:
        start = time.monotonic()
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        shape_name = request.model or self.settings.default_model
        image_name = None
        files: list[str] = []

        if request.image_b64:
            picture = decode_image(request.image_b64)
        else:
            image_name = request.image_model or self.settings.image_model
            painter = self._load(image_name, "image", report, 0.0)
            picture = painter.generate(request.prompt or "", seed=seed, report=span(report, 0, 0.3))
        picture.save(out / "reference.png")
        files.append("reference.png")

        report(0.3, "cutting out the object")
        framed = frame_object(remove_background(picture))
        framed.save(out / "object.png")
        files.append("object.png")

        shape_end = 0.75 if request.texture else 0.9
        shaper = self._load(shape_name, "shape", report, 0.31)
        mesh = shaper.generate(
            framed, seed=seed, steps=request.steps, report=span(report, 0.32, shape_end)
        )
        report(shape_end, "cleaning up the mesh")
        mesh = clean(mesh, request.faces)

        if request.texture:
            painter = self._load(self.settings.texture_model, "texture", report, shape_end + 0.01)
            mesh = painter.paint(mesh, framed, report=span(report, shape_end + 0.02, 0.98))

        mesh.export(out / "mesh.glb")
        files.append("mesh.glb")
        report(1.0, "done")
        return JobResult(
            model=shape_name,
            image_model=image_name,
            prompt=request.prompt,
            seed=seed,
            seconds=round(time.monotonic() - start, 2),
            textured=request.texture,
            triangles=len(mesh.faces),
            vertices=len(mesh.vertices),
            files={name: "" for name in files},
        )
