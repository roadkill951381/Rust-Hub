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

from hub_gen.backends.base import MODELS, Kind, Models
from hub_gen.colour import match_colours
from hub_gen.config import Settings
from hub_gen.imaging import SHEET_VIEWS, frame_object, remove_background, split_sheet
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

    def shape_model(self, request: GenerateRequest) -> str:
        if request.model:
            return request.model
        return self.settings.multiview_model if request.multiview else self.settings.default_model

    def validate(self, request: GenerateRequest) -> None:
        """Fails fast, before queueing, on names and inputs that cannot work."""
        try:
            spec = self.models.spec(self.shape_model(request), "shape")
            if request.multiview and not spec.multiview:
                multi = ", ".join(s.name for s in MODELS.values() if s.multiview)
                raise BadRequest(
                    f"{spec.name} builds from one picture; for several views use {multi}"
                )
            if not request.image_b64:
                self.models.spec(request.image_model or self.settings.image_model, "image")
            if request.texture:
                self.models.spec(self.settings.texture_model, "texture")
        except ValueError as e:
            raise BadRequest(str(e)) from e
        if request.image_b64:
            decode_image(request.image_b64)

    @staticmethod
    def _views(cut: Image.Image) -> dict[str, Image.Image]:
        """The views of a cut-out turnaround sheet, framed; just the first
        figure as the front when the sheet does not show three."""
        figures = split_sheet(cut)
        if not figures:
            raise ValueError("no object found in the picture after removing the background")
        if len(figures) < len(SHEET_VIEWS):
            log.warning(
                "the turnaround sheet shows %d figure(s), not %d; using the first as the front",
                len(figures),
                len(SHEET_VIEWS),
            )
            return {"front": frame_object(figures[0])}
        if len(figures) > len(SHEET_VIEWS):
            log.info("the sheet shows %d figures; using the first three", len(figures))
        return {
            name: frame_object(figure) for name, figure in zip(SHEET_VIEWS, figures, strict=False)
        }

    def _load(self, name: str, kind: Kind, report: Report, at: float) -> Any:
        if name not in self.models.loaded():
            from hub_gen.versions import needs_download

            extra = " (downloading it first; this happens once)" if needs_download(name) else ""
            report(at, f"loading {name}{extra}")
        return self.models.get(name, kind)

    def run(self, request: GenerateRequest, out: Path, report: Report) -> JobResult:
        start = time.monotonic()
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        shape_name = self.shape_model(request)
        image_name = None
        files: list[str] = []

        if request.image_b64:
            picture = decode_image(request.image_b64)
        else:
            image_name = request.image_model or self.settings.image_model
            painter = self._load(image_name, "image", report, 0.0)
            picture = painter.generate(
                request.prompt or "",
                seed=seed,
                report=span(report, 0, 0.3),
                sheet=request.multiview,
            )
        picture.save(out / "reference.png")
        files.append("reference.png")

        report(0.3, "cutting out the object")
        cut = remove_background(picture)
        views = self._views(cut) if request.multiview else {"front": frame_object(cut)}
        framed = views["front"]
        framed.save(out / "object.png")
        files.append("object.png")
        if len(views) > 1:
            for name, view in views.items():
                view.save(out / f"view-{name}.png")
                files.append(f"view-{name}.png")

        shape_end = 0.75 if request.texture else 0.9
        shaper = self._load(shape_name, "shape", report, 0.31)
        multiview = self.models.spec(shape_name, "shape").multiview
        mesh = shaper.generate(
            views if multiview else framed,
            seed=seed,
            steps=request.steps,
            report=span(report, 0.32, shape_end),
        )
        report(shape_end, "cleaning up the mesh")
        mesh = clean(mesh, request.faces)

        if request.texture:
            painter = self._load(self.settings.texture_model, "texture", report, shape_end + 0.01)
            reference = list(views.values()) if len(views) > 1 else framed
            mesh = painter.paint(mesh, reference, report=span(report, shape_end + 0.02, 0.98))
            report(0.98, "matching the texture's colours to the picture")
            mesh = match_colours(mesh, list(views.values()))

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
            views=list(views) if multiview else ["front"],
            triangles=len(mesh.faces),
            vertices=len(mesh.vertices),
            files={name: "" for name in files},
        )
