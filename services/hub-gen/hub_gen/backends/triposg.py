"""TripoSG (VAST, MIT licence). Its code is not packaged, so point
``HUB_GEN_TRIPOSG_DIR`` at a checkout of https://github.com/VAST-AI-Research/TripoSG.
Background removal is done by hub-gen (BiRefNet), not TripoSG's RMBG-1.4,
whose licence is non-commercial."""

from __future__ import annotations

import sys

import numpy as np
import trimesh
from PIL import Image

from hub_gen.config import Settings
from hub_gen.imaging import on_white
from hub_gen.progress import Report


class TripoSGShape:
    def __init__(self, settings: Settings, *, repo: str, steps: int) -> None:
        import torch
        from huggingface_hub import snapshot_download

        if settings.triposg_dir:
            sys.path.insert(0, str(settings.triposg_dir))
        try:
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        except ImportError as e:
            raise RuntimeError(
                "TripoSG's code was not found: clone https://github.com/VAST-AI-Research/TripoSG "
                "and set HUB_GEN_TRIPOSG_DIR to it"
            ) from e
        self._torch = torch
        self.device = settings.device
        self.steps = steps
        weights = snapshot_download(repo_id=repo)
        self.pipe = TripoSGPipeline.from_pretrained(weights).to(self.device, torch.float16)
        self._on_device = True

    def offload(self) -> None:
        if self._on_device:
            self.pipe.to("cpu")
            self._torch.cuda.empty_cache()
            self._on_device = False

    def generate(
        self, image: Image.Image, *, seed: int, steps: int | None, report: Report
    ) -> trimesh.Trimesh:
        torch = self._torch
        if not self._on_device:
            self.pipe.to(self.device, torch.float16)
            self._on_device = True
        n = steps or self.steps

        def on_step(_pipe: object, i: int, _t: object, kwargs: dict) -> dict:
            report((i + 1) / n * 0.8, "shaping")
            return kwargs

        with torch.inference_mode():
            vertices, faces = self.pipe(
                image=on_white(image),
                generator=torch.Generator(device=self.pipe.device).manual_seed(seed),
                num_inference_steps=n,
                guidance_scale=7.0,
                callback_on_step_end=on_step,
            ).samples[0]
        report(1.0, "surface extracted")
        return trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
