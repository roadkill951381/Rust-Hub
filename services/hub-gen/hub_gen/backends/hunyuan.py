"""Hunyuan3D 2 shape and texture models (Tencent), through the ``hy3dgen``
package from https://github.com/Tencent-Hunyuan/Hunyuan3D-2. Weights download
from Hugging Face on first use into the usual cache."""

from __future__ import annotations

from pathlib import Path

import trimesh
from PIL import Image

from hub_gen.backends.base import Views
from hub_gen.config import Settings
from hub_gen.hfdownload import exact_downloads
from hub_gen.progress import Report
from hub_gen.trust import trusted_pipelines_in


class HunyuanShape:
    def __init__(
        self,
        settings: Settings,
        *,
        repo: str,
        subfolder: str,
        steps: int,
        use_safetensors: bool = True,
        flashvdm_topk: str | None = None,
    ) -> None:
        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self._torch = torch
        self.device = settings.device
        self.steps = steps
        with exact_downloads():
            self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                repo,
                subfolder=subfolder,
                use_safetensors=use_safetensors,
                device=self.device,
            )
            # FlashVDM decodes the surface an order of magnitude faster. It
            # swaps in the turbo VAE, found by the repo's name.
            if flashvdm_topk:
                self.pipe.enable_flashvdm(topk_mode=flashvdm_topk)
            else:
                self.pipe.enable_flashvdm()
        self._on_device = True

    def offload(self) -> None:
        if self._on_device:
            self.pipe.to("cpu")
            self._torch.cuda.empty_cache()
            self._on_device = False

    def generate(
        self, image: Image.Image | Views, *, seed: int, steps: int | None, report: Report
    ) -> trimesh.Trimesh:
        torch = self._torch
        if isinstance(image, dict):
            # Hunyuan3D-2mv takes {"front": ..., "left": ..., "back": ...}.
            image = {name: view.convert("RGBA") for name, view in image.items()}
        else:
            image = image.convert("RGBA")
        if not self._on_device:
            self.pipe.to(self.device)
            self._on_device = True
        n = steps or self.steps

        def callback(step: int, _t: object, _outputs: object) -> None:
            report((step + 1) / n * 0.7, "shaping")

        report(0.0, "shaping")
        with torch.inference_mode():
            mesh = self.pipe(
                image=image,
                num_inference_steps=n,
                octree_resolution=380,
                num_chunks=20000,
                generator=torch.manual_seed(seed),
                output_type="trimesh",
                callback=callback,
                callback_steps=1,
            )[0]
        report(1.0, "surface extracted")
        return mesh


class HunyuanPaint:
    def __init__(self, settings: Settings, *, repo: str, subfolder: str) -> None:
        import hy3dgen
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        # Paint's multiview pipeline is code in hy3dgen's own package folder.
        bundled = Path(hy3dgen.__file__).parent
        with exact_downloads(), trusted_pipelines_in(bundled):
            self.pipe = Hunyuan3DPaintPipeline.from_pretrained(repo, subfolder=subfolder)
        if settings.low_vram:
            self.pipe.enable_model_cpu_offload()

    def paint(
        self, mesh: trimesh.Trimesh, image: Image.Image | list[Image.Image], *, report: Report
    ) -> trimesh.Trimesh:
        report(0.05, "painting the texture")
        # Several views (front first) paint the back and sides from the views
        # themselves rather than guessing them.
        if isinstance(image, list):
            image = [view.convert("RGBA") for view in image]
        else:
            image = image.convert("RGBA")
        textured = self.pipe(mesh, image=image)
        report(1.0, "painted")
        return textured
