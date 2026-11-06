"""Hunyuan3D 2 shape and texture models (Tencent), through the ``hy3dgen``
package from https://github.com/Tencent-Hunyuan/Hunyuan3D-2. Weights download
from Hugging Face on first use into the usual cache."""

from __future__ import annotations

import trimesh
from PIL import Image

from hub_gen.config import Settings
from hub_gen.progress import Report


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
        self.pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            repo,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            device=self.device,
        )
        # FlashVDM decodes the surface an order of magnitude faster.
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
        self, image: Image.Image, *, seed: int, steps: int | None, report: Report
    ) -> trimesh.Trimesh:
        torch = self._torch
        if not self._on_device:
            self.pipe.to(self.device)
            self._on_device = True
        n = steps or self.steps

        def callback(step: int, _t: object, _outputs: object) -> None:
            report((step + 1) / n * 0.7, "shaping")

        report(0.0, "shaping")
        with torch.inference_mode():
            mesh = self.pipe(
                image=image.convert("RGBA"),
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
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        self.pipe = Hunyuan3DPaintPipeline.from_pretrained(repo, subfolder=subfolder)
        if settings.low_vram:
            self.pipe.enable_model_cpu_offload()

    def paint(
        self, mesh: trimesh.Trimesh, image: Image.Image, *, report: Report
    ) -> trimesh.Trimesh:
        report(0.05, "painting the texture")
        textured = self.pipe(mesh, image=image.convert("RGBA"))
        report(1.0, "painted")
        return textured
