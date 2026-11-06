"""Text-to-image models that paint the reference picture a shape model builds from."""

from __future__ import annotations

from PIL import Image

from hub_gen.config import Settings
from hub_gen.progress import Report

#: Steers any prompt toward what image-to-3D models need.
SUFFIX = (
    ", a single complete object, full view, centered, plain white background, "
    "soft even studio lighting, 3D render"
)
NEGATIVE = (
    "text, watermark, cropped, out of frame, multiple objects, cluttered background, "
    "strong shadows, low quality, blurry, deformed"
)


class HunyuanDiT:
    def __init__(self, settings: Settings, *, repo: str) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self._torch = torch
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            enable_pag=True,
            pag_applied_layers=["blocks.(16|17|18|19)"],
        )
        if settings.low_vram:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(settings.device)

    def generate(self, prompt: str, *, seed: int, report: Report) -> Image.Image:
        steps = 25

        def on_step(_pipe: object, i: int, _t: object, kwargs: dict) -> dict:
            report((i + 1) / steps, "painting the reference image")
            return kwargs

        return self.pipe(
            prompt=prompt + SUFFIX,
            negative_prompt=NEGATIVE,
            num_inference_steps=steps,
            pag_scale=1.3,
            width=1024,
            height=1024,
            generator=self._torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=on_step,
        ).images[0]


class FluxSchnell:
    def __init__(self, settings: Settings, *, repo: str) -> None:
        import torch
        from diffusers import FluxPipeline, PipelineQuantizationConfig

        del settings  # always offloaded: the 4-bit model still needs the room
        self._torch = torch
        quantization = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        self.pipe = FluxPipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16, quantization_config=quantization
        )
        self.pipe.enable_model_cpu_offload()

    def generate(self, prompt: str, *, seed: int, report: Report) -> Image.Image:
        steps = 4

        def on_step(_pipe: object, i: int, _t: object, kwargs: dict) -> dict:
            report((i + 1) / steps, "painting the reference image")
            return kwargs

        return self.pipe(
            prompt=prompt + SUFFIX,
            guidance_scale=0.0,
            num_inference_steps=steps,
            max_sequence_length=256,
            width=1024,
            height=1024,
            generator=self._torch.Generator(device="cpu").manual_seed(seed),
            callback_on_step_end=on_step,
        ).images[0]
