"""Model interfaces, the registry of known models, and the loader that keeps
GPU memory in check."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import trimesh
from PIL import Image

from hub_gen.config import Settings
from hub_gen.progress import Report

log = logging.getLogger(__name__)

Kind = Literal["shape", "image", "texture"]


class ShapeModel(Protocol):
    def generate(
        self, image: Image.Image, *, seed: int, steps: int | None, report: Report
    ) -> trimesh.Trimesh:
        """Builds an untextured mesh (Y-up, centred) from an RGBA object image."""
        ...


class ImageModel(Protocol):
    def generate(self, prompt: str, *, seed: int, report: Report) -> Image.Image:
        """Paints a reference picture of the prompt's object."""
        ...


class TextureModel(Protocol):
    def paint(
        self, mesh: trimesh.Trimesh, image: Image.Image, *, report: Report
    ) -> trimesh.Trimesh:
        """Returns the mesh with UVs and a colour texture matching the image."""
        ...


class Offloadable(Protocol):
    def offload(self) -> None:
        """Moves weights off the GPU (to system memory) until next used."""
        ...


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: Kind
    #: ``module:Class``, constructed as ``Class(settings, **options)``.
    target: str
    #: Python modules that must be importable.
    requires: tuple[str, ...]
    #: Approximate GPU memory while running, in GB.
    vram_gb: float
    licence: str
    note: str
    options: dict[str, Any] = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec(
            "hunyuan3d-2",
            "shape",
            "hub_gen.backends.hunyuan:HunyuanShape",
            ("torch", "hy3dgen"),
            6.0,
            "Tencent Hunyuan 3D 2.0 Community (not EU, UK or South Korea)",
            "Hunyuan3D-DiT v2.0 Turbo with FlashVDM: detailed shapes in seconds.",
            {
                "repo": "tencent/Hunyuan3D-2",
                "subfolder": "hunyuan3d-dit-v2-0-turbo",
                "steps": 5,
                "use_safetensors": True,
            },
        ),
        ModelSpec(
            "hunyuan3d-2mini",
            "shape",
            "hub_gen.backends.hunyuan:HunyuanShape",
            ("torch", "hy3dgen"),
            5.0,
            "Tencent Hunyuan 3D 2.0 Community (not EU, UK or South Korea)",
            "Hunyuan3D-DiT v2 mini Turbo: smaller and faster, a little less detail.",
            {
                "repo": "tencent/Hunyuan3D-2mini",
                "subfolder": "hunyuan3d-dit-v2-mini-turbo",
                "steps": 5,
                "use_safetensors": False,
                "flashvdm_topk": "merge",
            },
        ),
        ModelSpec(
            "triposg",
            "shape",
            "hub_gen.backends.triposg:TripoSGShape",
            ("torch", "diso"),
            8.0,
            "MIT",
            "TripoSG 1.5B: sharp, clean geometry; needs a TripoSG checkout.",
            {"repo": "VAST-AI/TripoSG", "steps": 50},
        ),
        ModelSpec(
            "hunyuan-dit",
            "image",
            "hub_gen.backends.text2image:HunyuanDiT",
            ("torch", "diffusers"),
            7.0,
            "Tencent Hunyuan Community",
            "HunyuanDiT v1.1 distilled, as Hunyuan3D's own text-to-3D uses.",
            {"repo": "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"},
        ),
        ModelSpec(
            "flux-schnell",
            "image",
            "hub_gen.backends.text2image:FluxSchnell",
            ("torch", "diffusers", "bitsandbytes"),
            12.0,
            "Apache 2.0",
            "FLUX.1-schnell in 4-bit: follows long prompts best; slower to load.",
            {"repo": "black-forest-labs/FLUX.1-schnell"},
        ),
        ModelSpec(
            "hunyuan3d-paint",
            "texture",
            "hub_gen.backends.hunyuan:HunyuanPaint",
            ("torch", "hy3dgen", "custom_rasterizer"),
            11.0,
            "Tencent Hunyuan 3D 2.0 Community (not EU, UK or South Korea)",
            "Hunyuan3D-Paint v2.0 Turbo: colour texture from the reference image.",
            {"repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-paint-v2-0-turbo"},
        ),
        ModelSpec(
            "demo",
            "shape",
            "hub_gen.backends.demo:DemoShape",
            (),
            0.0,
            "MIT",
            "Not AI: a procedural stand-in for testing the pipeline without a GPU.",
        ),
        ModelSpec(
            "demo-image",
            "image",
            "hub_gen.backends.demo:DemoImage",
            (),
            0.0,
            "MIT",
            "Not AI: draws a silhouette, for testing without a GPU.",
        ),
        ModelSpec(
            "demo-texture",
            "texture",
            "hub_gen.backends.demo:DemoTexture",
            (),
            0.0,
            "MIT",
            "Not AI: projects the reference image onto the mesh.",
        ),
    ]
}


def missing_modules(spec: ModelSpec) -> list[str]:
    return [m for m in spec.requires if importlib.util.find_spec(m) is None]


class UnknownModel(ValueError):
    pass


class Models:
    """Creates models on first use and keeps them, parking idle ones in system
    memory when ``low_vram`` is set so one 16 GB card can run every stage."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded: dict[str, Any] = {}
        self._lock = threading.Lock()

    def spec(self, name: str, kind: Kind) -> ModelSpec:
        spec = MODELS.get(name)
        if spec is None or spec.kind != kind:
            names = ", ".join(s.name for s in MODELS.values() if s.kind == kind)
            raise UnknownModel(f"unknown {kind} model {name!r}; choose one of: {names}")
        if missing := missing_modules(spec):
            raise UnknownModel(
                f"{name} needs {', '.join(missing)}, which are not installed; "
                f"see the hub-gen README for its install command"
            )
        return spec

    def get(self, name: str, kind: Kind) -> Any:
        spec = self.spec(name, kind)
        with self._lock:
            if self.settings.low_vram:
                for other, model in self._loaded.items():
                    if other != name and hasattr(model, "offload"):
                        model.offload()
            model = self._loaded.get(name)
            if model is None:
                log.info("loading %s", name)
                module, _, cls = spec.target.partition(":")
                model = getattr(importlib.import_module(module), cls)(self.settings, **spec.options)
                self._loaded[name] = model
            return model

    def unload_all(self) -> list[str]:
        """Drops every model, giving back GPU and system memory. The next job
        loads what it needs again from the disk cache."""
        with self._lock:
            names = list(self._loaded)
            self._loaded.clear()
        if names:
            import gc
            import sys

            gc.collect()
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return names

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "kind": s.kind,
                "available": not missing_modules(s),
                "loaded": s.name in self._loaded,
                "vram_gb": s.vram_gb,
                "licence": s.licence,
                "note": s.note,
            }
            for s in MODELS.values()
        ]
