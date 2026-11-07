"""Settings, read from the environment (``HUB_GEN_*``) and the command line."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_output() -> Path:
    if os.name == "nt":
        # Not under AppData: Windows quietly moves AppData writes from the
        # Microsoft Store's Python into its own package folder, so the jobs
        # would not be where the log says.
        return Path.home() / "RustHub" / "hub-gen" / "jobs"
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "hub-gen" / "jobs"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    #: Required for requests when set; required to listen beyond this machine.
    token: str | None = None
    #: Where job folders (reference image, mesh) are written.
    output_dir: Path = field(default_factory=_default_output)
    #: Shape model used when a request names none.
    default_model: str = "hunyuan3d-2"
    #: Shape model for multiview requests that name none.
    multiview_model: str = "hunyuan3d-2mv"
    #: Text-to-image model for prompts.
    image_model: str = "hunyuan-dit"
    #: Texture model for ``texture: true``.
    texture_model: str = "hunyuan3d-paint"
    device: str = "cuda"
    #: Keep only the model in use on the GPU, parking the others in system
    #: memory. Needed on 16 GB cards when texturing; costs a few seconds.
    low_vram: bool = True
    #: Seconds without a job after which every model is unloaded, so the GPU
    #: is free for ComfyUI, games and so on; 0 keeps them loaded.
    idle_unload: float = 300.0
    #: Finished jobs kept on disk; older ones are deleted.
    keep_jobs: int = 50
    #: Checkout of https://github.com/VAST-AI-Research/TripoSG, for ``triposg``.
    triposg_dir: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        s = cls()
        s.host = env("HUB_GEN_HOST", s.host)
        s.port = int(env("HUB_GEN_PORT", str(s.port)))
        s.token = env("HUB_GEN_TOKEN") or None
        if out := env("HUB_GEN_OUTPUT"):
            s.output_dir = Path(out)
        s.default_model = env("HUB_GEN_MODEL", s.default_model)
        s.multiview_model = env("HUB_GEN_MULTIVIEW_MODEL", s.multiview_model)
        s.image_model = env("HUB_GEN_IMAGE_MODEL", s.image_model)
        s.texture_model = env("HUB_GEN_TEXTURE_MODEL", s.texture_model)
        s.device = env("HUB_GEN_DEVICE", s.device)
        s.low_vram = env("HUB_GEN_LOW_VRAM", "1").lower() not in ("0", "false", "no")
        s.keep_jobs = int(env("HUB_GEN_KEEP_JOBS", str(s.keep_jobs)))
        s.idle_unload = float(env("HUB_GEN_IDLE_UNLOAD", str(s.idle_unload)))
        if d := env("HUB_GEN_TRIPOSG_DIR"):
            s.triposg_dir = Path(d)
        return s
