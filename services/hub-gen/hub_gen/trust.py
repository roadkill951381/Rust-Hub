"""Loading pipeline code that ships inside an installed package.

diffusers 0.39 and newer refuse any ``custom_pipeline`` with its own Python
code unless ``trust_remote_code=True`` is passed, even a folder on disk.
Hunyuan3D-Paint loads its multiview pipeline that way from hy3dgen's own
``hunyuanpaint`` folder (the pinned, installed package, nothing downloaded),
and hy3dgen doesn't pass the flag. ``trusted_pipelines_in(root)`` passes it
for custom pipelines inside ``root`` only; anything else is left to diffusers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _inside(path: object, root: Path) -> bool:
    if not isinstance(path, str | Path):
        return False
    try:
        return Path(path).resolve().is_relative_to(root.resolve()) and Path(path).is_dir()
    except OSError:
        return False


@contextmanager
def trusted_pipelines_in(root: Path, pipeline_class: type | None = None) -> Iterator[None]:
    """While active, ``pipeline_class.from_pretrained`` (diffusers'
    ``DiffusionPipeline`` by default) trusts ``custom_pipeline`` folders
    under ``root``."""
    if pipeline_class is None:
        from diffusers import DiffusionPipeline

        pipeline_class = DiffusionPipeline
    original = pipeline_class.__dict__["from_pretrained"]
    load = original.__func__

    def from_pretrained(cls, *args, **kwargs):
        if _inside(kwargs.get("custom_pipeline"), root):
            kwargs.setdefault("trust_remote_code", True)
        return load(cls, *args, **kwargs)

    pipeline_class.from_pretrained = classmethod(from_pretrained)
    try:
        yield
    finally:
        pipeline_class.from_pretrained = original
