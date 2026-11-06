"""Hugging Face downloads of only the files a model loads.

Model repos often hold each weight file twice, as ``.safetensors`` and as a
``.bin`` or ``.ckpt`` (Hunyuan3D's paint UNet: 3.5 GB and 7.3 GB), and
hy3dgen fetches whole subfolders. While ``exact_downloads()`` is active, any
``snapshot_download`` skips the old formats wherever a ``.safetensors`` sits
in the same folder, which every loader here prefers anyway.
"""

from __future__ import annotations

import fnmatch
import logging
import posixpath
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

log = logging.getLogger(__name__)

#: Weight formats that a ``.safetensors`` in the same folder makes redundant.
OLD_FORMATS = (".bin", ".ckpt", ".pt", ".pth")


def needed_files(files: Iterable[str], patterns: Iterable[str] | None) -> list[str]:
    """The files matching ``patterns`` (all when None), without duplicates in
    older formats."""
    chosen = [f for f in files if patterns is None or any(fnmatch.fnmatch(f, p) for p in patterns)]
    safetensors = {posixpath.dirname(f) for f in chosen if f.endswith(".safetensors")}
    return [
        f for f in chosen if not (f.endswith(OLD_FORMATS) and posixpath.dirname(f) in safetensors)
    ]


@contextmanager
def exact_downloads() -> Iterator[None]:
    """Makes ``huggingface_hub.snapshot_download`` fetch only needed files."""
    import huggingface_hub

    original = huggingface_hub.snapshot_download

    def snapshot_download(repo_id, *args, allow_patterns=None, **kwargs):
        if isinstance(allow_patterns, str):
            allow_patterns = [allow_patterns]
        try:
            files = huggingface_hub.list_repo_files(
                repo_id,
                repo_type=kwargs.get("repo_type"),
                revision=kwargs.get("revision"),
                token=kwargs.get("token"),
            )
        except Exception as e:  # offline, or the Hub is down: use the cache as is
            log.debug("could not list %s (%s); downloading as asked", repo_id, e)
            return original(repo_id, *args, allow_patterns=allow_patterns, **kwargs)
        keep = needed_files(files, allow_patterns)
        skipped = len(
            [
                f
                for f in files
                if allow_patterns is None or any(fnmatch.fnmatch(f, p) for p in allow_patterns)
            ]
        ) - len(keep)
        if skipped:
            log.info("%s: skipping %d duplicate weight file(s)", repo_id, skipped)
        return original(repo_id, *args, allow_patterns=keep or allow_patterns, **kwargs)

    huggingface_hub.snapshot_download = snapshot_download
    try:
        yield
    finally:
        huggingface_hub.snapshot_download = original
