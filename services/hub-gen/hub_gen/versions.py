"""What hub-gen runs on, for the hub's update check, and pulling newer model
weights.

``report()`` reads package metadata and the Hugging Face cache only, so it is
quick, needs no network, and imports no model code. The hub compares it with
PyPI, GitHub and Hugging Face.
"""

from __future__ import annotations

import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from hub_gen import __version__
from hub_gen.backends.base import MODELS

_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?")
_EXTRA = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


def project_dir() -> Path:
    """The ``services/hub-gen`` folder that uv manages (hub-gen is installed
    from it in editable mode)."""
    return Path(__file__).resolve().parents[1]


def hf_cache() -> Path:
    """Hugging Face's hub cache, resolved the way ``huggingface_hub`` does."""
    env = os.environ
    if cache := env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE"):
        return Path(cache).expanduser()
    if home := env.get("HF_HOME"):
        return Path(home).expanduser() / "hub"
    base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "huggingface" / "hub"


def _requirements() -> list[dict[str, Any]]:
    try:
        lines = metadata.requires("hub-gen") or []
    except metadata.PackageNotFoundError:
        lines = []
    found: dict[str, dict[str, Any]] = {}
    for line in lines:
        spec, _, marker = line.partition(";")
        m = _NAME.match(spec)
        if not m:
            continue
        rest = spec[m.end() :].strip()
        url = rest[1:].strip() if rest.startswith("@") else None
        extra = _EXTRA.search(marker)
        # Anything but the extra, e.g. a platform condition.
        other = re.sub(r"\b(and|or)\b|[()\s]", "", _EXTRA.sub("", marker))
        found.setdefault(
            m.group(1).lower(),
            {
                "name": m.group(1),
                "wanted": "" if url else rest.strip("() "),
                "url": url,
                "extra": extra.group(1) if extra else None,
                "conditional": bool(other),
            },
        )
    return list(found.values())


def _installed_commit(name: str) -> str | None:
    try:
        text = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        return json.loads(text).get("vcs_info", {}).get("commit_id")
    except ValueError:
        return None


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def extras(reqs: list[dict[str, Any]] | None = None) -> list[str]:
    """Extras that were synced: every one of their packages is installed
    (platform-specific ones aside). A package another dependency pulled in
    does not make its extra count."""
    reqs = _requirements() if reqs is None else reqs
    by_extra: dict[str, list[dict[str, Any]]] = {}
    for r in reqs:
        if r["extra"]:
            by_extra.setdefault(r["extra"], []).append(r)
    return sorted(
        e
        for e, rs in by_extra.items()
        if all(_installed(r["name"]) for r in rs if not r["conditional"])
    )


def packages(
    reqs: list[dict[str, Any]] | None = None, active: list[str] | None = None
) -> list[dict[str, Any]]:
    """hub-gen's direct dependencies that are installed, with the range
    pyproject.toml allows, leaving out extras that were not synced."""
    reqs = _requirements() if reqs is None else reqs
    active = extras(reqs) if active is None else active
    out = []
    for req in reqs:
        if req["extra"] and req["extra"] not in active:
            continue
        installed = _installed(req["name"])
        if installed is None:
            continue
        out.append({**req, "installed": installed, "commit": _installed_commit(req["name"])})
    return out


def _repo_dir(repo: str) -> Path:
    return hf_cache() / ("models--" + repo.replace("/", "--"))


def _hy3dgen_copy(repo: str, subfolder: str | None) -> Path | None:
    """hy3dgen loads ``$HY3DGEN_MODELS/<repo>/<subfolder>`` (default
    ``~/.cache/hy3dgen``) instead of the Hugging Face cache when it exists."""
    if not subfolder:
        return None
    base = Path(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")).expanduser()
    path = base / repo / subfolder
    return path if path.is_dir() else None


def weights() -> list[dict[str, Any]]:
    """One entry per Hugging Face repo the models load from, with the commit
    in the cache (``None`` when it was never downloaded)."""
    repos: dict[str, dict[str, Any]] = {}
    for spec in MODELS.values():
        repo = spec.options.get("repo")
        if not repo:
            continue
        sub = spec.options.get("subfolder")
        entry = repos.setdefault(
            repo, {"repo": repo, "models": [], "revision": None, "downloaded": [], "copy": None}
        )
        entry["models"].append(spec.name)
        ref = _repo_dir(repo) / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text().strip()
            entry["revision"] = revision
            snapshot = _repo_dir(repo) / "snapshots" / revision
            if (snapshot / sub).is_dir() if sub else snapshot.is_dir():
                entry["downloaded"].append(spec.name)
        if copy := _hy3dgen_copy(repo, sub):
            entry["copy"] = str(copy)
    return list(repos.values())


def needs_download(model: str) -> bool:
    """Whether loading the model will download its weights first."""
    for w in weights():
        if model in w["models"]:
            return model not in w["downloaded"] and not w["copy"]
    return False


def report() -> dict[str, Any]:
    reqs = _requirements()
    active = extras(reqs)
    return {
        "version": __version__,
        "project_dir": str(project_dir()),
        "python": sys.executable,
        "extras": active,
        "packages": packages(reqs, active),
        "weights": weights(),
    }


def _patterns(names: list[str]) -> list[str] | None:
    """Files to fetch: the subfolders of the named models, or the whole repo
    when a model loads all of it."""
    subs = [MODELS[n].options.get("subfolder") for n in names]
    if any(s is None for s in subs):
        return None
    return [*sorted({f"{s}/*" for s in subs}), "*.json", "*.md"]


def pull(target: str, *, say=print) -> str:
    """Downloads the newest weights for a model or repo, then deletes the
    revisions they replace. Returns the new commit."""
    by_model = MODELS.get(target)
    repo = by_model.options.get("repo") if by_model else target
    entry = next((w for w in weights() if w["repo"] == repo), None)
    if entry is None:
        raise ValueError(f"{target!r} is not a model or repo hub-gen uses")
    # Whatever was downloaded before, plus what was asked for.
    names = sorted(set(entry["downloaded"]) | ({by_model.name} if by_model else set()))
    if not names:
        names = entry["models"]
    from huggingface_hub import snapshot_download

    kinds = {MODELS[n].kind for n in names}
    say(f"downloading {repo} for {', '.join(names)}")
    if kinds == {"image"}:
        # Only the files the pipeline loads, not single-file checkpoints too.
        from diffusers import DiffusionPipeline

        path = Path(DiffusionPipeline.download(repo))
    else:
        path = Path(snapshot_download(repo, allow_patterns=_patterns(names)))
    revision = path.name
    old = entry["revision"]
    if old and old != revision:
        say(f"{repo}: {old[:10]} -> {revision[:10]}")
        _delete_old(repo, revision, say)
    else:
        say(f"{repo} is at {revision[:10]}, the newest")
    if entry["copy"]:
        say(f"note: hy3dgen loads {entry['copy']} before the cache; delete it to use these")
    return revision


def _delete_old(repo: str, keep: str, say) -> None:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir(hf_cache())
    stale = [
        rev.commit_hash
        for r in cache.repos
        if r.repo_id == repo and r.repo_type == "model"
        for rev in r.revisions
        if rev.commit_hash != keep and not rev.refs
    ]
    if stale:
        strategy = cache.delete_revisions(*stale)
        say(f"removing the old revision ({strategy.expected_freed_size_str})")
        # Files a running hub-gen still has open are skipped with a warning.
        strategy.execute()
