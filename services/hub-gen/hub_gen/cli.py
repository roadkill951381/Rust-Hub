"""``hub-gen``: run the service, list models, or generate one model locally."""

from __future__ import annotations

import argparse
import base64
import ipaddress
import logging
import sys
from pathlib import Path

from hub_gen import __version__
from hub_gen.config import Settings


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if getattr(args, "demo", False):
        s.default_model, s.image_model, s.texture_model = "demo", "demo-image", "demo-texture"
    for name in ("host", "port", "token", "model", "image_model", "texture_model", "device"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(s, "default_model" if name == "model" else name, value)
    if getattr(args, "output", None):
        s.output_dir = Path(args.output)
    if getattr(args, "no_low_vram", False):
        s.low_vram = False
    return s


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from hub_gen.api import create_app

    s = _settings(args)
    if not _is_loopback(s.host) and not s.token and not args.insecure:
        print(
            "refusing to listen on a network address without --token "
            "(or HUB_GEN_TOKEN); pass --insecure to do it anyway",
            file=sys.stderr,
        )
        return 2
    logging.info(
        "hub-gen %s on http://%s:%d, models in %s", __version__, s.host, s.port, s.output_dir
    )
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")
    return 0


def models(args: argparse.Namespace) -> int:
    from hub_gen.backends.base import MODELS, missing_modules

    s = _settings(args)
    defaults = {s.default_model, s.image_model, s.texture_model}
    for spec in MODELS.values():
        missing = missing_modules(spec)
        state = "ready" if not missing else "needs " + ", ".join(missing)
        mark = "*" if spec.name in defaults else " "
        print(f"{mark} {spec.name:16} {spec.kind:8} {spec.vram_gb:4.0f} GB  {state:28} {spec.note}")
    print("\n* = default. Licences: run `hub-gen models --licences`." if not args.licences else "")
    if args.licences:
        for spec in MODELS.values():
            print(f"{spec.name:16} {spec.licence}")
    return 0


def versions(args: argparse.Namespace) -> int:
    import json

    from hub_gen import versions as v

    report = v.report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"hub-gen {report['version']} in {report['project_dir']}")
    print(f"extras: {', '.join(report['extras']) or 'none'}")
    for p in report["packages"]:
        pin = f"@{p['commit'][:10]}" if p["commit"] else ""
        print(f"  {p['name']:24} {p['installed'] + pin:24} {p['wanted'] or p['url'] or ''}")
    for w in report["weights"]:
        state = w["revision"][:10] if w["revision"] else "not downloaded"
        print(f"  {w['repo']:52} {state}")
    return 0


def pull(args: argparse.Namespace) -> int:
    from hub_gen import versions as v

    targets = args.targets
    if args.all:
        targets = [w["repo"] for w in v.weights() if w["downloaded"]]
    if not targets:
        print("name a model or repo, or pass --all", file=sys.stderr)
        return 2
    for target in targets:
        try:
            v.pull(target)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
    print("done; restart hub-gen if it is running so it loads the new weights")
    return 0


def generate(args: argparse.Namespace) -> int:
    from hub_gen.backends.base import Models
    from hub_gen.pipeline import Pipeline
    from hub_gen.schema import GenerateRequest

    s = _settings(args)
    image_b64 = None
    if args.image:
        image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()
    request = GenerateRequest(
        prompt=args.prompt,
        image_b64=image_b64,
        model=args.model,
        texture=args.texture,
        seed=args.seed,
        faces=args.faces,
    )
    pipeline = Pipeline(s, Models(s))
    pipeline.validate(request)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def report(fraction: float, stage: str) -> None:
        print(f"\r{fraction * 100:5.1f}%  {stage:40}", end="", file=sys.stderr, flush=True)

    result = pipeline.run(request, out, report)
    print(file=sys.stderr)
    print(result.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="hub-gen", description=__doc__)
    parser.add_argument("--version", action="version", version=f"hub-gen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", help="default shape model")
        p.add_argument("--image-model", dest="image_model", help="text-to-image model")
        p.add_argument("--texture-model", dest="texture_model", help="texture model")
        p.add_argument("--device", help="torch device, default cuda")
        p.add_argument("--demo", action="store_true", help="use the procedural demo models")
        p.add_argument("--no-low-vram", action="store_true", help="keep every model on the GPU")

    p = sub.add_parser("serve", help="run the HTTP service")
    common(p)
    p.add_argument("--host", help="address to listen on (default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8765)")
    p.add_argument("--token", help="require this bearer token")
    p.add_argument("--output", help="folder for job files")
    p.add_argument(
        "--insecure", action="store_true", help="allow a network address without a token"
    )
    p.set_defaults(run=serve)

    p = sub.add_parser("models", help="list models and whether they are installed")
    common(p)
    p.add_argument("--licences", action="store_true", help="show each model's licence")
    p.set_defaults(run=models)

    p = sub.add_parser("versions", help="show package and model weight versions")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(run=versions)

    p = sub.add_parser("pull", help="download the newest weights for models, removing the old ones")
    p.add_argument("targets", nargs="*", metavar="MODEL", help="model names or Hugging Face repos")
    p.add_argument("--all", action="store_true", help="every repo already downloaded")
    p.set_defaults(run=pull)

    p = sub.add_parser("generate", help="generate one model without the server")
    common(p)
    p.add_argument("prompt", nargs="?", help="what to make")
    p.add_argument("--image", help="picture of the object instead of a prompt")
    p.add_argument("--texture", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--faces", type=int)
    p.add_argument("--out-dir", default="hub-gen-out", help="folder for the results")
    p.set_defaults(run=generate)

    args = parser.parse_args(argv)
    sys.exit(args.run(args))
