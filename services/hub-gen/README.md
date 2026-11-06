# hub-gen

Rust-Hub's own 3D generation service: a text prompt or a picture becomes a 3D
model on your GPU, the way Meshy or Tripo do it, with nothing billed per model.
The hub (desktop app, `hub generate`, and Claude through MCP) talks to it over
a small HTTP API on your machine.

```
prompt ──► text-to-image ──► background removal ──► shape model ──► clean-up ──► (texture) ──► GLB
          HunyuanDiT/FLUX        BiRefNet            Hunyuan3D 2      floaters,     Hunyuan3D-Paint
                                                    or TripoSG       face count
```

A picture skips the first step. Models load on first use and stay loaded; with
`low_vram` on (the default) only the one running sits on the GPU.

## Models

| Name | Kind | VRAM | Licence | Notes |
|---|---|---|---|---|
| `hunyuan3d-2` (default) | shape | ~6 GB | Tencent Hunyuan 3D 2.0 Community | Turbo + FlashVDM: detailed shapes in seconds |
| `hunyuan3d-2mini` | shape | ~5 GB | same | Smaller and faster, a little less detail |
| `triposg` | shape | ~8 GB | MIT | Sharp, clean geometry; needs a TripoSG checkout |
| `hunyuan-dit` (default) | image | ~7 GB | Tencent Hunyuan Community | What Hunyuan3D's own text-to-3D uses |
| `flux-schnell` | image | ~12 GB | Apache 2.0 | Best at long prompts; 4-bit, slower to load |
| `hunyuan3d-paint` (default) | texture | ~11 GB | Tencent Hunyuan 3D 2.0 Community | Colour texture from the reference picture |
| `demo`, `demo-image`, `demo-texture` | all | none | MIT | Not AI: procedural stand-ins to test the pipeline |

`hub-gen models` shows which are ready on your machine and what each still
needs. Weights download from Hugging Face on first use (several GB each) into
the usual Hugging Face cache.

**Licences matter if you sell things.** The Hunyuan models' licence does not
cover the EU, the UK or South Korea, and needs a separate licence from Tencent
above 1 million monthly users. TripoSG (MIT) with FLUX.1-schnell (Apache 2.0)
is the fully permissive combination. Background removal uses BiRefNet (MIT),
not RMBG, whose licence is non-commercial.

## Install (Windows, also Linux)

You need an NVIDIA GPU with 16 GB or more for everything at once (8 GB runs
shapes without texture), a current NVIDIA driver, and
[uv](https://docs.astral.sh/uv/).

```powershell
cd services\hub-gen
uv sync --extra hunyuan            # PyTorch with CUDA 12.8, Hunyuan3D 2, HunyuanDiT
uv run hub-gen models              # check what is ready
uv run hub-gen serve               # http://127.0.0.1:8765
```

Add `--extra flux` for FLUX.1-schnell and `--extra triposg` for TripoSG.

### Texture painting

`hunyuan3d-paint` needs two small CUDA extensions from the Hunyuan3D 2 source.
Build them once, with the CUDA Toolkit 12.8 and (on Windows) the Visual Studio
2022 Build Tools installed, from an "x64 Native Tools" prompt:

```powershell
cd services\hub-gen
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 $env:TEMP\Hunyuan3D-2
uv pip install setuptools wheel
uv pip install --no-build-isolation $env:TEMP\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer
uv pip install --no-build-isolation $env:TEMP\Hunyuan3D-2\hy3dgen\texgen\differentiable_renderer
```

Shapes work without these; only `texture: true` needs them. Use
`uv sync --inexact` from then on, because a plain `uv sync` removes packages
it did not install itself.

### TripoSG

```powershell
git clone https://github.com/VAST-AI-Research/TripoSG C:\src\TripoSG
uv sync --extra hunyuan --extra triposg
$env:HUB_GEN_TRIPOSG_DIR = "C:\src\TripoSG"
uv run hub-gen serve --model triposg
```

## Try it without a GPU

```sh
uv sync
uv run hub-gen serve --demo
```

The demo models make simple lathed shapes, so the whole chain (service, hub,
library, print prep, MCP) can be tested on any machine.

## Settings

Every setting has an environment variable and a flag (`hub-gen serve --help`).

| Variable | Default | Meaning |
|---|---|---|
| `HUB_GEN_HOST` / `HUB_GEN_PORT` | `127.0.0.1` / `8765` | Where to listen |
| `HUB_GEN_TOKEN` | none | Bearer token clients must send. Required to listen beyond this machine |
| `HUB_GEN_MODEL` | `hunyuan3d-2` | Shape model when a request names none |
| `HUB_GEN_IMAGE_MODEL` | `hunyuan-dit` | Text-to-image model for prompts |
| `HUB_GEN_TEXTURE_MODEL` | `hunyuan3d-paint` | Texture model |
| `HUB_GEN_DEVICE` | `cuda` | PyTorch device |
| `HUB_GEN_LOW_VRAM` | `1` | Park idle models in system memory |
| `HUB_GEN_OUTPUT` | `%LOCALAPPDATA%\hub-gen\jobs` | Job folders |
| `HUB_GEN_KEEP_JOBS` | `50` | Finished jobs kept on disk |
| `HUB_GEN_TRIPOSG_DIR` | none | TripoSG checkout |

To use the GPU box from another computer on your network, start it with
`--host 0.0.0.0 --token <secret>` and set `HUB_GEN_URL=http://<box>:8765` and
`HUB_GEN_TOKEN=<secret>` where the hub runs.

## API

| Request | Does |
|---|---|
| `GET /health` | Version, device, models and whether each is loaded |
| `POST /v1/jobs` | Start a job: `{prompt?, image_b64?, model?, texture?, seed?, faces?}` |
| `GET /v1/jobs/{id}` | Status, progress (0 to 1), stage, and the result's files and info |
| `DELETE /v1/jobs/{id}` | Cancel a queued or running job |
| `GET /v1/jobs/{id}/files/{name}` | Download `mesh.glb` or `reference.png` |

Jobs run one at a time on the GPU; others wait in the queue.

## Development

```sh
uv sync
uv run ruff check && uv run ruff format --check
uv run pytest
```

The tests use the demo models, so they need no GPU.
