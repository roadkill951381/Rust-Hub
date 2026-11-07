# Rust-Hub

3D model tools for games and 3D printing, written in Rust. Every tool works two
ways: a person drives the desktop app, or Claude drives the same operations
through an MCP server.

Phases 1 and 2 of the [plan](https://claude.ai/artifact/Wg7qHb74eickMHqoNbEn4m)
are in: the foundation, and generating models on your own GPU and preparing
them for print. Rigging, texturing and scenes come next.

## What works today

- **Import and export** glTF/GLB, STL and 3MF. glTF keeps names, transforms
  and PBR materials for three.js and other engines. STL and 3MF are written in
  millimetres with Z up, ready for a slicer.
- **Generate** a model from a prompt or a picture on your own GPU with
  [hub-gen](services/hub-gen/README.md), a local service running Hunyuan3D 2
  or TripoSG. No per-model fees, nothing leaves your machine.
- **Game characters from a portrait**: draw front, side and back views that
  keep a portrait's face (FLUX with PuLID in ComfyUI), build the shape from
  all three (Hunyuan3D-2mv), paint it, and rig it like a character you
  already have, keeping its skeleton, rest pose and animations exactly (see
  [Make a game character](#make-a-game-character)).
- **Asset library**: every generated or saved model is kept with all its
  versions in plain folders (`~/RustHub/library/<name>/v0001.glb` and an
  `asset.json` that records how each version was made).
- **Print checks**: watertight (no open or non-manifold edges), faces pointing
  outward, zero-area faces, separate shells, internal cavities, wall thickness
  (0.8 mm by default), size, volume and surface area.
- **Repair**: removes zero-area and duplicate triangles, unifies face winding,
  turns inside-out shells outward, fills holes, and optionally deletes the
  floating specks AI models leave behind.
- **Hollow** for resin printing with walls of a set thickness. (Add drain
  holes in your slicer for now.)
- **Blender jobs**, run in the background: voxel remesh (fuses overlapping
  parts into one clean shell), decimate, UV unwrap, and rigging a model like
  an existing character.
- **Edits** with undo and redo: scale to a size in mm, place on the bed,
  move, rotate, set transform, apply transform, materials, primitives.
- **Desktop app** with a wgpu viewport (orbit, pan, zoom, click to select),
  a Generate panel, an object list, the library, an inspector, a print panel
  and mesh tools. Long jobs run in the background with a progress bar. Back
  faces render red, so holes and flipped faces are visible at a glance.
- **Claude interface**: an MCP server with a tool per command, a batch tool,
  and `render_preview`, which returns a PNG so Claude can see its own work.
- **CLI** for quick jobs and scripts.

## Quick start

You need a current stable Rust toolchain (the repo pins `stable` in
`rust-toolchain.toml`).

```sh
# Desktop app (optionally pass files to open)
cargo run --release -p hub-app -- path/to/model.glb

# Command line
cargo run --release -p hub-cli -- info model.glb
cargo run --release -p hub-cli -- check model.stl            # exit code 2 if not printable
cargo run --release -p hub-cli -- convert model.glb model.3mf --size-mm 80 --place-on-floor
cargo run --release -p hub-cli -- convert raw.glb print.3mf --repair --remove-small-shells 0.01 --hollow 2
cargo run --release -p hub-cli -- generate "a stone owl statue" --size-mm 80 -o owl.3mf
cargo run --release -p hub-cli -- assets                     # list the library
cargo run --release -p hub-cli -- render model.glb preview.png
cargo run --release -p hub-cli -- run script.json            # JSON array of commands
cargo run --release -p hub-cli -- schema                     # JSON Schema of a command
```

Install the CLI once with `cargo install --path crates/hub-cli`, which puts
`hub` on your `PATH`.

App controls: left-drag orbits, right-drag (or Shift + left-drag) pans, the
wheel zooms, click selects, F frames the selection, Ctrl+Z / Ctrl+Shift+Z undo
and redo, Delete removes, Ctrl+O imports, Ctrl+E exports. Files can be dropped
onto the window.

### Set up generation and Blender

Generation needs the hub-gen service running on a machine with an NVIDIA GPU
(16 GB for shape plus texture). Setup is in
[services/hub-gen/README.md](services/hub-gen/README.md); in short:

```powershell
cd services\hub-gen
uv sync --extra hunyuan
uv run hub-gen serve               # or: uv run hub-gen serve --demo  (no GPU, test shapes)
```

Remesh, decimate and UV unwrap need [Blender](https://www.blender.org/download/)
4.2 or newer. The hub finds it in the usual install folders or on the `PATH`.

| Variable | Default | Meaning |
|---|---|---|
| `HUB_GEN_URL` | `http://127.0.0.1:8765` | The generation service |
| `HUB_GEN_TOKEN` | none | Its token, if it was started with one |
| `HUB_LIBRARY` | `~/RustHub/library` | The asset library folder |
| `HUB_BLENDER` | found automatically | Blender's executable (or a Python with the `bpy` module) |

Nothing is contacted until a command needs it, and a command that needs a
missing engine says how to set it up.

### Use your ComfyUI

The hub can also generate with your own [ComfyUI](https://github.com/comfyanonymous/ComfyUI),
starting it if it isn't running (it looks in `C:\ComfyUI` and `~/ComfyUI`, or
set `HUB_COMFY_DIR`). A ComfyUI the hub started is stopped again after the job,
so the GPU and memory are free; one you started yourself is only asked to
unload its models. The hub's ComfyUI writes to `comfyui.log` in the
`rust-hub/comfyui` temp folder.

| Ask for | What happens |
|---|---|
| `--image-model comfyui` (app: Picture by "FLUX in ComfyUI") | FLUX in ComfyUI draws the picture, hub-gen's Hunyuan3D makes the shape |
| `--model comfyui` | Both steps in ComfyUI: FLUX, then ComfyUI's built-in Hunyuan3D 2.1 nodes |
| `--model comfyui:my-workflow` | Your own workflow does the whole job |

Built-in workflows: `flux-schnell` and `flux-dev` (FLUX GGUF through
ComfyUI-GGUF, with `t5xxl_fp8_e4m3fn`, `clip_l` and `ae`), `flux-dev-sheet`
(a front, side and back turnaround), `flux-pulid`, `flux-pulid-sheet` and
`flux-pulid-face` (keep a face; the last redraws a face close-up; need the ComfyUI_PuLID_Flux_ll nodes with `insightface`,
`facexlib`, `onnxruntime-gpu` and `facenet-pytorch --no-deps` in ComfyUI's
Python, plus `flux1-dev-Q5_K_S.gguf` and `pulid_flux_v0.9.1.safetensors` in
`models/pulid`), and `hunyuan3d-2.1`
(needs `hunyuan_3d_v2.1.safetensors` in `models/checkpoints`; ComfyUI's "Image
to Model (Hunyuan3D 2.1)" template offers the download). To use your own, open
it in ComfyUI, choose Workflow > Export (API), and save it as
`~/RustHub/comfyui/<name>.json` with `{{prompt}}`, `{{image}}` (a LoadImage
node's file) and `{{seed}}` where the hub should fill in values. A file named
like a built-in one replaces it. A workflow that saves a GLB makes models; one
that only saves images makes pictures.

| Variable | Default | Meaning |
|---|---|---|
| `HUB_COMFY_URL` | `http://127.0.0.1:8188` | Where ComfyUI listens |
| `HUB_COMFY_DIR` | found automatically | ComfyUI folder, so the hub can start it |
| `HUB_COMFY_WORKFLOWS` | `~/RustHub/comfyui` | Your exported workflows |
| `HUB_COMFY_PICTURE_WORKFLOW` / `HUB_COMFY_MODEL_WORKFLOW` | `flux-schnell` / `hunyuan3d-2.1` | Defaults for `comfyui` |
| `HUB_COMFY_KEEP_LOADED` | off | Keep ComfyUI running with its models loaded after a job |
| `HUB_IMAGE_MODEL` / `HUB_SHAPE_MODEL` | none | Defaults for every request, e.g. `HUB_IMAGE_MODEL=comfyui` |

### Make a game character

Given a portrait and a rigged character from your game, the hub makes a new
character that looks like the portrait and plays the game's animations:

```sh
hub generate "Mei, a young woman survivor, red jacket, cargo trousers, boots" \
    --face portrait.png --multiview --rig-like surv_mei_hi.glb \
    --materials surv_mei --faces 20000 -o mei.glb
```

1. Blender renders the reference character in its rest pose as a plain grey
   figure from the front, side and back. FLUX with PuLID in ComfyUI paints
   over that guide, so the new character stands exactly like the rig, and
   keeps the portrait's face. Without `--multiview` it draws one front view.
2. hub-gen cuts the sheet into views, builds the shape from all of them with
   Hunyuan3D-2mv and paints it with Hunyuan3D-Paint.
3. Blender fits the model to the reference character (height, facing, arm
   angles), copies the skin weights, and writes a GLB that keeps the
   reference's skeleton, inverse bind matrices and animations byte for byte.
   Skin is split from clothes by colour into `<materials>.body` and
   `<materials>.clothes`. The model is smooth-shaded, keeping edges sharper
   than 75° hard.
4. The shape model paints the whole body from a few small views, so the face
   comes out soft. Blender renders the rigged head close up and unlit, FLUX
   with PuLID redraws that render sharply with the portrait's face
   (repainting 0.65 of it, so the eyes and mouth stay put), and Blender
   paints the result back onto the texture through the same camera, fading
   it out toward the sides of the head and the neck. The result is
   `<name>-rigged-face.glb`; if this step fails, the rigged model is kept and
   the report says why.

To remake a character you already have, add `--keep-look`: the guide is the
reference with its own colours and clothes, and the picture repaints less of
it (0.7 instead of 0.88; set `--repaint` to choose), so the arms and outfit
stay where they are while the portrait sets the face.

`hub rig model.glb --like character.glb -o rigged.glb` runs step 3 alone on a
model you already have, and `hub sharpen-face rigged.glb --face portrait.png
--prompt "..." -o faced.glb` runs step 4 alone (`--repaint` from 0 to 1: more
is sharper and more like the portrait, less keeps the model's features where
they are). In the app these options are under Generate > Character; Claude
has them on the `generate`, `rig`, `sharpen_face` and `match_colours` tools.

Generated textures carry the colours of evenly lit pictures, which under a
game's lights often look washed out next to its own characters.
`hub match-colours rigged.glb --like character.glb -o matched.glb` darkens or
lightens the skin and the clothes, each as a whole and keeping their hues, to
the brightness of the same parts of the game's character (`--strength` from
0 to 1 goes part of the way; the clothes move at most from 0.15 to 2 times
their brightness, the skin from 0.6 to 1.4). `--max-texture 1024` shrinks the
texture to the game's size and `--jpeg` writes it as a JPEG, several times
smaller. Run it last, after `sharpen-face`.

### Build a character kit for an in-game customizer

A kit lets a game's players make their own character: pick a body, move
sliders for the body and face, and swap hair, outfits and shoes.
`hub kit humans.json -o kit/` (File > Build character kit in the app,
`build_character_kit` over MCP) builds one from a kit recipe in desktop
Blender with MPFB and MakeHuman's CC0 assets. For each base body it writes:

- `<body>/body.glb`: the body with its eyes, brows and teeth on the 52-bone
  Mixamo skeleton, every slider baked in as two morph targets (`weight-`,
  `weight+`), and a `_HIDE` vertex attribute that says which parts cover
  which skin, so a game can hide it and nothing pokes through;
- `<body>/<slot>/<part>.glb`: each hair style, outfit, pair of shoes or hat,
  rigged to the same skeleton and carrying the same morphs, so a slider moves
  the clothes with the body (hair textures are grey, for the game to tint);
- `<body>/skins/<skin>.jpg`: the other skin tones;
- `kit.json`: the sliders (with each body's starting value and range), the
  bone moves each morph needs so joints follow the shape, the parts and the
  colours, for cgen's `character-kit.js` to read.

Every slider is baked the way MPFB's own panel would make the change: the
macro (weight, muscle) or the shape targets are set to the slider's end, the
clothes and skeleton are refitted, and where every vertex and bone went is
recorded. `--body female` rebuilds one body and keeps the rest of the kit.
The recipe format is at the top of `crates/hub-blender/src/character_kit.py`;
cgen's `recipes/kits/humans.json` is the kit Brian's games use.

### Keep everything up to date

Help > Check for updates in the app (or `hub updates`, or the MCP tool
`check_updates`) checks every piece and shows the installed and newest
version of each, with an Update button where the hub can do it for you. Each
update runs in a window of its own so you can watch it.

| Piece | Compared with | Update |
|---|---|---|
| Rust-Hub | its git remote, and the commit this build came from | `git pull`, then a rebuild; the app closes and reopens itself |
| hub-gen's Python packages | PyPI, within the ranges in `pyproject.toml` | `uv sync --upgrade-package <name>` (stop hub-gen first) |
| Model weights (Hunyuan3D, HunyuanDiT, ...) | the repo's commits on Hugging Face | `hub-gen pull <repo>`, which also deletes the old copy |
| ComfyUI | its latest GitHub release | the portable build's own updater, or `git pull` (close ComfyUI first) |
| ComfyUI custom nodes | their git remotes | `git pull`; nodes not installed from git are listed but not checked |
| ComfyUI model files the built-in workflows load | present or missing | download into the right `models` folder |
| Blender | its release tags on projects.blender.org | `winget upgrade`, or blender.org |
| uv, Rust, NVIDIA driver | uv's releases, `rustup check` | `uv self update`, `rustup update`; the driver through the NVIDIA app |

"Held back" means a newer version exists but Rust-Hub pins the one it was
tested with, such as transformers below 5 and the Hunyuan3D-2 code commit.
Set `HUB_GITHUB_TOKEN` if GitHub's limit of 60 anonymous checks an hour runs
out, and `HF_TOKEN` for gated Hugging Face models.

### Connect Claude

Claude Code:

```sh
claude mcp add rust-hub -- hub mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rust-hub": { "command": "hub", "args": ["mcp"] }
  }
}
```

Then ask, for example: "Import C:\models\knight.glb, scale it to 80 mm tall,
check it is printable, repair it if needed, show me a preview, and export
knight.3mf." Or: "Generate a stone owl statue 80 mm tall, make it printable,
hollow it with 2 mm walls and export owl.3mf." Claude sees progress while the
GPU works.

## Architecture

```
     Person (hub-app, egui)        Claude (hub mcp)        Scripts (hub run)
                  \                       |                       /
                   +------ hub_core::Command (serde + JSON Schema) ------+
                                          |
              Session::execute, or begin → Job::run (any thread) → finish
                                          |  (undo / redo)
     +-------------+-------------+--------+---------+--------------------+
     |             |             |                  |                    |
   Scene    io (glTF, STL,   analysis, hollow   Engines (traits)     hub-render
            3MF)             (BVH, checks,      |      |       |     (wgpu, PNG)
                              repair)        Blender  hub-gen  Library
                                            (hub-blender) (hub-generate → HTTP → services/hub-gen)
```

| Crate | Role |
|---|---|
| `hub-core` | Scene model, commands, undo history, file formats, print checks, hollowing, the asset library, and the engine traits. No UI, GPU, process or network code. |
| `hub-blender` | Runs mesh jobs and rigging in Blender in the background (`MeshTools`, `Rigger`). |
| `hub-generate` | HTTP client for the hub-gen service (`Generator`). |
| `hub-comfy` | Drives the person's ComfyUI and its workflows (`Generator`, and pictures for hub-gen). |
| `hub-updates` | Checks every piece for newer versions and runs their upgrades. |
| `hub-engines` | Picks the engines from the environment, so every front end behaves the same. |
| `services/hub-gen` | The generation service itself, in Python (PyTorch). |
| `hub-render` | wgpu renderer used by the app and by headless previews. |
| `hub-mcp` | MCP server (official `rmcp` SDK) mapping tools to commands. |
| `hub-cli` | The `hub` binary: CLI subcommands and `hub mcp`. |
| `hub-app` | The desktop editor (eframe/egui on wgpu). |

Design rules:

- **One code path.** The GUI, CLI and MCP server all build a `Command` and call
  `Session::execute`. A feature exists for Claude the moment it exists for the GUI.
- **Atomic edits.** Each edit runs on a copy of the scene and commits only on
  success. Copies are cheap because meshes are shared behind `Arc`.
- **Long jobs never block.** Generation, hollowing and Blender jobs split into
  `begin` (quick, under the session), `Job::run` (slow, on any thread, with
  progress) and `finish` (one undo step). If the object was edited meanwhile,
  `finish` refuses rather than overwrite the newer edit.
- **Engines behind traits.** Blender and the generator are optional; tests use
  fakes, and each front end gets the same set from `hub-engines`.
- **Units at the edges.** The scene is metres, Y-up (glTF). Commands take
  millimetres and degrees. Print formats are millimetres, Z-up. Conversion
  happens only in `hub_core::io`.

## Known limits

- Textures from generation are kept in the library's copy of the model, but
  the scene does not load textures yet (phase 3), so exports from the scene
  are untextured.
- Hollowing does not punch drain holes yet; add them in your slicer.
- Face jobs fill a 12 GB card, so a ComfyUI the hub starts for one runs with
  `--cpu-vae`; one you started yourself keeps its own flags. With ComfyUI
  0.42 or newer, ComfyUI_PuLID_Flux_ll (commit 7c7362b) needs
  `timestep_zero_index=None,` added to `pulid_forward_orig` in
  `PulidFluxHook.py` until upstream fixes it.
- The GPU model adapters follow the upstream projects' own code but have only
  been run here through the demo models; first runs on a real GPU may need
  small fixes.

## Development

```sh
cargo fmt --all
cargo clippy --workspace --all-targets   # pedantic lints, kept at zero warnings
cargo test --workspace
```

Tests that need an engine skip unless it is there: set `HUB_BLENDER` to run
the Blender jobs, and `HUB_GEN_TEST_URL` (plus `HUB_GEN_TEST_TOKEN`) to run the
generation tests against a live service, e.g. `hub-gen serve --demo`. The
phase 2 gate (`crates/hub-mcp/tests/generate_gate.rs`) turns a prompt into a
hollowed, printable 3MF using MCP tool calls only.

The headless render test needs a GPU adapter. On Linux without one, install a
software Vulkan driver (`mesa-vulkan-drivers`); the test skips cleanly if no
adapter exists at all.

## Roadmap

1. **Foundation** (done): hub, viewport, formats, print checks, MCP.
2. **Generate and print** (this): local AI generation on your own GPU
   (Hunyuan3D 2, TripoSG), asset library, Blender job bridge, wall thickness,
   hole filling, hollowing.
3. **Character**: characters from a portrait rigged like an existing one
   (started), UniRig auto-rigging, PBR texturing, animation and retargeting.
4. **Scenes**: the level editor, exporting scenes for three.js.
