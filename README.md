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
  parts into one clean shell), decimate, and UV unwrap.
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
| `hub-blender` | Runs mesh jobs in Blender in the background (`MeshTools`). |
| `hub-generate` | HTTP client for the hub-gen service (`Generator`). |
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
3. **Character**: UniRig auto-rigging, PBR texturing, animation and retargeting.
4. **Scenes**: the level editor, exporting scenes for three.js.
