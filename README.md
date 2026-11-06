# Rust-Hub

3D model tools for games and 3D printing, written in Rust. Every tool works two
ways: a person drives the desktop app, or Claude drives the same operations
through an MCP server.

This is phase 1 of the [plan](https://claude.ai/artifact/Wg7qHb74eickMHqoNbEn4m):
the foundation that later phases (local AI generation, rigging, texturing,
scenes) build on.

## What works today

- **Import and export** glTF/GLB, STL and 3MF. glTF keeps names, transforms
  and PBR materials for three.js and other engines. STL and 3MF are written in
  millimetres with Z up, ready for a slicer.
- **Print checks**: watertight (no open or non-manifold edges), faces pointing
  outward, zero-area faces, separate shells, size, volume and surface area.
- **Repair**: removes zero-area and duplicate triangles, unifies face winding
  and turns inside-out shells outward. Holes are reported, not yet filled.
- **Edits** with undo and redo: scale to a size in mm, place on the bed,
  move, rotate, set transform, apply transform, materials, primitives.
- **Desktop app** with a wgpu viewport (orbit, pan, zoom, click to select),
  an object list, an inspector and a print panel. Back faces render red, so
  holes and flipped faces are visible at a glance.
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
knight.3mf."

## Architecture

```
     Person (hub-app, egui)        Claude (hub mcp)        Scripts (hub run)
                  \                       |                       /
                   +------ hub_core::Command (serde + JSON Schema) ------+
                                          |
                               Session::execute  (undo / redo)
                                          |
          +----------------+--------------+---------------+-------------------+
          |                |                              |                   |
        Scene        io (glTF, STL, 3MF)        analysis (print checks,   hub-render
  (Arc-shared meshes)                                   repair)         (wgpu, headless PNG)
```

| Crate | Role |
|---|---|
| `hub-core` | Scene model, commands, undo history, file formats, print checks. No UI or GPU code. |
| `hub-render` | wgpu renderer used by the app and by headless previews. |
| `hub-mcp` | MCP server (official `rmcp` SDK) mapping tools to commands. |
| `hub-cli` | The `hub` binary: CLI subcommands and `hub mcp`. |
| `hub-app` | The desktop editor (eframe/egui on wgpu). |

Design rules:

- **One code path.** The GUI, CLI and MCP server all build a `Command` and call
  `Session::execute`. A feature exists for Claude the moment it exists for the GUI.
- **Atomic edits.** Each edit runs on a copy of the scene and commits only on
  success. Copies are cheap because meshes are shared behind `Arc`.
- **Units at the edges.** The scene is metres, Y-up (glTF). Commands take
  millimetres and degrees. Print formats are millimetres, Z-up. Conversion
  happens only in `hub_core::io`.

## Development

```sh
cargo fmt --all
cargo clippy --workspace --all-targets   # pedantic lints, kept at zero warnings
cargo test --workspace
```

The headless render test needs a GPU adapter. On Linux without one, install a
software Vulkan driver (`mesa-vulkan-drivers`); the test skips cleanly if no
adapter exists at all.

## Roadmap

1. **Foundation** (this): hub, viewport, formats, print checks, MCP.
2. **Generate and print**: local AI generation service on your own GPU
   (Hunyuan3D 2, TripoSG), Blender job bridge, wall-thickness checks, hole filling.
3. **Character**: UniRig auto-rigging, PBR texturing, animation and retargeting.
4. **Scenes**: the level editor, exporting scenes for three.js.
