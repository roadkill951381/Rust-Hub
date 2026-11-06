# Rust-Hub: notes for Claude

Rust workspace for 3D tools (games and 3D printing) that a person or Claude can drive.
Read `README.md` for the architecture and roadmap.

## Rules that keep the design intact

- Every scene change is a `hub_core::Command` run through `Session::execute`
  (or `execute_coalescing` for GUI drags). Never mutate a `Scene` directly from
  the app, CLI or MCP server. A new feature means: add the `Command` variant,
  implement it in `session.rs`, then expose it in `hub-mcp` and the app.
- The scene is metres, Y-up. Commands take millimetres and degrees. STL/3MF are
  millimetres, Z-up. Only `hub_core::io` converts (`to_print_space`, `from_print_space`).
- Geometry checks weld vertices by position (`mesh::weld`) because glTF splits
  vertices along seams. 3MF export writes welded vertices; slicers need shared indices.
- Meshes live behind `Arc`. Replace the `Arc` when geometry changes; the
  renderer re-uploads only when the pointer changes.
- `hub-core` has no UI, GPU, process or network dependencies. Keep it that way.
  External tools sit behind the traits in `engine.rs` (`MeshTools` is Blender
  in `hub-blender`, `Generator` is the hub-gen client in `hub-generate`);
  front ends get them from `hub_engines::from_env()`.
- Slow commands (`Command::is_job`) go through `Session::begin`, then
  `Job::run` off the UI or async thread, then `Session::finish`. Never hold the
  session lock while a job runs. `execute` still works for them, inline.
- A missing engine is an `Error::Engine` that tells the person how to set it
  up; tests use fake engines, and live-engine tests skip unless `HUB_BLENDER`
  or `HUB_GEN_TEST_URL` is set.

## The generation service

`services/hub-gen` is Python (FastAPI, PyTorch), managed with uv:

```sh
cd services/hub-gen
uv sync && uv run ruff check && uv run ruff format --check && uv run pytest
uv run hub-gen serve --demo      # procedural models, no GPU; the Rust tests can use it
```

GPU adapters live in `hub_gen/backends/`; check them against the upstream
sources (Hunyuan3D-2 at the pinned commit, TripoSG) rather than memory.

## Before pushing

```sh
cargo fmt --all
cargo clippy --workspace --all-targets   # must stay at zero warnings (pedantic)
cargo test --workspace
```

Check crate APIs against the versions in `Cargo.lock` (wgpu 30, egui/eframe 0.37,
rmcp 3.9, glam 0.34); several changed shape recently, so read the source in
`~/.cargo/registry` rather than relying on memory.
