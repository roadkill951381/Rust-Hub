# Changes made for rust-hub

Upstream: Tencent-Hunyuan/Hunyuan3D-2 at f8db630, `hy3dgen/texgen/custom_rasterizer`.

hub-gen builds this extension with `uv sync --extra paint` against the PyTorch it
runs. On Windows, nvcc 12.x with MSVC 14.4x cannot compile a `.cu` file that
includes `<torch/extension.h>`: it stops in `torch/csrc/dynamo/compiled_autograd.h`
with `error C2872: 'std': ambiguous symbol`.

- `lib/custom_rasterizer_kernel/rasterizer.h`: includes `<torch/extension.h>`
  and declares the C++-only functions only outside nvcc (`#ifndef __CUDACC__`);
  `rasterize_image_gpu` is declared with `at::Tensor` (the same type as
  `torch::Tensor`).
- `lib/custom_rasterizer_kernel/rasterizer_gpu.cu`: uses ATen (`at::Tensor`,
  `at::zeros`, `at::kInt`, ...) instead of the `torch::` aliases.

Nothing else is changed; the kernels are as upstream wrote them.
