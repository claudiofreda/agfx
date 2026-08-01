<div align="center">

# AGFX

**Amélie's graphics library** — a small cross-platform RHI over D3D12, Metal 4 and Vulkan.

![C](https://img.shields.io/badge/C-99-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Backends](https://img.shields.io/badge/backends-D3D12%20%7C%20Metal%204%20%7C%20Vulkan%201.4%20-green)
![License](https://img.shields.io/badge/license-MIT-green)

![](.github/image.png)
![](.github/ez.png)
![](.github/tests.png)

</div>

AGFX is a small (<10k LOC) wrapper over D3D12, Metal 4 and Vulkan, designed to make it easier for indie developers to ship games on multiple platforms. It's MIT licensed.

## Design

AGFX takes the coding style of WebGPU — a C99 API, so it's easy to write bindings for other languages. A C++17 RAII header (`agfx.hpp`) and an immediate-mode layer for D3D11/OpenGL refugees (`agfx_ez.hpp`) ship alongside it.

It is designed to be **bindless first**, which is pretty reasonable considering any GPU that isn't older than the library's author should support it (I was born in 2006!). Resources are reached through plain integer handles pushed as constants — no descriptor sets, no root signatures, no argument encoders in sight.

Everything modern is exposed: inline raytracing, mesh shaders, GPU-driven rendering (multi-draw indirect), memory aliasing, timestamp queries, and a "write once, run everywhere" HLSL shader model. Native API objects are also exposed for things like upscaler integration.

AGFX code looks like this:

```c
agfxDeviceCreateInfo deviceCreateInfo = {};
deviceCreateInfo.allocate = agfxAlloc;
deviceCreateInfo.free = agfxDealloc;
deviceCreateInfo.tempAllocate = agfxAlloc;
deviceCreateInfo.tempFree = agfxDealloc;
deviceCreateInfo.enableValidation = true;

agfxDevice* device = agfxDeviceCreate(&deviceCreateInfo);

//
agfxCommandBufferReset(commandBuffer);
agfxCommandBufferBegin(commandBuffer);

agfxRenderPassCreateInfo renderPassCreateInfo = {};
renderPassCreateInfo.colorAttachmentCount = 1;
renderPassCreateInfo.colorAttachments[0].renderTarget = backBufferRenderTarget;
renderPassCreateInfo.colorAttachments[0].loadOp = AGFX_LOAD_OPERATION_CLEAR;
renderPassCreateInfo.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
renderPassCreateInfo.width = (uint32_t)drawableWidth;
renderPassCreateInfo.height = (uint32_t)drawableHeight;

uint32_t uniformHandle = agfxBufferViewGetHandle(constantView);

agfxRenderPass* renderPass = agfxRenderPassBegin(commandBuffer, &renderPassCreateInfo);
agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)drawableWidth, (float)drawableHeight, 0.0f, 1.0f);
agfxRenderPassSetScissor(pass, 0, 0, width, height);
agfxRenderPassSetPipeline(pass, pipeline);
agfxRenderPassPushConstants(pass, &uniformHandle, sizeof(uniformHandle));
agfxRenderPassDrawIndexed(pass, indexBuffer, indexCount, 1, indexOffset, 0, 0);
agfxRenderPassEnd(renderPass);
```

## What's included

| Folder | What it is |
| --- | --- |
| `.claude` | A porting agent and Claude skills to facilitate porting your game to AGFX |
| `agfx` | The main library (`agfx.h`, `agfx.hpp`, `agfx_ez.hpp`, one backend per platform) |
| `agfx_demo` | A self-contained demo using SDL3 and AGFX: physically based rendering, cascaded shadow maps, SSAO, HDR, raytraced mirror reflections, GPU-driven frustum culling |
| `agfx_ez_demo` | Simple self-contained demo showcasing `agfx_ez.hpp` |
| `agfx_imgui` | ImGui backend for AGFX |
| `agfx_shader` | Shader compiler library: HLSL → AGFX bytecode (DXIL, Metal IR, or SPIR-V) |
| `agfx_shader_cli` | A CLI to compile AGFX shaders |
| `agfx_tests` | An exhaustive RHI test suite (400+ cases, run against every backend) |
| `data/shaders` | The `agfx.h` HLSL include, plus shaders for `agfx_imgui`, the tests and the demo |

## Requirements

One backend compiles per platform — D3D12 on Windows, Metal 4 on macOS, Vulkan on Linux.

| Platform | Backend | Minimum spec |
| --- | --- | --- |
| Windows | D3D12 | A GPU with Shader Model 6.6 dynamic resources (Resource Binding Tier 3) and Enhanced Barriers: NVIDIA Turing (GTX 16 / RTX 20)+, AMD Polaris (RX 400)+, Intel Arc+. Ship the [Agility SDK](https://devblogs.microsoft.com/directx/directx12agility/) to unlock these on older Windows builds. |
| macOS | Metal 4 | Apple Silicon (M1+) on macOS 26+ |
| Linux | Vulkan | A Vulkan 1.4 driver with `VK_EXT_mutable_descriptor_type`: NVIDIA Pascal (GTX 10)+, AMD Polaris (RX 400)+ on RADV, Intel Xe+ — driver-dependent, so check `vulkaninfo`. Both X11 and Wayland are supported. |

Raytracing, mesh shaders and multi-draw indirect are optional capabilities queried through `agfxDeviceGetInfo()` — in practice they need NVIDIA RTX 20+, AMD RX 6000+, Intel Arc, or Apple M3+.

## Getting started

```sh
# install xmake first: https://xmake.io
xmake
xmake run agfx_demo
```

Done!

### Libraries to link (if you're using your own build system)

| Platform | `agfx` | `agfx_shader` |
| --- | --- | --- |
| Windows | `d3d12`, `dxgi` | `dxcompiler.lib` |
| macOS | Metal, QuartzCore, CoreGraphics | `dxcompiler`, `metalirconverter` (Metal Shader Converter dylib) |
| Linux | Nothing — Vulkan is loaded at runtime via volk | Nothing at link time — `libdxcompiler.so` is `dlopen`'d from `data/dlls` |

## Bindings

Bindings for Zig, Rust and Odin can be found [here](https://github.com/AmelieHeinrich/agfx-bindings) (may not be up to date with main).

## Missing features

- PS5, Switch 2 backend
- Sparse resources (unplanned)
- Support for Slang + BDA (unplanned)

## AI Notice

AI tools were used for a few specific backend feature implementations (aliasing, enhanced barrier port, most of the Vulkan backend), and wrote most of the tests and demos.

## Changelogs

| Version | Changes |
| --- | --- |
| v2.0.0 | Vulkan backend |
| v1.4.0 | Resource aliasing |
| v1.3.3 | D3D12 debug markers |
| v1.3.2 | Barrier rework |
| v1.3.1 | Pipeline cache update |
| v1.3.0 | Tests update |
| v1.2.0 | Multi draw indirect update |
| v1.1.0 | Raytracing update |
| v1.0.0 | Base Metal4/D3D12 backends with basic features, missing raytracing/draw indirect. Fully usable for a video game. |

## Projects that use AGFX

- **Eclipse** (Amélie Heinrich): Adventure puzzle game where you set out to vanquish an evil force trying to plunge the world into darkness.
- **Voxel Game** (RyDawgE): Adventure voxel game
