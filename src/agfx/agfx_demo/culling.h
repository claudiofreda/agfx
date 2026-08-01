/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-21 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

#include <agfx/agfx.h>
#include <glm/glm.hpp>

// GPU-driven frustum culling: one thread per scene primitive, appends surviving primitives'
// indexed-draw commands into an agfxIndirectBundle (AGFX_INDIRECT_BUNDLE_TYPE_DRAW_INDEXED).
// The caller owns barrier state: gpuSceneView must be readable and the bundle's commands/count
// buffers must be in AGFX_RESOURCE_STATE_UNORDERED_ACCESS on entry.
class AgfxCulling {
public:
    AgfxCulling(agfxDevice* device);
    ~AgfxCulling();

    AgfxCulling(const AgfxCulling&) = delete;
    AgfxCulling& operator=(const AgfxCulling&) = delete;

    // Encodes onto an already-open agfxComputePass, so the caller can share one pass/encoder with
    // the count-buffer reset and the indirect-bundle barriers/prepare step around it.
    // drawIndirectionHandle is a writeable raw view of a primitiveCount*4-byte buffer recording
    // slot -> primitive index; only the Vulkan shader path consumes it (its DrawIndex builtin is
    // linear, unlike D3D12/Metal's patched drawID), but it must be valid on every backend.
    void Cull(agfxComputePass* pass, agfxBufferView* gpuSceneView, uint32_t primitiveCount,
              const glm::vec4 frustumPlanes[6], uint64_t bundleHandle, uint32_t drawIndirectionHandle);

private:
    agfxDevice* m_device = nullptr;
    agfxShaderModule* computeShader = nullptr;
    agfxComputePipeline* pipeline = nullptr;
};
