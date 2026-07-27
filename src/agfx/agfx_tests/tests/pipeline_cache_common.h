/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#pragma once

// Shared scaffolding for the pipeline-cache tests: every one of the 7 shader-stage combinations
// (VS+PS, VS only, CS, MS+PS, MS only, TS+MS+PS, TS+MS only) follows the same recipe --
//
//   1. create pipeline A (no cache), draw/dispatch region 0 through it,
//   2. pull its cache blob via agfx{Render,Compute}PipelineGetCache, destroy A,
//   3. create pipeline B from that blob (identical desc otherwise), draw/dispatch region 1,
//   4. read back and confirm both regions are present -- proof B, built purely from A's cache,
//      behaves the same as A did.
//
// What's shared here is the pieces that recipe needs regardless of which stages are involved: the
// push-constant mirrors for pipeline_cache.hlsl/pipeline_cache_compute.hlsl, the cache-blob
// alloc/copy/free dance, and the depth-to-color reveal step the 3 "*-only" (no fragment shader)
// tests all need identically. Each test file still owns its own pipeline/pass construction, since
// the 7 combos differ enough in attachment and dispatch shape that forcing them through one shared
// "RunXxx" function would obscure more than it saves (see mesh_common.h/depth_common.h/blend_common.h
// for the same judgment call made elsewhere in this suite).

#include "agfx_tests/test_gpu.h"

#include <cstdint>

namespace agfxtest
{
    constexpr uint32_t kPipelineCacheWidth = 128;
    constexpr uint32_t kPipelineCacheHeight = 128;

    // Sample points used to assert the two regions pipeline_cache.hlsl's two triangles land in,
    // and a third point that lands in neither. Each has a comfortable double-digit-pixel margin
    // from every triangle edge (computed from the kPositions table in pipeline_cache.hlsl and the
    // standard NDC-to-pixel transform), so an 8x8 box centered on it never crosses an edge.
    constexpr uint32_t kPipelineCacheLeftSampleX = 28;
    constexpr uint32_t kPipelineCacheLeftSampleY = 92;
    constexpr uint32_t kPipelineCacheRightSampleX = 92;
    constexpr uint32_t kPipelineCacheRightSampleY = 92;
    constexpr uint32_t kPipelineCacheBackgroundSampleX = 60;
    constexpr uint32_t kPipelineCacheBackgroundSampleY = 2;
    constexpr uint32_t kPipelineCacheSampleBoxSize = 8;

    /// @brief Mirrors PipelineCachePushConstants in data/shaders/tests/pipeline_cache.hlsl. Field
    /// order and padding must match exactly -- this is memcpy'd into the push-constant block.
    struct PipelineCacheConstants
    {
        float tint[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        uint32_t half_ = 0;
        float depthValue = 0.0f;
        uint32_t padding0 = 0;
        uint32_t padding1 = 0;
    };
    static_assert(sizeof(PipelineCacheConstants) == 32, "PipelineCacheConstants must match the HLSL layout");

    /// @brief Mirrors PipelineCacheComputePushConstants in data/shaders/tests/pipeline_cache_compute.hlsl.
    struct PipelineCacheComputeConstants
    {
        uint32_t rwBuffer = 0;
        uint32_t slot = 0;
        uint32_t value = 0;
        uint32_t padding0 = 0;
    };
    static_assert(sizeof(PipelineCacheComputeConstants) == 16,
                  "PipelineCacheComputeConstants must match the HLSL layout");

    /// @brief Calls agfxRenderPipelineGetCache, copies the result into a vector, and frees the
    /// original buffer (the test device's allocator is plain malloc/free). Empty on failure.
    std::vector<uint8_t> GetRenderPipelineCacheBytes(agfxDevice* device, agfxRenderPipeline* pipeline);

    /// @brief As GetRenderPipelineCacheBytes, for a compute pipeline.
    std::vector<uint8_t> GetComputePipelineCacheBytes(agfxDevice* device, agfxComputePipeline* pipeline);

    /// @brief Reads a DEPTH32F texture back as a golden-comparable RGBA32F image, by reinterpreting
    /// it as R32F and running it through sampling_comparison.hlsl:main_sample_depth_cs -- the same
    /// technique test_sample_depth_texture.cpp uses to verify a depth-only draw actually landed.
    /// `depth` must be in AGFX_RESOURCE_STATE_DEPTH_WRITE on entry; left in
    /// AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE on return (nothing after this reads it further).
    bool SampleDepthToImage(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* depth,
                            uint32_t width, uint32_t height, Image& outImage);

    /// @brief True if every pixel in the box is within `tolerance` of (r,g,b) (alpha ignored).
    /// Tolerates readback's 8-bit UNORM quantization, unlike an exact compare.
    bool ColorRegionIs(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float r,
                       float g, float b, float tolerance = 2.0f / 255.0f);

    /// @brief True if every pixel in the box's red channel equals `expected` exactly. Depth values
    /// round-tripped through float32 are exact, unlike 8-bit color, so this needs no tolerance.
    bool DepthRegionIs(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float expected);
} // namespace agfxtest
