/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "pipeline_cache_common.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace agfxtest
{
    namespace
    {
        constexpr agfxTextureFormat kSampleDepthViewFormat = AGFX_TEXTURE_FORMAT_R32F;
        constexpr agfxTextureFormat kSampleDestFormat = AGFX_TEXTURE_FORMAT_RGBA32F;
        constexpr uint32_t kSampleGroupSize = 8; // Matches sampling_comparison.hlsl's [numthreads(8,8,1)].

        /// @brief Mirrors SamplingComparisonPushConstants in sampling_comparison.hlsl.
        struct SampleDepthPushConstants
        {
            uint32_t source = 0;
            uint32_t samplerId = 0;
            uint32_t destination = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t bandCount = 1;
            uint32_t padding0 = 0;
            uint32_t padding1 = 0;
            float references[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // unused by main_sample_depth_cs
        };
        static_assert(sizeof(SampleDepthPushConstants) == 48,
                      "SampleDepthPushConstants must match sampling_comparison.hlsl's layout");

        /// @brief Records `record` onto a throwaway command buffer on `queue` and blocks until it
        /// completes, mirroring test_gpu.cpp's private RecordAndWait -- so this helper works from a
        /// GpuFixture's queue, a raw agfx::CommandQueue, or an ez::Context's, without needing one
        /// of its own already-open command buffers.
        template<typename Fn>
        void RecordAndWait(agfxDevice* device, agfxCommandQueue* queue, Fn&& record)
        {
            agfxCommandBuffer* cmd = agfxCommandBufferCreate(device, queue);
            agfxFence* fence = agfxFenceCreate(device);

            agfxCommandBufferBegin(cmd);
            record(cmd);
            agfxCommandBufferEnd(cmd);

            agfxCommandBuffer* buffers[] = {cmd};
            agfxCommandQueueSubmit(queue, buffers, 1);
            agfxCommandQueueSignal(queue, fence, 1);
            agfxFenceWait(fence, 1, UINT64_MAX);

            agfxFenceDestroy(device, fence);
            agfxCommandBufferDestroy(device, cmd);
        }
    } // namespace

    std::vector<uint8_t> GetRenderPipelineCacheBytes(agfxDevice* device, agfxRenderPipeline* pipeline)
    {
        uint64_t size = 0;
        uint8_t* bytes = agfxRenderPipelineGetCache(device, pipeline, &size);
        if (!bytes) {
            return {};
        }
        std::vector<uint8_t> result(bytes, bytes + size);
        free(bytes);
        return result;
    }

    std::vector<uint8_t> GetComputePipelineCacheBytes(agfxDevice* device, agfxComputePipeline* pipeline)
    {
        uint64_t size = 0;
        uint8_t* bytes = agfxComputePipelineGetCache(device, pipeline, &size);
        if (!bytes) {
            return {};
        }
        std::vector<uint8_t> result(bytes, bytes + size);
        free(bytes);
        return result;
    }

    bool SampleDepthToImage(agfxDevice* device, agfxCommandQueue* queue, agfxTexture* depth,
                            uint32_t width, uint32_t height, Image& outImage)
    {
        const CompiledShader csShader =
            CompileTestShader("sampling_comparison.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_sample_depth_cs");
        if (!csShader.Valid()) {
            return false;
        }

        agfxTextureCreateInfo destInfo{};
        destInfo.type = AGFX_TEXTURE_TYPE_2D;
        destInfo.format = kSampleDestFormat;
        destInfo.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_STORAGE | AGFX_TEXTURE_USAGE_SAMPLED);
        destInfo.width = width;
        destInfo.height = height;
        destInfo.depthOrArrayLayers = 1;
        destInfo.mipLevels = 1;
        agfxTexture* dest = agfxTextureCreate(device, &destInfo);

        agfxTextureViewCreateInfo depthViewInfo{};
        depthViewInfo.texture = depth;
        depthViewInfo.format = kSampleDepthViewFormat; // the D32 -> R32F reinterpretation.
        depthViewInfo.type = AGFX_TEXTURE_TYPE_2D;
        depthViewInfo.mipLevelCount = 1;
        depthViewInfo.arrayLayerCount = 1;
        depthViewInfo.writeable = 0;
        agfxTextureView* depthSrv = dest ? agfxTextureViewCreate(device, &depthViewInfo) : nullptr;

        agfxTextureViewCreateInfo destViewInfo{};
        destViewInfo.texture = dest;
        destViewInfo.format = kSampleDestFormat;
        destViewInfo.type = AGFX_TEXTURE_TYPE_2D;
        destViewInfo.mipLevelCount = 1;
        destViewInfo.arrayLayerCount = 1;
        destViewInfo.writeable = 1;
        agfxTextureView* destUav = dest ? agfxTextureViewCreate(device, &destViewInfo) : nullptr;

        agfxSamplerCreateInfo samplerInfo{};
        samplerInfo.filter = AGFX_SAMPLER_FILTER_NEAREST;
        samplerInfo.addressModeU = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = AGFX_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.comparisonFunction = AGFX_COMPARISON_FUNCTION_ALWAYS; // not a comparison sampler.
        agfxSampler* sampler = agfxSamplerCreate(device, &samplerInfo);

        agfxShaderModule* csModule =
            CreateShaderModule(device, csShader, "main_sample_depth_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
        agfxComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.name = "pipeline cache depth sample";
        pipelineInfo.computeShader = csModule;
        pipelineInfo.groupSizeX = kSampleGroupSize;
        pipelineInfo.groupSizeY = kSampleGroupSize;
        pipelineInfo.groupSizeZ = 1;
        agfxComputePipeline* pipeline = csModule ? agfxComputePipelineCreate(device, &pipelineInfo) : nullptr;
        if (csModule) {
            agfxShaderModuleDestroy(device, csModule);
        }

        bool ok = dest != nullptr && depthSrv != nullptr && destUav != nullptr && sampler != nullptr &&
                  pipeline != nullptr;
        if (ok) {
            agfxDeviceMakeResourcesResident(device);

            SampleDepthPushConstants constants{};
            constants.source = (uint32_t)agfxTextureViewGetHandle(depthSrv);
            constants.samplerId = (uint32_t)agfxSamplerGetHandle(sampler);
            constants.destination = (uint32_t)agfxTextureViewGetHandle(destUav);
            constants.width = width;
            constants.height = height;

            const uint32_t groupsX = (width + kSampleGroupSize - 1) / kSampleGroupSize;
            const uint32_t groupsY = (height + kSampleGroupSize - 1) / kSampleGroupSize;

            RecordAndWait(device, queue, [&](agfxCommandBuffer* cmd) {
                agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                                AGFX_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                agfxCommandBufferTextureBarrier(cmd, dest, AGFX_RESOURCE_STATE_COMMON,
                                                AGFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                                AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
                agfxComputePass* pass = agfxComputePassBegin(cmd, "pipeline cache depth sample");
                agfxComputePassSetPipeline(pass, pipeline);
                agfxComputePassPushConstants(pass, &constants, sizeof(constants));
                agfxComputePassDispatch(pass, groupsX, groupsY, 1);
                agfxComputePassEnd(pass);
            });

            ok = ReadbackTexture2D(device, queue, dest, width, height, kSampleDestFormat,
                                   AGFX_RESOURCE_STATE_UNORDERED_ACCESS, outImage);
        }

        if (pipeline) agfxComputePipelineDestroy(device, pipeline);
        if (sampler) agfxSamplerDestroy(device, sampler);
        if (destUav) agfxTextureViewDestroy(device, destUav);
        if (depthSrv) agfxTextureViewDestroy(device, depthSrv);
        if (dest) agfxTextureDestroy(device, dest);
        return ok;
    }

    bool ColorRegionIs(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float r,
                       float g, float b, float tolerance)
    {
        if (!image.Valid() || x + w > image.width || y + h > image.height) {
            return false;
        }
        for (uint32_t py = y; py < y + h; ++py) {
            for (uint32_t px = x; px < x + w; ++px) {
                const size_t i = (size_t(py) * image.width + px) * 4;
                if (std::fabs(image.pixels[i + 0] - r) > tolerance ||
                    std::fabs(image.pixels[i + 1] - g) > tolerance ||
                    std::fabs(image.pixels[i + 2] - b) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    bool DepthRegionIs(const Image& image, uint32_t x, uint32_t y, uint32_t w, uint32_t h, float expected)
    {
        if (!image.Valid() || x + w > image.width || y + h > image.height) {
            return false;
        }
        for (uint32_t py = y; py < y + h; ++py) {
            for (uint32_t px = x; px < x + w; ++px) {
                if (image.pixels[(size_t(py) * image.width + px) * 4] != expected) {
                    return false;
                }
            }
        }
        return true;
    }
} // namespace agfxtest
