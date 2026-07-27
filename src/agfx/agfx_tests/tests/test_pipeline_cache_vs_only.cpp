/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "pipeline cache: VS only".
//
// A depth-only pipeline (fragmentShader == nullptr) -- a separate code path in
// agfxRenderPipelineCreate from the VS+PS one, and one the cache blob must also round-trip through.
// Pipeline A draws the left triangle at one depth; agfxRenderPipelineGetCache pulls its blob; A is
// destroyed; pipeline B is built from that blob alone and draws the right triangle at a different
// depth, into the same depth buffer with a LOAD op so A's write survives. Neither draw has a pixel
// shader, so there is nothing to read back directly -- the result is revealed by reinterpreting the
// depth buffer as R32F and sampling it out with a compute pass, the same technique
// test_sample_depth_texture.cpp uses.

#include "agfx_tests/test_gpu.h"
#include "pipeline_cache_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr agfxTextureFormat kDepthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
    constexpr float kFarDepth = 0.75f;    // Clear value / background.
    constexpr float kDepthValueA = 0.25f; // Left triangle, pipeline A.
    constexpr float kDepthValueB = 0.35f; // Right triangle, pipeline B (from cache).
    constexpr const char* kGolden = "pipeline_cache_vs_only.hdr";

    agfxTextureCreateInfo DepthInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kDepthFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kPipelineCacheWidth;
        info.height = kPipelineCacheHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "pipeline cache vs only";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 1;
        info.depthWriteEnable = 1;
        info.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS; // unconditional write, as in test_sample_depth_texture.cpp
        info.depthFormat = kDepthFormat;
        info.colorAttachmentCount = 0;
        info.vertexShader = vs;
        info.fragmentShader = nullptr; // the code path under test
        return info;
    }

    agfxRenderPassCreateInfo PassInfo(agfxRenderTarget* depthTarget, agfxLoadOperation loadOp)
    {
        agfxRenderPassCreateInfo info{};
        info.colorAttachmentCount = 0;
        info.hasDepthAttachment = 1;
        info.depthAttachment.renderTarget = depthTarget;
        info.depthAttachment.loadOp = loadOp;
        info.depthAttachment.storeOp = AGFX_STORE_OPERATION_STORE;
        info.depthAttachment.clearDepth = kFarDepth;
        info.width = kPipelineCacheWidth;
        info.height = kPipelineCacheHeight;
        info.name = "pipeline cache vs only";
        return info;
    }

    PipelineCacheConstants ConstantsFor(float depthValue, uint32_t half_)
    {
        PipelineCacheConstants constants;
        constants.depthValue = depthValue;
        constants.half_ = half_;
        return constants;
    }

    void ExpectAccumulated(TestContext& ctx, const Image& image)
    {
        AGFX_EXPECT_MSG(DepthRegionIs(image, kPipelineCacheLeftSampleX, kPipelineCacheLeftSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize, kDepthValueA),
                       "pipeline A's depth write (left) is missing or wrong");
        AGFX_EXPECT_MSG(DepthRegionIs(image, kPipelineCacheRightSampleX, kPipelineCacheRightSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize, kDepthValueB),
                       "pipeline B's depth write (right, built from A's cache) is missing or wrong");
        AGFX_EXPECT_MSG(DepthRegionIs(image, kPipelineCacheBackgroundSampleX, kPipelineCacheBackgroundSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize, kFarDepth),
                       "the background depth was overwritten by one of the draws");
    }
} // namespace

AGFX_TEST_TEXTURE(PipelineCacheVsOnly, C, kPipelineCacheWidth, kPipelineCacheHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile pipeline_cache.hlsl:main_vs");

    const agfxTextureCreateInfo depthInfo = DepthInfo();
    agfxTexture* depth = agfxTextureCreate(device, &depthInfo);
    AGFX_EXPECT_NOT_NULL(depth);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth;
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfxRenderTarget* depthTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(depthTarget);

    agfxShaderModule* vsModuleA = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    const agfxRenderPipelineCreateInfo pipelineInfoA = PipelineInfo(vsModuleA);
    agfxRenderPipeline* pipelineA = agfxRenderPipelineCreate(device, &pipelineInfoA);
    agfxShaderModuleDestroy(device, vsModuleA);
    AGFX_EXPECT_NOT_NULL(pipelineA);

    agfxDeviceMakeResourcesResident(device);

    const PipelineCacheConstants constantsA = ConstantsFor(kDepthValueA, 0);
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, depth, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_DEPTH_WRITE,
                                        AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, 1);
        const agfxRenderPassCreateInfo passInfo = PassInfo(depthTarget, AGFX_LOAD_OPERATION_CLEAR);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        agfxRenderPassSetPipeline(pass, pipelineA);
        agfxRenderPassPushConstants(pass, &constantsA, sizeof(constantsA));
        agfxRenderPassDraw(pass, 3, 1, 0, 0);
        agfxRenderPassEnd(pass);
    });

    const std::vector<uint8_t> cache = GetRenderPipelineCacheBytes(device, pipelineA);
    AGFX_EXPECT_MSG(!cache.empty(), "agfxRenderPipelineGetCache returned an empty cache blob");
    agfxRenderPipelineDestroy(device, pipelineA);

    agfxShaderModule* vsModuleB = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModuleB);
    pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
    pipelineInfoB.cacheSize = cache.size();
    agfxRenderPipeline* pipelineB = agfxRenderPipelineCreate(device, &pipelineInfoB);
    agfxShaderModuleDestroy(device, vsModuleB);
    AGFX_EXPECT_NOT_NULL(pipelineB);

    const PipelineCacheConstants constantsB = ConstantsFor(kDepthValueB, 1);
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        const agfxRenderPassCreateInfo passInfo = PassInfo(depthTarget, AGFX_LOAD_OPERATION_LOAD);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        agfxRenderPassSetPipeline(pass, pipelineB);
        agfxRenderPassPushConstants(pass, &constantsB, sizeof(constantsB));
        agfxRenderPassDraw(pass, 3, 1, 0, 0);
        agfxRenderPassEnd(pass);
    });

    Image image;
    const bool sampledOk = SampleDepthToImage(device, gpu.Queue(), depth, kPipelineCacheWidth, kPipelineCacheHeight, image);

    agfxRenderPipelineDestroy(device, pipelineB);
    agfxRenderTargetDestroy(device, depthTarget);
    agfxTextureDestroy(device, depth);

    AGFX_EXPECT_MSG(sampledOk, "depth sample readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(PipelineCacheVsOnly, Cpp, kPipelineCacheWidth, kPipelineCacheHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile pipeline_cache.hlsl:main_vs");

    agfx::Texture depth = device.CreateTexture(DepthInfo());
    AGFX_EXPECT_NOT_NULL(depth.Get());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth;
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfx::RenderTarget depthTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(depthTarget.Get());

    agfx::RenderPipeline pipelineA;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(vsModule));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    const PipelineCacheConstants constantsA = ConstantsFor(kDepthValueA, 0);
    cmd.Begin();
    cmd.TextureBarrier(depth, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_DEPTH_WRITE,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(depthTarget, AGFX_LOAD_OPERATION_CLEAR));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineA);
        pass.PushConstants(&constantsA, sizeof(constantsA));
        pass.Draw(3);
    }
    cmd.End();
    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    const std::vector<uint8_t> cache = GetRenderPipelineCacheBytes(device.Get(), pipelineA.Get());
    AGFX_EXPECT_MSG(!cache.empty(), "agfxRenderPipelineGetCache returned an empty cache blob");
    pipelineA.Reset();

    agfx::RenderPipeline pipelineB;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModule);
        pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateRenderPipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    const PipelineCacheConstants constantsB = ConstantsFor(kDepthValueB, 1);
    cmd.Begin();
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(depthTarget, AGFX_LOAD_OPERATION_LOAD));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineB);
        pass.PushConstants(&constantsB, sizeof(constantsB));
        pass.Draw(3);
    }
    cmd.End();
    queue.Submit(cmd);
    queue.Signal(fence, 2);
    fence.Wait(2, UINT64_MAX);

    Image image;
    const bool sampledOk = SampleDepthToImage(device.Get(), queue, depth, kPipelineCacheWidth, kPipelineCacheHeight, image);
    AGFX_EXPECT_MSG(sampledOk, "depth sample readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(PipelineCacheVsOnly, Ez, kPipelineCacheWidth, kPipelineCacheHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kPipelineCacheWidth;
    contextInfo.height = kPipelineCacheHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    AGFX_EXPECT_MSG(vsShader.Valid(), "failed to compile pipeline_cache.hlsl:main_vs");

    // A depth-only pipeline has no ez equivalent at all (ez::PipelineDesc always builds a pipeline
    // through SetPipeline(), which owns it internally) -- same reasoning as the VS+PS Ez variant,
    // drop to a raw render pass on the frame's command buffer.
    agfx::ez::Texture2D depth = context.CreateTexture2D(
        kPipelineCacheWidth, kPipelineCacheHeight, kDepthFormat,
        (agfxTextureUsage)(AGFX_TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED));

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth.Raw();
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfx::RenderTarget depthTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(depthTarget.Get());

    agfx::RenderPipeline pipelineA;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(vsModule));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    const PipelineCacheConstants constantsA = ConstantsFor(kDepthValueA, 0);
    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(depth, AGFX_RESOURCE_STATE_DEPTH_WRITE);
        agfx::RenderPass pass = context.GetCurrentCommandBuffer().BeginRenderPass(PassInfo(depthTarget, AGFX_LOAD_OPERATION_CLEAR));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineA);
        pass.PushConstants(&constantsA, sizeof(constantsA));
        pass.Draw(3);
    }
    context.DrainGPU();

    const std::vector<uint8_t> cache = GetRenderPipelineCacheBytes(device.Get(), pipelineA.Get());
    AGFX_EXPECT_MSG(!cache.empty(), "agfxRenderPipelineGetCache returned an empty cache blob");
    pipelineA.Reset();

    agfx::RenderPipeline pipelineB;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModule);
        pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateRenderPipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    const PipelineCacheConstants constantsB = ConstantsFor(kDepthValueB, 1);
    {
        agfx::ez::Frame frame = context.BeginFrame();
        agfx::RenderPass pass = context.GetCurrentCommandBuffer().BeginRenderPass(PassInfo(depthTarget, AGFX_LOAD_OPERATION_LOAD));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineB);
        pass.PushConstants(&constantsB, sizeof(constantsB));
        pass.Draw(3);
    }
    context.DrainGPU();

    Image image;
    const bool sampledOk = SampleDepthToImage(device.Get(), context.GetGraphicsQueue(), depth.Raw(),
                                              kPipelineCacheWidth, kPipelineCacheHeight, image);
    AGFX_EXPECT_MSG(sampledOk, "depth sample readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
