/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "pipeline cache: MS only".
//
// Same recipe as test_pipeline_cache_vs_only.cpp, but through a mesh pipeline with no fragment
// shader: pipeline A draws the left triangle's depth via agfxRenderPassDrawMesh,
// agfxRenderPipelineGetCache pulls its blob, pipeline B is built from that blob alone and draws the
// right triangle's depth. Revealed via the same depth-to-color compute sample as the VS-only test.
// Mesh shading is optional, so this skips rather than fails where unsupported -- see mesh_common.h.

#include "agfx_tests/test_gpu.h"
#include "mesh_common.h"
#include "pipeline_cache_common.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr agfxTextureFormat kDepthFormat = AGFX_TEXTURE_FORMAT_DEPTH32F;
    constexpr float kFarDepth = 0.75f;    // Clear value / background.
    constexpr float kDepthValueA = 0.25f; // Left triangle, pipeline A.
    constexpr float kDepthValueB = 0.35f; // Right triangle, pipeline B (from cache).
    constexpr const char* kGolden = "pipeline_cache_mesh_only.hdr";

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

    agfxRenderPipelineCreateInfo PipelineInfo(const CompiledShader& meshShader, agfxShaderModule* ms)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "pipeline cache ms only";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 1;
        info.depthWriteEnable = 1;
        info.depthCompareOp = AGFX_COMPARISON_FUNCTION_ALWAYS;
        info.depthFormat = kDepthFormat;
        info.colorAttachmentCount = 0;
        info.meshShader = ms;
        info.fragmentShader = nullptr; // the code path under test
        info.meshGroupSizeX = meshShader.meshSizeX;
        info.meshGroupSizeY = meshShader.meshSizeY;
        info.meshGroupSizeZ = meshShader.meshSizeZ;
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
        info.name = "pipeline cache ms only";
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

AGFX_TEST_TEXTURE(PipelineCacheMeshOnly, C, kPipelineCacheWidth, kPipelineCacheHeight)
{
    if (!DeviceSupportsMeshShaders()) {
        ctx.Skip("device reports no mesh shader support");
        return;
    }

    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader msShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_MESH, "main_ms");
    AGFX_EXPECT_MSG(msShader.Valid(), "failed to compile pipeline_cache.hlsl:main_ms");

    const agfxTextureCreateInfo depthInfo = DepthInfo();
    agfxTexture* depth = agfxTextureCreate(device, &depthInfo);
    AGFX_EXPECT_NOT_NULL(depth);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = depth;
    rtInfo.format = kDepthFormat;
    rtInfo.isDepth = 1;
    agfxRenderTarget* depthTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(depthTarget);

    agfxShaderModule* msModuleA = CreateShaderModule(device, msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH);
    const agfxRenderPipelineCreateInfo pipelineInfoA = PipelineInfo(msShader, msModuleA);
    agfxRenderPipeline* pipelineA = agfxRenderPipelineCreate(device, &pipelineInfoA);
    agfxShaderModuleDestroy(device, msModuleA);
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
        agfxRenderPassDrawMesh(pass, 1, 1, 1);
        agfxRenderPassEnd(pass);
    });

    const std::vector<uint8_t> cache = GetRenderPipelineCacheBytes(device, pipelineA);
    AGFX_EXPECT_MSG(!cache.empty(), "agfxRenderPipelineGetCache returned an empty cache blob");
    agfxRenderPipelineDestroy(device, pipelineA);

    agfxShaderModule* msModuleB = CreateShaderModule(device, msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH);
    agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(msShader, msModuleB);
    pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
    pipelineInfoB.cacheSize = cache.size();
    agfxRenderPipeline* pipelineB = agfxRenderPipelineCreate(device, &pipelineInfoB);
    agfxShaderModuleDestroy(device, msModuleB);
    AGFX_EXPECT_NOT_NULL(pipelineB);

    const PipelineCacheConstants constantsB = ConstantsFor(kDepthValueB, 1);
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        const agfxRenderPassCreateInfo passInfo = PassInfo(depthTarget, AGFX_LOAD_OPERATION_LOAD);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        agfxRenderPassSetPipeline(pass, pipelineB);
        agfxRenderPassPushConstants(pass, &constantsB, sizeof(constantsB));
        agfxRenderPassDrawMesh(pass, 1, 1, 1);
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

AGFX_TEST_TEXTURE(PipelineCacheMeshOnly, Cpp, kPipelineCacheWidth, kPipelineCacheHeight)
{
    if (!DeviceSupportsMeshShaders()) {
        ctx.Skip("device reports no mesh shader support");
        return;
    }

    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(agfx::CommandQueueType::Graphics);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader msShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_MESH, "main_ms");
    AGFX_EXPECT_MSG(msShader.Valid(), "failed to compile pipeline_cache.hlsl:main_ms");

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
        agfx::ShaderModule msModule(device.Get(),
            CreateShaderModule(device.Get(), msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(msShader, msModule));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    const PipelineCacheConstants constantsA = ConstantsFor(kDepthValueA, 0);
    cmd.Begin();
    cmd.TextureBarrier(depth, agfx::ResourceState::Common, agfx::ResourceState::DepthWrite,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, true);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(depthTarget, AGFX_LOAD_OPERATION_CLEAR));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineA);
        pass.PushConstants(&constantsA, sizeof(constantsA));
        pass.DrawMesh(1, 1, 1);
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
        agfx::ShaderModule msModule(device.Get(),
            CreateShaderModule(device.Get(), msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(msShader, msModule);
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
        pass.DrawMesh(1, 1, 1);
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

AGFX_TEST_TEXTURE(PipelineCacheMeshOnly, Ez, kPipelineCacheWidth, kPipelineCacheHeight)
{
    if (!DeviceSupportsMeshShaders()) {
        ctx.Skip("device reports no mesh shader support");
        return;
    }

    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kPipelineCacheWidth;
    contextInfo.height = kPipelineCacheHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader msShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_MESH, "main_ms");
    AGFX_EXPECT_MSG(msShader.Valid(), "failed to compile pipeline_cache.hlsl:main_ms");

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
        agfx::ShaderModule msModule(device.Get(),
            CreateShaderModule(device.Get(), msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(msShader, msModule));
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
        pass.DrawMesh(1, 1, 1);
    }
    context.DrainGPU();

    const std::vector<uint8_t> cache = GetRenderPipelineCacheBytes(device.Get(), pipelineA.Get());
    AGFX_EXPECT_MSG(!cache.empty(), "agfxRenderPipelineGetCache returned an empty cache blob");
    pipelineA.Reset();

    agfx::RenderPipeline pipelineB;
    {
        agfx::ShaderModule msModule(device.Get(),
            CreateShaderModule(device.Get(), msShader, "main_ms", AGFX_SHADER_MODULE_TYPE_MESH));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(msShader, msModule);
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
        pass.DrawMesh(1, 1, 1);
    }
    context.DrainGPU();

    Image image;
    const bool sampledOk = SampleDepthToImage(device.Get(), context.GetGraphicsQueue(), depth.Raw(),
                                              kPipelineCacheWidth, kPipelineCacheHeight, image);
    AGFX_EXPECT_MSG(sampledOk, "depth sample readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
