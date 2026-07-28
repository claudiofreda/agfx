/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "pipeline cache: VS + PS".
//
// Pipeline A (fresh, no cache) draws the left triangle in one tint; agfxRenderPipelineGetCache
// pulls its blob; pipeline A is destroyed; pipeline B is created from that blob alone (identical
// desc otherwise) and draws the right triangle in a different tint, into the same target with a
// LOAD op so A's triangle survives. The golden is both triangles together -- proof B, built purely
// from A's cache, renders exactly as A would have.

#include "agfx_tests/test_gpu.h"
#include "pipeline_cache_common.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    constexpr agfxTextureFormat kFormat = AGFX_TEXTURE_FORMAT_RGBA8_UNORM;
    constexpr float kClearColor[4] = {0.05f, 0.05f, 0.05f, 1.0f};
    constexpr float kTintA[4] = {0.8f, 0.2f, 0.2f, 1.0f}; // left triangle, pipeline A
    constexpr float kTintB[4] = {0.2f, 0.3f, 0.9f, 1.0f}; // right triangle, pipeline B (from cache)
    constexpr const char* kGolden = "pipeline_cache_vs_ps.png";

    agfxTextureCreateInfo TargetInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kFormat;
        info.usage = (agfxTextureUsage)(AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT | AGFX_TEXTURE_USAGE_SAMPLED);
        info.width = kPipelineCacheWidth;
        info.height = kPipelineCacheHeight;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxRenderPipelineCreateInfo PipelineInfo(agfxShaderModule* vs, agfxShaderModule* ps)
    {
        agfxRenderPipelineCreateInfo info{};
        info.name = "pipeline cache vs+ps";
        info.fillMode = AGFX_FILL_MODE_SOLID;
        info.cullMode = AGFX_CULL_MODE_NONE;
        info.frontFace = AGFX_FRONT_FACE_COUNTER_CLOCKWISE;
        info.topology = AGFX_TOPOLOGY_TRIANGLES;
        info.depthTestEnable = 0;
        info.depthWriteEnable = 0;
        info.colorAttachmentCount = 1;
        info.colorFormats[0] = kFormat;
        info.vertexShader = vs;
        info.fragmentShader = ps;
        return info;
    }

    agfxRenderPassCreateInfo PassInfo(agfxRenderTarget* renderTarget, agfxLoadOperation loadOp)
    {
        agfxRenderPassCreateInfo info{};
        info.colorAttachmentCount = 1;
        info.colorAttachments[0].renderTarget = renderTarget;
        info.colorAttachments[0].loadOp = loadOp;
        info.colorAttachments[0].storeOp = AGFX_STORE_OPERATION_STORE;
        memcpy(info.colorAttachments[0].clearColor, kClearColor, sizeof(kClearColor));
        info.width = kPipelineCacheWidth;
        info.height = kPipelineCacheHeight;
        info.name = "pipeline cache vs+ps";
        return info;
    }

    PipelineCacheConstants ConstantsFor(const float tint[4], uint32_t half_)
    {
        PipelineCacheConstants constants;
        memcpy(constants.tint, tint, sizeof(constants.tint));
        constants.half_ = half_;
        return constants;
    }

    void ExpectAccumulated(TestContext& ctx, const Image& image)
    {
        AGFX_EXPECT_MSG(ColorRegionIs(image, kPipelineCacheLeftSampleX, kPipelineCacheLeftSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize,
                                     kTintA[0], kTintA[1], kTintA[2]),
                       "pipeline A's triangle (left) is missing or the wrong color");
        AGFX_EXPECT_MSG(ColorRegionIs(image, kPipelineCacheRightSampleX, kPipelineCacheRightSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize,
                                     kTintB[0], kTintB[1], kTintB[2]),
                       "pipeline B's triangle (right, built from A's cache) is missing or the wrong color");
        AGFX_EXPECT_MSG(ColorRegionIs(image, kPipelineCacheBackgroundSampleX, kPipelineCacheBackgroundSampleY,
                                     kPipelineCacheSampleBoxSize, kPipelineCacheSampleBoxSize,
                                     kClearColor[0], kClearColor[1], kClearColor[2]),
                       "the background was overdrawn by one of the triangles");
    }
} // namespace

AGFX_TEST_TEXTURE(PipelineCacheVsPs, C, kPipelineCacheWidth, kPipelineCacheHeight)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile pipeline_cache.hlsl");

    const agfxTextureCreateInfo targetInfo = TargetInfo();
    agfxTexture* target = agfxTextureCreate(device, &targetInfo);
    AGFX_EXPECT_NOT_NULL(target);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = kFormat;
    agfxRenderTarget* renderTarget = agfxRenderTargetCreate(device, &rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget);

    agfxShaderModule* vsModuleA = CreateShaderModule(device, vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX);
    agfxShaderModule* psModuleA = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    const agfxRenderPipelineCreateInfo pipelineInfoA = PipelineInfo(vsModuleA, psModuleA);
    agfxRenderPipeline* pipelineA = agfxRenderPipelineCreate(device, &pipelineInfoA);
    agfxShaderModuleDestroy(device, vsModuleA);
    agfxShaderModuleDestroy(device, psModuleA);
    AGFX_EXPECT_NOT_NULL(pipelineA);

    agfxDeviceMakeResourcesResident(device);

    const PipelineCacheConstants constantsA = ConstantsFor(kTintA, 0);
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferTextureBarrier(cmd, target, AGFX_RESOURCE_STATE_COMMON,
                                        AGFX_RESOURCE_STATE_RENDER_TARGET, 0, 0, 0);
        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget, AGFX_LOAD_OPERATION_CLEAR);
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
    agfxShaderModule* psModuleB = CreateShaderModule(device, psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT);
    agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModuleB, psModuleB);
    pipelineInfoB.cache = cache.data() ? const_cast<uint8_t*>(cache.data()) : nullptr;
    pipelineInfoB.cacheSize = cache.size();
    agfxRenderPipeline* pipelineB = agfxRenderPipelineCreate(device, &pipelineInfoB);
    agfxShaderModuleDestroy(device, vsModuleB);
    agfxShaderModuleDestroy(device, psModuleB);
    AGFX_EXPECT_NOT_NULL(pipelineB);

    const PipelineCacheConstants constantsB = ConstantsFor(kTintB, 1);
    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        const agfxRenderPassCreateInfo passInfo = PassInfo(renderTarget, AGFX_LOAD_OPERATION_LOAD);
        agfxRenderPass* pass = agfxRenderPassBegin(cmd, &passInfo);
        agfxRenderPassSetViewport(pass, 0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight, 0.0f, 1.0f);
        agfxRenderPassSetScissor(pass, 0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        agfxRenderPassSetPipeline(pass, pipelineB);
        agfxRenderPassPushConstants(pass, &constantsB, sizeof(constantsB));
        agfxRenderPassDraw(pass, 3, 1, 0, 0);
        agfxRenderPassEnd(pass);
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), target, kPipelineCacheWidth, kPipelineCacheHeight,
                                          kFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, image);

    agfxRenderPipelineDestroy(device, pipelineB);
    agfxRenderTargetDestroy(device, renderTarget);
    agfxTextureDestroy(device, target);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(PipelineCacheVsPs, Cpp, kPipelineCacheWidth, kPipelineCacheHeight)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(agfx::CommandQueueType::Graphics);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile pipeline_cache.hlsl");

    agfx::Texture target = device.CreateTexture(TargetInfo());
    AGFX_EXPECT_NOT_NULL(target.Get());

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target;
    rtInfo.format = kFormat;
    agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget.Get());

    agfx::RenderPipeline pipelineA;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    const PipelineCacheConstants constantsA = ConstantsFor(kTintA, 0);
    cmd.Begin();
    cmd.TextureBarrier(target, agfx::ResourceState::Common, agfx::ResourceState::RenderTarget,
                       AGFX_SUBRESOURCE_ALL_MIPS, AGFX_SUBRESOURCE_ALL_LAYERS, false);
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget, AGFX_LOAD_OPERATION_CLEAR));
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
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModule, psModule);
        pipelineInfoB.cache = cache.data() ? const_cast<uint8_t*>(cache.data()) : nullptr;
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateRenderPipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    const PipelineCacheConstants constantsB = ConstantsFor(kTintB, 1);
    cmd.Begin();
    {
        agfx::RenderPass pass = cmd.BeginRenderPass(PassInfo(renderTarget, AGFX_LOAD_OPERATION_LOAD));
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
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), queue, target, kPipelineCacheWidth, kPipelineCacheHeight,
                                      kFormat, AGFX_RESOURCE_STATE_RENDER_TARGET, image),
                    "texture readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}

AGFX_TEST_TEXTURE(PipelineCacheVsPs, Ez, kPipelineCacheWidth, kPipelineCacheHeight)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kPipelineCacheWidth;
    contextInfo.height = kPipelineCacheHeight;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader vsShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_VERTEX, "main_vs");
    const CompiledShader psShader = CompileTestShader("pipeline_cache.hlsl", AGFX_SHADER_STAGE_FRAGMENT, "main_ps");
    AGFX_EXPECT_MSG(vsShader.Valid() && psShader.Valid(), "failed to compile pipeline_cache.hlsl");

    // ez's SetPipeline() only accepts a PipelineDesc, which builds and owns its own pipeline
    // internally and never touches agfxRenderPipelineGetCache -- there is no ez-idiomatic way to
    // bind a manually created agfx::RenderPipeline. So this test drops to a raw render pass on the
    // frame's command buffer instead, the same way the compute tests drop to a raw compute pass.
    agfx::ez::Texture2D target = context.CreateTexture2D(kPipelineCacheWidth, kPipelineCacheHeight, kFormat,
                                                         AGFX_TEXTURE_USAGE_COLOR_ATTACHMENT);

    agfxRenderTargetCreateInfo rtInfo{};
    rtInfo.texture = target.Raw();
    rtInfo.format = kFormat;
    agfx::RenderTarget renderTarget = device.CreateRenderTarget(rtInfo);
    AGFX_EXPECT_NOT_NULL(renderTarget.Get());

    agfx::RenderPipeline pipelineA;
    {
        agfx::ShaderModule vsModule(device.Get(),
            CreateShaderModule(device.Get(), vsShader, "main_vs", AGFX_SHADER_MODULE_TYPE_VERTEX));
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        pipelineA = device.CreateRenderPipeline(PipelineInfo(vsModule, psModule));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    const PipelineCacheConstants constantsA = ConstantsFor(kTintA, 0);
    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionTexture(target, AGFX_RESOURCE_STATE_RENDER_TARGET);
        agfx::RenderPass pass = context.GetCurrentCommandBuffer().BeginRenderPass(PassInfo(renderTarget, AGFX_LOAD_OPERATION_CLEAR));
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
        agfx::ShaderModule psModule(device.Get(),
            CreateShaderModule(device.Get(), psShader, "main_ps", AGFX_SHADER_MODULE_TYPE_FRAGMENT));
        agfxRenderPipelineCreateInfo pipelineInfoB = PipelineInfo(vsModule, psModule);
        pipelineInfoB.cache = cache.data() ? const_cast<uint8_t*>(cache.data()) : nullptr;
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateRenderPipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    const PipelineCacheConstants constantsB = ConstantsFor(kTintB, 1);
    {
        agfx::ez::Frame frame = context.BeginFrame();
        agfx::RenderPass pass = context.GetCurrentCommandBuffer().BeginRenderPass(PassInfo(renderTarget, AGFX_LOAD_OPERATION_LOAD));
        pass.SetViewport(0.0f, 0.0f, (float)kPipelineCacheWidth, (float)kPipelineCacheHeight);
        pass.SetScissor(0, 0, kPipelineCacheWidth, kPipelineCacheHeight);
        pass.SetPipeline(pipelineB);
        pass.PushConstants(&constantsB, sizeof(constantsB));
        pass.Draw(3);
    }
    context.DrainGPU();

    Image image;
    AGFX_EXPECT_MSG(ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), target.Raw(), kPipelineCacheWidth,
                                      kPipelineCacheHeight, kFormat, target.State(), image),
                    "texture readback failed");
    ExpectAccumulated(ctx, image);
    ExpectImageMatchesGolden(ctx, kGolden, image);
}
