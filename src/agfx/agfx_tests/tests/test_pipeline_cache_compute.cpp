/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-27 00:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "pipeline cache: compute".
//
// Pipeline A (fresh, no cache) dispatches and stores a value into buffer slot 0;
// agfxComputePipelineGetCache pulls its blob; A is destroyed; pipeline B is built from that blob
// alone and dispatches again, storing a different value into slot 1 -- no reset in between, so
// slot 0 survives. The golden is both slots together -- proof B, built purely from A's cache,
// dispatches exactly as A would have.

#include "agfx_tests/test_gpu.h"
#include "pipeline_cache_common.h"

#include <agfx/agfx_ez.hpp>

namespace
{
    using namespace agfxtest;

    constexpr uint64_t kBufferSize = 2 * sizeof(uint32_t);
    constexpr uint32_t kValueA = 111; // slot 0, pipeline A
    constexpr uint32_t kValueB = 222; // slot 1, pipeline B (from cache)
    constexpr const char* kGolden = "pipeline_cache_compute.bin";

    agfxBufferCreateInfo BufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kBufferSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxComputePipelineCreateInfo ComputePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "pipeline cache compute";
        info.computeShader = module;
        info.groupSizeX = 1;
        info.groupSizeY = 1;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

AGFX_TEST_BUFFER(PipelineCacheCompute, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader shader = CompileTestShader("pipeline_cache_compute.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile pipeline_cache_compute.hlsl:main_cs");

    const agfxBufferCreateInfo bufferInfo = BufferInfo();
    agfxBuffer* buffer = agfxBufferCreate(device, &bufferInfo);
    AGFX_EXPECT_NOT_NULL(buffer);

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = buffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    viewInfo.offset = 0;
    viewInfo.writeable = 1;
    agfxBufferView* view = agfxBufferViewCreate(device, &viewInfo);
    AGFX_EXPECT_NOT_NULL(view);

    agfxShaderModule* moduleA = CreateShaderModule(device, shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo pipelineInfoA = ComputePipelineInfo(moduleA);
    agfxComputePipeline* pipelineA = agfxComputePipelineCreate(device, &pipelineInfoA);
    agfxShaderModuleDestroy(device, moduleA);
    AGFX_EXPECT_NOT_NULL(pipelineA);

    agfxDeviceMakeResourcesResident(device);

    PipelineCacheComputeConstants constantsA;
    constantsA.rwBuffer = (uint32_t)agfxBufferViewGetHandle(view);
    constantsA.slot = 0;
    constantsA.value = kValueA;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferBufferBarrier(cmd, buffer, AGFX_RESOURCE_STATE_COMMON,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 0);
        agfxComputePass* pass = agfxComputePassBegin(cmd, "pipeline cache compute a");
        agfxComputePassSetPipeline(pass, pipelineA);
        agfxComputePassPushConstants(pass, &constantsA, sizeof(constantsA));
        agfxComputePassDispatch(pass, 1, 1, 1);
        agfxComputePassEnd(pass);
    });

    const std::vector<uint8_t> cache = GetComputePipelineCacheBytes(device, pipelineA);
    AGFX_EXPECT_MSG(!cache.empty(), "agfxComputePipelineGetCache returned an empty cache blob");
    agfxComputePipelineDestroy(device, pipelineA);

    agfxShaderModule* moduleB = CreateShaderModule(device, shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxComputePipelineCreateInfo pipelineInfoB = ComputePipelineInfo(moduleB);
    pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
    pipelineInfoB.cacheSize = cache.size();
    agfxComputePipeline* pipelineB = agfxComputePipelineCreate(device, &pipelineInfoB);
    agfxShaderModuleDestroy(device, moduleB);
    AGFX_EXPECT_NOT_NULL(pipelineB);

    PipelineCacheComputeConstants constantsB;
    constantsB.rwBuffer = (uint32_t)agfxBufferViewGetHandle(view);
    constantsB.slot = 1;
    constantsB.value = kValueB;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxComputePass* pass = agfxComputePassBegin(cmd, "pipeline cache compute b");
        agfxComputePassSetPipeline(pass, pipelineB);
        agfxComputePassPushConstants(pass, &constantsB, sizeof(constantsB));
        agfxComputePassDispatch(pass, 1, 1, 1);
        agfxComputePassEnd(pass);
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, pipelineB);
    agfxBufferViewDestroy(device, view);
    agfxBufferDestroy(device, buffer);

    AGFX_EXPECT_MSG(readOk, "buffer readback failed");
    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(PipelineCacheCompute, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(AGFX_COMMAND_QUEUE_TYPE_GRAPHICS);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader shader = CompileTestShader("pipeline_cache_compute.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile pipeline_cache_compute.hlsl:main_cs");

    agfx::Buffer buffer = device.CreateBuffer(BufferInfo());
    AGFX_EXPECT_NOT_NULL(buffer.Get());

    agfxBufferViewCreateInfo viewInfo{};
    viewInfo.buffer = buffer;
    viewInfo.type = AGFX_BUFFER_VIEW_TYPE_RAW;
    viewInfo.offset = 0;
    viewInfo.writeable = 1;
    agfx::BufferView view = device.CreateBufferView(viewInfo);
    AGFX_EXPECT_NOT_NULL(view.Get());

    agfx::ComputePipeline pipelineA;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipelineA = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    PipelineCacheComputeConstants constantsA;
    constantsA.rwBuffer = (uint32_t)view.GetHandle();
    constantsA.slot = 0;
    constantsA.value = kValueA;

    cmd.Begin();
    cmd.BufferBarrier(buffer, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("pipeline cache compute a");
        pass.SetPipeline(pipelineA);
        pass.PushConstants(constantsA);
        pass.Dispatch(1, 1, 1);
    }
    cmd.End();
    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    const std::vector<uint8_t> cache = GetComputePipelineCacheBytes(device.Get(), pipelineA.Get());
    AGFX_EXPECT_MSG(!cache.empty(), "agfxComputePipelineGetCache returned an empty cache blob");
    pipelineA.Reset();

    agfx::ComputePipeline pipelineB;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfxComputePipelineCreateInfo pipelineInfoB = ComputePipelineInfo(module);
        pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateComputePipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    PipelineCacheComputeConstants constantsB;
    constantsB.rwBuffer = (uint32_t)view.GetHandle();
    constantsB.slot = 1;
    constantsB.value = kValueB;

    cmd.Begin();
    {
        agfx::ComputePass pass = cmd.BeginComputePass("pipeline cache compute b");
        pass.SetPipeline(pipelineB);
        pass.PushConstants(constantsB);
        pass.Dispatch(1, 1, 1);
    }
    cmd.End();
    queue.Submit(cmd);
    queue.Signal(fence, 2);
    fence.Wait(2, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, buffer, kBufferSize,
                                       AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}

AGFX_TEST_BUFFER(PipelineCacheCompute, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    const CompiledShader shader = CompileTestShader("pipeline_cache_compute.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_cs");
    AGFX_EXPECT_MSG(shader.Valid(), "failed to compile pipeline_cache_compute.hlsl:main_cs");

    agfx::Device& device = context.GetDevice();

    agfx::ez::Buffer buffer =
        context.CreateStructuredBuffer(nullptr, kBufferSize, sizeof(uint32_t), /*shaderWritable*/ true);
    agfx::BufferView& view = buffer.View(AGFX_BUFFER_VIEW_TYPE_RAW, /*writeable*/ true);
    AGFX_EXPECT_NOT_NULL(view.Get());

    agfx::ComputePipeline pipelineA;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        pipelineA = device.CreateComputePipeline(ComputePipelineInfo(module));
    }
    AGFX_EXPECT_NOT_NULL(pipelineA.Get());

    device.MakeResourcesResident();

    PipelineCacheComputeConstants constantsA;
    constantsA.rwBuffer = (uint32_t)view.GetHandle();
    constantsA.slot = 0;
    constantsA.value = kValueA;

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(buffer, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        // ez deliberately has no compute-pass sugar; drop to the frame's raw command buffer.
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("pipeline cache compute a");
        pass.SetPipeline(pipelineA);
        pass.PushConstants(&constantsA, sizeof(constantsA));
        pass.Dispatch(1, 1, 1);
    }
    context.DrainGPU();

    const std::vector<uint8_t> cache = GetComputePipelineCacheBytes(device.Get(), pipelineA.Get());
    AGFX_EXPECT_MSG(!cache.empty(), "agfxComputePipelineGetCache returned an empty cache blob");
    pipelineA.Reset();

    agfx::ComputePipeline pipelineB;
    {
        agfx::ShaderModule module(device.Get(),
            CreateShaderModule(device.Get(), shader, "main_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfxComputePipelineCreateInfo pipelineInfoB = ComputePipelineInfo(module);
        pipelineInfoB.cache = const_cast<uint8_t*>(cache.data());
        pipelineInfoB.cacheSize = cache.size();
        pipelineB = device.CreateComputePipeline(pipelineInfoB);
    }
    AGFX_EXPECT_NOT_NULL(pipelineB.Get());

    PipelineCacheComputeConstants constantsB;
    constantsB.rwBuffer = (uint32_t)view.GetHandle();
    constantsB.slot = 1;
    constantsB.value = kValueB;

    {
        agfx::ez::Frame frame = context.BeginFrame();
        agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("pipeline cache compute b");
        pass.SetPipeline(pipelineB);
        pass.PushConstants(&constantsB, sizeof(constantsB));
        pass.Dispatch(1, 1, 1);
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), buffer.Raw(), kBufferSize,
                                       buffer.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "buffer readback failed");

    ExpectBufferMatchesGolden(ctx, kGolden, bytes);
}
