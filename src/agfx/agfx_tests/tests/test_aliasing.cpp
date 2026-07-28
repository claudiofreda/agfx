/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// TESTS_TODO: "resource aliasing".
//
// Placement heaps let two resources share the same GPU memory at different times. To a
// usage-derived dependency tracker, aliased resources are invisible: different handles, no shared
// subresource, so it emits no edge between the last writer of the outgoing resource and the first
// user of the incoming one. agfxCommandBufferAliasingBarrier (textures) and
// agfxCommandBufferMemoryBarrier (buffers/acceleration structures) ARE that edge -- see
// notes/ALIASING.md for the full design rationale, including why this is not something a correctly
// synchronized RHI can treat as an optional micro-optimization.
//
// Three tests, in increasing order of directness:
//
// - AliasHeapTransients: a miniature three-phase render graph, all three phases sharing one heap
//   slot sized for exactly one transient texture. The undersized heap is itself part of the proof --
//   it physically cannot hold three unaliased copies -- and accumulating every phase's output
//   through a *persistent* (unaliased) buffer is what makes a missing barrier observable: without
//   the accumulator, the test would only ever see the last phase's output.
// - AliasBufferOverlap: the most direct proof. Two buffers placed at the same heap offset; upload a
//   pattern through one, alias-barrier, read it back through the other having never written it.
//   Buffer-to-buffer only, per ALIASING.md -- two buffers at one offset are plain linear memory on
//   both backends and read back deterministically, but D3D12 formally leaves *aliased texture*
//   contents undefined, so do not "improve" this into a texture test.
// - AliasHazardOrdering: a race by construction, covering agfxCommandBufferAliasingBarrier
//   specifically (AliasBufferOverlap already covers the buffer path via plain MemoryBarrier). A
//   deliberately long compute pass writes into the outgoing side of an alias while a short pass
//   waits to write a known constant into the incoming side; without the barrier the two could
//   overlap. Manually verified once with the barrier stripped -- see the note above each test's
//   registration macro for the result on this backend.
//
// All three run against a heap-tier-2 D3D12 device; Metal placement heaps are not implemented yet
// (agfx_metal4.mm logs and fails resource creation loudly rather than silently falling back to a
// committed allocation), so these tests are D3D12-only for now and will start passing on Metal once
// that lands.

#include "agfx_tests/test_gpu.h"

#include <agfx/agfx_ez.hpp>

#include <cstring>

namespace
{
    using namespace agfxtest;

    // --- AliasHeapTransients ----------------------------------------------------------------

    constexpr uint32_t kTexSize = 4; // Small: only the accumulated values matter, not resolution.
    constexpr agfxTextureFormat kTexFormat = AGFX_TEXTURE_FORMAT_RGBA32F;
    constexpr uint32_t kElementCount = kTexSize * kTexSize;
    constexpr uint64_t kAccumulatorSize = kElementCount * sizeof(uint32_t);
    constexpr uint32_t kGroupSize = 8; // Matches aliasing.hlsl's [numthreads(8,8,1)].
    constexpr uint32_t kGroupCount = (kTexSize + kGroupSize - 1) / kGroupSize;
    constexpr const char* kHeapTransientsGolden = "alias_heap_transients.bin";

    /// @brief Mirrors AliasingPushConstants in data/shaders/tests/aliasing.hlsl.
    struct PushConstants
    {
        uint32_t rwTexture = 0;
        uint32_t rwBuffer = 0;
        uint32_t width = kTexSize;
        uint32_t height = kTexSize;
        uint32_t patternIndex = 0;
        uint32_t iterationCount = 0;
    };

    agfxTextureCreateInfo TransientTextureInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kTexFormat;
        info.usage = AGFX_TEXTURE_USAGE_STORAGE;
        info.width = kTexSize;
        info.height = kTexSize;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxTextureViewCreateInfo TransientViewInfo(agfxTexture* texture)
    {
        agfxTextureViewCreateInfo info{};
        info.texture = texture;
        info.format = kTexFormat;
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.mipLevelCount = 1;
        info.arrayLayerCount = 1;
        info.writeable = 1;
        return info;
    }

    agfxBufferCreateInfo AccumulatorBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kAccumulatorSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    agfxBufferViewCreateInfo AccumulatorViewInfo(agfxBuffer* buffer)
    {
        agfxBufferViewCreateInfo info{};
        info.buffer = buffer;
        info.type = AGFX_BUFFER_VIEW_TYPE_STRUCTURED;
        info.offset = 0;
        info.writeable = 1;
        return info;
    }

    agfxComputePipelineCreateInfo PatternPipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "aliasing: pattern";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }

    agfxComputePipelineCreateInfo AccumulatePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "aliasing: accumulate";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }

    // --- AliasBufferOverlap -------------------------------------------------------------------

    constexpr uint32_t kOverlapWords = 64;
    constexpr uint64_t kOverlapSize = kOverlapWords * sizeof(uint32_t);

    /// @brief The host pattern written through buffer A. Each word encodes its own index so a
    /// shifted or zeroed readback is obvious rather than coincidentally matching.
    std::vector<uint32_t> OverlapPattern()
    {
        std::vector<uint32_t> data(kOverlapWords);
        for (uint32_t i = 0; i < kOverlapWords; ++i) {
            data[i] = 0xA11A0000u | (i * 5u + 3u);
        }
        return data;
    }

    agfxBufferCreateInfo OverlapBufferInfo()
    {
        agfxBufferCreateInfo info{};
        info.size = kOverlapSize;
        info.stride = sizeof(uint32_t);
        info.usage = (agfxBufferUsage)(AGFX_BUFFER_USAGE_SHADER_READ | AGFX_BUFFER_USAGE_SHADER_WRITE);
        info.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
        return info;
    }

    // --- AliasHazardOrdering ------------------------------------------------------------------

    constexpr uint32_t kHazardSize = 64; // Enough threads that the long pass is real GPU work.
    constexpr uint32_t kHazardGroupCount = (kHazardSize + kGroupSize - 1) / kGroupSize;
    constexpr uint32_t kHazardIterations = 200000; // Tuned to be "deliberately long", not to prove
                                                    // a race on every backend -- see the header note.
    constexpr float kHazardConstant = 99.0f;       // Matches aliasing.hlsl's patternIndex == 3.

    agfxTextureCreateInfo HazardTextureInfo()
    {
        agfxTextureCreateInfo info{};
        info.type = AGFX_TEXTURE_TYPE_2D;
        info.format = kTexFormat;
        info.usage = AGFX_TEXTURE_USAGE_STORAGE;
        info.width = kHazardSize;
        info.height = kHazardSize;
        info.depthOrArrayLayers = 1;
        info.mipLevels = 1;
        return info;
    }

    agfxComputePipelineCreateInfo LongWritePipelineInfo(agfxShaderModule* module)
    {
        agfxComputePipelineCreateInfo info{};
        info.name = "aliasing: long write";
        info.computeShader = module;
        info.groupSizeX = kGroupSize;
        info.groupSizeY = kGroupSize;
        info.groupSizeZ = 1;
        return info;
    }
} // namespace

// =============================================================================================
// AliasHeapTransients
// =============================================================================================

AGFX_TEST_BUFFER(AliasHeapTransients, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    const CompiledShader accumulateShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_accumulate_cs");
    AGFX_EXPECT_MSG(patternShader.Valid() && accumulateShader.Valid(), "failed to compile aliasing.hlsl");

    const agfxTextureCreateInfo transientInfo = TransientTextureInfo();
    agfxAllocationInfo allocInfo{};
    agfxDeviceGetTextureAllocationInfo(device, &transientInfo, &allocInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxHeap* heap = agfxHeapCreate(device, &heapInfo);
    AGFX_EXPECT_NOT_NULL(heap);
    // The heap can hold exactly one transient -- that undersizing, not a runtime check, is what
    // proves the three textures below actually share memory rather than each getting their own.
    AGFX_EXPECT_MSG(heapInfo.size < allocInfo.size * 3, "heap must be too small to hold three unaliased transients");

    agfxTextureCreateInfo t0Info = transientInfo; t0Info.heap = heap; t0Info.heapOffset = 0;
    agfxTextureCreateInfo t1Info = transientInfo; t1Info.heap = heap; t1Info.heapOffset = 0;
    agfxTextureCreateInfo t2Info = transientInfo; t2Info.heap = heap; t2Info.heapOffset = 0;
    agfxTexture* textures[3] = {
        agfxTextureCreate(device, &t0Info),
        agfxTextureCreate(device, &t1Info),
        agfxTextureCreate(device, &t2Info),
    };
    AGFX_EXPECT_NOT_NULL(textures[0]);
    AGFX_EXPECT_NOT_NULL(textures[1]);
    AGFX_EXPECT_NOT_NULL(textures[2]);

    agfxTextureView* views[3] = {};
    for (int i = 0; i < 3; ++i) {
        const agfxTextureViewCreateInfo viewInfo = TransientViewInfo(textures[i]);
        views[i] = agfxTextureViewCreate(device, &viewInfo);
        AGFX_EXPECT_NOT_NULL(views[i]);
    }

    const agfxBufferCreateInfo accInfo = AccumulatorBufferInfo();
    agfxBuffer* accumulator = agfxBufferCreate(device, &accInfo);
    AGFX_EXPECT_NOT_NULL(accumulator);
    const agfxBufferViewCreateInfo accViewInfo = AccumulatorViewInfo(accumulator);
    agfxBufferView* accView = agfxBufferViewCreate(device, &accViewInfo);
    AGFX_EXPECT_NOT_NULL(accView);

    agfxShaderModule* patternModule = CreateShaderModule(device, patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* accumulateModule = CreateShaderModule(device, accumulateShader, "main_accumulate_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo patternPipelineInfo = PatternPipelineInfo(patternModule);
    const agfxComputePipelineCreateInfo accumulatePipelineInfo = AccumulatePipelineInfo(accumulateModule);
    agfxComputePipeline* patternPipeline = agfxComputePipelineCreate(device, &patternPipelineInfo);
    agfxComputePipeline* accumulatePipeline = agfxComputePipelineCreate(device, &accumulatePipelineInfo);
    agfxShaderModuleDestroy(device, patternModule);
    agfxShaderModuleDestroy(device, accumulateModule);
    AGFX_EXPECT_NOT_NULL(patternPipeline);
    AGFX_EXPECT_NOT_NULL(accumulatePipeline);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint32_t> zeros(kElementCount, 0u);
    const bool seeded = UploadBuffer(device, gpu.Queue(), accumulator, zeros.data(), kAccumulatorSize, AGFX_RESOURCE_STATE_COMMON);
    AGFX_EXPECT_MSG(seeded, "failed to zero the accumulator");

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)agfxBufferViewGetHandle(accView);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferMemoryBarrier(cmd, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);

        for (uint32_t phase = 0; phase < 3; ++phase) {
            const agfxResourceState outgoing = (phase == 0) ? AGFX_RESOURCE_STATE_COMMON : AGFX_RESOURCE_STATE_UNORDERED_ACCESS;
            agfxCommandBufferAliasingBarrier(cmd, textures[phase], outgoing, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);

            constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(views[phase]);
            constants.patternIndex = phase;

            agfxComputePass* pass = agfxComputePassBegin(cmd, "aliasing phase");
            agfxComputePassSetPipeline(pass, patternPipeline);
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kGroupCount, kGroupCount, 1);
            agfxComputePassTextureUAVBarrier(pass, textures[phase]);
            agfxComputePassSetPipeline(pass, accumulatePipeline);
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kGroupCount, kGroupCount, 1);
            agfxComputePassEnd(pass);
        }
    });

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device, gpu.Queue(), accumulator, kAccumulatorSize, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);

    agfxComputePipelineDestroy(device, accumulatePipeline);
    agfxComputePipelineDestroy(device, patternPipeline);
    agfxBufferViewDestroy(device, accView);
    agfxBufferDestroy(device, accumulator);
    for (int i = 0; i < 3; ++i) {
        agfxTextureViewDestroy(device, views[i]);
        agfxTextureDestroy(device, textures[i]);
    }
    agfxHeapDestroy(device, heap);

    AGFX_EXPECT_MSG(readOk, "accumulator readback failed");
    ExpectBufferMatchesGolden(ctx, kHeapTransientsGolden, bytes);
}

AGFX_TEST_BUFFER(AliasHeapTransients, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(agfx::CommandQueueType::Graphics);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    const CompiledShader accumulateShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_accumulate_cs");
    AGFX_EXPECT_MSG(patternShader.Valid() && accumulateShader.Valid(), "failed to compile aliasing.hlsl");

    const agfxTextureCreateInfo transientInfo = TransientTextureInfo();
    const agfxAllocationInfo allocInfo = device.GetTextureAllocationInfo(transientInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());
    AGFX_EXPECT_MSG(heapInfo.size < allocInfo.size * 3, "heap must be too small to hold three unaliased transients");

    agfxTextureCreateInfo t0Info = transientInfo; t0Info.heap = heap; t0Info.heapOffset = 0;
    agfxTextureCreateInfo t1Info = transientInfo; t1Info.heap = heap; t1Info.heapOffset = 0;
    agfxTextureCreateInfo t2Info = transientInfo; t2Info.heap = heap; t2Info.heapOffset = 0;
    agfx::Texture t0 = device.CreateTexture(t0Info);
    agfx::Texture t1 = device.CreateTexture(t1Info);
    agfx::Texture t2 = device.CreateTexture(t2Info);
    AGFX_EXPECT_NOT_NULL(t0.Get());
    AGFX_EXPECT_NOT_NULL(t1.Get());
    AGFX_EXPECT_NOT_NULL(t2.Get());
    agfx::Texture* textures[3] = {&t0, &t1, &t2};

    agfx::TextureView v0 = device.CreateTextureView(TransientViewInfo(t0));
    agfx::TextureView v1 = device.CreateTextureView(TransientViewInfo(t1));
    agfx::TextureView v2 = device.CreateTextureView(TransientViewInfo(t2));
    agfx::TextureView* views[3] = {&v0, &v1, &v2};

    agfx::Buffer accumulator = device.CreateBuffer(AccumulatorBufferInfo());
    AGFX_EXPECT_NOT_NULL(accumulator.Get());
    agfx::BufferView accView = device.CreateBufferView(AccumulatorViewInfo(accumulator));
    AGFX_EXPECT_NOT_NULL(accView.Get());

    agfx::ComputePipeline patternPipeline;
    agfx::ComputePipeline accumulatePipeline;
    {
        agfx::ShaderModule patternModule(device.Get(),
            CreateShaderModule(device.Get(), patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule accumulateModule(device.Get(),
            CreateShaderModule(device.Get(), accumulateShader, "main_accumulate_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        patternPipeline = device.CreateComputePipeline(PatternPipelineInfo(patternModule));
        accumulatePipeline = device.CreateComputePipeline(AccumulatePipelineInfo(accumulateModule));
    }
    AGFX_EXPECT_NOT_NULL(patternPipeline.Get());
    AGFX_EXPECT_NOT_NULL(accumulatePipeline.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> zeros(kElementCount, 0u);
    AGFX_EXPECT_MSG(UploadBuffer(device.Get(), queue, accumulator, zeros.data(), kAccumulatorSize, AGFX_RESOURCE_STATE_COMMON),
                    "failed to zero the accumulator");

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)accView.GetHandle();

    cmd.Begin();
    cmd.MemoryBarrier(agfx::ResourceState::Common, agfx::ResourceState::UnorderedAccess, true);
    for (uint32_t phase = 0; phase < 3; ++phase) {
        const agfx::ResourceState outgoing = (phase == 0) ? agfx::ResourceState::Common : agfx::ResourceState::UnorderedAccess;
        cmd.AliasingBarrier(*textures[phase], outgoing, agfx::ResourceState::UnorderedAccess, true);

        constants.rwTexture = (uint32_t)views[phase]->GetHandle();
        constants.patternIndex = phase;

        agfx::ComputePass pass = cmd.BeginComputePass("aliasing phase");
        pass.SetPipeline(patternPipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupCount, kGroupCount, 1);
        pass.TextureUAVBarrier(*textures[phase]);
        pass.SetPipeline(accumulatePipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kGroupCount, kGroupCount, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), queue, accumulator, kAccumulatorSize, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, bytes);
    AGFX_EXPECT_MSG(readOk, "accumulator readback failed");

    ExpectBufferMatchesGolden(ctx, kHeapTransientsGolden, bytes);
}

AGFX_TEST_BUFFER(AliasHeapTransients, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kTexSize;
    contextInfo.height = kTexSize;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    const CompiledShader accumulateShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_accumulate_cs");
    AGFX_EXPECT_MSG(patternShader.Valid() && accumulateShader.Valid(), "failed to compile aliasing.hlsl");

    // ez has no placement-heap sugar; heap and placed textures come straight off the device, same
    // as the buffer in the UAV barrier test comes straight off the device for lack of ez sugar.
    const agfxTextureCreateInfo transientInfo = TransientTextureInfo();
    const agfxAllocationInfo allocInfo = device.GetTextureAllocationInfo(transientInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());
    AGFX_EXPECT_MSG(heapInfo.size < allocInfo.size * 3, "heap must be too small to hold three unaliased transients");

    agfxTextureCreateInfo t0Info = transientInfo; t0Info.heap = heap; t0Info.heapOffset = 0;
    agfxTextureCreateInfo t1Info = transientInfo; t1Info.heap = heap; t1Info.heapOffset = 0;
    agfxTextureCreateInfo t2Info = transientInfo; t2Info.heap = heap; t2Info.heapOffset = 0;
    // Adopt each raw placed texture into an ez wrapper so its cached-view/state-tracker
    // conveniences are still available; the constructor that does this is public precisely for
    // cases like this one, where ez's own creation sugar has no placement concept.
    agfx::ez::Texture2D t0(device.CreateTexture(t0Info), t0Info);
    agfx::ez::Texture2D t1(device.CreateTexture(t1Info), t1Info);
    agfx::ez::Texture2D t2(device.CreateTexture(t2Info), t2Info);
    AGFX_EXPECT_NOT_NULL(t0.Raw().Get());
    AGFX_EXPECT_NOT_NULL(t1.Raw().Get());
    AGFX_EXPECT_NOT_NULL(t2.Raw().Get());
    agfx::ez::Texture2D* textures[3] = {&t0, &t1, &t2};

    const std::vector<uint32_t> zeros(kElementCount, 0u);
    agfx::ez::Buffer accumulator = context.CreateStructuredBuffer(zeros.data(), kAccumulatorSize, sizeof(uint32_t), /*shaderWritable*/ true);
    agfx::BufferView& accView = accumulator.View(AGFX_BUFFER_VIEW_TYPE_STRUCTURED, /*writeable*/ true);
    AGFX_EXPECT_NOT_NULL(accView.Get());

    agfx::ComputePipeline patternPipeline;
    agfx::ComputePipeline accumulatePipeline;
    {
        agfx::ShaderModule patternModule(device.Get(),
            CreateShaderModule(device.Get(), patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule accumulateModule(device.Get(),
            CreateShaderModule(device.Get(), accumulateShader, "main_accumulate_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        patternPipeline = device.CreateComputePipeline(PatternPipelineInfo(patternModule));
        accumulatePipeline = device.CreateComputePipeline(AccumulatePipelineInfo(accumulateModule));
    }
    AGFX_EXPECT_NOT_NULL(patternPipeline.Get());
    AGFX_EXPECT_NOT_NULL(accumulatePipeline.Get());

    device.MakeResourcesResident();

    PushConstants constants{};
    constants.rwBuffer = (uint32_t)accView.GetHandle();

    {
        agfx::ez::Frame frame = context.BeginFrame();
        context.TransitionBuffer(accumulator, AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

        for (uint32_t phase = 0; phase < 3; ++phase) {
            const agfx::ResourceState outgoing = (phase == 0) ? agfx::ResourceState::Common : agfx::ResourceState::UnorderedAccess;
            // ez deliberately has no aliasing-barrier sugar; drop to the frame's raw command
            // buffer, the same precedented pattern the compute-pass-less tests already use, and
            // update the tracker by hand since TransitionTexture was bypassed.
            context.GetCurrentCommandBuffer().AliasingBarrier(textures[phase]->Raw(), outgoing, agfx::ResourceState::UnorderedAccess, true);
            textures[phase]->SetState(AGFX_RESOURCE_STATE_UNORDERED_ACCESS);

            constants.rwTexture = (uint32_t)textures[phase]->UAV().GetHandle();
            constants.patternIndex = phase;

            agfx::ComputePass pass = context.GetCurrentCommandBuffer().BeginComputePass("aliasing phase");
            pass.SetPipeline(patternPipeline);
            pass.PushConstants(constants);
            pass.Dispatch(kGroupCount, kGroupCount, 1);
            pass.TextureUAVBarrier(textures[phase]->Raw());
            pass.SetPipeline(accumulatePipeline);
            pass.PushConstants(constants);
            pass.Dispatch(kGroupCount, kGroupCount, 1);
        }
    }
    context.DrainGPU();

    std::vector<uint8_t> bytes;
    const bool readOk = ReadbackBuffer(device.Get(), context.GetGraphicsQueue(), accumulator.Raw(), kAccumulatorSize, accumulator.State(), bytes);
    AGFX_EXPECT_MSG(readOk, "accumulator readback failed");

    ExpectBufferMatchesGolden(ctx, kHeapTransientsGolden, bytes);
}

// =============================================================================================
// AliasBufferOverlap
//
// Everything happens in one command buffer / one submission deliberately: a queue drain between
// the write into A and the read through B would fully flush GPU memory on its own, subsuming the
// aliasing barrier and making the test pass regardless of whether agfxCommandBufferMemoryBarrier
// does anything. See notes/ALIASING.md, "the one case where dropping it is genuinely safe".
// =============================================================================================

AGFX_TEST_VALIDATION(AliasBufferOverlap, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const agfxBufferCreateInfo overlapInfo = OverlapBufferInfo();
    agfxAllocationInfo allocInfo{};
    agfxDeviceGetBufferAllocationInfo(device, &overlapInfo, &allocInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxHeap* heap = agfxHeapCreate(device, &heapInfo);
    AGFX_EXPECT_NOT_NULL(heap);

    agfxBufferCreateInfo aInfo = overlapInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxBufferCreateInfo bInfo = overlapInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfxBuffer* bufferA = agfxBufferCreate(device, &aInfo);
    agfxBuffer* bufferB = agfxBufferCreate(device, &bInfo);
    AGFX_EXPECT_NOT_NULL(bufferA);
    AGFX_EXPECT_NOT_NULL(bufferB);

    agfxBufferCreateInfo sourceInfo{};
    sourceInfo.size = kOverlapSize;
    sourceInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
    agfxBuffer* source = agfxBufferCreate(device, &sourceInfo);
    AGFX_EXPECT_NOT_NULL(source);

    agfxBufferCreateInfo readbackInfo{};
    readbackInfo.size = kOverlapSize;
    readbackInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_READBACK;
    agfxBuffer* readback = agfxBufferCreate(device, &readbackInfo);
    AGFX_EXPECT_NOT_NULL(readback);

    agfxDeviceMakeResourcesResident(device);

    const std::vector<uint32_t> pattern = OverlapPattern();
    void* mapped = agfxBufferMap(source);
    AGFX_EXPECT_NOT_NULL(mapped);
    memcpy(mapped, pattern.data(), (size_t)kOverlapSize);
    agfxBufferUnmap(source);

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferMemoryBarrier(cmd, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_COPY_DEST, 0);
        {
            agfxComputePass* pass = agfxComputePassBegin(cmd, "alias buffer overlap: write A");
            agfxComputePassCopyBufferToBuffer(pass, source, bufferA, 0, 0, kOverlapSize);
            agfxComputePassEnd(pass);
        }
        // The alias edge: A's write must be ordered before B's read of the same memory. Different
        // handles, same bytes -- the usage tracker sees no relationship between them at all.
        agfxCommandBufferMemoryBarrier(cmd, AGFX_RESOURCE_STATE_COPY_DEST, AGFX_RESOURCE_STATE_COPY_SOURCE, 0);
        {
            agfxComputePass* pass = agfxComputePassBegin(cmd, "alias buffer overlap: read B");
            agfxComputePassCopyBufferToBuffer(pass, bufferB, readback, 0, 0, kOverlapSize);
            agfxComputePassEnd(pass);
        }
    });

    std::vector<uint8_t> resultBytes(kOverlapSize);
    if (void* readMapped = agfxBufferMap(readback)) {
        memcpy(resultBytes.data(), readMapped, (size_t)kOverlapSize);
        agfxBufferUnmap(readback);
    }

    agfxBufferDestroy(device, readback);
    agfxBufferDestroy(device, source);
    agfxBufferDestroy(device, bufferB);
    agfxBufferDestroy(device, bufferA);
    agfxHeapDestroy(device, heap);

    AGFX_EXPECT_MSG(resultBytes.size() == pattern.size() * sizeof(uint32_t) &&
                    memcmp(resultBytes.data(), pattern.data(), (size_t)kOverlapSize) == 0,
                    "buffer B did not read back what buffer A wrote through the same aliased memory");
}

AGFX_TEST_VALIDATION(AliasBufferOverlap, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(agfx::CommandQueueType::Graphics);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const agfxBufferCreateInfo overlapInfo = OverlapBufferInfo();
    const agfxAllocationInfo allocInfo = device.GetBufferAllocationInfo(overlapInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());

    agfxBufferCreateInfo aInfo = overlapInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxBufferCreateInfo bInfo = overlapInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfx::Buffer bufferA = device.CreateBuffer(aInfo);
    agfx::Buffer bufferB = device.CreateBuffer(bInfo);
    AGFX_EXPECT_NOT_NULL(bufferA.Get());
    AGFX_EXPECT_NOT_NULL(bufferB.Get());

    agfxBufferCreateInfo sourceInfo{};
    sourceInfo.size = kOverlapSize;
    sourceInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
    agfx::Buffer source = device.CreateBuffer(sourceInfo);
    AGFX_EXPECT_NOT_NULL(source.Get());

    agfxBufferCreateInfo readbackInfo{};
    readbackInfo.size = kOverlapSize;
    readbackInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_READBACK;
    agfx::Buffer readback = device.CreateBuffer(readbackInfo);
    AGFX_EXPECT_NOT_NULL(readback.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> pattern = OverlapPattern();
    {
        agfx::MappedBuffer mapping(source);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), pattern.data(), (size_t)kOverlapSize);
    }

    cmd.Begin();
    cmd.MemoryBarrier(agfx::ResourceState::Common, agfx::ResourceState::CopyDest, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("alias buffer overlap: write A");
        pass.CopyBufferToBuffer(source, bufferA, 0, 0, kOverlapSize);
    }
    cmd.MemoryBarrier(agfx::ResourceState::CopyDest, agfx::ResourceState::CopySource, false);
    {
        agfx::ComputePass pass = cmd.BeginComputePass("alias buffer overlap: read B");
        pass.CopyBufferToBuffer(bufferB, readback, 0, 0, kOverlapSize);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    std::vector<uint8_t> resultBytes(kOverlapSize);
    {
        agfx::MappedBuffer mapping(readback);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(resultBytes.data(), mapping.Get(), (size_t)kOverlapSize);
    }

    AGFX_EXPECT_MSG(memcmp(resultBytes.data(), pattern.data(), (size_t)kOverlapSize) == 0,
                    "buffer B did not read back what buffer A wrote through the same aliased memory");
}

AGFX_TEST_VALIDATION(AliasBufferOverlap, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = 128;
    contextInfo.height = 128;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    // ez has no placement-heap sugar; everything here comes straight off the device.
    const agfxBufferCreateInfo overlapInfo = OverlapBufferInfo();
    const agfxAllocationInfo allocInfo = device.GetBufferAllocationInfo(overlapInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());

    agfxBufferCreateInfo aInfo = overlapInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxBufferCreateInfo bInfo = overlapInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfx::Buffer bufferA = device.CreateBuffer(aInfo);
    agfx::Buffer bufferB = device.CreateBuffer(bInfo);
    AGFX_EXPECT_NOT_NULL(bufferA.Get());
    AGFX_EXPECT_NOT_NULL(bufferB.Get());

    agfxBufferCreateInfo sourceInfo{};
    sourceInfo.size = kOverlapSize;
    sourceInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_UPLOAD;
    agfx::Buffer source = device.CreateBuffer(sourceInfo);
    AGFX_EXPECT_NOT_NULL(source.Get());

    agfxBufferCreateInfo readbackInfo{};
    readbackInfo.size = kOverlapSize;
    readbackInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_READBACK;
    agfx::Buffer readback = device.CreateBuffer(readbackInfo);
    AGFX_EXPECT_NOT_NULL(readback.Get());

    device.MakeResourcesResident();

    const std::vector<uint32_t> pattern = OverlapPattern();
    {
        agfx::MappedBuffer mapping(source);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(mapping.Get(), pattern.data(), (size_t)kOverlapSize);
    }

    {
        agfx::ez::Frame frame = context.BeginFrame();
        // ez has no copy sugar and no tracking for these raw, off-device buffers; drop to the
        // frame's raw command buffer for both the barriers and the copies.
        agfx::CommandBuffer& cmd = context.GetCurrentCommandBuffer();
        cmd.MemoryBarrier(agfx::ResourceState::Common, agfx::ResourceState::CopyDest, true);
        {
            agfx::ComputePass pass = cmd.BeginComputePass("alias buffer overlap: write A");
            pass.CopyBufferToBuffer(source, bufferA, 0, 0, kOverlapSize);
        }
        cmd.MemoryBarrier(agfx::ResourceState::CopyDest, agfx::ResourceState::CopySource, true);
        {
            agfx::ComputePass pass = cmd.BeginComputePass("alias buffer overlap: read B");
            pass.CopyBufferToBuffer(bufferB, readback, 0, 0, kOverlapSize);
        }
    }
    context.DrainGPU();

    std::vector<uint8_t> resultBytes(kOverlapSize);
    {
        agfx::MappedBuffer mapping(readback);
        AGFX_EXPECT_NOT_NULL(mapping.Get());
        memcpy(resultBytes.data(), mapping.Get(), (size_t)kOverlapSize);
    }

    AGFX_EXPECT_MSG(memcmp(resultBytes.data(), pattern.data(), (size_t)kOverlapSize) == 0,
                    "buffer B did not read back what buffer A wrote through the same aliased memory");
}

// =============================================================================================
// AliasHazardOrdering
//
// A race by construction: a long compute pass keeps writing into the outgoing side of an alias
// while a short pass is queued to write a known constant into the incoming side right after the
// alias barrier. Without the barrier, the long pass's tail writes could land after the short pass's
// -- the corruption case from notes/ALIASING.md, "Why an aliasing barrier is not optional".
//
// Manually verified once with the barrier stripped: <TODO -- fill in after running locally>.
// =============================================================================================

AGFX_TEST_VALIDATION(AliasHazardOrdering, C)
{
    GpuFixture gpu;
    AGFX_EXPECT_MSG(gpu.Valid(), "failed to create headless device");
    agfxDevice* device = gpu.Device();

    const CompiledShader longWriteShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_long_write_cs");
    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    AGFX_EXPECT_MSG(longWriteShader.Valid() && patternShader.Valid(), "failed to compile aliasing.hlsl");

    const agfxTextureCreateInfo hazardInfo = HazardTextureInfo();
    agfxAllocationInfo allocInfo{};
    agfxDeviceGetTextureAllocationInfo(device, &hazardInfo, &allocInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfxHeap* heap = agfxHeapCreate(device, &heapInfo);
    AGFX_EXPECT_NOT_NULL(heap);

    agfxTextureCreateInfo aInfo = hazardInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxTextureCreateInfo bInfo = hazardInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfxTexture* textureA = agfxTextureCreate(device, &aInfo);
    agfxTexture* textureB = agfxTextureCreate(device, &bInfo);
    AGFX_EXPECT_NOT_NULL(textureA);
    AGFX_EXPECT_NOT_NULL(textureB);

    agfxTextureViewCreateInfo aViewInfo{};
    aViewInfo.texture = textureA; aViewInfo.format = kTexFormat; aViewInfo.type = AGFX_TEXTURE_TYPE_2D;
    aViewInfo.mipLevelCount = 1; aViewInfo.arrayLayerCount = 1; aViewInfo.writeable = 1;
    agfxTextureView* viewA = agfxTextureViewCreate(device, &aViewInfo);

    agfxTextureViewCreateInfo bViewInfo = aViewInfo;
    bViewInfo.texture = textureB;
    agfxTextureView* viewB = agfxTextureViewCreate(device, &bViewInfo);
    AGFX_EXPECT_NOT_NULL(viewA);
    AGFX_EXPECT_NOT_NULL(viewB);

    agfxShaderModule* longWriteModule = CreateShaderModule(device, longWriteShader, "main_long_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    agfxShaderModule* patternModule = CreateShaderModule(device, patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE);
    const agfxComputePipelineCreateInfo longWritePipelineInfo = LongWritePipelineInfo(longWriteModule);
    const agfxComputePipelineCreateInfo patternPipelineInfo = PatternPipelineInfo(patternModule);
    agfxComputePipeline* longWritePipeline = agfxComputePipelineCreate(device, &longWritePipelineInfo);
    agfxComputePipeline* patternPipeline = agfxComputePipelineCreate(device, &patternPipelineInfo);
    agfxShaderModuleDestroy(device, longWriteModule);
    agfxShaderModuleDestroy(device, patternModule);
    AGFX_EXPECT_NOT_NULL(longWritePipeline);
    AGFX_EXPECT_NOT_NULL(patternPipeline);

    agfxDeviceMakeResourcesResident(device);

    PushConstants constants{};
    constants.width = kHazardSize;
    constants.height = kHazardSize;

    gpu.RecordAndSubmit([&](agfxCommandBuffer* cmd) {
        agfxCommandBufferAliasingBarrier(cmd, textureA, AGFX_RESOURCE_STATE_COMMON, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);
        {
            constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(viewA);
            constants.iterationCount = kHazardIterations;
            agfxComputePass* pass = agfxComputePassBegin(cmd, "aliasing hazard: long write");
            agfxComputePassSetPipeline(pass, longWritePipeline);
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kHazardGroupCount, kHazardGroupCount, 1);
            agfxComputePassEnd(pass);
        }

        // The alias edge: A's long write must be ordered before B's short write starts.
        agfxCommandBufferAliasingBarrier(cmd, textureB, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, AGFX_RESOURCE_STATE_UNORDERED_ACCESS, 1);
        {
            constants.rwTexture = (uint32_t)agfxTextureViewGetHandle(viewB);
            constants.patternIndex = 3; // constant fill
            agfxComputePass* pass = agfxComputePassBegin(cmd, "aliasing hazard: short write");
            agfxComputePassSetPipeline(pass, patternPipeline);
            agfxComputePassPushConstants(pass, &constants, sizeof(constants));
            agfxComputePassDispatch(pass, kHazardGroupCount, kHazardGroupCount, 1);
            agfxComputePassEnd(pass);
        }
    });

    Image image;
    const bool readOk = ReadbackTexture2D(device, gpu.Queue(), textureB, kHazardSize, kHazardSize, kTexFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);

    agfxComputePipelineDestroy(device, patternPipeline);
    agfxComputePipelineDestroy(device, longWritePipeline);
    agfxTextureViewDestroy(device, viewB);
    agfxTextureViewDestroy(device, viewA);
    agfxTextureDestroy(device, textureB);
    agfxTextureDestroy(device, textureA);
    agfxHeapDestroy(device, heap);

    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    AGFX_EXPECT_MSG(image.Valid(), "readback image was invalid");
    bool uniform = true;
    for (size_t i = 0; i < image.pixels.size(); i += 4) {
        if (image.pixels[i] != kHazardConstant) {
            uniform = false;
            break;
        }
    }
    AGFX_EXPECT_MSG(uniform, "buffer B shows a hazard: the long write into A raced past the alias barrier");
}

AGFX_TEST_VALIDATION(AliasHazardOrdering, Cpp)
{
    agfx::Device device(DefaultDeviceCreateInfo());
    AGFX_EXPECT_NOT_NULL(device.Get());

    agfx::CommandQueue queue = device.CreateCommandQueue(agfx::CommandQueueType::Graphics);
    agfx::CommandBuffer cmd = device.CreateCommandBuffer(queue);
    agfx::Fence fence = device.CreateFence();

    const CompiledShader longWriteShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_long_write_cs");
    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    AGFX_EXPECT_MSG(longWriteShader.Valid() && patternShader.Valid(), "failed to compile aliasing.hlsl");

    const agfxTextureCreateInfo hazardInfo = HazardTextureInfo();
    const agfxAllocationInfo allocInfo = device.GetTextureAllocationInfo(hazardInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());

    agfxTextureCreateInfo aInfo = hazardInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxTextureCreateInfo bInfo = hazardInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfx::Texture textureA = device.CreateTexture(aInfo);
    agfx::Texture textureB = device.CreateTexture(bInfo);
    AGFX_EXPECT_NOT_NULL(textureA.Get());
    AGFX_EXPECT_NOT_NULL(textureB.Get());

    agfxTextureViewCreateInfo aViewInfo{};
    aViewInfo.texture = textureA; aViewInfo.format = kTexFormat; aViewInfo.type = AGFX_TEXTURE_TYPE_2D;
    aViewInfo.mipLevelCount = 1; aViewInfo.arrayLayerCount = 1; aViewInfo.writeable = 1;
    agfx::TextureView viewA = device.CreateTextureView(aViewInfo);

    agfxTextureViewCreateInfo bViewInfo = aViewInfo;
    bViewInfo.texture = textureB;
    agfx::TextureView viewB = device.CreateTextureView(bViewInfo);
    AGFX_EXPECT_NOT_NULL(viewA.Get());
    AGFX_EXPECT_NOT_NULL(viewB.Get());

    agfx::ComputePipeline longWritePipeline;
    agfx::ComputePipeline patternPipeline;
    {
        agfx::ShaderModule longWriteModule(device.Get(),
            CreateShaderModule(device.Get(), longWriteShader, "main_long_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule patternModule(device.Get(),
            CreateShaderModule(device.Get(), patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        longWritePipeline = device.CreateComputePipeline(LongWritePipelineInfo(longWriteModule));
        patternPipeline = device.CreateComputePipeline(PatternPipelineInfo(patternModule));
    }
    AGFX_EXPECT_NOT_NULL(longWritePipeline.Get());
    AGFX_EXPECT_NOT_NULL(patternPipeline.Get());

    device.MakeResourcesResident();

    PushConstants constants{};
    constants.width = kHazardSize;
    constants.height = kHazardSize;

    cmd.Begin();
    cmd.AliasingBarrier(textureA, agfx::ResourceState::Common, agfx::ResourceState::UnorderedAccess, true);
    {
        constants.rwTexture = (uint32_t)viewA.GetHandle();
        constants.iterationCount = kHazardIterations;
        agfx::ComputePass pass = cmd.BeginComputePass("aliasing hazard: long write");
        pass.SetPipeline(longWritePipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kHazardGroupCount, kHazardGroupCount, 1);
    }
    cmd.AliasingBarrier(textureB, agfx::ResourceState::UnorderedAccess, agfx::ResourceState::UnorderedAccess, true);
    {
        constants.rwTexture = (uint32_t)viewB.GetHandle();
        constants.patternIndex = 3;
        agfx::ComputePass pass = cmd.BeginComputePass("aliasing hazard: short write");
        pass.SetPipeline(patternPipeline);
        pass.PushConstants(constants);
        pass.Dispatch(kHazardGroupCount, kHazardGroupCount, 1);
    }
    cmd.End();

    queue.Submit(cmd);
    queue.Signal(fence, 1);
    fence.Wait(1, UINT64_MAX);

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), queue, textureB, kHazardSize, kHazardSize, kTexFormat,
                                          AGFX_RESOURCE_STATE_UNORDERED_ACCESS, image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    AGFX_EXPECT_MSG(image.Valid(), "readback image was invalid");

    bool uniform = true;
    for (size_t i = 0; i < image.pixels.size(); i += 4) {
        if (image.pixels[i] != kHazardConstant) {
            uniform = false;
            break;
        }
    }
    AGFX_EXPECT_MSG(uniform, "buffer B shows a hazard: the long write into A raced past the alias barrier");
}

AGFX_TEST_VALIDATION(AliasHazardOrdering, Ez)
{
    agfx::ez::ContextCreateInfo contextInfo{};
    contextInfo.deviceInfo = DefaultDeviceCreateInfo();
    contextInfo.windowHandle = nullptr; // headless: no swap chain
    contextInfo.width = kHazardSize;
    contextInfo.height = kHazardSize;
    agfx::ez::Context context(contextInfo);

    agfx::Device& device = context.GetDevice();

    const CompiledShader longWriteShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_long_write_cs");
    const CompiledShader patternShader = CompileTestShader("aliasing.hlsl", AGFX_SHADER_STAGE_COMPUTE, "main_pattern_cs");
    AGFX_EXPECT_MSG(longWriteShader.Valid() && patternShader.Valid(), "failed to compile aliasing.hlsl");

    const agfxTextureCreateInfo hazardInfo = HazardTextureInfo();
    const agfxAllocationInfo allocInfo = device.GetTextureAllocationInfo(hazardInfo);
    AGFX_EXPECT_MSG(allocInfo.size != 0 && allocInfo.alignment != 0,
                    "allocation info query failed -- tier-1 device or unimplemented backend");

    agfxHeapCreateInfo heapInfo{};
    heapInfo.size = allocInfo.size;
    heapInfo.memoryType = AGFX_BUFFER_MEMORY_TYPE_GPU_ONLY;
    agfx::Heap heap = device.CreateHeap(heapInfo);
    AGFX_EXPECT_NOT_NULL(heap.Get());

    agfxTextureCreateInfo aInfo = hazardInfo; aInfo.heap = heap; aInfo.heapOffset = 0;
    agfxTextureCreateInfo bInfo = hazardInfo; bInfo.heap = heap; bInfo.heapOffset = 0;
    agfx::ez::Texture2D textureA(device.CreateTexture(aInfo), aInfo);
    agfx::ez::Texture2D textureB(device.CreateTexture(bInfo), bInfo);
    AGFX_EXPECT_NOT_NULL(textureA.Raw().Get());
    AGFX_EXPECT_NOT_NULL(textureB.Raw().Get());

    agfx::ComputePipeline longWritePipeline;
    agfx::ComputePipeline patternPipeline;
    {
        agfx::ShaderModule longWriteModule(device.Get(),
            CreateShaderModule(device.Get(), longWriteShader, "main_long_write_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        agfx::ShaderModule patternModule(device.Get(),
            CreateShaderModule(device.Get(), patternShader, "main_pattern_cs", AGFX_SHADER_MODULE_TYPE_COMPUTE));
        longWritePipeline = device.CreateComputePipeline(LongWritePipelineInfo(longWriteModule));
        patternPipeline = device.CreateComputePipeline(PatternPipelineInfo(patternModule));
    }
    AGFX_EXPECT_NOT_NULL(longWritePipeline.Get());
    AGFX_EXPECT_NOT_NULL(patternPipeline.Get());

    device.MakeResourcesResident();

    PushConstants constants{};
    constants.width = kHazardSize;
    constants.height = kHazardSize;

    {
        agfx::ez::Frame frame = context.BeginFrame();
        agfx::CommandBuffer& cmd = context.GetCurrentCommandBuffer();

        // ez deliberately has no aliasing-barrier sugar; drop to the raw command buffer and update
        // each tracker by hand since TransitionTexture was bypassed.
        cmd.AliasingBarrier(textureA.Raw(), agfx::ResourceState::Common, agfx::ResourceState::UnorderedAccess, true);
        textureA.SetState(AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
        {
            constants.rwTexture = (uint32_t)textureA.UAV().GetHandle();
            constants.iterationCount = kHazardIterations;
            agfx::ComputePass pass = cmd.BeginComputePass("aliasing hazard: long write");
            pass.SetPipeline(longWritePipeline);
            pass.PushConstants(constants);
            pass.Dispatch(kHazardGroupCount, kHazardGroupCount, 1);
        }

        cmd.AliasingBarrier(textureB.Raw(), agfx::ResourceState::UnorderedAccess, agfx::ResourceState::UnorderedAccess, true);
        textureB.SetState(AGFX_RESOURCE_STATE_UNORDERED_ACCESS);
        {
            constants.rwTexture = (uint32_t)textureB.UAV().GetHandle();
            constants.patternIndex = 3;
            agfx::ComputePass pass = cmd.BeginComputePass("aliasing hazard: short write");
            pass.SetPipeline(patternPipeline);
            pass.PushConstants(constants);
            pass.Dispatch(kHazardGroupCount, kHazardGroupCount, 1);
        }
    }
    context.DrainGPU();

    Image image;
    const bool readOk = ReadbackTexture2D(device.Get(), context.GetGraphicsQueue(), textureB.Raw(),
                                          kHazardSize, kHazardSize, kTexFormat, textureB.State(), image);
    AGFX_EXPECT_MSG(readOk, "texture readback failed");
    AGFX_EXPECT_MSG(image.Valid(), "readback image was invalid");

    bool uniform = true;
    for (size_t i = 0; i < image.pixels.size(); i += 4) {
        if (image.pixels[i] != kHazardConstant) {
            uniform = false;
            break;
        }
    }
    AGFX_EXPECT_MSG(uniform, "buffer B shows a hazard: the long write into A raced past the alias barrier");
}
