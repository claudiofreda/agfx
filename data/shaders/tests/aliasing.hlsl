//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader for resource aliasing (see test_aliasing.cpp).
//
// Every value produced here is an exact small integer stored through an RGBA32F texel: integers
// under 2^24 round-trip exactly through fp32, which keeps the AliasHeapTransients golden bit-stable
// instead of one float-rounding ULP away from flaky.

#include "data/shaders/agfx.h"

struct AliasingPushConstants
{
    ResourceHandle rwTexture;  // AGFXRWTexture2D<float4>, the aliased transient/target texture.
    ResourceHandle rwBuffer;   // AGFXRWStructuredBuffer<uint>, the persistent accumulator.
    uint width;
    uint height;
    uint patternIndex;         // 0 gradient, 1 checkerboard, 2 radial, 3 constant fill.
    uint iterationCount;       // main_long_write_cs only.
};

AGFX_PUSH_CONSTANTS(AliasingPushConstants, g_Constants);

// Writes one of four deterministic patterns into rwTexture, one thread per texel.
[numthreads(8, 8, 1)]
void main_pattern_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRWTexture2D<float4> tex = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);

    uint value = 0;
    if (g_Constants.patternIndex == 0) {
        value = id.x + id.y; // gradient
    } else if (g_Constants.patternIndex == 1) {
        value = ((id.x ^ id.y) & 1u) * 7u; // checkerboard
    } else if (g_Constants.patternIndex == 2) {
        int dx = int(id.x) - int(g_Constants.width / 2);
        int dy = int(id.y) - int(g_Constants.height / 2);
        value = uint(dx * dx + dy * dy) >> 4; // radial
    } else {
        value = 99u; // constant fill -- AliasHazardOrdering's "short pass" into the incoming alias.
    }

    tex.Store(int2(id.xy), float4((float)value, 0.0f, 0.0f, 1.0f));
}

// Folds rwTexture's red channel into the persistent rwBuffer accumulator, one thread per texel.
// acc[i] = acc[i]*3 + texel -- the *3 fold makes a dropped or reordered phase distinguishable from a
// lost one, the same trick multi_dispatch.hlsl uses with its passIndex fold.
[numthreads(8, 8, 1)]
void main_accumulate_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRWTexture2D<float4> tex = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);
    AGFXRWStructuredBuffer<uint> acc = AGFXRWStructuredBuffer<uint>::Create(g_Constants.rwBuffer);

    const uint index = id.y * g_Constants.width + id.x;
    const uint texel = (uint)tex.Load(int2(id.xy)).r;
    acc.Store(index, acc.Load(index) * 3u + texel);
}

// A deliberately long pass: iterationCount data-dependent loop iterations before a single store, so
// the compiler can't hoist or shortcut it away. This is AliasHazardOrdering's long-running write into
// the outgoing side of an alias, raced against a short write into the incoming side.
[numthreads(8, 8, 1)]
void main_long_write_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= g_Constants.width || id.y >= g_Constants.height) {
        return;
    }

    AGFXRWTexture2D<float4> tex = AGFXRWTexture2D<float4>::Create(g_Constants.rwTexture);

    uint value = id.x + id.y + 1u;
    [loop]
    for (uint i = 0; i < g_Constants.iterationCount; ++i) {
        value = (value * 2654435761u) ^ i;
    }

    tex.Store(int2(id.xy), float4((float)(value & 0xffu), 0.0f, 0.0f, 1.0f));
}
