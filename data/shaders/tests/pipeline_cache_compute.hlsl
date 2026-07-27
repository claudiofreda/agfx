//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the compute-only pipeline behind the pipeline-cache compute test. One thread, one
// store -- the dispatch under test writes a single value into a single buffer slot chosen by the
// `slot` push constant, so two dispatches through two different pipeline objects (one fresh, one
// rebuilt from the first one's cache blob) can write two adjacent, independently checkable slots
// without racing each other.

#include "data/shaders/agfx.h"

struct PipelineCacheComputePushConstants
{
    ResourceHandle rwBuffer; // AGFX_BUFFER_VIEW_TYPE_RAW, writeable.
    uint slot;               // Which 4-byte word to write: 0 or 1.
    uint value;              // The value to store there.
    uint padding0;
};

AGFX_PUSH_CONSTANTS(PipelineCacheComputePushConstants, g_Constants);

[numthreads(1, 1, 1)]
void main_cs()
{
    AGFXRWByteAddressBuffer dst = AGFXRWByteAddressBuffer::Create(g_Constants.rwBuffer);
    dst.Store(g_Constants.slot * 4u, g_Constants.value);
}
