//
// @ Author: Amélie Heinrich @ Amélie Heinrich
// @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
//
// Test shader: the classic (VS+PS), mesh (MS[+PS]) and task+mesh (TS+MS[+PS]) pipelines behind the
// pipeline-cache tests. Every one of those pipelines draws exactly one of two fixed triangles --
// chosen by the `half_` push constant rather than by SV_VertexID/SV_GroupID -- because each test
// draws twice, once through a freshly created pipeline and once through a pipeline rebuilt purely
// from the first one's cache blob, and needs the two draws to land in different, non-overlapping
// regions of the target so the final image visibly shows both draws, not just the second one
// overwriting the first.
//
// depthValue is written to SV_Position.z unconditionally. The color pipelines run with depth testing
// off, so it is inert there; the depth-only pipelines (fragmentShader == nullptr) use it as the only
// observable output, read back afterwards via a depth-to-color compute pass
// (sampling_comparison.hlsl:main_sample_depth_cs), the same technique test_sample_depth_texture.cpp
// uses to verify a depth-only draw actually landed.

#include "data/shaders/agfx.h"

struct PipelineCachePushConstants
{
    float4 tint;       // Color pipelines: the flat fragment output.
    uint half_;        // 0 = left triangle, 1 = right triangle.
    float depthValue;  // Depth-only pipelines: clip-space z, i.e. SV_Position.z.
    uint padding0;
    uint padding1;
};

AGFX_PUSH_CONSTANTS(PipelineCachePushConstants, g_Constants);

// The same left/right triangle pair raster.hlsl and mesh.hlsl use, so the pipeline-cache tests share
// their proven-safe, well-separated geometry rather than inventing new coordinates.
static const float2 kPositions[2][3] = {
    { float2(-0.9f, -0.8f), float2(-0.1f, -0.8f), float2(-0.5f,  0.8f) },
    { float2( 0.1f, -0.8f), float2( 0.5f,  0.8f), float2( 0.9f, -0.8f) }
};

float4 main_vs(uint vertexID : SV_VertexID) : SV_Position
{
    const uint half_ = min(g_Constants.half_, 1u);
    return float4(kPositions[half_][vertexID], g_Constants.depthValue, 1.0f);
}

float4 main_ps() : SV_Target0
{
    return g_Constants.tint;
}

struct ms_out
{
    float4 position : SV_Position;
};

// Shared by both mesh entry points below, same as mesh.hlsl's EmitTriangle.
void EmitTriangle(uint half_, uint threadID, out vertices ms_out verts[3], out indices uint3 tris[1])
{
    const uint h = min(half_, 1u);
    verts[threadID].position = float4(kPositions[h][threadID], g_Constants.depthValue, 1.0f);
    if (threadID == 0) {
        tris[0] = uint3(0, 1, 2);
    }
}

// One group, one triangle -- unlike mesh.hlsl's amplification demo, this always dispatches exactly
// one mesh group per draw, so which triangle comes out is decided entirely by the half_ push
// constant rather than by SV_GroupID.
[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main_ms(uint threadID : SV_GroupThreadID, out vertices ms_out verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    EmitTriangle(g_Constants.half_, threadID, verts, tris);
}

// Trivial payload: the task stage here only needs to prove the amplification stage is present and
// forwards to one mesh group -- mesh.hlsl's task test already covers routing real per-group payload
// data, so this one doesn't need to repeat that.
struct MeshPayload
{
    uint unused;
};

groupshared MeshPayload s_payload;

[numthreads(1, 1, 1)]
void main_as()
{
    s_payload.unused = 0;
    DispatchMesh(1, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main_ms_payload(uint threadID : SV_GroupThreadID, in payload MeshPayload payload,
                     out vertices ms_out verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    EmitTriangle(g_Constants.half_, threadID, verts, tris);
}
