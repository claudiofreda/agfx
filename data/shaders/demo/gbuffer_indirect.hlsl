/**
 * @ Author: Amélie Heinrich @ Amélie Heinrich
 * @ Create Time: 2026-07-18 23:00:00
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

// Split out of gbuffer.hlsl: the SPIR-V backend rejects a translation unit that declares more
// than one [[vk::push_constant]] block even when only one entry point (and its push-constant
// struct) is actually selected for compilation, so the regular and GPU-driven indirect paths
// -- each with their own push-constant struct at b0 -- need separate files on Vulkan.

#include "data/shaders/agfx.h"

struct SceneVertex {
    float3 pos;
    float3 normal;
    float4 tangent;
    float2 uv;
};

struct GBufferSceneConstants {
    float4x4 viewProj;
};

struct vs_out {
    float4 position : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float4 tangent : TEXCOORD2;
    float2 uv : TEXCOORD3;
    nointerpolation uint drawID : TEXCOORD4;
};

struct ps_out {
    float4 albedo : SV_TARGET0;
    float4 normal : SV_TARGET1;
    float4 mra : SV_TARGET2;
};

struct GPUSceneInstance {
    float4x4 worldMatrix;
    uint vertexOffset;
    uint indexOffset;
    uint albedoTex;
    uint normalTex;
    uint metallicRoughnessTex;
    float metallicFactor;
    float roughnessFactor;
    uint indexCount;
    float3 boundsMin;
    float _pad0;
    float3 boundsMax;
    float _pad1;
};

struct GBufferIndirectPushConstants {
    ResourceHandle vertexBuffer;
    ResourceHandle gpuScene;
    ResourceHandle textureSampler;
    ResourceHandle sceneCB;
    // Consumed on Vulkan only: slot -> GPU-scene index, filled by culling.hlsl (see its
    // CullingPushConstants::drawIndirection).
    ResourceHandle drawIndirection;
};

AGFX_PUSH_CONSTANTS(GBufferIndirectPushConstants, g_IndirectConstants);
AGFX_DECLARE_DRAW_ID();

// GPU-driven path: per-draw data (worldMatrix, textures, factors) comes from GPUSceneInstance
// indexed by AGFX_DRAW_ID() instead of push constants, since a single ExecuteIndirect call shares
// one push-constants blob across every draw in the bundle. See notes/mdi.md.
vs_out main_vs_indirect(uint vertexID : SV_VertexID) {
    AGFXStructuredBuffer<SceneVertex> vertices = AGFXStructuredBuffer<SceneVertex>::Create(g_IndirectConstants.vertexBuffer);
    AGFXStructuredBuffer<GPUSceneInstance> gpuScene = AGFXStructuredBuffer<GPUSceneInstance>::Create(g_IndirectConstants.gpuScene);
    AGFXStructuredBuffer<GBufferSceneConstants> sceneCB = AGFXStructuredBuffer<GBufferSceneConstants>::Create(g_IndirectConstants.sceneCB);
    GBufferSceneConstants scene = sceneCB.Load(0);
#if defined(AGFX_VULKAN)
    // Vulkan's AGFX_DRAW_ID() is the linear DrawIndex in the post-culling bundle, not the GPU-scene
    // index the culling shader wrote as drawId -- resolve it through the indirection buffer.
    AGFXByteAddressBuffer indirection = AGFXByteAddressBuffer::Create(g_IndirectConstants.drawIndirection);
    const uint drawID = indirection.Load(AGFX_DRAW_ID() * 4);
#else
    const uint drawID = AGFX_DRAW_ID();
#endif
    GPUSceneInstance inst = gpuScene.Load(drawID);

    SceneVertex v = vertices.Load(vertexID + inst.vertexOffset);

    float4 worldPos = mul(inst.worldMatrix, float4(v.pos, 1.0f));

    vs_out output;
    output.position = mul(scene.viewProj, worldPos);
    output.worldPos = worldPos.xyz;
    output.normal = normalize(mul((float3x3)inst.worldMatrix, v.normal));
    output.tangent = float4(normalize(mul((float3x3)inst.worldMatrix, v.tangent.xyz)), v.tangent.w);
    output.uv = v.uv;
    output.drawID = drawID;
    return output;
}

// AGFX_DRAW_ID() isn't called here: on Vulkan it resolves to the SPIR-V DrawIndex builtin, which
// is only valid in vertex/mesh/task stages, so the vertex shader forwards it via vs_out instead.
ps_out main_ps_indirect(vs_out input) {
    AGFXStructuredBuffer<GPUSceneInstance> gpuScene = AGFXStructuredBuffer<GPUSceneInstance>::Create(g_IndirectConstants.gpuScene);
    GPUSceneInstance inst = gpuScene.Load(input.drawID);

    AGFXTexture2D<float4> albedoTex = AGFXTexture2D<float4>::Create(inst.albedoTex);
    AGFXTexture2D<float4> normalTex = AGFXTexture2D<float4>::Create(inst.normalTex);
    AGFXTexture2D<float4> metallicRoughnessTex = AGFXTexture2D<float4>::Create(inst.metallicRoughnessTex);
    AGFXSampler samp = AGFXSampler::Create(g_IndirectConstants.textureSampler);

    float4 albedoSample = albedoTex.Sample(samp, input.uv);
    if (albedoSample.a < 0.5f) discard;
    float3 albedo = pow(albedoSample.rgb, 2.2f);

    float3 n = normalize(input.normal);
    float3 t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
    float3 b = cross(n, t) * input.tangent.w;
    float3x3 tbn = float3x3(t, b, n);

    float3 mapNormal = normalTex.Sample(samp, input.uv).xyz * 2.0f - 1.0f;
    float3 worldNormal = normalize(mul(mapNormal, tbn));

    float3 mrSample = metallicRoughnessTex.Sample(samp, input.uv).rgb;
    float metallic = saturate(mrSample.b * inst.metallicFactor);
    float roughness = saturate(mrSample.g * inst.roughnessFactor);

    ps_out output;
    output.albedo = float4(albedo, 1.0f);
    output.normal = float4(worldNormal * 0.5f + 0.5f, 1.0f);
    output.mra = float4(metallic, roughness, 1.0f, 1.0f);
    return output;
}
