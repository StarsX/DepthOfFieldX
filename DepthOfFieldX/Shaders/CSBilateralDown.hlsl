//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cb
{
	uint g_level;
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
RWTexture2D<float3> g_rwDst;
RWTexture2D<float> g_rwCoC;

Texture2D g_txSrc;
Texture2D<float> g_txCoC;

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState g_sampler;

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	float2 imageSize;
	g_rwDst.GetDimensions(imageSize.x, imageSize.y);

	const float2 uv = (DTid + 0.5) / imageSize;

	float4x4 gathers = float4x4(
		g_txSrc.GatherRed(g_sampler, uv),
		g_txSrc.GatherGreen(g_sampler, uv),
		g_txSrc.GatherBlue(g_sampler, uv),
		g_txCoC.GatherRed(g_sampler, uv)
		);

	const float4 wb = BilinearDomainWeights(g_txSrc, uv);

	float4x4 srcs = transpose(gathers);
	const float r = CalcMipLevelRadius(g_level);

	float4 dst = 0.0;
	float ws = 0.0, w_max = 0.0;

	[unroll]
	for (uint i = 0; i < 4; ++i)
	{
		const int br = CoCRadius(srcs[i].w);
		float w = Gaussian(r, br);
		w *= wb[i];

		dst += srcs[i] * w;
		ws += w;
	}

	if (ws > 0.0)
	{
		dst /= ws;

		g_rwDst[DTid] = dst.xyz;
		g_rwCoC[DTid] = dst.w;
	}
	else g_rwCoC[DTid] = 0.0;
}
