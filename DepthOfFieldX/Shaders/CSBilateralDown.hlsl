//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cb
{
	float g_level;
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
RWTexture2D<float3> g_rwDst;
RWTexture2D<float> g_rwCoC;

Texture2D<float3> g_txSrc;
Texture2D<float> g_txCoC;

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState g_sampler;

//--------------------------------------------------------------------------------------
// Get domain location of bilinear filter
//--------------------------------------------------------------------------------------
float2 BilinearDomainLoc(Texture2D<float3> tx, float2 uv)
{
	float2 texSize;
	tx.GetDimensions(texSize.x, texSize.y);

	return frac(uv * texSize - 0.5);
}

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

	const float2 domain = BilinearDomainLoc(g_txSrc, uv);
	const float2 domainInv = 1.0 - domain;
	// |3|2|
	// |0|1|
	const float2 domains[] =
	{
		float2(domain.x, domainInv.y),
		float2(domainInv.x, domainInv.y),
		float2(domainInv.x, domain.y),
		float2(domain.x, domain.y),
	};
	const float4 wb =
	{
		domainInv.x * domain.y,
		domain.x * domain.y,
		domain.x * domainInv.y,
		domainInv.x * domainInv.y
	};

	float4x4 srcs = transpose(gathers);
	const float r = CalcMipLevelRadius(g_level);

	float4 dst = 0.0;
	float ws = 0.0, w_max = 0.0;

	[unroll]
	for (uint i = 0; i < 4; ++i)
	{
		srcs[i].w = abs(srcs[i].w);
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
