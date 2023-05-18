//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

#ifndef _MULTI_FINER_
#define _MULTI_FINER_ 1
#endif

//#define _WAVELET_

#define MAX_RADIUS 256

//--------------------------------------------------------------------------------------
// Structure
//--------------------------------------------------------------------------------------
struct WeightData
{
	float Finers[9];
	float Coarsers[4];
};

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cbPerPass
{
	uint g_level;
};

//--------------------------------------------------------------------------------------
// Textures and buffer
//--------------------------------------------------------------------------------------
RWTexture2D<float3> g_rwDst;

Texture2D g_txCoarser				: register (t0);
Texture2D<float> g_txCoCCoarser		: register (t1);
Texture2D<float3> g_txSrc			: register (t2);
Texture2D<float> g_txCoC			: register (t3);
Texture2D<float3> g_txSrcCoarser	: register (t4);

StructuredBuffer<WeightData> g_roWeights : register (t5);

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState g_sampler;

//--------------------------------------------------------------------------------------
// Get domain location of bilinear filter
//--------------------------------------------------------------------------------------
float2 BilinearDomainLoc(Texture2D tx, float2 uv)
{
	float2 texSize;
	tx.GetDimensions(texSize.x, texSize.y);

	return frac(uv * texSize - 0.5);
}

//--------------------------------------------------------------------------------------
// Calculate the radius for the corresponding mip level
//--------------------------------------------------------------------------------------
float CalcMipLevelRadius(uint level, float len = 0.5)
{
	return (1u << level) * len * sqrt(2.0);
}

float CalcMipLevelRadius(float2 domain, uint level)
{
	return CalcMipLevelRadius(level, length(domain));
}

//--------------------------------------------------------------------------------------
// 3x3 finer samples
//--------------------------------------------------------------------------------------
void Fetch3x3(out float4 samples3x3[9], uint2 pos)
{
	uint i = 0;
	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			const uint2 idx = (int2)pos + int2(x, y);
			samples3x3[i].xyz = g_txSrc[idx];
			samples3x3[i++].w = g_txCoC[idx];
		}
	}
}

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	float2 imageSize;
	g_rwDst.GetDimensions(imageSize.x, imageSize.y);

	const float cocC = g_txCoC[DTid];
	const float2 uv = (DTid + 0.5) / imageSize;

	const float4x4 gathers =
	{
		g_txCoarser.GatherRed(g_sampler, uv),
		g_txCoarser.GatherGreen(g_sampler, uv),
		g_txCoarser.GatherBlue(g_sampler, uv),
		g_txCoCCoarser.GatherRed(g_sampler, uv)
	};

	const float3x4 gatherRGBs =
	{
		g_txSrcCoarser.GatherRed(g_sampler, uv),
		g_txSrcCoarser.GatherGreen(g_sampler, uv),
		g_txSrcCoarser.GatherBlue(g_sampler, uv)
	};

	float4 finers[9];
	Fetch3x3(finers, DTid);

	// Load weight data
	const uint radius = CoCRadius(cocC);
	const WeightData weightData = g_roWeights[MAX_RADIUS * g_level + radius];

	const float4x4 coarsers = transpose(gathers);
	const float4x3 coarserColors = transpose(gatherRGBs);

	const float2 domain = BilinearDomainLoc(g_txCoarser, uv);
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

	const float r = CalcMipLevelRadius(g_level + 1);
	float4 src = float4(finers[4].xyz, 1.0); // Fallback to the center sample
	float4 dst = 0.0;
	float wr = 1.0;

	uint i;
#if _MULTI_FINER_ == 1
	[unroll]
	for (i = 0; i < 9; ++i)
	{
		if (i != 4)
		{
			const float4 finer = finers[i];
			const float coc = abs(finer.w);
			float w = weightData.Finers[i];
			const int br = max((abs(coc) - 1.0) * 3.0 + 1.0, 0.0);
			const float we = Gaussian(r / 2.0, br);

			wr -= w;
			w *= we;

			// 3x3 bilateral filter
			dst.xyz += finer.xyz * w;
			dst.w += w;

			src.xyz += finer.xyz * we;
			src.w += we;
		}
	}

	src.xyz = src.w > 0.0 ? src.xyz / src.w : finers[4].xyz;
#endif

	[unroll]
	for (i = 0; i < 4; ++i)
	{
		//const float radius = CalcMipLevelRadius(domains[i], g_level + 1);
		float w = weightData.Coarsers[i];
		const float coc = abs(coarsers[i].w);
		const int br = max((abs(coc) - 1.0) * 3.0 + 1.0, 0.0);
		const float we = Gaussian(r, br);

		// Apply the convolution weight with edge-stopping function
		const float3 coarser = lerp(src.xyz, coarsers[i].xyz, we);

#ifndef _WAVELET_
		w *= wb[i];
		wr -= w;
		w *= we;

		dst.xyz += coarser * w;
		dst.w += w;
#else
		const float wf = weightData.Finers[4];
		const float3 h = finers[4].xyz - coarserColors[i] * we;
		dst.xyz += (wf * h + coarser * we) * wb[i];
		dst.w += (wf * (1.0 - we) + we) * wb[i];
#endif
	}

#ifndef _WAVELET_
	// Center sample
	dst.xyz += finers[4].xyz * wr;
	dst.w += wr;
#endif

	dst.xyz = dst.w > 0.0 ? dst.xyz / dst.w : src.xyz;

	g_rwDst[DTid] = dst.xyz;
}
