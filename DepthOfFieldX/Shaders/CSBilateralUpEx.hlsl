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
RWTexture2D<float> g_rwCoC;

Texture2D g_txCoarser				: register (t0);
Texture2D<float> g_txCoCCoarser		: register (t1);
Texture2D g_txSrc					: register (t2);
Texture2D g_txCoC					: register (t3);
Texture2D<float3> g_txSrcCoarser	: register (t4);

StructuredBuffer<WeightData> g_roWeights : register (t5);

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

	float4 finers[9], finerCoCs[9];
	Fetch3x3(finers, g_txSrc, DTid);
	Fetch3x3(finerCoCs, g_txCoC, DTid);

	// Load weight data
	const uint radius = CoCRadius(finerCoCs[4].x);
	const WeightData weightData = g_roWeights[MAX_RADIUS * g_level + radius];

	const float4x4 coarsers = transpose(gathers);
	const float4x3 coarserColors = transpose(gatherRGBs);

	const float4 wb = BilinearDomainWeights(g_txCoarser, uv);

	const float r = CalcMipLevelRadius(g_level + 1);
	float4 src = float4(finers[4].xyz, 1.0); // Fallback to the center sample
	float4 dst = 0.0;
	float ws = 0.0, wr = 1.0;

	uint i;
#if _MULTI_FINER_ == 1
	[unroll]
	for (i = 0; i < 9; ++i)
	{
		if (i != 4)
		{
			float w = weightData.Finers[i];
			const int br = CoCRadius(finerCoCs[i].x);
			const float we = Gaussian(r / 2.0, br);

			wr -= w;
			w *= we;

			// 3x3 bilateral filter
			dst.xyz += finers[i].xyz * w;
			dst.w += w;

			src.xyz += finers[i].xyz * we;
			src.w += we;
		}
	}

	src.xyz = src.w > 0.0 ? src.xyz / src.w : finers[4].xyz;
#endif

	[unroll]
	for (i = 0; i < 4; ++i)
	{
		int br = CoCRadius(coarsers[i].w);
		br = coarsers[i].w < 0.0 ? max(radius, br) : radius;
		float w = g_roWeights[MAX_RADIUS * g_level + br].Coarsers[i];
		float we = Gaussian(r, br);

		// Apply the convolution weight with edge-stopping function
		const float3 coarser = lerp(src.xyz, coarsers[i].xyz, we);

#ifndef _WAVELET_
		w *= wb[i];
		wr -= w;
		w *= we;

		dst += float4(coarser, coarsers[i].w) * w;
		ws += w;
#else
		const float wf = weightData.Finers[4];
		const float3 h = finers[4].xyz - coarserColors[i] * we;
		dst.xyz += (wf * h + coarser * we) * wb[i];
		dst.w += (wf * (1.0 - we) + we) * wb[i];
#endif
	}

#ifndef _WAVELET_
	// Center sample
	dst += float4(finers[4].xyz, finerCoCs[4].x) * wr;
	ws += wr;
#endif

	dst = ws > 0.0 ? dst / ws : float4(src.xyz, finerCoCs[4].x);

	g_rwDst[DTid] = dst.xyz;
	g_rwCoC[DTid] = dst.w;
}
