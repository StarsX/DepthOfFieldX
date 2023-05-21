//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

#ifndef _MULTI_FINER_
#define _MULTI_FINER_ 1
#endif

//#define _WAVELET_

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cbPerPass
{
	uint g_level;
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
RWTexture2D<float3> g_rwDst;

Texture2D g_txCoarser				: register (t0);
Texture2D<float> g_txCoCCoarser		: register (t1);
Texture2D g_txSrc					: register (t2);
Texture2D g_txCoC					: register (t3);
Texture2D<float3> g_txSrcCoarser	: register (t4);

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState g_sampler;

//--------------------------------------------------------------------------------------
// Calculate radius for the MIP level
//--------------------------------------------------------------------------------------
float CalcMipLevelRadius(float2 domain, uint level)
{
	return CalcMipLevelRadius(level, length(domain));
}

//--------------------------------------------------------------------------------------
// Calculate blending weight
//--------------------------------------------------------------------------------------
float MipGaussianBlendWeight(uint level, int radius)
{
	// Compute deviation
	const float sigma = GaussianSigmaFromRadius(radius);
	const float sigma_sq = sigma * sigma;

	// Gaussian-approximating Haar coefficients (weights of box filters)
	const float l = level + 0.7;
	const float c = 2.0 * PI * sigma_sq;
	const float numerator = pow(16.0, l) * log(4.0);
	const float denorminator = c * (pow(4.0, l) + c);
	//const float numerator = pow(2.0, level * 4.0) * log(4.0);
	//const float denorminator = c * (pow(2.0, level * 2.0) + c);
	//const float numerator = (1u << (level * 4)) * log(4.0);
	//const float denorminator = c * ((1u << (level * 2)) + c);
	//const float numerator = (1u << (level << 2)) * log(4.0);
	//const float denorminator = c * ((1u << (level << 1)) + c);

	return saturate(numerator / denorminator);
}

float4 GetSampleIn2x2From3x3(float4 samples3x3[9], uint2 i)
{
	// |0|1|2|
	// |3|4|5|
	// |6|7|8|

	// |3|2|
	// |0|1|
	static const uint4x4 m =
	{
		uint4(6, 7, 4, 3),
		uint4(7, 8, 5, 4),
		uint4(4, 5, 2, 1),
		uint4(3, 4, 1, 0)
	};

	const uint idx = m[i.x][i.y];

	return samples3x3[idx];
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

	const float4x4 coarsers = transpose(gathers);
	const float4x3 coarserColors = transpose(gatherRGBs);

	const float4 wb = BilinearDomainWeights(g_txCoarser, uv);

	// Calculate Gaussian weight
	const uint radius = CoCRadius(finerCoCs[4].x);
	const float r = CalcMipLevelRadius(g_level + 1);
	const float wf = MipGaussianBlendWeight(g_level, radius);
	const float wc = 1.0 - wf;

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
			const int br = CoCRadius(finerCoCs[i].x);
			const float we = Gaussian(r / 2.0, br);

			src.xyz += finers[i].xyz * we;
			src.w += we;
		}
	}

	src.xyz = src.w > 0.0 ? src.xyz / src.w : finers[4].xyz;
#endif

	[unroll]
	for (i = 0; i < 4; ++i)
	{
		//const float radius = CalcMipLevelRadius(domains[i], g_level + 1);
		float w = wc;
		const int br = CoCRadius(coarsers[i].w);
		float we = Gaussian(r, br);
		//we = 1.0;

		// Apply the convolution weight with edge-stopping function
		const float3 coarser = lerp(src.xyz, coarsers[i].xyz, we);

#ifndef _WAVELET_
		w *= wb[i];
		wr -= w;
		w *= we;

		dst.xyz += coarser * w;
		dst.w += w;
#else
		const float3 h = finers[4].xyz - coarserColors[i] * we;
		dst.xyz += (wf * h + coarser * we) * wb[i];
		dst.w += (wf * (1.0 - we) + we) * wb[i];
#endif
		//dst.xyz += coarser * we * wb[i];
		//dst.w += we * wb[i];
	}

#ifndef _WAVELET_
	// Center sample
	dst.xyz += finers[4].xyz * wr;
	dst.w += wr;
#endif

	dst.xyz = dst.w > 0.0 ? dst.xyz / dst.w : src.xyz;
	//dst.xyz = lerp(dst.xyz, finers[4].xyz, w);

	g_rwDst[DTid] = dst.xyz;
}
