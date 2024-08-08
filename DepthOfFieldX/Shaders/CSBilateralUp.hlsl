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
RWTexture2D<float> g_rwCoC;

Texture2D g_txCoarser		: register (t0);
Texture2D g_txCoCCoarser	: register (t1);
Texture2D g_txSrc			: register (t2);
Texture2D g_txCoC			: register (t3);
Texture2D g_txSrcCoarser	: register (t4);

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

float MipGaussianBlendWeightCoarse(uint level, int radius)
{
	// Compute deviation
	const float sigma = GaussianSigmaFromRadius(radius);
	const float sigma_sq = sigma * sigma;

	// Gaussian-approximating Haar coefficients (weights of box filters)
	const float d = pow(2.0, level);
	const float r = 0.5 * d - 0.5;
	const float r1 = d - 0.5;

	//return exp(-0.5 * r1 * r1 / sigma_sq) / exp(-0.5 * r * r / sigma_sq);
	return exp(0.5 * (r * r - r1 * r1) / sigma_sq);
}

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	float4 finers[9], finerCoCs[9], coarsers[9], coarserCoCs[9], coarserColors[9], coarserCocs[9];
	Fetch3x3(finers, g_txSrc, DTid);
	Fetch3x3(finerCoCs, g_txCoC, DTid);

	const uint2 posCC = DTid / 2; // Center texcoord position of the coarser layer 
	Fetch3x3(coarsers, g_txCoarser, posCC);
	Fetch3x3(coarserCoCs, g_txCoCCoarser, posCC);
	Fetch3x3(coarserColors, g_txSrcCoarser, posCC);
	//Fetch3x3(coarserCocs, g_txCocCoarser, posCC);

	// Calculate domain weights for 3x3 linear interpolation
	float wd[9];
	DomainWeights(wd, DTid);

	// Calculate Gaussian weight
	const int radius = CoCRadius(finerCoCs[4].x);
	float r = CalcMipLevelRadius(g_level, 0.5);
	const float wc = MipGaussianBlendWeightCoarse(g_level, 24);
	//const float wc = 1.0 - MipGaussianBlendWeight(g_level, 24);

	float4 src = float4(finers[4].xyz, 1.0); // Fallback to the center sample
	float4 dst = 0.0;
	float ws = 0.0;
#ifndef _WAVELET_
	float wf = 1.0;
#endif

	uint i;
#if _MULTI_FINER_ == 1
	[unroll]
	for (i = 0; i < 9; ++i)
	{
		if (i != 4)
		{
			const int br = CoCRadius(finerCoCs[i].x);
			const float fr = Gaussian(r, br);

			src.xyz += finers[i].xyz * fr;
			src.w += fr;
		}
	}

	src.xyz = src.w > 0.0 ? src.xyz / src.w : finers[4].xyz;
#endif

	const float4 finer = float4(finers[4].xyz, finerCoCs[4].x);
	r = CalcMipLevelRadius(g_level + 1, 0.5);

	[unroll]
	for (i = 0; i < 9; ++i)
	{
		const bool isOccluded = finerCoCs[4].x < coarserCoCs[i].x;
		const int cr = CoCRadius(coarserCoCs[i].x);

		float w = wc;
		const int br = isOccluded ? radius : cr;
		const float fr = Gaussian(r, br);

		w *= MipGaussianBlendWeightCoarse(g_level, max(radius, cr));
		//w *= 1.0 - MipGaussianBlendWeight(g_level, max(radius, cr));

		// Apply the convolution weight with edge-stopping function
		//const float coc = isOccluded ? finerCoCs[4].x : coarserCoCs[i].x;
		const float coc = coarserCoCs[i].x;
		float4 coarser = float4(coarsers[i].xyz, coc);
		coarser.xyz = lerp(src.xyz, coarser.xyz, fr);

#ifndef _WAVELET_
		w *= wd[i];
		wf -= w;
		w *= fr;

		dst += coarser * w;
		ws += w;
#else
		const float wf = 1.0 - w;
		const float4 h = finer - coarser * fr;
		dst += (wf * h + coarser * fr) * wd[i];
		ws += (wf * (1.0 - fr) + fr) * wd[i];
#endif
	}

#ifndef _WAVELET_
	// Center sample
	dst += finer * wf;
	ws += wf;
#endif

	dst = ws > 0.0 ? dst / ws : finer;

	g_rwDst[DTid] = dst.xyz;
	g_rwCoC[DTid] = dst.w;
}
