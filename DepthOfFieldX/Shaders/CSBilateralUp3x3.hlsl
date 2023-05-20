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

Texture2D g_txCoarser		: register (t0);
Texture2D g_txCoCCoarser	: register (t1);
Texture2D g_txSrc			: register (t2);
Texture2D g_txCoC			: register (t3);
Texture2D g_txSrcCoarser	: register (t4);

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

float CalcMipLevelRadius(float2 domain, uint level)
{
	return CalcMipLevelRadius3x3(level, length(domain));
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
	const float d = pow(3.0, level);
	const float r = 0.5 * d - 0.5;
	const float r1 = 1.5 * d - 0.5;

	//return exp(-0.5 * r1 * r1 / sigma_sq) / exp(-0.5 * r * r / sigma_sq);
	return exp(0.5 * (r * r - r1 * r1) / sigma_sq);
}

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	float4 finers[9], finerCoCs[9], coarsers[9], coarserCoCs[9], coarserColors[9];
	Fetch3x3(finers, g_txSrc, DTid);
	Fetch3x3(finerCoCs, g_txCoC, DTid);

	const uint2 posCC = DTid / 3; // Center texcoord position of the coarser layer 
	Fetch3x3(coarsers, g_txCoarser, posCC);
	Fetch3x3(coarserCoCs, g_txCoCCoarser, posCC);
	Fetch3x3(coarserColors, g_txSrcCoarser, posCC);

	// Calculate Gaussian weight
	const uint radius = CoCRadius(finerCoCs[4].x);
	const float r = CalcMipLevelRadius(g_level + 1);
	const float wc = MipGaussianBlendWeightCoarse(g_level, radius);
	const float wf = 1.0 - wc;

	float4 src = float4(finers[4].xyz, 1.0); // Fallback to the center sample
	float4 dst = 0.0;
	float wcs = 0.0, wbs = 0.0;

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

	const int2 offset = int2(DTid % 3) - 1;
	i = 0;

	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			float w = wc;
			const int br = CoCRadius(coarserCoCs[i].x);
			float we = Gaussian(r, br);

			// Apply the convolution weight with edge-stopping function
			const float3 coarser = lerp(src.xyz, coarsers[i].xyz, we);
			//const float3 coarser = coarsers[i].xyz;

			const float2 d = 1.0 - abs(int2(x, y) * 2 - offset) / 4.0;
			const float wd = d.x * d.y;

#ifndef _WAVELET_
			w *= wd;
			wcs += w;
			wbs += wd;
			w *= we;

			dst.xyz += coarser * w;
			dst.w += w;
#else
			const float3 h = finers[4].xyz - coarserColors[i].xyz * we;
			dst.xyz += (wf * h + coarser * we) * wd;
			dst.w += (wf * (1.0 - we) + we) * wd;
#endif
			++i;
		}
	}

#ifndef _WAVELET_
	// Center sample
	const float wr = 1.0 - wcs / wbs;
	dst.xyz += finers[4].xyz * wr;
	dst.w += wr;
#endif

	dst.xyz = dst.w > 0.0 ? dst.xyz / dst.w : src.xyz;

	g_rwDst[DTid] = dst.xyz;
}
