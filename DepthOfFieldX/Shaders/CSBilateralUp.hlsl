//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

#ifndef _MULTI_FINER_
#define _MULTI_FINER_ 1
#endif

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
Texture2D<float3> g_txSrc			: register (t2);
Texture2D<float> g_txCoC			: register (t3);

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState g_smpLinear;

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
#ifdef _IN_PLACE_
			samples3x3[i].xyz = g_rwDst[idx];
#else
			samples3x3[i].xyz = g_txSrc[idx];
#endif
			samples3x3[i++].w = g_txCoC[idx];
		}
	}
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

	const float cocC = g_txCoC[DTid];
	const float2 uv = (DTid + 0.5) / imageSize;

	const float4x4 gathers =
	{
		g_txCoarser.GatherRed(g_smpLinear, uv),
		g_txCoarser.GatherGreen(g_smpLinear, uv),
		g_txCoarser.GatherBlue(g_smpLinear, uv),
		g_txCoCCoarser.GatherRed(g_smpLinear, uv)
	};
	float4 finers[9];
	Fetch3x3(finers, DTid);

	const float4x4 coarsers = transpose(gathers);

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

	// Calculate Gaussian weight
	const uint radius = max((abs(cocC) - 1.0) * 3.0 + 1.0, 0.0);
	const float r = CalcMipLevelRadius(g_level + 1);
	const float w = MipGaussianBlendWeight(g_level, radius);

	float4 src = finers[4];

	uint i;
#if _MULTI_FINER_ == 1
	src = 0.0;

	[unroll]
	for (i = 0; i < 9; ++i)
	{
		const float4 finer = finers[i];
		const float coc = abs(finer.w);
		const int br = max((abs(coc) - 1.0) * 3.0 + 1.0, 0.0);
		const float w = Gaussian(r / 2.0, br);

		src.xyz += finer.xyz * w;
		src.w += w;
	}

	src.xyz = src.w > 0.0 ? src.xyz / src.w : finers[4].xyz;
#endif

	float4 dst = 0.0;

	[unroll]
	for (i = 0; i < 4; ++i)
	{
		//const float radius = CalcMipLevelRadius(domains[i], g_level + 1);
		const float coc = abs(coarsers[i].w);
		const int br = max((abs(coc) - 1.0) * 3.0 + 1.0, 0.0);
		float we = Gaussian(r, br);

		const float3 coarser = lerp(src.xyz, coarsers[i].xyz, we);
		dst.xyz += lerp(coarser * we, finers[4].xyz, w) * wb[i];
		dst.w += lerp(we, 1.0, w) * wb[i];
		//dst.xyz += coarser * we * wb[i];
		//dst.w += we * wb[i];
	}

	dst.xyz = dst.w > 0.0 ? dst.xyz / dst.w : src.xyz;
	//dst.xyz = lerp(dst.xyz, finers[4].xyz, w);

	g_rwDst[DTid] = dst.xyz;
}
