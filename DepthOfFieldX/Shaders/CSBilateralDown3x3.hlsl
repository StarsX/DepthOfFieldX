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
Texture2D g_txCoC;

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	const uint2 posFC = DTid * 3 + 1; // Center texcoord position of the finer layer

	// Gather samples
	float4 srcs[9], cocs[9];
	Fetch3x3(srcs, g_txSrc, posFC);
	Fetch3x3(cocs, g_txCoC, posFC);

	const float r = CalcMipLevelRadius3x3(g_level);

	// Down sampling with 3x3 bilateral filtering
	float4 dst = 0.0;
	float ws = 0.0;

	uint i = 0;

	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			srcs[i].w = cocs[i].x;
			const int br = CoCRadius(srcs[i].w);
			float w = Gaussian(r, br);

			dst += srcs[i] * w;
			ws += w;
			++i;
		}
	}

	if (ws > 0.0)
	{
		dst /= ws;

		g_rwDst[DTid] = dst.xyz;
		g_rwCoC[DTid] = dst.w;
	}
	else g_rwCoC[DTid] = 0.0;
}
