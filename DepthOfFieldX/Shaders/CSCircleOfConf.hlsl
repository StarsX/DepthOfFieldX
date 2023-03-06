//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DofCommon.hlsli"

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cb
{
	float g_cocScale;
	float g_cocBias;
	float g_cocToImageSpace;
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
RWTexture2D<float2> g_rwCoc;
Texture2D<float> g_txDepth;

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	// Calculate CoC
	const float depth = g_txDepth[DTid];
	const float coc = depth * g_cocScale + g_cocBias;

	g_rwCoc[DTid] = coc * g_cocToImageSpace;
}
