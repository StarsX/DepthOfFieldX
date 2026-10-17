//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Definitions
//--------------------------------------------------------------------------------------

#include "XUSGAdvanced.h"

// Use YCoCg dependently
#if	_USE_YCOCG_
#define GET_LUMA4(v)	((v).x)
#else
#define GET_LUMA4(v)	dot(v, g_luma4Base)
#endif

static const min16float3 g_luma4Base = { 1.0, 2.0, 1.0 };

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
XUSG_VK_BINDING(0, XUSG_VK_SPACE_UAV)
RWTexture2D<float4> g_rwDst;

XUSG_VK_BINDING(0, XUSG_VK_SPACE_SRV)
Texture2D g_txSrc;

//--------------------------------------------------------------------------------------
// A fast invertible tone map that preserves color (Reinhard)
//--------------------------------------------------------------------------------------
min16float3 TM(float3 hdr)
{
	min16float3 color = min16float3(hdr);
#if _USE_YCOCG_
	color = rgbToYCoCg(color);
#endif

	return color / (4.0 + GET_LUMA4(color));
}

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	const float4 src = g_txSrc[DTid];

	g_rwDst[DTid] = float4(TM(src.xyz), src.w);
}
