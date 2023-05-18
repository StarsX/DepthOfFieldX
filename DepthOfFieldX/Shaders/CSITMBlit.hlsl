//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Definitions
//--------------------------------------------------------------------------------------

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
RWTexture2D<float4> g_rwDst;
Texture2D g_txSrc;

//--------------------------------------------------------------------------------------
// Inverse of preceding function
//--------------------------------------------------------------------------------------
min16float3 ITM(min16float3 color)
{
	color *= 4.0 / (1.0 - GET_LUMA4(color));

#if _USE_YCOCG_
	return yCoCgToRGB(color);
#else
	return color;
#endif
}

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	const float4 src = g_txSrc[DTid];

	g_rwDst[DTid] = float4(ITM(min16float3(src.xyz)), src.w);
}
