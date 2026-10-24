//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "Common.hlsli"

//--------------------------------------------------------------------------------------
// Definitions
//--------------------------------------------------------------------------------------
#ifndef _VARIANCE_AABB_
#define	_VARIANCE_AABB_	1
#endif

#ifndef _USE_YCOCG_
#define _USE_YCOCG_		1
#endif

#ifndef _R11G11B10_
#define _R11G11B10_		1
#endif

#ifndef _VH_ONLY_
#define	_VH_ONLY_		0
#endif

#ifndef _CN_ONLY_
#define	_CN_ONLY_		0
#endif

#if	_VH_ONLY_ || _CN_ONLY_
#define	NUM_NEIGHBORS	4
#else
#define	NUM_NEIGHBORS	8
#endif
#define	NUM_SAMPLES		(NUM_NEIGHBORS + 1)
#define	NUM_NEIGHBORS_H	4

#ifndef ALPHA_BOUND
#define ALPHA_BOUND		(1.0 / 255.0)
#endif

// Use YCoCg dependently
#if	_USE_YCOCG_
#define GET_LUMA4(v)	((v).x)
#else
#define GET_LUMA4(v)	dot(v, g_luma4Base)
#endif

//--------------------------------------------------------------------------------------
// Constant buffers
//--------------------------------------------------------------------------------------
XUSG_VK_BINDING(0, XUSG_VK_SPACE_CBV)
cbuffer cbImmutable : register(b0)
{
	float2 g_viewport;
};

static const uint g_historyBits = 4;
static const uint g_historyMask = (1u << g_historyBits) - 1;
static const float g_historyMax = g_historyMask;
static const float g_channelMax = (1 << 8) - 1.0;

static const float3 g_luma4Base = { 1.0, 2.0, 1.0 };
static const int2 g_texOffsets[] =
{
#if	!_CN_ONLY_
	int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1),
#endif
	int2(-1, -1), int2(1, -1), int2(1, 1), int2(-1, 1)
};

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
XUSG_VK_BINDING(0, XUSG_VK_SPACE_UAV)
RWTexture2D<float4> g_rwColor;

#if _R11G11B10_
XUSG_VK_BINDING(1, XUSG_VK_SPACE_UAV)
RWTexture2D<float>	g_rwMetadata;

XUSG_VK_BINDING(0, XUSG_VK_SPACE_SRV)
Texture2D<float3>	g_txCurrent;
XUSG_VK_BINDING(1, XUSG_VK_SPACE_SRV)
Texture2D<float3>	g_txHistory;
XUSG_VK_BINDING(2, XUSG_VK_SPACE_SRV)
Texture2D<float2>	g_txVelocity;
XUSG_VK_BINDING(3, XUSG_VK_SPACE_SRV)
Texture2D<float>	g_txShadeAmt;
XUSG_VK_BINDING(4, XUSG_VK_SPACE_SRV)
Texture2D<float>	g_txHistMeta;

#ifdef _HAS_DOF_
XUSG_VK_BINDING(5, XUSG_VK_SPACE_SRV)
Texture2D<float>	g_txCoC;
#endif

#else
XUSG_VK_BINDING(0, XUSG_VK_SPACE_SRV)
Texture2D			g_txCurrent;
XUSG_VK_BINDING(1, XUSG_VK_SPACE_SRV)
Texture2D			g_txHistory;
XUSG_VK_BINDING(2, XUSG_VK_SPACE_SRV)
Texture2D<float2>	g_txVelocity;

#ifdef _HAS_DOF_
XUSG_VK_BINDING(3, XUSG_VK_SPACE_SRV)
Texture2D<float>	g_txCoC;
#endif

#endif

//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
XUSG_VK_BINDING(0, XUSG_VK_SPACE_SAMPLER)
SamplerState		g_smpLinear;

//--------------------------------------------------------------------------------------
// RGB to YCgCo
//--------------------------------------------------------------------------------------
min16float3 rgbToYCoCg(min16float3 rgb)
{
	const min16float y = dot(rgb, min16float3(1.0, 2.0, 1.0));
	const min16float co = dot(rgb.xz, min16float2(2.0, -2.0));
	const min16float cg = dot(rgb, min16float3(-1.0, 2.0, -1.0));

	return min16float3(y, co, cg);
}

//--------------------------------------------------------------------------------------
// YCgCo to RGB
//--------------------------------------------------------------------------------------
min16float3 yCoCgToRGB(min16float3 yCoCg)
{
	yCoCg *= 0.25;
	const min16float r = yCoCg.x + yCoCg.y - yCoCg.z;
	const min16float g = yCoCg.x + yCoCg.z;
	const min16float b = yCoCg.x - yCoCg.y - yCoCg.z;

	return min16float3(r, g, b);
}

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

//--------------------------------------------------------------------------------------
// Get current pixel data
//--------------------------------------------------------------------------------------
float4 GetCurrent(int2 pos)
{
#if _R11G11B10_
	float4 current;
	current.xyz = g_txCurrent[pos];
	current.w = g_txShadeAmt[pos];

	return current;
#else
	return g_txCurrent[pos];
#endif
}

//--------------------------------------------------------------------------------------
// Get historic metadata
//--------------------------------------------------------------------------------------
float GetHistoricMetadata(float2 uv, int2 uvOffsets = 0)
{
#if _R11G11B10_
	return g_txHistMeta[int2(uv * g_viewport) + uvOffsets];
#else
	return g_txHistory[int2(uv * g_viewport) + uvOffsets].w;
#endif
}

//--------------------------------------------------------------------------------------
// Get history pixel data
//--------------------------------------------------------------------------------------
float4 GetHistory(float2 uv)
{
	const float3 history = g_txHistory.SampleLevel(g_smpLinear, uv, 0.0).xyz;
	const float metaData = GetHistoricMetadata(uv);
	
	return float4(history, metaData);
}

//--------------------------------------------------------------------------------------
// Encode history
//--------------------------------------------------------------------------------------
float EncodeHistory(uint hist, float shadeAmt)
{
	// About 4-bit mask and 4-bit weight
	hist |= uint(round(shadeAmt * g_historyMax)) << g_historyBits;

	return hist / g_channelMax;
}

//--------------------------------------------------------------------------------------
// Decode history
//--------------------------------------------------------------------------------------
float DecodeHistory(inout float history)
{
	// About 4-bit mask and 4-bit weight
	const uint hist = history * g_channelMax;
	const float shadeAmt = (hist >> g_historyBits) / g_historyMax;
	history = hist & g_historyMask;

	return shadeAmt;
}

//--------------------------------------------------------------------------------------
// Maxinum velocity of 3x3
//--------------------------------------------------------------------------------------
float4 VelocityMax(int2 pos)
{
	const float2 velocity = g_txVelocity[pos];
	float2 velocities[NUM_NEIGHBORS_H];
	uint i;

	[unroll]
	for (i = 0; i < NUM_NEIGHBORS_H; ++i)
		velocities[i] = g_txVelocity[pos + g_texOffsets[i + NUM_NEIGHBORS_H]];

	float4 velocityMax = velocity.xyxy;
	float speed_sq = dot(velocityMax.xy, velocityMax.xy);

	//[unroll]
	for (i = 0; i < NUM_NEIGHBORS_H; ++i)
	{
		const float2 neighbor = velocities[i];
#if 0
		velocityMax.xy = max(neighbor, velocityMax.xy);
#else
		const float speedN_sq = dot(neighbor, neighbor);
		if (speedN_sq > speed_sq)
		{
			velocityMax.xy = neighbor;
			speed_sq = speedN_sq;
		}
#endif
	}

	return velocityMax;
}

//--------------------------------------------------------------------------------------
// Extract historical shade amount, value and bias
//--------------------------------------------------------------------------------------
float2 HistoryShadeAmount(inout float history, float2 uv)
{
	float histories[NUM_NEIGHBORS_H];
	uint i;

	[unroll]
	for (i = 0; i < NUM_NEIGHBORS_H; ++i)
		histories[i] = GetHistoricMetadata(uv, g_texOffsets[i]);

	const float shadeAmt = DecodeHistory(history);
	float shadeBias = 0.0;

	//[unroll]
	for (i = 0; i < NUM_NEIGHBORS_H; ++i)
	{
		shadeBias += abs(DecodeHistory(histories[i]) - shadeAmt);
		history = min(histories[i], history);
	}

	return float2(shadeAmt, shadeBias);
}

//--------------------------------------------------------------------------------------
// Minimum and maxinum of the neighbor samples, returning Gaussian blurred color
//--------------------------------------------------------------------------------------
min16float4 NeighborMinMax(out float4 neighborMin, out float4 neighborMax,
	float4 current, int2 pos, float gamma = 1.0)
{
	static const float weights[] =
	{
#if	!_CN_ONLY_
		0.5, 0.5, 0.5, 0.5,
#endif
		0.25, 0.25, 0.25, 0.25
	};

	float4 neighbors[NUM_NEIGHBORS];
	uint i;

	[unroll]
	for (i = 0; i < NUM_NEIGHBORS; ++i)
		neighbors[i] = GetCurrent(pos + g_texOffsets[i]);

	min16float3 mu = min16float3(current.xyz);
	min16float3 color = mu;
	min16float3 nMin = mu;
	min16float3 nMax = mu;
	current.w = current.w < ALPHA_BOUND ? 0.0 : 1.0;

#if	_VARIANCE_AABB_
	min16float3 m1 = mu;
	min16float3 m2 = m1 * m1;
#endif

	//[unroll]
	for (i = 0; i < NUM_NEIGHBORS; ++i)
	{
		const min16float3 neighbor = TM(neighbors[i].xyz);
		const float neighborW = neighbors[i].w < ALPHA_BOUND ? 0.0 : 1.0;
		color += neighbor * min16float(weights[i]);
		current.w += neighborW * weights[i];

#if	_VARIANCE_AABB_
		m1 += neighbor;
		m2 += neighbor * neighbor;
#else
		nMin = min(neighbor, nMin);
		nMax = max(neighbor, nMax);
#endif
	}

#if	_VH_ONLY_
	color /= 3.0;
	current.w /= 3.0;
#elif _CN_ONLY_
	color /= 2.0;
	current.w /= 2.0;
#else
	color /= 4.0;
	current.w /= 4.0;
#endif

#if	_VARIANCE_AABB_
	m1 /= NUM_SAMPLES;
	m2 /= NUM_SAMPLES;
	const min16float3 sigma = sqrt(abs(m2 - m1 * m1));
	const min16float3 gsigma = min16float(gamma) * sigma;
	nMin = m1 - gsigma;
	nMax = m1 + gsigma;
	nMin = min(nMin, color);
	nMax = max(nMax, color);
	neighborMin.w = GET_LUMA4(m1 - sigma);
	neighborMax.w = GET_LUMA4(m1 + sigma);
#else
	neighborMin.w = GET_LUMA4(nMin);
	neighborMax.w = GET_LUMA4(nMax);
#endif
	neighborMin.xyz = nMin;
	neighborMax.xyz = nMax;

	return min16float4(color, current.w);
}

//--------------------------------------------------------------------------------------
// Clip color
//--------------------------------------------------------------------------------------
min16float3 clipColor(min16float3 color, min16float3 minColor, min16float3 maxColor)
{
	const min16float3 cent = 0.5 * (maxColor + minColor);
	const min16float3 dist = 0.5 * (maxColor - minColor);

	const min16float3 disp = color - cent;
	const min16float3 dir = abs(disp / dist);
	const float maxComp = max(dir.x, max(float(dir.y), dir.z));

	if (maxComp > 1.0) return cent + disp / min16float(maxComp);
	else return color;
}

//--------------------------------------------------------------------------------------
// Clip color with blend
//--------------------------------------------------------------------------------------
float historyClamp(float3 history, float3 filtered, float3 nMin, float3 nMax)
{
	float3 rayDir = filtered - history;
	rayDir = select(abs(rayDir) < (1.0 / 65536.0), (1.0 / 65536.0), rayDir);
	const float3 invRayDir = rcp(rayDir);

	const float3 minIntersect = (nMin - history) * invRayDir;
	const float3 maxIntersect = (nMax - history) * invRayDir;
	const float3 enterIntersect = min(minIntersect, maxIntersect);

	return max(max(enterIntersect.x, enterIntersect.y), enterIntersect.z);
}

//--------------------------------------------------------------------------------------
// Compute shader
//--------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	// Load G-buffers
	const int2 pos = DTid;
	const float2 uv = (DTid + 0.5) / g_viewport;
	const float4 current = GetCurrent(pos);
	const float4 velocity = VelocityMax(pos);
	const float2 uvBack = uv - velocity.xy;
	float4 history = GetHistory(uvBack);

	// Speed to history blur
	static const float2 historyBlurAmp = 4.0 * g_viewport;
	const float2 historyBlurs = abs(velocity.xy) * historyBlurAmp;
	float curHistoryBlur = saturate(historyBlurs.x + historyBlurs.y);

	// Decode history weight (that indicates the convergence) and shade amount from metadata
	const float2 prevShadeAmt = HistoryShadeAmount(history.w, uvBack);
	float historyBlur = 1.0 - history.w / g_historyMax;
	historyBlur = max(historyBlur, curHistoryBlur);
	history.w += 1.0;

	// Compute color-space AABB
	float4 neighborMin, neighborMax;
	const min16float3 currentTM = TM(current.xyz);
	const float curShadeAmt = (current.w - ALPHA_BOUND) / (1.0 - ALPHA_BOUND);
#ifdef _FORCE_GAMMA_
	const float gamma = _FORCE_GAMMA_;
#else
#if _HAS_DOF_
	const bool hasBokeh = abs(g_txCoC[DTid]) > 1.0;
#else
	const bool hasBokeh = false;
#endif
	const float gamma = historyBlur > 0.0 || current.w < ALPHA_BOUND || hasBokeh ||
		abs(prevShadeAmt.x - curShadeAmt) + prevShadeAmt.y > 1.0 / g_historyMax ? 1.0 : 16.0;
#endif
	min16float4 filtered = NeighborMinMax(neighborMin, neighborMax, float4(currentTM, current.w), pos, gamma);

	// Clip historical color
	min16float3 historyTM = TM(history.xyz);
#if _USE_YCOCG_
	historyTM = clamp(historyTM, min16float3(neighborMin.xyz), min16float3(neighborMax.xyz));
#else
	historyTM = clipColor(historyTM, min16float3(neighborMin.xyz), min16float3(neighborMax.xyz));
	//const float clampBlend = historyClamp(historyTM, filtered.xyz, neighborMin.xyz, neighborMax.xyz);
	//historyTM = lerp(historyTM, filtered.xyz, saturate(clampBlend));
#endif
	const float contrast = neighborMax.w - neighborMin.w;
	//curHistoryBlur *= filtered.w;

	// Add aliasing
#if _USE_YCOCG_
	static const float lumContrastFactor = 32.0 * 4.0;
#else
	static const float lumContrastFactor = 32.0;
#endif
	float addAlias = historyBlur * 0.5 + 0.25;
	addAlias = saturate(addAlias + 1.0 / (1.0 + contrast * lumContrastFactor));
	filtered.xyz = lerp(filtered.xyz, currentTM, min16float(addAlias));

	// Calculate blend factor
	const float lumHist = GET_LUMA4(historyTM);
	const float distToClamp = min(abs(neighborMin.w - lumHist), abs(neighborMax.w - lumHist));
#if 0
	const float historyAmt = 1.0 / history.w + historyBlur / 8.0;
	const float historyFactor = distToClamp * historyAmt * (1.0 + historyBlur * historyAmt * 8.0);
	float blend = historyFactor / (distToClamp + contrast);
#else
	const float historyAmt = min(1.0 / history.w + historyBlur / 8.0, 1.0);
	float blend = 0.25 / lerp(8.0, distToClamp + contrast, historyAmt);
#endif
	//blend = historyBlur > 0.0 || filtered.w < 1.0 ? blend : 0.03125;
	blend = min(blend, 0.25);

	min16float3 result = ITM(lerp(historyTM, filtered.xyz, min16float(blend)));
	result = any(isnan(float3(result))) ? ITM(filtered.xyz) : result;
	history.w = min(history.w, (1.0 - curHistoryBlur) * g_historyMax);
	history.w = EncodeHistory(history.w, curShadeAmt);

	g_rwColor[DTid] = float4(result, history.w);
#if _R11G11B10_
	g_rwMetadata[DTid] = history.w;
#endif
}
