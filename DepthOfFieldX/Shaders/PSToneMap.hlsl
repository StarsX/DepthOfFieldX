//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "Common.hlsli"

//--------------------------------------------------------------------------------------
// Definitions
//--------------------------------------------------------------------------------------
#define	A	shoulderStrength
#define	B	linearStrength
#define	C	linearAngle
#define	D	toeStrength
#define	E	toeNumerator
#define	F	toeDenominator
#define W	linearWhite
#define PI	3.1415926535897

#ifndef TONE_MAPPER
#define TONE_MAPPER AMDTonemapper
#endif

//--------------------------------------------------------------------------------------
// Constants
//--------------------------------------------------------------------------------------
static const min16float3 g_greyFactor = { 0.27, 0.67, 0.06 };
static const min16float2 g_exposure = { 0.18, 1.4 };

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
Texture2D<float3> g_txImage;
StructuredBuffer<float> g_roLogLum;
Texture2D<float> g_txCoC;

//--------------------------------------------------------------------------------------
// Unsharp
//--------------------------------------------------------------------------------------
min16float3 unsharp(const int2 pos)
{
	const min16float3 center = min16float3(g_txImage[pos]);
	const min16float3 left = min16float3(g_txImage[pos + int2(-1, 0)]);
	const min16float3 right = min16float3(g_txImage[pos + int2(1, 0)]);
	const min16float3 top = min16float3(g_txImage[pos + int2(0, -1)]);
	const min16float3 bottom = min16float3(g_txImage[pos + int2(0, 1)]);

	const min16float3 laplacian = 4.0 * center - (left + right + top + bottom);

	return max(laplacian * (abs(g_txCoC[pos]) > 0.5 ? 0.0 : 0.32) + center, 0.0);
}

//--------------------------------------------------------------------------------------
// Hable's filmic tone mapping
//--------------------------------------------------------------------------------------
min16float4 Uncharted2TonemapOp(min16float4 x)
{
	const min16float  A = 0.15;
	const min16float  B = 0.50;
	const min16float  C = 0.10;
	const min16float  D = 0.20;
	const min16float  E = 0.02;
	const min16float  F = 0.30;

	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

min16float3 Uncharted2Tonemap(min16float3 color)
{
	const float W = 11.2;

	const min16float4 result = Uncharted2TonemapOp(min16float4(color.xyz * 2.0, W));

	return result.xyz / result.w;
}

//--------------------------------------------------------------------------------------
// ACES filmic tone mapping
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
//--------------------------------------------------------------------------------------
min16float3 ACESFilmOp(min16float3 x)
{
	const min16float A = 2.51;
	const min16float B = 0.03;
	const min16float C = 2.43;
	const min16float D = 0.59;
	const min16float E = 0.14;

	return saturate((x * (A * x + B)) / (x * (C * x + D) + E));
}

min16float3 ACESFilm(min16float3 color)
{
	return ACESFilmOp(color.xyz * 0.6);
}

//--------------------------------------------------------------------------------------
// Reinhard
//--------------------------------------------------------------------------------------
min16float3 Reinhard(min16float3 color)
{
	return color / (color + 1.0);
}

//--------------------------------------------------------------------------------------
// The tone mapper used in HDRToneMappingCS11
//--------------------------------------------------------------------------------------
min16float3 DX11DSK(min16float3 color)
{
	const min16float MIDDLE_GRAY = 0.72;
	const min16float LUM_WHITE = 1.5;

	// Tone mapping
	color *= MIDDLE_GRAY;
	color *= (1.0 + color / LUM_WHITE);
	color /= 1.0 + color;

	return color;
}

//--------------------------------------------------------------------------------------
// AMD Tonemapper
//--------------------------------------------------------------------------------------
// General tonemapping operator, build 'b' term.
min16float ColToneB(min16float hdrMax, min16float contrast, min16float shoulder, min16float midIn, min16float midOut)
{
	return
		-((-pow(midIn, contrast) + (midOut * (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) -
			pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut)) /
			(pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut)) /
			(pow(midIn, contrast * shoulder) * midOut));
}

// General tonemapping operator, build 'c' term.
min16float ColToneC(min16float hdrMax, min16float contrast, min16float shoulder, min16float midIn, min16float midOut)
{
	return (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) - pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut) /
		(pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut);
}

// General tonemapping operator, p := { contrast,shoulder,b,c }.
min16float ColTone(min16float x, min16float4 p)
{
	min16float z = pow(x, p.x);

	return z / (pow(z, p.y) * p.z + p.w);
}

min16float3 AMDTonemapper(min16float3 color)
{
	const min16float hdrMax = 25.0;		// How much HDR range before clipping. HDR modes likely need this pushed up to say 25.0.
	const min16float contrast = 1.25;	// Use as a baseline to tune the amount of contrast the tonemapper has.
	const min16float shoulder = 1.0;	// Likely don¡¯t need to mess with this factor, unless matching existing tonemapper is not working well..
	const min16float midIn = 0.18;		// most games will have a {0.0 to 1.0} range for LDR so midIn should be 0.18.
	const min16float midOut = 0.18;		// Use for LDR. For HDR10 10:10:10:2 use maybe 0.18/25.0 to start. For scRGB, I forget what a good starting point is, need to re-calculate.

	const min16float b = ColToneB(hdrMax, contrast, shoulder, midIn, midOut);
	const min16float c = ColToneC(hdrMax, contrast, shoulder, midIn, midOut);

#define EPS 1e-6
	min16float peak = max(color.x, max(color.y, color.z));
	peak = max(EPS, peak);

	float3 ratio = color / float(peak);
	peak = ColTone(peak, min16float4(contrast, shoulder, b, c));

	// then process ratio
	// probably want send these pre-computed (so send over saturation/crossSaturation as a constant)
	const min16float crosstalk = 2.0; // controls amount of channel crosstalk
	const min16float saturation = contrast; // full tonal range saturation control
	const min16float crossSaturation = contrast * 16.0; // crosstalk saturation

	const min16float white = 1.0;

	// wrap crosstalk in transform
	ratio = pow(abs(ratio), saturation / crossSaturation);
	ratio = lerp(ratio, white, pow(peak, crosstalk));
	ratio = pow(abs(ratio), crossSaturation);

	// then apply ratio to peak
	return min16float3(peak * ratio);
}

//--------------------------------------------------------------------------------------
// Vignetting
//--------------------------------------------------------------------------------------
min16float3 vignette(float2 winBias)
{
	// convert window coord to [-1, 1] range
	const float2 winPos = winBias * 2.0 - 1.0;

	// calculate distance from origin
	return 1.0 - smoothstep(0.8, 1.5, min16float(length(winPos))) * 0.8;
}

//--------------------------------------------------------------------------------------
// Pixel shader that performs tone mapping
//--------------------------------------------------------------------------------------
float4 main(const PS_Input input) : SV_TARGET
{
	const int2 pos = input.Pos.xy;
	const float avgLum = g_roLogLum[0];
	min16float3 color = unsharp(pos);

	// Auto exposure adjustment
	const min16float luma = dot(g_greyFactor, color);
	color *= g_exposure.x / min16float(avgLum);

	// Tone mapping
	color = TONE_MAPPER(color);
	color = saturate(color);

	/*const min16float b = g_exposure.y * 4.0 - 1.0;
	const min16float a = 1.0 - b;
	luma *= luma * a + b;*/
	//color *= pow(luma, 0.1);
	color = sqrt(color);

	color *= vignette(input.UV);

	return float4(color, 1.0);
}
