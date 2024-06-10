//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#define BOKEH_RADIUS 32
#define PI 3.141592654

float CoCWeight(float coc, float radius)
{
	return radius <= coc ? 1.0 : 0.0;
	//return saturate((coc - radius + 2.0) / 2.0);
}

int CoCRadius(float coc)
{
	return abs(coc) * 3.0;
}

float GaussianSigmaFromRadius(int radius)
{
	return (radius + 1) / 3.0;
}

float Gaussian(float r, float sigma)
{
	const float a = r / sigma;

	return exp(-0.5 * a * a);
}

float Gaussian(float r, int radius)
{
	const float sigma = GaussianSigmaFromRadius(radius);

	return Gaussian(r, sigma);
}

//--------------------------------------------------------------------------------------
// Calculate the radius for the corresponding mip level
//--------------------------------------------------------------------------------------
float CalcMipLevelRadius(uint level, float bias = 0.0, float len = 0.5)
{
	return (pow(2.0 + bias, level) * len - 0.5);
}

//--------------------------------------------------------------------------------------
// Calculate the radius for the corresponding mip level
//--------------------------------------------------------------------------------------
float CalcMipLevelRadius3x3(uint level, float bias = 0.0, float len = 0.5)
{
	return (pow(3.0 + bias, level) * len - 0.5);
}

//--------------------------------------------------------------------------------------
// Fetch 3x3 samples
//--------------------------------------------------------------------------------------
void Fetch3x3(out float4 samples3x3[9], Texture2D txSrc, uint2 pos)
{
	int2 texSize;
	txSrc.GetDimensions(texSize.x, texSize.y);

	uint i = 0;
	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			const uint2 idx = clamp((int2)pos + int2(x, y), 0, texSize - 1);
			samples3x3[i++] = txSrc[idx];
		}
	}
}

//--------------------------------------------------------------------------------------
// Calculate domain weights for bilinear interpolation
//--------------------------------------------------------------------------------------
float4 BilinearDomainWeights(Texture2D tex, float2 uv)
{
	float2 texSize;
	tex.GetDimensions(texSize.x, texSize.y);

	const float2 domain = frac(uv * texSize - 0.5);
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

	return float4(
		domainInv.x * domain.y,
		domain.x * domain.y,
		domain.x * domainInv.y,
		domainInv.x * domainInv.y);
}

//--------------------------------------------------------------------------------------
// Calculate domain weights for 2x2 to 3x3 linear interpolation
//--------------------------------------------------------------------------------------
void DomainWeights(out float wd[9], uint2 idx)
{
	const int2 offset = int2(idx % 2) * 2 - 1;
	uint i = 0;

	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			const float2 d = min(6 - abs(int2(x, y) * 4 - offset), 4.0) / 8.0;
			wd[i] = d.x * d.y;
			++i;
		}
	}
}

//--------------------------------------------------------------------------------------
// Calculate domain weights for 3x3 linear interpolation
//--------------------------------------------------------------------------------------
void DomainWeights3x3(out float wd[9], uint2 idx)
{
	const int2 offset = int2(idx % 3) - 1;
	uint i = 0;
	float wds = 0.0;

	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			const float2 d = 1.0 - abs(int2(x, y) * 2 - offset) / 4.0;
			const float w = d.x * d.y;
			wd[i] = w;
			wds += w;
			++i;
		}
	}

	[unroll]
	for (i = 0; i < 9; ++i) wd[i] /= wds;
}
