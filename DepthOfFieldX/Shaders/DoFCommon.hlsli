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
	return max((abs(coc) - 1.0) * 3.0 + 1.0, 0.0);
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
float CalcMipLevelRadius(uint level, float len = 0.5)
{
	return ((1u << level) * len - 0.5) * sqrt(2.0);
}

//--------------------------------------------------------------------------------------
// Calculate the radius for the corresponding mip level
//--------------------------------------------------------------------------------------
float CalcMipLevelRadius3x3(uint level, float len = 0.5)
{
	return (pow(3.0, level) * len - 0.5) * sqrt(2.0);
}

//--------------------------------------------------------------------------------------
// Fetch 3x3 samples
//--------------------------------------------------------------------------------------
void Fetch3x3(out float4 samples3x3[9], Texture2D txSrc, uint2 pos)
{
	uint i = 0;
	[unroll]
	for (int y = -1; y <= 1; ++y)
	{
		[unroll]
		for (int x = -1; x <= 1; ++x)
		{
			const uint2 idx = (int2)pos + int2(x, y);
			samples3x3[i++] = txSrc[idx];
		}
	}
}
