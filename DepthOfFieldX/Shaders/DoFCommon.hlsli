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
