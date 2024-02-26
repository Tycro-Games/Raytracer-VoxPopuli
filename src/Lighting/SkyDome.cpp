#include "precomp.h"
#include "SkyDome.h"

#include <stb_image.h>

SkyDome::SkyDome()
{
	// load HDR sky
	int skyBpp;
	skyPixels = stbi_loadf("assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0);
	for (int i = 0; i < skyWidth * skyHeight * 3; i++)
	{
		skyPixels[i] = sqrtf(skyPixels[i]);
	}
}

float3 SkyDome::SampleSky(Ray& ray) const
{
	// Sample sky
	const float uFloat = static_cast<float>(skyWidth) * atan2f(ray.D.z, ray.D.x) * INV2PI - 0.5f;
	const float vFloat = static_cast<float>(skyHeight) * acosf(ray.D.y) * INVPI - 0.5f;

	const int u = static_cast<int>(uFloat);
	const int v = static_cast<int>(vFloat);
	//TODO maybe FIX this
	const int skyIdx = max(0, u + v * skyWidth);
	return HDRLightContribution * float3(skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2]);
}
