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

float3 SkyDome::SampleSky(const float3& direction) const
{
	// Sample sky
	const float uFloat = static_cast<float>(skyWidth) * atan2f(direction.z, direction.x) * INV2PI - 0.5f;
	const float vFloat = static_cast<float>(skyHeight) * acosf(direction.y) * INVPI - 0.5f;

	const int u = static_cast<int>(uFloat);
	const int v = static_cast<int>(vFloat);

	const int skyIdx = max(0, u + v * skyWidth) * 3;
	return HDRLightContribution * float3(skyPixels[skyIdx], skyPixels[skyIdx + 1], skyPixels[skyIdx + 2]);
}
