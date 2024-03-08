#pragma once
//from Jacco: https://jacco.ompf2.com/2022/05/27/how-to-build-a-bvh-part-8-whitted-style/
class SkyDome
{
public:
	SkyDome();
	float3 SampleSky(const float3& direction) const;
	float HDRLightContribution = 5.9f;

private:
	int skyWidth, skyHeight;
	float* skyPixels;
};
