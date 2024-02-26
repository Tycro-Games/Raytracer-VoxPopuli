#pragma once
#include "common.h"
#include "tmpl8math.h"

//based on https://ogldev.org/www/tutorial21/tutorial21.html
struct SpotLightData
{
	float3 position;
	float3 direction;
	float3 color;
	float angle;
};

class SpotLight : public Light
{
public:
	float3 Evaluate(Ray& ray, Scene& scene) override;
	SpotLightData data{{0, 0, 0}, {1, 0, 0}, {1, 1, 1}, CosDegrees(45.0f)};

private:
};
