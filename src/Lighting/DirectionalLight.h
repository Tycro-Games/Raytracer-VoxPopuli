#pragma once
struct DirectionalLightData
{
	float3 direction;
	float3 color;
};

class DirectionalLight
{
public:
	//float3 Evaluate(Ray& ray, Scene& scene) override;
	DirectionalLightData data{{1, 0, 0}, {0, 0, 0}};
};
