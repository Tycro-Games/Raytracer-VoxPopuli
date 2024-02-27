#pragma once

class Material
{
public:
	virtual ~Material() = default;
	virtual float3 GetAlbedo() = 0;
	virtual float GetRoughness() = 0;
};


class ReflectivityMaterial : public Material
{
public:
	ReflectivityMaterial(float3 albedo, float rougness = 1.0f);

	float3 GetAlbedo() override;
	float GetRoughness() override;
	float3 albedo{1};
	float roughness{0};
};

//class Reflective : public Material
//{
//public:
//	DiffuseMaterial(float3 albedo);
//
//	float3 Evaluate(float3 normal, float3 intersectionP, Ray& outRay) override;
//	float3 GetAlbedo() override;
//	float3 albedo{1};
//};
