#pragma once

class Material
{
public:
	virtual ~Material() = default;
	virtual float3 GetAlbedo() = 0;
	virtual float GetReflectivity() = 0;
};

class DiffuseMaterial : public Material
{
public:
	DiffuseMaterial(float3 albedo);

	float3 GetAlbedo() override;
	float3 albedo{1};
	float GetReflectivity() override;
};

class ReflectivityMaterial : public Material
{
public:
	ReflectivityMaterial(float3 albedo, float reflectivity = 1.0f);

	float3 GetAlbedo() override;
	float GetReflectivity() override;
	float3 albedo{1};
	float reflectivity;
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
