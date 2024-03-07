﻿#pragma once

class Material
{
public:
	virtual ~Material() = default;
	explicit Material(float3 _albedo, float _roughness = 1.0f);
	float3 albedo{1};
	float roughness{0};
	float emissiveStrength{0};
	float IOR{1.5f};
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
