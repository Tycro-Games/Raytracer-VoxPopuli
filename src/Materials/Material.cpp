#include "precomp.h"
#include "Material.h"


ReflectivityMaterial::ReflectivityMaterial(float3 _albedo, float _roughness)
{
	albedo = _albedo;
	roughness = _roughness;
}

float3 ReflectivityMaterial::GetAlbedo()
{
	return albedo;
}

float ReflectivityMaterial::GetRoughness()
{
	return roughness;
}
