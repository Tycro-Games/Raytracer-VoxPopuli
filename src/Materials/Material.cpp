#include "precomp.h"
#include "Material.h"


ReflectivityMaterial::ReflectivityMaterial(float3 albedo, float rougness): albedo(albedo),
                                                                           roughness(rougness)
{
}

float3 ReflectivityMaterial::GetAlbedo()
{
	return albedo;
}

float ReflectivityMaterial::GetRoughness()
{
	return roughness;
}
