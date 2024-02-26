#include "precomp.h"
#include "Material.h"

DiffuseMaterial::DiffuseMaterial(float3 albedo): albedo(albedo)
{
}


float3 DiffuseMaterial::GetAlbedo()
{
	return albedo;
}

float DiffuseMaterial::GetReflectivity()
{
	return 0;
}

ReflectivityMaterial::ReflectivityMaterial(float3 albedo, float reflectivity): albedo(albedo),
                                                                               reflectivity(reflectivity)
{
}

float3 ReflectivityMaterial::GetAlbedo()
{
	return albedo;
}

float ReflectivityMaterial::GetReflectivity()
{
	return reflectivity;
}
