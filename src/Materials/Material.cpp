#include "precomp.h"
#include "Material.h"


ReflectivityMaterial::ReflectivityMaterial(float3 _albedo, float _roughness)
{
	albedo = _albedo;
	roughness = _roughness;
}
