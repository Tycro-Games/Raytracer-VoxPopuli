#include "precomp.h"
#include "SpotLight.h"

float3 SpotLight::Evaluate(Ray& ray, Scene& scene)
{
	const float3 intersectionPoint = ray.O + ray.t * ray.D;
	const float3 dir = data.position - intersectionPoint;
	float dst = length(dir);
	float3 dirNormalized = dir / dst;

	float3 normal = ray.GetNormal();
	//light angle
	float cosTheta = dot(dirNormalized, data.direction);
	if (cosTheta <= data.angle)
		return 0;

	float alphaCutOff = 1.0f - (1.0f - cosTheta) * 1.0f / (1.0f - data.angle);

	float3 lightIntensity = max(0.0f, cosTheta) * data.color / (dst * dst);
	//material evaluation
	Ray newRay;
	const float3 k = ray.GetAlbedo(scene);


	Ray shadowRay(OffsetRay(intersectionPoint, normal), dirNormalized);
	shadowRay.t = dst;
	if (scene.IsOccluded(shadowRay))
		return 0;

	return lightIntensity * k * alphaCutOff;
}
