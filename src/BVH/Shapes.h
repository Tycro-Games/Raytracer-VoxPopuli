#pragma once
#include "scene.h"

struct Sphere
{
	Sphere(float3 c, float r, MaterialType::MatType m) : center(c), radius(r), material(m)
	{
	}

	Sphere();
	// For ray/sphere intersection: https://gamedev.stackexchange.com/questions/96459/fast-ray-sphere-collision-code

	void Hit(Ray& r)
	{
		const float3 toRay = r.O - center;
		const float b{dot(toRay, r.D)};
		const float c{sqrLength(toRay) - (radius * radius)};
		const float discriminant = b * b - c;
		if (c > 0.0f && b > 0.0f)
		{
			return;
		}
		if (discriminant < 0)
			return;

		const float rayLength{-b - sqrt(discriminant)};
		if (rayLength > r.t)
			return;
		// Calculate the outwards normal at the intersection point
		const float3 intersectionPoint{r.O + rayLength * r.D};
		const float3 outwardNormal{(intersectionPoint - center) / radius};

		// Inverse the normal if the ray is coming from inside the sphere
		const bool rayOutsideSphere{dot(r.D, outwardNormal) < 0};
		r.rayNormal = {rayOutsideSphere ? outwardNormal : -outwardNormal};
		r.isInsideGlass = !rayOutsideSphere;
		r.t = rayLength;
		r.indexMaterial = material;
	}

	bool IsHit(const Ray& r) const
	{
		const float3 toRay = r.O - center;
		const float b{dot(toRay, r.D)};
		const float c{sqrLength(toRay) - (radius * radius)};
		const float discriminant = b * b - c;
		if (c > 0.0f && b > 0.0f)
		{
			return false;
		}
		if (discriminant < 0)
			return false;

		const float rayLength{-b - sqrt(discriminant)};
		if (rayLength > r.t)
			return false;
		return true;
	}

	float3 center;
	float radius;
	MaterialType::MatType material = MaterialType::NONE;
};
