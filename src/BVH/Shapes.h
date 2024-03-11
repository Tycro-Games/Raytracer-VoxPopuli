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
		if (rayLength < 0)
		{
			return;
		}
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
		if (rayLength < 0)
			return false;
		if (rayLength > r.t)
			return false;
		return true;
	}

	float3 center;
	float radius;
	MaterialType::MatType material = MaterialType::NONE;
};

//based on https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct Triangle
{
	void SetPos(const float3& pos);
	Triangle(MaterialType::MatType m);
	Triangle();


	void Hit(Ray& ray) const
	{
		float3 p1 = position, p2 = position, p3 = position;
		p1 += vertex0;
		p2 += vertex1;
		p3 += vertex2;
		const float3 edge1 = p2 - p1;
		const float3 edge2 = p3 - p1;
		const float3 h = cross(ray.D, edge2);
		const float a = dot(edge1, h);
		if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
		const float f = 1 / a;
		const float3 s = ray.O - p1;
		const float u = f * dot(s, h);
		if (u < 0 || u > 1) return;
		const float3 q = cross(s, edge1);
		const float v = f * dot(ray.D, q);
		if (v < 0 || u + v > 1) return;
		const float t = f * dot(edge2, q);
		if (t > 0.0001f)
		{
			if (ray.t > t)
			{
				ray.t = t;
				ray.indexMaterial = material;
				float3 normal = normalize(cross(edge1, edge2));
				const bool rayOutsideSphere{dot(ray.D, normal) < 0};
				ray.rayNormal = rayOutsideSphere ? normal : -normal;
			}
		}
	}

	bool IsHit(const Ray& ray) const
	{
		float3 p1 = position, p2 = position, p3 = position;
		p1 += vertex0;
		p2 += vertex1;
		p3 += vertex2;
		const float3 edge1 = p2 - p1;
		const float3 edge2 = p3 - p1;
		const float3 h = cross(ray.D, edge2);
		const float a = dot(edge1, h);
		if (a > -0.0001f && a < 0.0001f) return false; // ray parallel to triangle
		const float f = 1 / a;
		const float3 s = ray.O - p1;
		const float u = f * dot(s, h);
		if (u < 0 || u > 1) return false;
		const float3 q = cross(s, edge1);
		const float v = f * dot(ray.D, q);
		if (v < 0 || u + v > 1) return false;
		const float t = f * dot(edge2, q);
		if (t < 0.0001f)
		{
			return false;
		}
		if (t > ray.t)
		{
			return false;
		}
		return true;
	}

	float3 vertex0, vertex1, vertex2; //12
	float3 centroid; //16
	float3 position; //20
	MaterialType::MatType material = MaterialType::NONE; //24
};
