#include "precomp.h"
#include "Shapes.h"

Sphere::Sphere()
{
	center = {0};
	radius = 1;
	material = MaterialType::NON_METAL_WHITE;
}

void Triangle::SetPos(const float3& pos)
{
	position = pos;
}

Triangle::Triangle(MaterialType::MatType m): material(m)
{
	const float3 r0(RandomFloat(), RandomFloat(), RandomFloat());
	const float3 r1(RandomFloat(), RandomFloat(), RandomFloat());
	const float3 r2(RandomFloat(), RandomFloat(), RandomFloat());
	vertex0 = r0 * 2 - float3(1);
	vertex1 = vertex0 + r1;
	vertex2 = vertex0 + r2;
	centroid = (vertex0 + vertex1 + vertex2) * 0.3333f;
	position = {0};
}

Triangle::Triangle()
{
	const float3 r0(RandomFloat(), RandomFloat(), RandomFloat());
	const float3 r1(RandomFloat(), RandomFloat(), RandomFloat());
	const float3 r2(RandomFloat(), RandomFloat(), RandomFloat());
	vertex0 = r0 * 9 - float3(5);
	vertex1 = vertex0 + r1;
	vertex2 = vertex0 + r2;
	centroid = (vertex0 + vertex1 + vertex2) * 0.3333f;
	position = {0};
	material = MaterialType::NON_METAL_WHITE;
}
