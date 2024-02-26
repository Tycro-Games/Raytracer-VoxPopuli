#pragma once
class VoxelCube
{
public:
	VoxelCube() = default;
	VoxelCube(const float3 pos, const float3 size);
	float Intersect(const Ray& ray) const;
	bool Contains(const float3& pos) const;
	float3 b[2];
};
