#pragma once
//following this tutorial: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
static constexpr int countVoxel = 2;

struct BVHNode
{
	float3 aabbMin, aabbMax;
	uint leftFirst, triCount;

	bool isLeaf() const
	{
		return triCount > 0;
	}
};

class BasicBVH
{
public:
	void UpdateBounds();
	BasicBVH();
	static void Grow(float3 p, float3& bmin, float3& bmax);
	void SetTransform(mat4& invTransform, uint32_t id);


	bool IntersectAABB(const Ray& ray, float3 bmin, float3 bmax);
	void IntersectBVH(Ray& ray, const uint nodeIdx, int32_t& idVolume);
	void BuildBVH();
	void UpdateNodeBounds(uint nodeIdx);
	void Subdivide(uint nodeIdx);
	std::array<Scene, countVoxel> voxelVolumes;
	uint32_t voxIdx[countVoxel];
	BVHNode bvhNode[countVoxel * 2 - 1];
	uint32_t rootNodeIdx = 0, nodesUsed = 1;

private:
};
