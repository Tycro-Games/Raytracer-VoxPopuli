#pragma once
//following this tutorial: https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct Tri
{
	float3 vertex0, vertex1, vertex2;
	float3 centroid;
};

static constexpr int countTri = 64;

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
	BasicBVH();
	void IntersectTri(Ray& ray, const Tri& _tri);


	bool IntersectAABB(const Ray& ray, float3 bmin, float3 bmax);
	void IntersectBVH(Ray& ray, uint nodeIdx);
	void BuildBVH();
	void UpdateNodeBounds(uint nodeIdx);
	void Subdivide(uint nodeIdx);
	Tri tri[countTri];
	uint triIdx[countTri];
	BVHNode bvhNode[countTri * 2 - 1];
	uint rootNodeIdx = 0, nodesUsed = 1;

private:
};
