#include "precomp.h"
#include "BasicBVH.h"

void BasicBVH::UpdateBounds()
{
	for (auto& voxelVolume : voxelVolumes)
	{
		if (voxelVolume.flag)
		{
			voxelVolume.flag = false;
			SetTransform(voxelVolume.cube.invMatrix, voxelVolume.index);
		}
	}
}

BasicBVH::BasicBVH()
{
	constexpr int sizeX = 2;
	constexpr int sizeY = 1;
	constexpr int sizeZ = 1;
	int id = 0;
	const array powersTwo = {64, 2, 4, 8, 16, 32, 64};
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			for (int k = 0; k < sizeZ; k++)
			{
				const int index = (k + i + j) % powersTwo.size();
				voxelVolumes[id].cube.position = float3{
					static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)
				};

				voxelVolumes[id].SetWorldSize(powersTwo[index]);

				voxelVolumes[id].GenerateSomeNoise(0.03f);
				voxelVolumes[id].SetTransform(float3{0});

				id++;
			}
		}
	}
	BuildBVH();
	UpdateBounds();
}

void BasicBVH::Grow(float3 p, float3& bmin, float3& bmax)
{
	bmin = fminf(bmin, p);
	bmax = fmaxf(bmax, p);
}

void BasicBVH::SetTransform(mat4& invTransform, uint32_t id)
{
	// calculate world-space bounds using the new matrix
	BVHNode& node = bvhNode[id];
	float3 bmin = node.aabbMin, bmax = node.aabbMax;
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (int i = 0; i < 8; i++)
		Grow((TransformPosition(float3(i & 1 ? bmax.x : bmin.x,
		                               i & 2 ? bmax.y : bmin.y, i & 4 ? bmax.z : bmin.z), invTransform.Inverted())),
		     node.aabbMin,
		     node.aabbMax);
}

bool BasicBVH::IntersectAABB(const Ray& ray, const float3 bmin, const float3 bmax)
{
	float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
	float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
	float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
	tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
	float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
	tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
	return tmax >= tmin && tmin < ray.t && tmax > 0;
}

void BasicBVH::IntersectBVH(Ray& ray, const uint nodeIdx, int32_t& idVolume)
{
	BVHNode& node = bvhNode[nodeIdx];
	if (!IntersectAABB(ray, node.aabbMin, node.aabbMax))
		return;
	if (node.isLeaf())
	{
		for (uint i = 0; i < node.triCount; i++)
		{
			const int32_t index = voxIdx[node.leftFirst + i];

			/*	Ray backupRay = ray;
				mat4 invTransform = voxelVolumes[voxIdx[index]].cube.invMatrix;
				ray.O = TransformPosition(ray.O, invTransform);
				ray.D = TransformVector(ray.D, invTransform);
				ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
				ray.Dsign = ray.ComputeDsign(ray.D);*/

			if (voxelVolumes[voxIdx[index]].FindNearest(ray))
				idVolume = index;
			/*backupRay.t = ray.t;
			ray = backupRay;*/
		}
	}
	else
	{
		IntersectBVH(ray, node.leftFirst, idVolume);
		IntersectBVH(ray, node.leftFirst + 1, idVolume);
	}
}

void BasicBVH::BuildBVH()
{
	for (int i = 0; i < countVoxel; i++)
		voxIdx[i] = i;

	for (int i = 0; i < countVoxel; i++)
		voxelVolumes[i].centroid = voxelVolumes[i].GetCenter();
	// assign all triangles to root node
	BVHNode& root = bvhNode[rootNodeIdx];
	root.leftFirst = 0, root.triCount = countVoxel;

	UpdateNodeBounds(rootNodeIdx);
	// subdivide recursively
	Subdivide(rootNodeIdx);
}

void BasicBVH::UpdateNodeBounds(uint nodeIdx)
{
	BVHNode& node = bvhNode[nodeIdx];
	node.aabbMin = float3(1e30f);
	node.aabbMax = float3(-1e30f);
	for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
	{
		uint leafTriIdx = voxIdx[first + i];
		Scene& leafTri = voxelVolumes[leafTriIdx];
		float3 position = leafTri.cube.position;
		node.aabbMin = fminf(node.aabbMin, leafTri.cube.b[0]),
			node.aabbMax = fmaxf(node.aabbMax, leafTri.cube.b[1]);
	}
}

void BasicBVH::Subdivide(uint nodeIdx)
{
	// terminate recursion
	BVHNode& node = bvhNode[nodeIdx];
	if (node.triCount <= 1)
	{
		voxelVolumes[node.leftFirst].index = nodeIdx;
		return;
	}
	// determine split axis and position
	float3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (voxelVolumes[voxIdx[i]].centroid[axis] < splitPos)
			i++;
		else
			swap(voxIdx[i], voxIdx[j--]);
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || static_cast<uint>(leftCount) == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodesUsed++;
	int rightChildIdx = nodesUsed++;
	bvhNode[leftChildIdx].leftFirst = node.leftFirst;
	bvhNode[leftChildIdx].triCount = leftCount;
	bvhNode[rightChildIdx].leftFirst = i;
	bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;
	UpdateNodeBounds(leftChildIdx);
	UpdateNodeBounds(rightChildIdx);
	// recurse
	Subdivide(leftChildIdx);
	Subdivide(rightChildIdx);
}
