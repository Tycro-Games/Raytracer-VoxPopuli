#include "precomp.h"


#pragma warning(push, 0)
#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"
#pragma warning(pop)
float3 Ray::ComputeDsign(const float3& _D) const
{
	const uint x_sign = (*(uint*)&_D.x >> 31);
	const uint y_sign = (*(uint*)&_D.y >> 31);
	const uint z_sign = (*(uint*)&_D.z >> 31);

	return (float3(static_cast<float>(x_sign) * 2 - 1, static_cast<float>(y_sign) * 2 - 1,
	               static_cast<float>(z_sign) * 2 - 1) + 1) * 0.5f;
}

Ray::Ray(const float3 origin, const float3 direction, const float rayLength, const int) : O(origin), t(rayLength)
{
	D = normalize(direction);
	// calculate reciprocal ray direction for triangles and AABBs
	// TODO: prevent NaNs - or don't
	rD = float3(1 / D.x, 1 / D.y, 1 / D.z);
	//if (std::isnan(rD.x) || isnan(rD.y) || isnan(rD.z))
	//	std::cout << "NaN in rD.x" << std::endl;
	//faster than reinterpret cast by about 1%
	Dsign = ComputeDsign(D);
}

//added comments with chatGPT
float3 Ray::GetNormalVoxel(const uint32_t worldSize, const mat4& invMatrix) const
{
	const float3 ori = TransformPosition(O, invMatrix);

	const float3 dir = (TransformVector(D, invMatrix));

	float3 intersectionPoint = invMatrix.TransformPoint(IntersectionPoint());
	// Calculate the intersection point
	const float3 I1 = intersectionPoint * static_cast<float>(worldSize);

	// Calculate fractional part of I1
	const float3 fG = fracf(I1);

	// Calculate distances to boundaries											  
	const float3 d = min3(fG, 1.0f - fG);
	const float mind = min(min(d.x, d.y), d.z);

	// Calculate signs
	const float3 sign = ComputeDsign(dir) * 2 - 1;

	// Determine the normal based on the minimum distance
	float3 normal = float3(mind == d.x ? sign.x : 0, mind == d.y ? sign.y : 0, mind == d.z ? sign.z : 0);

	// Transform the normal from object space to world space
	normal = normalize(TransformVector(normal, invMatrix.Inverted()));


	return (normal);
	// TODO:
	// - *only* in case the profiler flags this as a bottleneck:
	// - This function might benefit from SIMD.
}

float3 Ray::UintToFloat3(uint col) const
{
	const uint8_t red = (col >> 16) & 0xFF; // Red component
	const uint8_t green = (col >> 8) & 0xFF; // Green component
	const uint8_t blue = col & 0xFF; // Blue component

	// Normalize color components to the range [0, 1]
	const float normRed = static_cast<float>(red) / 255.0f;
	const float normGreen = static_cast<float>(green) / 255.0f;
	const float normBlue = static_cast<float>(blue) / 255.0f;

	// Return the color as a float3
	return float3(normRed, normGreen, normBlue);
}


float3 Ray::GetAlbedo(const Renderer& scene) const
{
	return scene.materials[indexMaterial]->albedo;
}

float Ray::GetEmissive(const Renderer& scene) const
{
	return scene.materials[indexMaterial]->emissiveStrength;
}

float Ray::GetRefractivity(const Renderer& scene) const
{
	return scene.materials[indexMaterial]->IOR;
}

float Ray::GetRoughness(const Renderer& scene) const
{
	return scene.materials[indexMaterial]->roughness;
}


float Cube::Intersect(const Ray& ray) const
{
	//rewritten by chatgpt
	// Determine the signs of ray direction components
	const int signx = ray.D.x < 0;
	const int signy = ray.D.y < 0;
	const int signz = ray.D.z < 0;

	// Calculate t-values for intersection with each face of the cube
	float tmin_x = (b[signx].x - ray.O.x) * ray.rD.x;
	float tmax_x = (b[1 - signx].x - ray.O.x) * ray.rD.x;

	const float tmin_y = (b[signy].y - ray.O.y) * ray.rD.y;
	const float tmax_y = (b[1 - signy].y - ray.O.y) * ray.rD.y;

	// Check for intersection with Y faces
	if (tmin_x > tmax_y || tmin_y > tmax_x)
		return 1e34f; // No intersection

	// Update tmin and tmax
	tmin_x = std::max(tmin_x, tmin_y);
	tmax_x = std::min(tmax_x, tmax_y);

	const float tmin_z = (b[signz].z - ray.O.z) * ray.rD.z;
	const float tmax_z = (b[1 - signz].z - ray.O.z) * ray.rD.z;

	// Check for intersection with Z faces
	if (tmin_x > tmax_z || tmin_z > tmax_x)
		return 1e34f; // No intersection

	// Final intersection
	tmin_x = std::max(tmin_x, tmin_z);
	if (tmin_x > 0)
		return tmin_x;

	return 1e34f; // No intersection
}


bool Cube::Contains(const float3& pos) const
{
	// test if pos is inside the cube
	return pos.x >= b[0].x && pos.y >= b[0].y && pos.z >= b[0].z &&
		pos.x <= b[1].x && pos.y <= b[1].y && pos.z <= b[1].z;
}


void Scene::SetCubeBoundaries(const float3& position)
{
	cube.b[0] = position;
	cube.b[1] = position + cube.scale;
}


void Scene::SetTransform(const mat4& transform)
{
	cube.invMatrix = transform.Inverted();
	// calculate world-space bounds using the new matrix
	float3 bmin = {cube.b[0]}, bmax = {cube.b[1]};
	cube.b[0] = 1e30f;
	cube.b[1] = -1e30f;
	for (int i = 0; i < 8; i++)
		cube.Grow(TransformPosition(float3(i & 1 ? bmax.x : bmin.x,
		                                   i & 2 ? bmax.y : bmin.y, i & 4 ? bmax.z : bmin.z), transform));
}

// Function to set scaling of the voxel cube
void Scene::SetScale(const float3& scl)
{
	cube.scale = scl; // Store the scaling factors
}

void Scene::GenerateSomeNoise(float frequency = 0.03f)
{
	ResetGrid();
	//from https://github.com/Auburn/FastNoise2/wiki/3:-Getting-started-using-FastNoise2

	const auto fnPerlin = FastNoise::New<FastNoise::Perlin>();


	// Create an array of floats to store the noise output in
	std::vector<float> noiseOutput(GRIDSIZE3);
	fnPerlin->GenUniformGrid3D(noiseOutput.data(), 0, 0, 0, WORLDSIZE, WORLDSIZE, WORLDSIZE, frequency, RandomUInt());


	for (uint32_t z = 0; z < WORLDSIZE; z++)
	{
		for (uint32_t y = 0; y < WORLDSIZE; y++)
		{
			for (uint32_t x = 0; x < WORLDSIZE; x++)
			{
				const float n = noiseOutput[x + y * WORLDSIZE + z * WORLDSIZE * WORLDSIZE];
				// Sample noise from pre-generated vector
				MaterialType::MatType color = MaterialType::NONE;


				if (n <= 0.04f)
				{
					color = MaterialType::NONE;
				}
				else if (n < 0.08)
				{
					color = static_cast<MaterialType::MatType>(Rand(static_cast<float>(MaterialType::NONE)));
				}
				else if (n < 0.2)
				{
					color = MaterialType::NON_METAL_RED;
				}
				else if (n < 0.17f)
				{
					color = MaterialType::NON_METAL_WHITE;
				}
				else if (n < 0.3)
				{
					color = MaterialType::GLASS;
				}
				else if (n < 0.5f)
					color = MaterialType::METAL_HIGH;
				else if (n < 0.7f)
					color = MaterialType::METAL_MID;
				else if (n < 0.9f)
					color = MaterialType::METAL_LOW;


				Set(x, y, z, color); // Assuming Set function is defined elsewhere
			}
		}
	}
}

void Scene::ResetGrid(MaterialType::MatType type)
{
	std::fill(grid.begin(), grid.end(), type);
}

void Scene::SetTransform(float3& rotation)
{
	//as Max (230184) explained how I could rotate around a pivot

	// Translate the object to the pivot point (center of the cube)
	const mat4 translateToPivot = mat4::Translate((cube.b[0] + cube.b[1]) * 0.5f);

	// Translate back to the original position after rotation
	const mat4 translateBack = mat4::Translate(-((cube.b[0] + cube.b[1]) * 0.5f));

	// Scale the object
	const mat4 scale = mat4::Scale(cube.scale);

	// Rotate the object around the pivot point
	const mat4 rot = mat4::RotateX(rotation.x) * mat4::RotateY(rotation.y) * mat4::RotateZ(rotation.z);

	// Calculate the inverse transformation matrix
	cube.invMatrix = (translateToPivot * rot * scale * translateBack).Inverted();
}

Scene::Scene(const float3& position, const uint32_t worldSize) : WORLDSIZE(worldSize), GRIDSIZE(worldSize),
                                                                 GRIDSIZE2(worldSize * worldSize),
                                                                 GRIDSIZE3(worldSize * worldSize * worldSize)

{
	//sets the cube
	grid.resize(GRIDSIZE3);
	SetCubeBoundaries(position);
	ResetGrid(MaterialType::NON_METAL_BLUE);
	// initialize the mainScene using Perlin noise, parallel over z
	//LoadModel("assets/teapot.vox");
	if (worldSize > 1)
		GenerateSomeNoise();
}


// a helper function to load a magica voxel scene given a filename from https://github.com/jpaver/opengametools/blob/master/demo/demo_vox.cpp
void Scene::LoadModel(Renderer& renderer, const char* filename, uint32_t scene_read_flags)
{
	ResetGrid();
	// open the file
#if defined(_MSC_VER) && _MSC_VER >= 1400
	FILE* fp;
	if (0 != fopen_s(&fp, filename, "rb"))
		fp = 0;
#else
	FILE* fp = fopen(filename, "rb");
#endif
	if (!fp)
		return;

	// get the buffer size which matches the size of the file
	fseek(fp, 0, SEEK_END);
	const uint32_t buffer_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// load the file into a memory buffer
	uint8_t* buffer = new uint8_t[buffer_size];
	fread(buffer, buffer_size, 1, fp);
	fclose(fp);

	// construct the scene from the buffer
	const auto scene = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags)->models[0];
	const auto scenePallete = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags)->palette;
	//created using chatgpt promts
	// Assign colors based on the loaded scene
	// Define scaling factors for each dimension
	if (scene->size_x > GRIDSIZE)
	{
		// Modify scaleModel variables based on the comparison
		scaleModel.x *= static_cast<float>(GRIDSIZE) / static_cast<float>(scene->size_x);
		scaleModel.y *= static_cast<float>(GRIDSIZE) / static_cast<float>(scene->size_y);
		scaleModel.z *= static_cast<float>(GRIDSIZE) / static_cast<float>(scene->size_z);
	}
	//do stuff
	for (uint32_t z = 0; z < scene->size_z; ++z)
	{
		for (uint32_t y = 0; y < scene->size_y; ++y)
		{
			for (uint32_t x = 0; x < scene->size_x; ++x)
			{
				// Calculate the scaled position
				// Calculate the scaled position
				const int scaledX = static_cast<int>((static_cast<float>(x) *
					scaleModel.x));
				const int scaledY = static_cast<int>(static_cast<float>(z) *
					scaleModel.y);
				const int scaledZ = static_cast<int>(static_cast<float>(y) *
					scaleModel.z);


				// Assume each voxel has a color index, and map that to MatType
				MaterialType::MatType materialIndex = MaterialType::NONE;
				// Calculate index into voxel_data based on the current position
				const uint32_t index = x + y * scene->size_x + z * scene->size_x * scene->size_y;
				// Access color index from voxel_data
				uint8_t voxelColorIndex = scene->voxel_data[index];
				const auto col = scenePallete.color[voxelColorIndex];

				if (voxelColorIndex == 0)
				{
					continue;
				}

				materialIndex = static_cast<MaterialType::MatType>(voxelColorIndex);
				renderer.materials[materialIndex]->albedo = (float3(static_cast<float>(col.r) / 255.0f,
				                                                    static_cast<float>(col.g) / 255.0f,
				                                                    static_cast<float>(col.b) / 255.0f));
				renderer.materials[materialIndex]->roughness = 1.0f;
				// Set the color at position (x, y, z)
				Set(scaledX, scaledY, scaledZ, materialIndex);
			}
		}
	}

	// Cleanup
	delete[] buffer;
}

void Scene::CreateEmmisiveSphere(MaterialType::MatType mat, float radiusEmissiveSphere)
{
	//ResetGrid();
	//based on Lynn's implementation
	// When looping over (x, y, z) during scene creation

	const float worldCenter{(static_cast<float>(WORLDSIZE) / 2.0f)};

	for (uint32_t z = 0; z < WORLDSIZE; ++z)
	{
		for (uint32_t y = 0; y < WORLDSIZE; ++y)
		{
			for (uint32_t x = 0; x < WORLDSIZE; ++x)
			{
				const float3 point{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
				const float distanceSquared{length(worldCenter - point)}; // Distance from the center of the world


				if (distanceSquared < radiusEmissiveSphere)
				{
					// The voxel exists
					Set(x, y, z, mat);
				}
			}
		}
	}
}

void Scene::Set(const uint x, const uint y, const uint z, const MaterialType::MatType v)
{
	grid[GetVoxel(x, y, z)] = v;
}

//This changes to any position now
bool Scene::Setup3DDDA(Ray& ray, DDAState& state) const
{
	Ray backupRay = ray;
	ray.O = TransformPosition(ray.O, cube.invMatrix);

	ray.D = TransformVector(ray.D, cube.invMatrix);

	ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
	// if ray is not inside the world: advance until it is
	state.t = 0;
	if (!cube.Contains(ray.O))
	{
		state.t = cube.Intersect(ray);
		if (state.t > 1e33f)
		{
			backupRay.t = ray.t;
			ray = backupRay;
			return false;
		}
		// ray misses voxel data entirely
	}
	//expressed in world space
	const float3 voxelMinBounds = cube.b[0];
	const float3 voxelMaxBounds = cube.b[1] - cube.b[0];
	const auto gridsizeFloat = static_cast<float>(GRIDSIZE);
	const float cellSize = 1.0f / gridsizeFloat;
	state.step = make_int3(1 - ray.Dsign * 2);
	//based on our cube position
	const float3 posInGrid = gridsizeFloat * ((ray.O - voxelMinBounds) + (state.t + 0.00005f) * ray.D) /
		voxelMaxBounds;
	const float3 gridPlanes = (ceilf(posInGrid) - ray.Dsign) * cellSize;
	const int3 P = clamp(make_int3(posInGrid), 0, GRIDSIZE - 1);
	state.X = P.x, state.Y = P.y, state.Z = P.z;
	state.tdelta = cellSize * float3(state.step) * ray.rD;
	state.tmax = ((gridPlanes * voxelMaxBounds) - (ray.O - voxelMinBounds)) * ray.rD;

	backupRay.t = ray.t;
	ray = backupRay;
	return true;
}

void Scene::FindNearest(Ray& ray) const
{
	//TODO maybe try to move this

	// setup Amanatides & Woo grid traversal
	DDAState s;

	if (!Setup3DDDA(ray, s))
	{
		return;
	}
	// start stepping
	while (s.t < ray.t)
	{
		const MaterialType::MatType cell = grid[GetVoxel(s.X, s.Y, s.Z)];
		if (cell != MaterialType::NONE && s.t < ray.t)
		{
			ray.t = s.t;
			ray.rayNormal = ray.GetNormalVoxel(WORLDSIZE, cube.invMatrix);
			ray.indexMaterial = cell;
			break;
		}
		if (s.tmax.x < s.tmax.y)
		{
			if (s.tmax.x < s.tmax.z)
			{
				s.t = s.tmax.x, s.X += s.step.x;
				if (s.X >= GRIDSIZE) break;
				s.tmax.x += s.tdelta.x;
			}
			else
			{
				s.t = s.tmax.z, s.Z += s.step.z;
				if (s.Z >= GRIDSIZE) break;
				s.tmax.z += s.tdelta.z;
			}
		}
		else
		{
			if (s.tmax.y < s.tmax.z)
			{
				s.t = s.tmax.y, s.Y += s.step.y;
				if (s.Y >= GRIDSIZE) break;
				s.tmax.y += s.tdelta.y;
			}
			else
			{
				s.t = s.tmax.z, s.Z += s.step.z;
				if (s.Z >= GRIDSIZE) break;
				s.tmax.z += s.tdelta.z;
			}
		}
	}

	// TODO:
	// - A nested grid will let rays skip empty space much faster.
	// - Coherent rays can traverse the grid faster together.
	// - Perhaps s.X / s.Y / s.Z (the integer grid coordinates) can be stored in a single uint?
	// - Loop-unrolling may speed up the while loop.
	// - This code can be ported to GPU.
}

bool Scene::FindMaterialExit(Ray& ray, MaterialType::MatType matType) const
{
	//TODO maybe try to move this
	// setup Amanatides & Woo grid traversal
	DDAState s;
	if (!Setup3DDDA(ray, s))
	{
		// proceed with traversal

		return false;
	}
	// start stepping
	while (1)
	{
		const MaterialType::MatType cell = grid[GetVoxel(s.X, s.Y, s.Z)];
		if (cell != matType)
		{
			ray.t = s.t;
			ray.rayNormal = ray.GetNormalVoxel(WORLDSIZE, cube.invMatrix);
			ray.indexMaterial = cell;
			return true;
		}
		if (s.tmax.x < s.tmax.y)
		{
			if (s.tmax.x < s.tmax.z)
			{
				s.t = s.tmax.x, s.X += s.step.x;
				if (s.X >= GRIDSIZE) break;
				s.tmax.x += s.tdelta.x;
			}
			else
			{
				s.t = s.tmax.z, s.Z += s.step.z;
				if (s.Z >= GRIDSIZE) break;
				s.tmax.z += s.tdelta.z;
			}
		}
		else
		{
			if (s.tmax.y < s.tmax.z)
			{
				s.t = s.tmax.y, s.Y += s.step.y;
				if (s.Y >= GRIDSIZE) break;
				s.tmax.y += s.tdelta.y;
			}
			else
			{
				s.t = s.tmax.z, s.Z += s.step.z;
				if (s.Z >= GRIDSIZE) break;
				s.tmax.z += s.tdelta.z;
			}
		}
	}
	ray.O = ray.O + ray.D * s.t;
	ray.t = 0;

	//ray.rayNormal = ray.GetNormalVoxel();

	// TODO:
	// - A nested grid will let rays skip empty space much faster.
	// - Coherent rays can traverse the grid faster together.
	// - Perhaps s.X / s.Y / s.Z (the integer grid coordinates) can be stored in a single uint?
	// - Loop-unrolling may speed up the while loop.
	// - This code can be ported to GPU.
	return false;
}


bool Scene::IsOccluded(Ray& ray) const
{
	// setup Amanatides & Woo grid traversal
	DDAState s;
	if (!Setup3DDDA(ray, s)) return false;
	// start stepping
	while (s.t < ray.t)
	{
		const auto cell = grid[GetVoxel(s.X, s.Y, s.Z)];
		if (cell != MaterialType::NONE) /* we hit a solid voxel */ return s.t < ray.t;
		if (s.tmax.x < s.tmax.y)
		{
			if (s.tmax.x < s.tmax.z)
			{
				if ((s.X += s.step.x) >= GRIDSIZE) return false;
				s.t = s.tmax.x, s.tmax.x += s.tdelta.x;
			}
			else
			{
				if ((s.Z += s.step.z) >= GRIDSIZE) return false;
				s.t = s.tmax.z, s.tmax.z += s.tdelta.z;
			}
		}
		else
		{
			if (s.tmax.y < s.tmax.z)
			{
				if ((s.Y += s.step.y) >= GRIDSIZE) return false;
				s.t = s.tmax.y, s.tmax.y += s.tdelta.y;
			}
			else
			{
				if ((s.Z += s.step.z) >= GRIDSIZE) return false;
				s.t = s.tmax.z, s.tmax.z += s.tdelta.z;
			}
		}
	}
	return false;
}
