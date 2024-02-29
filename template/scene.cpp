#include "precomp.h"


#pragma warning(push, 0)
#define OGT_VOX_IMPLEMENTATION
#include "ogt_vox.h"
#pragma warning(pop)
Ray::Ray(const float3 origin, const float3 direction, const float rayLength, const int): O(origin), t(rayLength)
{
	D = normalize(direction);
	// calculate reciprocal ray direction for triangles and AABBs
	// TODO: prevent NaNs - or don't
	rD = float3(1 / D.x, 1 / D.y, 1 / D.z);
	//if (std::isnan(rD.x) || isnan(rD.y) || isnan(rD.z))
	//	std::cout << "NaN in rD.x" << std::endl;
	//faster than reinterpret cast by about 1%
	uint x_sign = (*(uint*)&D.x >> 31);
	uint y_sign = (*(uint*)&D.y >> 31);
	uint z_sign = (*(uint*)&D.z >> 31);

	Dsign = (float3((float)x_sign * 2 - 1, (float)y_sign * 2 - 1, (float)z_sign * 2 - 1) + 1) * 0.5f;
}


Ray Ray::GetRefractedRay(const Ray& ray, const float IORRatio)
{
	const float3 rayIn = ray.D;
	const float3 rayNormal = ray.GetNormal();

	const float cosTheta = min(dot(-rayIn, rayNormal), 1.0f);
	const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	bool cannotRefract = IORRatio * sinTheta > 1.0f;

	const float3 rayOutPerpendicular = IORRatio * (rayIn + cosTheta * rayNormal);
	const float3 rayOutParallel = -sqrtf(fabsf(1.0f - sqrLength(rayOutPerpendicular))) *
		rayNormal;
	float3 direction;
	if (cannotRefract || Reflectance(cosTheta, IORRatio) > RandomFloat())
	{
		direction = {ray.D - 2 * rayNormal * dot(rayNormal, ray.D)};
	}
	else
	{
		direction = rayOutPerpendicular + rayOutParallel;
	}
	return {OffsetRay(ray.IntersectionPoint(), rayNormal), direction};
}

float3 Ray::GetNormal() const
{
	// return the voxel normal at the nearest intersection
	const float3 I1 = (O + t * D) * WORLDSIZE; // our mainScene size is (1,1,1), so this scales each voxel to (1,1,1)
	const float3 fG = fracf(I1);
	const float3 d = min3(fG, 1.0f - fG);
	const float mind = min(min(d.x, d.y), d.z);
	const float3 sign = Dsign * 2 - 1;
	return float3(mind == d.x ? sign.x : 0, mind == d.y ? sign.y : 0, mind == d.z ? sign.z : 0);
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


float3 Ray::GetAlbedo(const Scene& scene) const
{
	return scene.materials[indexMaterial]->albedo;
}

float3 Ray::GetEmissive(const Scene& scene) const
{
	return scene.materials[indexMaterial]->emissiveStrength;
}

float Ray::GetRefractivity(const Scene& scene) const
{
	return scene.materials[indexMaterial]->IOR;
}

float Ray::GetRoughness(const Scene& scene) const
{
	return scene.materials[indexMaterial]->roughness;
}


Cube::Cube(const float3 pos, const float3 size)
{
	// set cube bounds
	b[0] = pos;
	b[1] = pos + size;
}

float Cube::Intersect(const Ray& ray) const
{
	// test if the ray intersects the cube
	const int signx = ray.D.x < 0, signy = ray.D.y < 0, signz = ray.D.z < 0;
	float tmin = (b[signx].x - ray.O.x) * ray.rD.x;
	float tmax = (b[1 - signx].x - ray.O.x) * ray.rD.x;
	const float tymin = (b[signy].y - ray.O.y) * ray.rD.y;
	const float tymax = (b[1 - signy].y - ray.O.y) * ray.rD.y;
	if (tmin > tymax || tymin > tmax) goto miss;
	tmin = max(tmin, tymin), tmax = min(tmax, tymax);
	const float tzmin = (b[signz].z - ray.O.z) * ray.rD.z;
	const float tzmax = (b[1 - signz].z - ray.O.z) * ray.rD.z;
	if (tmin > tzmax || tzmin > tmax) goto miss; // yeah c has 'goto' ;)
	if ((tmin = max(tmin, tzmin)) > 0) return tmin;
miss:
	return 1e34f;
}

bool Cube::Contains(const float3& pos) const
{
	// test if pos is inside the cube
	return pos.x >= b[0].x && pos.y >= b[0].y && pos.z >= b[0].z &&
		pos.x <= b[1].x && pos.y <= b[1].y && pos.z <= b[1].z;
}

void Scene::GenerateSomeNoise(float frequency = 0.03f)
{
	ResetGrid();
	//from https://github.com/Auburn/FastNoise2/wiki/3:-Getting-started-using-FastNoise2

	const auto fnPerlin = FastNoise::New<FastNoise::Perlin>();


	// Create an array of floats to store the noise output in
	std::vector<float> noiseOutput(GRIDSIZE3);
	fnPerlin->GenUniformGrid3D(noiseOutput.data(), 0, 0, 0, WORLDSIZE, WORLDSIZE, WORLDSIZE, frequency, RandomUInt());


	for (int z = 0; z < WORLDSIZE; z++)
	{
		for (int y = 0; y < WORLDSIZE; y++)
		{
			for (int x = 0; x < WORLDSIZE; x++)
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

void Scene::ResetGrid()
{
	grid.fill(MaterialType::NONE);
}

Scene::Scene()
{
	// the voxel world sits in a 1x1x1 cube
	cube = Cube(float3(0, 0, 0), float3(1, 1, 1));
	ResetGrid();
	// initialize the mainScene using Perlin noise, parallel over z
	//LoadModel("assets/teapot.vox");
	//GenerateSomeNoise();
}

// a helper function to load a magica voxel scene given a filename from https://github.com/jpaver/opengametools/blob/master/demo/demo_vox.cpp
void Scene::LoadModel(const char* filename, uint32_t scene_read_flags)
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
				materials[materialIndex]->albedo = (float3(static_cast<float>(col.r) / 255.0f,
				                                           static_cast<float>(col.g) / 255.0f,
				                                           static_cast<float>(col.b) / 255.0f));
				materials[materialIndex]->roughness = 1.0f;
				// Set the color at position (x, y, z)
				Set(scaledX, scaledY, scaledZ, materialIndex);
			}
		}
	}

	// Cleanup
	delete[] buffer;
}

void Scene::CreateEmmisiveSphere(MaterialType::MatType mat)
{
	//ResetGrid();
	//based on Lynn's implementation
	// When looping over (x, y, z) during scene creation

	constexpr float worldCenter{(WORLDSIZE / 2.0f)};

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
	grid[x + y * GRIDSIZE + z * GRIDSIZE2] = v;
}

bool Scene::Setup3DDDA(const Ray& ray, DDAState& state) const
{
	// if ray is not inside the world: advance until it is
	state.t = 0;
	if (!cube.Contains(ray.O))
	{
		state.t = cube.Intersect(ray);
		if (state.t > 1e33f) return false; // ray misses voxel data entirely
	}
	// setup amanatides & woo - assume world is 1x1x1, from (0,0,0) to (1,1,1)
	static const float cellSize = 1.0f / GRIDSIZE;
	state.step = make_int3(1 - ray.Dsign * 2);
	const float3 posInGrid = GRIDSIZE * (ray.O + (state.t + 0.00005f) * ray.D);
	const float3 gridPlanes = (ceilf(posInGrid) - ray.Dsign) * cellSize;
	const int3 P = clamp(make_int3(posInGrid), 0, GRIDSIZE - 1);
	state.X = P.x, state.Y = P.y, state.Z = P.z;
	state.tdelta = cellSize * float3(state.step) * ray.rD;
	state.tmax = (gridPlanes - ray.O) * ray.rD;
	// proceed with traversal

	return true;
}

void Scene::FindNearest(Ray& ray) const
{
	// setup Amanatides & Woo grid traversal
	DDAState s;
	if (!Setup3DDDA(ray, s)) return;
	// start stepping
	while (1)
	{
		const MaterialType::MatType cell = grid[s.X + s.Y * GRIDSIZE + s.Z * GRIDSIZE2];
		if (cell != MaterialType::NONE)
		{
			ray.t = s.t;
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
	// setup Amanatides & Woo grid traversal
	DDAState s;
	if (!Setup3DDDA(ray, s)) return false;
	// start stepping
	while (1)
	{
		const MaterialType::MatType cell = grid[s.X + s.Y * GRIDSIZE + s.Z * GRIDSIZE2];
		if (cell != matType)
		{
			ray.t = s.t;
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
	// TODO:
	// - A nested grid will let rays skip empty space much faster.
	// - Coherent rays can traverse the grid faster together.
	// - Perhaps s.X / s.Y / s.Z (the integer grid coordinates) can be stored in a single uint?
	// - Loop-unrolling may speed up the while loop.
	// - This code can be ported to GPU.
	return false;
}


bool Scene::IsOccluded(const Ray& ray) const
{
	// setup Amanatides & Woo grid traversal
	DDAState s;
	if (!Setup3DDDA(ray, s)) return false;
	// start stepping
	while (s.t < ray.t)
	{
		const auto cell = grid[s.X + s.Y * GRIDSIZE + s.Z * GRIDSIZE2];
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