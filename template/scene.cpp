#include "precomp.h"

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

float Ray::UintToFloat3EmmisionStrength(uint col) const
{
	const uint8_t emmision = (col >> 24) & 0xFF;


	// Normalize color components to the range [0, 1]
	const float normRed = static_cast<float>(emmision) / 255.0f;


	return normRed;
}

float3 Ray::GetAlbedo(Scene& scene)
{
	return scene.materials[indexMaterial]->GetAlbedo();
}

float Ray::GetReflectivity(Scene& scene) const
{
	return scene.materials[indexMaterial]->GetReflectivity();
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
				float n = noiseOutput[x + y * WORLDSIZE + z * WORLDSIZE * WORLDSIZE];
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
					color = MaterialType::DIFFUSE_RED;
				}
				else if (n < 0.17f)
				{
					color = MaterialType::PARTIAL_MIRROR;
				}
				else if (n < 0.3)
				{
					color = MaterialType::DIFFUSE_BLUE;
				}
				else if (n < 0.5f)
					color = MaterialType::MIRROR_HIGH_REFLECTIVITY;
				else if (n < 0.7f)
					color = MaterialType::MIRROR_MID_REFLECTIVITY;
				else if (n < 0.9f)
					color = MaterialType::MIRROR_LOW_REFLECTIVITY;


				Set(x, y, z, color); // Assuming Set function is defined elsewhere
			}
		}
	}
}

Scene::Scene()
{
	// the voxel world sits in a 1x1x1 cube
	cube = Cube(float3(0, 0, 0), float3(1, 1, 1));
	// initialize the mainScene using Perlin noise, parallel over z
	grid = static_cast<MaterialType::MatType*>(MALLOC64(GRIDSIZE3 * sizeof(MaterialType::MatType)));
	memset(grid, MaterialType::NONE, GRIDSIZE3 * sizeof(MaterialType::MatType));
	//GenerateSomeNoise();
}

// a helper function to load a magica voxel scene given a filename.
void Scene::load_and_assign_vox_scene(const char* filename, uint32_t scene_read_flags = 0)
{
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
	uint32_t buffer_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// load the file into a memory buffer
	uint8_t* buffer = new uint8_t[buffer_size];
	fread(buffer, buffer_size, 1, fp);
	fclose(fp);

	// construct the scene from the buffer
	const ogt_vox_scene* scene = ogt_vox_read_scene_with_flags(buffer, buffer_size, scene_read_flags);

	// Assign colors based on the loaded scene
	for (int x = 0; x < scene->x; ++x)
	{
		for (int y = 0; y < scene->size_y; ++y)
		{
			for (int z = 0; z < scene->size_z; ++z)
			{
				// Assume each voxel has a color index, and map that to MatType
				MatType color;
				// Example: color assignment logic based on voxel properties
				// Replace this with your own logic based on the loaded scene
				uint8_t voxel_color_index = scene->voxels[x][y][z].color_index;
				// Map voxel color index to MatType
				switch (voxel_color_index)
				{
				case 0: color = DIFFUSE_WHITE;
					break;
				case 1: color = DIFFUSE_RED;
					break;
				case 2: color = DIFFUSE_BLUE;
					break;
				case 3: color = DIFFUSE_GREEN;
					break;
				// Add more cases as needed
				default: color = NONE;
					break;
				}
				// Set the color at position (x, y, z)
				Set(x, y, z, color);
			}
		}
	}

	// Cleanup
	delete[] buffer;
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
