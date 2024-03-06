#pragma once

// high level settings
// #define TWOLEVEL
//#define USE_MORTON 
constexpr auto WORLDSIZE = 128; // power of 2. Warning: max 512 for a 512x512x512x4 bytes = 512MB world!;
// #define USE_SIMD
// #define USE_FMA3
// #define SKYDOME
// #define WHITTED
// #define DOF

// low-level / derived
#define WORLDSIZE2	(WORLDSIZE*WORLDSIZE)
#ifdef TWOLEVEL
// feel free to replace with whatever suits your two-level implementation,
// should you chose this challenge.
#define BRICKSIZE	8
#define BRICKSIZE2	(BRICKSIZE*BRICKSIZE)
#define BRICKSIZE3	(BRICKSIZE*BRICKSIZE*BRICKSIZE)
#define GRIDSIZE	(WORLDSIZE/BRICKSIZE)
#define VOXELSIZE	(1.0f/WORLDSIZE)
#else
#define GRIDSIZE	WORLDSIZE
#endif
#define GRIDSIZE2	(GRIDSIZE*GRIDSIZE)
#define GRIDSIZE3	(GRIDSIZE*GRIDSIZE*GRIDSIZE)
/* 3D coordinate to morton code. */
constexpr uint64_t BMI_3D_X_MASK = 0x9249249249249249;
constexpr uint64_t BMI_3D_Y_MASK = 0x2492492492492492;
constexpr uint64_t BMI_3D_Z_MASK = 0x4924924924924924;
constexpr uint64_t BMI_3D_MASKS[3] = {BMI_3D_X_MASK, BMI_3D_Y_MASK, BMI_3D_Z_MASK};

namespace MaterialType
{
	enum MatType :uint32_t
	{
		NON_METAL_WHITE,
		NON_METAL_RED,
		NON_METAL_BLUE,
		NON_METAL_GREEN,
		NON_METAL_PINK,
		METAL_HIGH,
		METAL_MID,
		METAL_LOW,
		GLASS,
		EMISSIVE,
		NONE = 256
	};
}

class ogt_vox_scene;

namespace Tmpl8
{
	class Ray
	{
	public:
		Ray() = default;

		Ray(const float3 origin, const float3 direction, const float rayLength = 1e34f, const int = 0);

		float3 IntersectionPoint() const
		{
			return O + t * D;
		}

		//from Ray tracing in one weekend
		static Ray GetRefractedRay(const Ray& ray, const float IORRatio, bool& isReflected);

		float3 GetNormalVoxel() const;
		float3 UintToFloat3(uint col) const;
		float3 GetAlbedo(const Scene& scene) const;
		float GetEmissive(const Scene& scene) const;
		float GetRoughness(const Scene& scene) const;
		float GetRefractivity(const Scene& scene) const;
		//E reflected = E incoming multiplied by C material

		float3 GetAbsorption(const float3& I) const; // TODO: implement
		float3 Evaluate(const float3& I) const; // TODO: implement
		// ray data
		float3 O; // ray origin //12 bytes
		float3 rD; // reciprocal ray direction
		float3 D = float3(0); // ray direction
		float3 Dsign = float3(1); // 48 bytes
		float3 rayNormal{0}; //60 bytes

		float t = 1e34f; // ray length
		bool isInsideGlass = false;
		MaterialType::MatType indexMaterial = MaterialType::NONE; //replace grid color with material index


		int8_t depth = 5;

	private:
		// min3 is used in normal reconstruction.
		__inline static float3 min3(const float3& a, const float3& b)
		{
			return float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
		}
	};

	class Cube
	{
	public:
		Cube() = default;
		Cube(const float3 pos, const float3 size);
		float Intersect(const Ray& ray) const;
		bool Contains(const float3& pos) const;
		float3 b[2];
	};

	class Scene
	{
	public:
		struct DDAState
		{
			int3 step; // 16 bytes
			uint X, Y, Z; // 12 bytes
			float t; // 4 bytes
			float3 tdelta;
			float dummy1 = 0; // 16 bytes
			float3 tmax;
			float dummy2 = 0; // 16 bytes, 64 bytes in total
		};

		void GenerateSomeNoise(float frequency);
		void LoadModel(const char* filename, uint32_t scene_read_flags = 0);
		void CreateEmmisiveSphere(MaterialType::MatType mat);
		void ResetGrid();
		Scene();
		void FindNearest(Ray& ray) const;
		bool FindMaterialExit(Ray& ray, MaterialType::MatType matType) const;
		// morton order from Coppen, Max (230184)


		inline uint64_t MortonEncode(const uint32_t x, const uint32_t y, const uint32_t z) const
		{
			return _pdep_u64((uint64_t)x, BMI_3D_X_MASK) | _pdep_u64((uint64_t)y, BMI_3D_Y_MASK) |
				_pdep_u64((uint64_t)z, BMI_3D_Z_MASK);
		}

		inline uint64_t GetVoxel(const uint32_t x, const uint32_t y, const uint32_t z) const
		{
#ifdef USE_MORTON
			return MortonEncode(x, y, z);
# else
			return x + y * GRIDSIZE + z * GRIDSIZE2;
#endif
		}

		bool IsOccluded(const Ray& ray) const;
		void Set(const uint x, const uint y, const uint z, const MaterialType::MatType v);

		std::array<MaterialType::MatType, GRIDSIZE3> grid{};

		float3 scaleModel{1.0f};
		Cube cube;
		std::vector<shared_ptr<Material>> materials;
		float radiusEmissiveSphere = 1.0f;

	private:
		bool Setup3DDDA(const Ray& ray, DDAState& state) const;
	};
}
