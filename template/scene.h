#pragma once

namespace Tmpl8
{
	class Renderer;
}

// high level settings
// #define TWOLEVEL
//#define USE_MORTON 
//constexpr uint32_t WORLDSIZE = 128; // power of 2. Warning: max 512 for a 512x512x512x4 bytes = 512MB world!;
#define SIMD	 //slower in release??
// #define USE_FMA3
// #define SKYDOME
// #define WHITTED
// #define DOF

// low-level / derived
//#define WORLDSIZE2	(WORLDSIZE*WORLDSIZE)
//#ifdef TWOLEVEL

//// feel free to replace with whatever suits your two-level implementation,
//// should you chose this challenge.
//#define BRICKSIZE	8
//#define BRICKSIZE2	(BRICKSIZE*BRICKSIZE)
//#define BRICKSIZE3	(BRICKSIZE*BRICKSIZE*BRICKSIZE)
//#define GRIDSIZE	(WORLDSIZE/BRICKSIZE)


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
		SMOKE_LOW_DENSITY,
		SMOKE_LOW2_DENSITY,
		SMOKE_MID_DENSITY,
		SMOKE_MID2_DENSITY,
		SMOKE_HIGH_DENSITY,
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
		void CopyToPrevRay(Ray& ray);
		Ray() = default;
		Ray(const Ray& ray);

		float3 ComputeDsign(const float3& _D) const;
		float3 ComputeDsign_SSE(const __m128& m) const;
		Ray(const float3 origin, const float3 direction, const float rayLength = 1e34f, const int = 0);

		float3 IntersectionPoint() const
		{
			return O + t * D;
		}

		//from Ray tracing in one weekend
		static Ray GetRefractedRay(const Ray& ray, const float IORRatio, bool& isReflected);

		float3 GetNormalVoxel(const uint32_t worldSize, const mat4& matrix) const;
		float3 UintToFloat3(uint col) const;
		float3 GetAlbedo(const Renderer& scene) const;
		float GetEmissive(const Renderer& scene) const;
		float GetRoughness(const Renderer& scene) const;
		float GetRefractivity(const Renderer& scene) const;
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
		float Intersect(const Ray& ray) const;
		bool Contains(const float3& pos) const;
		float3 b[2] = {{0}, {1}};
		float3 rotation{0};
		float3 position{0};
		float3 scale{1};
		mat4 invMatrix;
		mat4 matrix;

		void Grow(float3 p)
		{
			b[0] = fminf(b[0], p);
			b[1] = fmaxf(b[1], p);
		}
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

		//Assumes size of 1
		void SetCubeBoundaries(const float3& position);
		void SetTransform(const float3& rotation);
		void SetScale(const float3& scl);

		void GenerateSomeNoise(float frequency);
		void GenerateSomeSmoke(float frequency);
		void CreateEmmisiveSphere(MaterialType::MatType mat, float radiusEmissiveSphere);
		void ResetGrid(MaterialType::MatType type = MaterialType::NONE);
		Scene(const float3& position, uint32_t worldSize = 64);
		void LoadModel(Renderer& scene, const char* filename, uint32_t scene_read_flags = 0);
		bool FindNearest(Ray& ray) const;
		bool FindMaterialExit(Ray& ray, MaterialType::MatType matType) const;
		bool FindSmokeExit(Ray& ray) const;
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

		bool IsOccluded(Ray& ray) const;
		void Set(const uint x, const uint y, const uint z, const MaterialType::MatType v);
		const uint32_t WORLDSIZE; // power of 2. Warning: max 512 for a 512x512x512x4 bytes = 512MB world!;
		const uint32_t GRIDSIZE;
		const uint32_t GRIDSIZE2;
		const uint32_t GRIDSIZE3;

		/* 3D coordinate to morton code. */
		std::vector<MaterialType::MatType> grid{};

		float3 scaleModel{1.0f};
		Cube cube;

	private:
		bool Setup3DDDA(Ray& ray, DDAState& state) const;
	};
}
