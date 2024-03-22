#include "precomp.h"

#include <execution>
#include <filesystem>
#include <stb_image.h>

// YOU GET:
// 1. A fast voxel renderer in plain C/C++
// 2. Normals and voxel colors
// FROM HERE, TASKS COULD BE:							FOR SUFFICIENT
// * Materials:
//   - Reflections and diffuse reflections				<===
//   - Transmission with Snell, Fresnel					<===
//   - Textures, Minecraft-style						<===
//   - Beer's Law
//   - Normal maps
//   - Emissive materials with postproc bloom
//   - Glossy reflections (BASIC)
//   - Glossy reflections (microfacet)
// * Light transport:
//   - Point lights										<===
//   - Spot lights										<===
//   - Area lights										<===
//	 - Sampling multiple lights with 1 ray
//   - Importance-sampling
//   - Image based lighting: sky
// * Camera:
//   - Depth of field									<===
//   - Anti-aliasing									<===
//   - Panini, fish-eye etc.
//   - Post-processing: now also chromatic				<===
//   - Spline cam, follow cam, fixed look-at cam
//   - Low-res cam with CRT shader
// * Scene:
//   - HDR skydome										<===
//   - Spheres											<===
//   - Smoke & trilinear interpolation
//   - Signed Distance Fields
//   - Voxel instances with transform
//   - Triangle meshes (with a BVH)
//   - High-res: nested grid
//   - Procedural art: shapes & colors
//   - Multi-threaded Perlin / Voronoi
// * Various:
//   - Object picking
//   - Ray-traced physics
//   - Profiling & optimization
// * GPU:
//   - GPU-side Perlin / Voronoi
//   - GPU rendering *not* allowed!
// * Advanced:
//   - Ambient occlusion
//   - Denoising for soft shadows
//   - Reprojection for AO / soft shadows
//   - Line lights, tube lights, ...
//   - Bilinear interpolation and MIP-mapping
// * Simple game:										
//   - 3D Arkanoid										<===
//   - 3D Snake?
//   - 3D Tank Wars for two players
//   - Chess
// REFERENCE IMAGES:
// https://www.rockpapershotgun.com/minecraft-ray-tracing
// https://assetsio.reedpopcdn.com/javaw_2019_04_20_23_52_16_879.png
// https://www.pcworld.com/wp-content/uploads/2023/04/618525e8fa47b149230.56951356-imagination-island-1-on-100838323-orig.jpg


void Renderer::InitMultithreading()
{
#ifdef 	PROFILE
	SetThreadAffinityMask(GetCurrentThread(), 1ULL << (std::thread::hardware_concurrency() - 1));
#endif

	const auto numThreads = thread::hardware_concurrency();
	cout << "Number of threads: " << numThreads << '\n';
	std::vector<uint32_t> threads;
	for (uint i = 0; i < numThreads; i++)
		threads.push_back(i);
	for_each(execution::par, threads.begin(), threads.end(),
	         [this](const uint32_t x)
	         {
		         x;
		         //using chatgpt to convert the id to a seed
		         const std::thread::id threadId = std::this_thread::get_id();

		         // Convert std::thread::id to uint
		         const uint id = static_cast<uint>(std::hash<std::thread::id>{}(threadId));

		         InitSeed(id);
	         });
}

void Renderer::SetUpLights()
{
}

float3 Renderer::PointLightEvaluate(Ray& ray, const PointLightData& lightData)
{
	//Getting the intersection point
	const float3 intersectionPoint = ray.IntersectionPoint();
	const float3 dir = lightData.position - intersectionPoint;
	const float dst = length(dir);


	const float3 dirNormalized = dir * (1.0f / dst);

	const float3 normal = ray.rayNormal;
	//Having a negative dot product means the light is behind the point
	const float cosTheta = dot(dirNormalized, normal);
	if (cosTheta <= 0.0f)
		return {0.0f};
	//the  formula for distance attenuation 
	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color * (1.0f / (dst * dst));
	//materi
	const float3 originRay = OffsetRay(intersectionPoint, normal);
	const float3 k = ray.GetAlbedo(*this);

	Ray shadowRay(originRay, dirNormalized);
	// we do not shoot the ray behind the light source
	shadowRay.t = dst;
	if (IsOccluded(shadowRay))
		return {0.0f};


	return lightIntensity * k;
}

float3 Renderer::SpotLightEvaluate(const Ray& ray, const SpotLightData& lightData) const
{
	const float3 intersectionPoint = ray.IntersectionPoint();
	const float3 dir = lightData.position - intersectionPoint;
	const float dst = length(dir);
	const float3 dirNormalized = dir / dst;

	const float3 normal = ray.rayNormal;
	//light angle
	const float cosTheta = dot(dirNormalized, lightData.direction);
	if (cosTheta <= lightData.angle)
		return 0;

	const float alphaCutOff = 1.0f - (1.0f - cosTheta) * 1.0f / (1.0f - lightData.angle);

	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color / (dst * dst);
	//material evaluation
	const float3 k = ray.GetAlbedo(*this);


	Ray shadowRay(OffsetRay(intersectionPoint, normal), dirNormalized);
	shadowRay.t = dst;
	if (IsOccluded(shadowRay))
		return 0;

	return lightIntensity * k * alphaCutOff;
}

float3 Renderer::AreaLightEvaluation(Ray& ray, const SphereAreaLightData& lightData) const
{
	const float3 intersectionPoint = ray.IntersectionPoint();
	const float3 normal = ray.rayNormal;
	const float3 center = lightData.position;
	const float radius = lightData.radius;
	float3 incomingLight{0};
	const float3 k = ray.GetAlbedo(*this);
	float3 point = OffsetRay(intersectionPoint, normal);

	//the same as before, we get all the needed variables


	//we check the shadow for a number of points and then average by the sample count
	for (int i = 0; i < numCheckShadowsAreaLight; i++)
	{
		float3 randomPoint = RandomDirection();

		randomPoint *= radius;
		randomPoint += center;

		const float3 dir = randomPoint - intersectionPoint;
		const float dst = length(dir);
		const float3 dirNormalized = dir * (1 / dst);

		const float cosTheta = dot(dirNormalized, normal);
		if (cosTheta <= 0)
		{
			continue;
		}
		Ray shadowRay(point, dirNormalized);
		shadowRay.t = dst;
		if (IsOccluded(shadowRay))
			continue;
		//https://www.physicsforums.com/threads/luminance-of-a-lambertian-sphere-formula.449703/
		const float3 lightIntensity = cosTheta * lightData.color * lightData.colorMultiplier * (1.0f / (dst *
				dst)) * (radius * radius) *
			PI;


		incomingLight += lightIntensity;
	}
	incomingLight /= static_cast<float>(numCheckShadowsAreaLight);


	return incomingLight * k;
}

bool Renderer::IsOccluded(Ray& ray) const
{
	for (auto& scene : voxelVolumes)
	{
		Ray backupRay = ray;
		ray.O = TransformPosition(ray.O, scene.invMatrix);

		ray.D = TransformVector(ray.D, scene.invMatrix);

		ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
		ray.Dsign = ray.ComputeDsign(ray.D);
		if (scene.IsOccluded(ray))
		{
			backupRay.t = ray.t;
			backupRay.CopyToPrevRay(ray);
			return true;
		}
		backupRay.t = ray.t;
		backupRay.CopyToPrevRay(ray);
	}


	for (auto& sphere : spheres)
	{
		if (sphere.IsHit(ray))
			return true;
	}
	for (auto& tri : triangles)
	{
		if (tri.IsHit(ray))
			return true;
	}

	return false;
}

bool Renderer::IsOccludedSpheres(Ray& ray) const
{
	//if (mainScene.IsOccluded(ray))
	//	return true;

	for (auto& sphere : spheres)
	{
		if (sphere.IsHit(ray))
			return true;
	}
	//for (auto& tri : triangles)
	//{
	//	if (tri.IsHit(ray))
	//		return true;
	//}

	return false;
}

float3 Renderer::DirectionalLightEvaluate(Ray& ray, const DirectionalLightData& lightData)
{
	const float3 intersectionPoint = ray.IntersectionPoint();
	const float3 dir = -lightData.direction;
	const float3 attenuation = dir;

	const float3 normal = ray.rayNormal;

	//light angle
	const float cosTheta = dot(attenuation, normal);
	if (cosTheta <= 0)
		return 0;
	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color;
	//material evaluation
	const float3 k = ray.GetAlbedo(*this);

	Ray shadowRay(OffsetRay(intersectionPoint, normal), (dir));


	if (IsOccluded(shadowRay))
		return {0};

	return lightIntensity * k;
}

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::ResetAccumulator()
{
	numRenderedFrames = 0;
}

void Renderer::MaterialSetUp()
{
	const auto materialDifWhite = make_shared<Material>(float3(1, 1, 1));
	const auto materialDifRed = make_shared<Material>(float3(1, 0, 0));
	const auto materialDifBlue = make_shared<Material>(float3(0, 0, 1));
	const auto materialDifGreen = make_shared<Material>(float3(0, 1, 0), 0.0f);
	const auto partialMetal = make_shared<Material>(float3(1, 1, 1), 0.75f);

	//Mirror
	const auto materialDifReflectivity = make_shared<Material>(float3(1));
	const auto materialDifRefMid = make_shared<Material>(float3(0, 1, 1), 0.5f);
	const auto materialDifRefLow = make_shared<Material>(float3(1, 1, 0), 0.1f);
	//partial mirror
	const auto glass = make_shared<Material>(float3(1, 1, 1));
	glass->IOR = 1.45f;
	float3 smokeColor = float3{1.0f, 0.8f, 0.0f};

	const auto smoke = make_shared<Material>(smokeColor);
	smoke->IOR = 1.0f;
	smoke->emissiveStrength = 8.0f;
	const auto smoke1 = make_shared<Material>(smokeColor);
	smoke1->IOR = 1.0f;
	smoke1->emissiveStrength = 12.0f;
	const auto smoke2 = make_shared<Material>(smokeColor);
	smoke2->IOR = 1.0f;
	smoke2->emissiveStrength = 15.0f;
	const auto smoke3 = make_shared<Material>(smokeColor);
	smoke3->IOR = 1.0f;
	smoke3->emissiveStrength = 20.0f;
	const auto smoke4 = make_shared<Material>(smokeColor);
	smoke4->IOR = 1.0f;
	smoke4->emissiveStrength = 22.0f;

	const auto emissive = make_shared<Material>(smokeColor);
	emissive->emissiveStrength = 5.0f;

	nonMetalMaterials.push_back(materialDifWhite);
	nonMetalMaterials.push_back(materialDifRed);
	nonMetalMaterials.push_back(materialDifBlue);
	nonMetalMaterials.push_back(materialDifGreen);
	nonMetalMaterials.push_back(partialMetal);
	//metals
	metalMaterials.push_back(materialDifReflectivity);
	metalMaterials.push_back(materialDifRefMid);
	metalMaterials.push_back(materialDifRefLow);

	dielectricsMaterials.push_back(glass);
	smokeMaterials.push_back(smoke);
	smokeMaterials.push_back(smoke1);
	smokeMaterials.push_back(smoke2);
	smokeMaterials.push_back(smoke3);
	smokeMaterials.push_back(smoke4);
	emissiveMaterials.push_back(emissive);

	for (auto& mat : nonMetalMaterials)
		materials.push_back(mat);
	for (auto& mat : metalMaterials)
		materials.push_back(mat);
	for (auto& mat : dielectricsMaterials)
		materials.push_back(mat);
	for (auto& mat : smokeMaterials)
		materials.push_back(mat);
	for (auto& mat : emissiveMaterials)
		materials.push_back(mat);

	if (materials.size() < MaterialType::NONE)
	{
		size_t i = materials.size();
		materials.resize(MaterialType::NONE);
		for (; i < MaterialType::NONE; ++i)
		{
			materials[i] = std::make_shared<Material>(float3(1, 1, 1), 1.f);
		}
	}
}

void Renderer::AddSphere()
{
	spheres.push_back(Sphere(float3{0}, .5, static_cast<MaterialType::MatType>(Rand(MaterialType::EMISSIVE))));
}

void Renderer::RemoveLastSphere()
{
	spheres.pop_back();
}

void Renderer::AddTriangle()
{
	triangles.push_back(Triangle{static_cast<MaterialType::MatType>(Rand(MaterialType::EMISSIVE))});
}

void Renderer::RemoveTriangle()
{
	triangles.pop_back();
}

void Renderer::RemoveVoxelVolume()
{
	voxelVolumes.pop_back();
}

void Renderer::AddVoxelVolume()
{
	voxelVolumes.emplace_back(Scene({0}));
	//const float3 rot = float3{15.0f, 0.0f, 0.0f};
	//voxelVolumes[0].cube.rotation = rot;
	//voxelVolumes[0].SetTransform(rot * DEG2RAD);
}

void Renderer::ShapesSetUp()
{
	//AddSphere();
	AddVoxelVolume();
	constexpr int sizeX = 6;
	constexpr int sizeY = 1;
	constexpr int sizeZ = 2;
	const array powersTwo = {1, 2, 4, 8, 16, 32, 64};
	for (int i = 0; i < sizeX; i++)
	{
		for (int j = 0; j < sizeY; j++)
		{
			for (int k = 1; k < sizeZ; k++)
			{
				const int index = (k + i + j) % powersTwo.size();
				voxelVolumes.emplace_back(Scene({static_cast<float>(i), static_cast<float>(j), static_cast<float>(k)},
				                                powersTwo[index]));
			}
		}
	}
}

void Renderer::Init()
{
	CopyToPrevCamera();
	int skyBpp;
	skyPixels = stbi_loadf("assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0);
	for (int i = 0; i < skyWidth * skyHeight * 3; i++)
	{
		skyPixels[i] = sqrtf(skyPixels[i]);
	}
	//sizeof(Ray);
	InitSeed(static_cast<uint>(time(nullptr)));

	//multiply by 16 because float4 consists of 16 bytes
	accumulator = static_cast<float4*>(MALLOC64(SCRWIDTH * SCRHEIGHT * 16));


	memset(accumulator, 0, SCRWIDTH * SCRHEIGHT * 16);
	// try to load a camera
	FILE* f = fopen("camera.bin", "rb");
	if (f)
	{
		fread(&camera, 1, sizeof(Camera), f);
		fclose(f);
	}
	//init multithreading
	InitMultithreading();
	vertIterator.resize(SCRHEIGHT);
	for (int i = 0; i < SCRHEIGHT; i++)
		vertIterator[i] = i;
	for (const auto& entry : std::filesystem::directory_iterator("assets"))
	{
		if (entry.path().extension() == ".vox")
		{
			voxFiles.push_back(entry.path().filename().string());
		}
	}

	//Lighting set-up
	SetUpLights();
	//shape set-up
	ShapesSetUp();
	//Material set-up
	MaterialSetUp();
}

void Renderer::Illumination(Ray& ray, float3& incLight)
{
	const size_t randLightIndex = static_cast<size_t>(Rand(LIGHT_COUNT));
	//map every index to a certain light element				
	if (randLightIndex < POINT_LIGHTS)
	{
		const auto p = (randLightIndex);
		incLight = PointLightEvaluate(ray, pointLights[p].data);
	}
	else if (randLightIndex < AREA_LIGHTS + POINT_LIGHTS)
	{
		const auto a = randLightIndex - POINT_LIGHTS;
		incLight = AreaLightEvaluation(ray, areaLights[a].data);
	}
	else if (randLightIndex < AREA_LIGHTS + SPOT_LIGHTS + POINT_LIGHTS)
	{
		const auto s = randLightIndex - SPOT_LIGHTS - POINT_LIGHTS;
		incLight = SpotLightEvaluate(ray, spotLights[s].data);
	}
	//1 is the only directional
	else
	{
		incLight = DirectionalLightEvaluate(ray, dirLight.data);
	}

	incLight *= LIGHT_COUNT;
}


float3 Renderer::Reflect(const float3 direction, const float3 normal)
{
	return direction - 2 * normal * dot(normal, direction);
}

//from ray tracing in one weekend
float3 Renderer::Refract(const float3 direction, const float3 normal, const float IORRatio)
{
	const float cosTheta = min(dot(-direction, normal), 1.0f);
	const float3 rPer = IORRatio * (direction + cosTheta * normal);
	const float3 rPar = -sqrtf(fabsf(1.0f - sqrLength(rPer))) * normal;
	return rPer + rPar;
}

// use this this for even faster reciprocal
//https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
__m128 Renderer::FastReciprocal(__m128& x)
{
	__m128 res = _mm_rcp_ps(x);
	const __m128 muls = _mm_mul_ps(x, _mm_mul_ps(res, res));
	return res = _mm_sub_ps(_mm_add_ps(res, res), muls);
}

__m128 Renderer::SlowReciprocal(__m128& dirSSE)
{
	return _mm_div_ps(_mm_set_ps1(1.0f), dirSSE);
}

int32_t Renderer::FindNearest(Ray& ray)
{
	int32_t voxelIndex = -2;

#if 1

	__m128 oriSSE = ray.O4;
	__m128 dirSSE = ray.D4;
#endif

	int32_t voxelCount = static_cast<int32_t>(voxelVolumes.size());
	for (int32_t i = 0; i < voxelCount; i++)

	{
		Ray backupRay = ray;


		mat4 invMat = voxelVolumes[i].invMatrix;
#if 1
		ray.O4 = TransformPosition_SSEM(oriSSE, invMat);

		ray.D4 = TransformVector_SSEM(dirSSE, invMat);

		//for my machine the fast reciprocal is a bit slower

#if 0
		__m128 rDSSE = SlowReciprocal(ray.D4);
#else
		__m128 rDSSE = FastReciprocal(ray.D4);

#endif

		ray.Dsign = ray.ComputeDsign_SSE(ray.D4);

		ray.rD4 = rDSSE;

#else
		ray.O = TransformPosition(ray.O, invMat);


		ray.D = TransformVector(ray.D, invMat);

		ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
		ray.Dsign = ray.ComputeDsign(ray.D);
#endif

		if (voxelVolumes[i].FindNearest(ray))
		{
			voxelIndex = i;
		}
		backupRay.t = ray.t;
		backupRay.CopyToPrevRay(ray);
	}

	//get the nearest t
	//{
	//	Ray sphereHit{ray.O, ray.D};

	//	for (auto& sphere : spheres)
	//	{
	//		sphere.Hit(sphereHit);
	//	}
	//	for (auto& triangle : triangles)
	//	{
	//		triangle.Hit(sphereHit);
	//	}
	//	//change to the closest ray information
	//	if (ray.t > sphereHit.t)
	//	{
	//		ray.t = sphereHit.t;
	//		ray.indexMaterial = sphereHit.indexMaterial;
	//		ray.rayNormal = sphereHit.rayNormal;
	//		ray.isInsideGlass = sphereHit.isInsideGlass;
	//		voxelIndex = -1;
	//	}
	//}
	return voxelIndex;
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace(Ray& ray, int depth)
{
	if (depth < 0)
	{
		return {0};
	}
	//Find nearest BVH


	int32_t voxIndex = FindNearest(ray);
	//return { 0 };

	//evaluate materials and trace again for reflections and refraction

	// Break early if no intersection
	if (ray.indexMaterial == MaterialType::NONE)
	{
		return SampleSky(ray.D);
	}

	//return .5f * (ray.rayNormal + 1);


	switch (ray.indexMaterial)
	{
	//metals
	case MaterialType::METAL_MID:
	case MaterialType::METAL_HIGH:
	case MaterialType::METAL_LOW:
		{
			Ray newRay;
			float3 reflectedDirection = Reflect(ray.D, ray.rayNormal);
			newRay = Ray{
				OffsetRay(ray.IntersectionPoint(), ray.rayNormal),
				reflectedDirection + ray.GetRoughness(*this) * RandomSphereSample()
			};
			return Trace(newRay, depth - 1) * ray.GetAlbedo(*this);
		}
	//non-metal

	case MaterialType::NON_METAL_WHITE:
	case MaterialType::NON_METAL_PINK:
	case MaterialType::NON_METAL_RED:
	case MaterialType::NON_METAL_BLUE:
	case MaterialType::NON_METAL_GREEN:
		{
			Ray newRay;
			float3 color{0};
			if (RandomFloat() > SchlickReflectanceNonMetal(dot(-ray.D, ray.rayNormal)))
			{
				float3 incLight{0};
				float3 randomDirection = RandomLambertianReflectionVector(ray.rayNormal);
				Illumination(ray, incLight);
				newRay = Ray{OffsetRay(ray.IntersectionPoint(), ray.rayNormal), randomDirection};
				color += incLight;
				color += Trace(newRay, depth - 1) * ray.GetAlbedo(*this);
			}
			else
			{
				float3 reflectedDirection = Reflect(ray.D, ray.rayNormal);
				newRay = Ray{
					OffsetRay(ray.IntersectionPoint(), ray.rayNormal),
					reflectedDirection + ray.GetRoughness(*this) * RandomSphereSample()
				};
				color = Trace(newRay, depth - 1);
			}
			return color;
		}
	//mostly based on Ray tracing in one weekend
	case MaterialType::GLASS:
		{
			float3 color{1.0f};
			//code for glass
			bool isInGlass = ray.isInsideGlass;
			float IORMaterial = ray.GetRefractivity(*this); //1.45
			//get the IOR
			float refractionRatio = isInGlass ? IORMaterial : 1.0f / IORMaterial;
			//we need to get to the next voxel
			bool isInsideVolume = true;
			if (isInGlass)
			{
				color = ray.GetAlbedo(*this);
				//only the first one has glass
				isInsideVolume = voxelVolumes[voxIndex].FindMaterialExit(ray, MaterialType::GLASS);
			}
			if (!isInsideVolume)
			{
				ray.O = ray.O + ray.D * ray.t;
				ray.t = 0;
			}

			float cosTheta = min(dot(-ray.D, ray.rayNormal), 1.0f);
			float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

			bool cannotRefract = refractionRatio * sinTheta > 1.0f;

			float3 resultingDirection;

			//this may be negative if we refract
			float3 resultingNormal;
			if (cannotRefract || SchlickReflectance(cosTheta, refractionRatio) > RandomFloat())
			{
				//reflect!
				resultingDirection = Reflect(ray.D, ray.rayNormal);
				resultingNormal = ray.rayNormal;
			}
			else
			{
				//we are exiting or entering the glass
				resultingDirection = Refract(ray.D, ray.rayNormal, refractionRatio);
				isInGlass = !isInGlass;
				resultingNormal = -ray.rayNormal;
			}
			Ray newRay;
			newRay = {OffsetRay(ray.IntersectionPoint(), resultingNormal), resultingDirection};
			newRay.isInsideGlass = isInGlass;


			return Trace(newRay, depth - 1) * color;
		}
	//from Erik Cupak
	case MaterialType::SMOKE_LOW_DENSITY:
	case MaterialType::SMOKE_LOW2_DENSITY:
	case MaterialType::SMOKE_MID_DENSITY:
	case MaterialType::SMOKE_MID2_DENSITY:
	case MaterialType::SMOKE_HIGH_DENSITY:
		{
			float3 color{1.0f};
			//code for glass
			bool isInGlass = ray.isInsideGlass;
			//float IORMaterial = ray.GetRefractivity(*this); //1.45
			//get the IOR
			float refractionRatio = 1.0f;
			//we need to get to the next voxel
			bool isInsideVolume = true;
			float intensity = 0;
			float distanceTraveled = 0;

			if (isInGlass)
			{
				color = ray.GetAlbedo(*this);
				intensity = ray.GetEmissive(*this);
				//only the first one has glass
				Ray backupRay = ray;

				float3 origin = ray.O;
				float3 dir = ray.D;
				mat4 invMat = voxelVolumes[voxIndex].invMatrix;
				ray.O = TransformPosition(ray.O, invMat);


				ray.D = TransformVector(ray.D, invMat);

				ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
				ray.Dsign = ray.ComputeDsign(ray.D);
				isInsideVolume = voxelVolumes[voxIndex].FindSmokeExit(ray);
				backupRay.t = ray.t;
				backupRay.CopyToPrevRay(ray);
				distanceTraveled = ray.t;
			}
			//simple density functions
			float threshold = RandomFloat() * 100 - intensity;
			if (RandomFloat() * distanceTraveled > threshold)
			{
				ray.O = ray.O + ray.D * Rand(ray.t);

				ray.D = RandomDirection();
				ray.t = 0;
			}
			color = Absorption(color, intensity, distanceTraveled);

			if (!isInsideVolume)
			{
				ray.O = ray.O + ray.D * ray.t;
				ray.t = 0;
			}

			float3 resultingDirection;

			//this may be negative if we refract
			float3 resultingNormal;


			resultingDirection = Refract(ray.D, ray.rayNormal, refractionRatio);
			isInGlass = !isInGlass;
			resultingNormal = -ray.rayNormal;

			Ray newRay;
			newRay = {OffsetRay(ray.IntersectionPoint(), resultingNormal), resultingDirection};
			newRay.isInsideGlass = isInGlass;


			return Trace(newRay, depth - 1) * color;
		}
	case MaterialType::EMISSIVE:
		return ray.GetAlbedo(*this) * ray.GetEmissive(*this);

	//random materials from the models
	default:
		float3 incLight{0};
		float3 randomDirection = DiffuseReflection(ray.rayNormal);
		Illumination(ray, incLight);
		Ray newRay;

		newRay = Ray{OffsetRay(ray.IntersectionPoint(), ray.rayNormal), randomDirection};
		return Trace(newRay, depth - 1) * ray.GetAlbedo(*this) + incLight;
	}
}

//From raytracing in one weekend
float Renderer::SchlickReflectance(const float cosine, const float indexOfRefraction)
{
	// Use Schlick's approximation for reflectance.
	auto r0 = (1 - indexOfRefraction) / (1 + indexOfRefraction);
	r0 = r0 * r0;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

float3 Renderer::Absorption(const float3& color, float intensity, float distanceTraveled)
{
	// Combining 'e' and 'c' terms into a single "density" value (stored as intensity in the material).
	// [Credit] https://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_3_Refractions_and_Beers_Law.shtml
	const float3 flipped_color{1.0f - color};
	float3 exponent{
		-distanceTraveled
		* intensity
		* flipped_color

	};
	return {expf(exponent.x), expf(exponent.y), expf(exponent.z)};
}

//From Remi
float Renderer::SchlickReflectanceNonMetal(const float cosine)
{
	//for diffuse
	constexpr static float r0 = 0.04f;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}


float4 Renderer::SamplePreviousFrameColor(const float2& screenPosition)
{
	const int x = static_cast<int>((screenPosition.x * (SCRWIDTH)));
	const int y = static_cast<int>((screenPosition.y * (SCRHEIGHT)));

	const size_t pixelIndex = x + y * SCRWIDTH;


	return accumulator[pixelIndex];
}

float4 Renderer::BlendColor(const float4& currentColor, const float4& previousColor, float blendFactor)
{
	return currentColor * (1.0f - blendFactor) + previousColor * blendFactor;
}

bool Renderer::IsValid(const float2& uv)
{
	return uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f;
}

void Renderer::Update()
{
	//do only once
	//c++ 17 onwards parallel for loop
	weight = 1.0f / (static_cast<float>(numRenderedFrames) + 1.0f);
	static __m256 weightSSE = _mm256_set1_ps(weight);
	static __m256 antiAliasingStrengthSSE = _mm256_set1_ps(antiAliasingStrength);
#ifdef PROFILE
	for (uint32_t y = 0; y < SCRHEIGHT; y++)
	{
#else
	for_each(execution::par, vertIterator.begin(), vertIterator.end(),
	         [this](const uint32_t y)
	         {
#endif

		         const uint32_t pitch = y * SCRWIDTH;
#if 1
		         __m256 ySSE = _mm256_set1_ps(static_cast<float>(y));
		         //avx2 
		         for (uint32_t x = 0; x < SCRWIDTH; x += 8)
		         {
			         /* float3 newPixel{0};
			          float3 newPixel1{0};
			          float3 newPixel2{0};
			          float3 newPixel3{0};
			          float3 newPixel4{0};
			          float3 newPixel5{0};
			          float3 newPixel6{0};
			          float3 newPixel7{0};*/
			         __m256 xSSE = _mm256_set_ps(static_cast<float>(x + 7), static_cast<float>(x + 6),
			                                     static_cast<float>(x + 5),
			                                     static_cast<float>(x + 4), static_cast<float>(x + 3),
			                                     static_cast<float>(x + 2), static_cast<float>(x + 1),
			                                     static_cast<float>(x));


			         __m256 randomXDirSSE = _mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat(),
			                                              RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat());
			         __m256 randomYDirSSE = _mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat(),
			                                              RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat());
			         randomXDirSSE = _mm256_mul_ps(randomXDirSSE, antiAliasingStrengthSSE);
			         randomYDirSSE = _mm256_mul_ps(randomYDirSSE, antiAliasingStrengthSSE);
			         ////AA
			         //float randomXDir = RandomFloat() * antiAliasingStrength;
			         //float randomYDir = RandomFloat() * antiAliasingStrength;
			         randomXDirSSE = _mm256_add_ps(randomXDirSSE, xSSE);
			         randomYDirSSE = _mm256_add_ps(randomYDirSSE, ySSE);
			         Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x) + randomXDir,
			                                               static_cast<float>(y) + randomYDir);
			         //get new pixel


			         Ray primaryRay1 = camera.GetPrimaryRay(static_cast<float>(x + 1) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);
			         Ray primaryRay2 = camera.GetPrimaryRay(static_cast<float>(x + 2) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);
			         Ray primaryRay3 = camera.GetPrimaryRay(static_cast<float>(x + 3) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);

			         Ray primaryRay4 = camera.GetPrimaryRay(static_cast<float>(x + 4) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);
			         Ray primaryRay5 = camera.GetPrimaryRay(static_cast<float>(x + 5) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);
			         Ray primaryRay6 = camera.GetPrimaryRay(static_cast<float>(x + 6) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);
			         Ray primaryRay7 = camera.GetPrimaryRay(static_cast<float>(x + 7) + randomXDir,
			                                                static_cast<float>(y) + randomYDir);


			         newPixel = Trace(primaryRay, maxBounces);
			         newPixel1 = Trace(primaryRay1, maxBounces);
			         newPixel2 = Trace(primaryRay2, maxBounces);
			         newPixel3 = Trace(primaryRay3, maxBounces);
			         newPixel4 = Trace(primaryRay4, maxBounces);
			         newPixel5 = Trace(primaryRay5, maxBounces);
			         newPixel6 = Trace(primaryRay6, maxBounces);
			         newPixel7 = Trace(primaryRay7, maxBounces);

			         float4 pixel = newPixel;
			         float4 pixel1 = newPixel1;
			         float4 pixel2 = newPixel2;
			         float4 pixel3 = newPixel3;
			         float4 pixel4 = newPixel4;
			         float4 pixel5 = newPixel5;
			         float4 pixel6 = newPixel6;
			         float4 pixel7 = newPixel7;


			         pixel = BlendColor(pixel, accumulator[x + pitch], 1.0f - weight);
			         pixel1 = BlendColor(pixel1, accumulator[x + 1 + pitch], 1.0f - weight);
			         pixel2 = BlendColor(pixel2, accumulator[x + 2 + pitch], 1.0f - weight);
			         pixel3 = BlendColor(pixel3, accumulator[x + 3 + pitch], 1.0f - weight);
			         pixel4 = BlendColor(pixel4, accumulator[x + 4 + pitch], 1.0f - weight);
			         pixel5 = BlendColor(pixel5, accumulator[x + 5 + pitch], 1.0f - weight);
			         pixel6 = BlendColor(pixel6, accumulator[x + 6 + pitch], 1.0f - weight);
			         pixel7 = BlendColor(pixel7, accumulator[x + 7 + pitch], 1.0f - weight);

			         //display
			         accumulator[x + pitch] = pixel;
			         accumulator[x + 1 + pitch] = pixel1;
			         accumulator[x + 2 + pitch] = pixel2;
			         accumulator[x + 3 + pitch] = pixel3;
			         accumulator[x + 4 + pitch] = pixel4;
			         accumulator[x + 5 + pitch] = pixel5;
			         accumulator[x + 6 + pitch] = pixel6;
			         accumulator[x + 7 + pitch] = pixel7;

			         pixel = ApplyReinhardJodie(pixel);
			         pixel1 = ApplyReinhardJodie(pixel1);
			         pixel2 = ApplyReinhardJodie(pixel2);
			         pixel3 = ApplyReinhardJodie(pixel3);
			         pixel4 = ApplyReinhardJodie(pixel4);
			         pixel5 = ApplyReinhardJodie(pixel5);
			         pixel6 = ApplyReinhardJodie(pixel6);
			         pixel7 = ApplyReinhardJodie(pixel7);

			         screen->pixels[x + pitch] = RGBF32_to_RGB8(&pixel);
			         screen->pixels[x + 1 + pitch] = RGBF32_to_RGB8(&pixel1);
			         screen->pixels[x + 2 + pitch] = RGBF32_to_RGB8(&pixel2);
			         screen->pixels[x + 3 + pitch] = RGBF32_to_RGB8(&pixel3);
			         screen->pixels[x + 4 + pitch] = RGBF32_to_RGB8(&pixel4);
			         screen->pixels[x + 5 + pitch] = RGBF32_to_RGB8(&pixel5);
			         screen->pixels[x + 6 + pitch] = RGBF32_to_RGB8(&pixel6);
			         screen->pixels[x + 7 + pitch] = RGBF32_to_RGB8(&pixel7);
		         }
#else

		         for (uint32_t x = 0; x < SCRWIDTH; x++)
		         {
			         float3 newPixel{0};


			         //AA
			         const float randomXDir = RandomFloat() * antiAliasingStrength;
			         const float randomYDir = RandomFloat() * antiAliasingStrength;

			         Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x) + randomXDir,
			                                               static_cast<float>(y) + randomYDir);
			         //get new pixel
			         newPixel = Trace(primaryRay, maxBounces);
			         float4 pixel = newPixel;

			         //////use this for reprojection?
			         //const float2 previousPixelCoordinate = prevCamera.PointToUV(primaryRay.IntersectionPoint());
			         //if (IsValid(previousPixelCoordinate) && !staticCamera)
			         //{
			         //	float4 previousFrameColor = SamplePreviousFrameColor(
			         //		previousPixelCoordinate);


			         //	/*         if (staticCamera)
			         //				 weight = 1.0f / (static_cast<float>(numRenderedFrames) + 1.0f);*/
			         //	//weight is usually 0.1, but it is the inverse of the usual 0.9 theta behind the scenes
			         //	const float4 blendedColor = BlendColor(newPixel, previousFrameColor,
			         //	                                       1.0f - weight);
			         //	pixel = blendedColor;
			         //}


			         pixel = BlendColor(pixel, accumulator[x + pitch], 1.0f - weight);

			         //display
			         accumulator[x + pitch] = pixel;

			         pixel = ApplyReinhardJodie(pixel);

			         screen->pixels[x + pitch] = RGBF32_to_RGB8(&pixel);
		         }
#endif

#ifdef PROFILE
	}
#else
	         });
#endif

	numRenderedFrames++;
}

void Renderer::CopyToPrevCamera()
{
	prevCamera.camPos = camera.camPos;

	prevCamera.bottomNormal = camera.bottomNormal;
	prevCamera.leftNormal = camera.leftNormal;
	prevCamera.rightNormal = camera.rightNormal;
	prevCamera.topNormal = camera.topNormal;
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick(const float deltaTime)
{
	if (camera.HandleInput(deltaTime))
	{
		ResetAccumulator();
	}
	camera.SetFrustumNormals();

	// pixel loop
	const Timer t;

	//DOF from Remi

	Ray focusRay = camera.GetPrimaryRay(SCRWIDTH / 2, SCRHEIGHT / 2);
	for (auto& scene : voxelVolumes)
		scene.FindNearest(focusRay);

	camera.focalDistance = clamp(focusRay.t, -1.0f, 1e4f);


	Update();

	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000.0f / avg, rps = (SCRWIDTH * SCRHEIGHT) / avg;
	printf("%5.2fms (%.1ffps) - %.1fMrays/s\n", avg, fps, rps / 1000);
	// handle user input


	CopyToPrevCamera();
}


//this is from Lynn's code
// [CREDITS] Article on tonemapping https://64.github.io/tonemapping/#aces
float3 Renderer::ApplyReinhardJodie(const float3& color)
{
	const float luminance{GetLuminance(color)};
	const float3 reinhardAdjustment{color / (1.0f + color)};
	const float3 luminanceAdjustment{color / (1.0f + luminance)};

	return
	{
		lerp(luminanceAdjustment.x, reinhardAdjustment.x, reinhardAdjustment.x),
		lerp(luminanceAdjustment.y, reinhardAdjustment.y, reinhardAdjustment.y),
		lerp(luminanceAdjustment.z, reinhardAdjustment.z, reinhardAdjustment.z)
	};
}

// [CREDITS] Article on tonemapping https://64.github.io/tonemapping/#aces
float Renderer::GetLuminance(const float3& color)
{
	return dot(color, {0.2126f, 0.7152f, 0.0722f});
}


// -----------------------------------------------------------
// User wants to close down
// -----------------------------------------------------------
void Renderer::Shutdown()
{
	// save current camera
	FILE* f = fopen("camera.bin", "wb");
	fwrite(&camera, 1, sizeof(Camera), f);
	fclose(f);
}

void Renderer::MouseDown(int button)
{
	std::cout << "Button: " << button << std::endl;
	Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(mousePos.x),
	                                      static_cast<float>(mousePos.y));
	Trace(primaryRay, 5);
}

float3 Renderer::SampleSky(const float3& direction) const
{
	if (!activateSky)
	{
		return {0.392f, 0.584f, 0.829f};
	}
	// Sample sky
	const float uFloat = static_cast<float>(skyWidth) * atan2_approximation2(direction.z, direction.x) * INV2PI
		- 0.5f;
	const int u = static_cast<int>(uFloat);

	const float vFloat = static_cast<float>(skyHeight) * FastAcos(direction.y) * INVPI - 0.5f;

	const int v = static_cast<int>(vFloat);

	const int skyIdx = max(0, u + v * skyWidth) * 3;

	return HDRLightContribution * float3{skyPixels[skyIdx], skyPixels[skyIdx + 1], skyPixels[skyIdx + 2]};
	//const float uFloat = static_cast<float>(skyWidth) * atan2f(direction.z, direction.x) * INV2PI - 0.5f;
	//const float vFloat = static_cast<float>(skyHeight) * acosf(direction.y) * INVPI - 0.5f;

	//const int u = static_cast<int>(uFloat);
	//const int v = static_cast<int>(vFloat);

	//const int skyIdx = max(0, u + v * skyWidth) * 3;
	//return HDRLightContribution * float3(skyPixels[skyIdx], skyPixels[skyIdx + 1], skyPixels[skyIdx + 2]);
}

void Renderer::HandleImguiPointLights()

{
	if (!ImGui::CollapsingHeader("Point Lights"))

		return;

	int pointIndex = 0;

	for (auto& light : pointLights)

	{
		ImGui::SliderFloat3(("Light position:" + to_string(pointIndex)).c_str(), light.data.position.cell, -1.0f, 2.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::ColorEdit3(("Light Color:" + to_string(pointIndex)).c_str(), light.data.color.cell);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		pointIndex++;
	}
}


void Renderer::HandleImguiAreaLights()

{
	if (!ImGui::CollapsingHeader("Area Lights"))

		return;

	int pointIndex = 0;

	for (auto& light : areaLights)

	{
		ImGui::SliderFloat3(("Area Lights position:" + to_string(pointIndex)).c_str(), light.data.position.cell, -5.0f,

		                    6.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::ColorEdit3(("Area Lights Color:" + to_string(pointIndex)).c_str(), light.data.color.cell);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::SliderFloat(("Area Lights color multiplier:" + to_string(pointIndex)).c_str(),

		                   &light.data.colorMultiplier,

		                   0.f, 10.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::SliderFloat(("Area Lights Radius:" + to_string(pointIndex)).c_str(), &light.data.radius, 0.f, 10.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::SliderInt(("Area Lights checks per shadow:" + to_string(pointIndex)).c_str(), &numCheckShadowsAreaLight,

		                 1, 100);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		pointIndex++;
	}
}


void Renderer::HandleImguiSpotLights()

{
	if (!ImGui::CollapsingHeader("Spot lights"))

		return;

	int spotIndex = 0;

	for (auto& light : spotLights)

	{
		ImGui::SliderFloat3(("Spot  position:" + to_string(spotIndex)).c_str(), light.data.direction.cell, 0, 1);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::SliderFloat3(("Spot light position:" + to_string(spotIndex)).c_str(), light.data.position.cell, -1.0f,

		                    2.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::ColorEdit3(("Spot light color:" + to_string(spotIndex)).c_str(), light.data.color.cell);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		ImGui::SliderFloat(("Spot light angle:" + to_string(spotIndex)).c_str(), &light.data.angle, 0.0f, 1.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}

		spotIndex++;
	}
}


void Renderer::HandleImguiDirectionalLight()

{
	if (!ImGui::CollapsingHeader("Directional light"))

		return;

	ImGui::SliderFloat3("Dir light direction:", dirLight.data.direction.cell, -1.0f, 1.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::ColorEdit3("Dir light:", dirLight.data.color.cell);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
}


//void Renderer::HandleImguiAmbientLight()

//{

//	if (!ImGui::CollapsingHeader("Ambient light"))

//		return;

//	ImGui::ColorEdit3("Ambient light:", ambientLight.data.color.cell);

//}


void Renderer::HandleImguiCamera()

{
	ImGui::SliderFloat("Accumulation weight", &weight, 0.001f, 1.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderFloat(" threshold rejection", &colorThreshold, 0.001f, 1.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
	ImGui::Checkbox("Accumulate", &staticCamera);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderFloat("HDR contribution", &HDRLightContribution, 0.1f, 10.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
	ImGui::Checkbox("Activate sky", &activateSky);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderInt("Max Bounces", &maxBounces, 1, 300);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	/*ImGui::SliderInt("Max Rays per Pixel", &maxRayPerPixel, 1, 200);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}*/

	ImGui::SliderFloat("DOF strength", &camera.defocusJitter, 0.0f, 50.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::DragFloat("Focal Point distance", &camera.focalTargetDistance, .1f, 0.3f, 100.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}


	ImGui::SliderFloat("AA strength", &antiAliasingStrength, 0.0f, 1.0f);
	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}


	ImGui::Text("Camera look ahead %.2f,  %.2f,  %.2f:", camera.ahead.x, camera.ahead.y,

	            camera.ahead.z);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
}


void Renderer::MaterialEdit(int index, vector<shared_ptr<Material>>::value_type& material)
{
	ImGui::ColorEdit3(("albedo:" + to_string(index)).c_str(), material->albedo.cell);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderFloat(("roughness :" + to_string(index)).c_str(), &material->roughness, 0,
	                   1.0f);
	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
	ImGui::SliderFloat(("IOR :" + to_string(index)).c_str(), &material->IOR, 0.0f,

	                   2.4f);
	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
	ImGui::SliderFloat(("Emmisive strength :" + to_string(index)).c_str(), &material->emissiveStrength, 0.0f,

	                   100.0f);
	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}
}

void Renderer::HandleImguiMaterials()

{
	int index = 0;


	if (ImGui::CollapsingHeader("Non-Metals"))

	{
		for (auto& material : nonMetalMaterials)

		{
			MaterialEdit(index, material);
			index++;
		}
	}

	if (ImGui::CollapsingHeader("Metals"))

	{
		for (auto& material : metalMaterials)

		{
			MaterialEdit(index, material);
			index++;
		}
	}

	if (ImGui::CollapsingHeader("Dielectrics"))

	{
		for (auto& material : dielectricsMaterials)

		{
			MaterialEdit(index, material);
			index++;
		}
	}
	if (ImGui::CollapsingHeader("Smoke"))

	{
		for (auto& material : smokeMaterials)

		{
			MaterialEdit(index, material);
			index++;
		}
	}
	if (ImGui::CollapsingHeader("Emissive"))

	{
		for (auto& material : emissiveMaterials)

		{
			MaterialEdit(index, material);
			index++;
		}
	}
}

void Renderer::HandleImguiSpheres()
{
	if (!ImGui::CollapsingHeader("Spheres"))

		return;

	int sphereIndex = 0;

	for (auto& sphere : spheres)
	{
		ImGui::SliderFloat3(("Sphere position:" + to_string(sphereIndex)).c_str(), sphere.center.cell, -5.0f, 5.0f);

		if (ImGui::IsItemEdited())
		{
			ResetAccumulator();
		}

		ImGui::SliderFloat(("Sphere radius:" + to_string(sphereIndex)).c_str(), &sphere.radius, 0.001f, 5.5f);

		if (ImGui::IsItemEdited())
		{
			ResetAccumulator();
		}
		ImGui::SliderInt(("Material Type" + to_string(sphereIndex)).c_str(), reinterpret_cast<int*>(&sphere.material),
		                 0,
		                 MaterialType::EMISSIVE);
		if (ImGui::IsItemEdited())
		{
			ResetAccumulator();
		}

		sphereIndex++;
	}
	if (ImGui::Button("Create new Sphere"))
	{
		AddSphere();
		ResetAccumulator();
	}
	if (ImGui::Button("Delete last Sphere"))
	{
		RemoveLastSphere();
		ResetAccumulator();
	}
}


// -----------------------------------------------------------

// Update user interface (imgui)

// -----------------------------------------------------------

void Renderer::HandleImguiTriangles()
{
	if (!ImGui::CollapsingHeader("Triangles"))

		return;

	int sphereIndex = 0;

	for (auto& triangle : triangles)
	{
		float3 pos = triangle.position;
		if (ImGui::SliderFloat3(("Triangle position:" + to_string(sphereIndex)).c_str(), pos.cell, -5.0f, 5.0f))
		{
			triangle.SetPos(pos);
		}

		if (ImGui::IsItemEdited())
		{
			ResetAccumulator();
		}

		ImGui::SliderInt(("Triangle Material Type" + to_string(sphereIndex)).c_str(),
		                 reinterpret_cast<int*>(&triangle.material),
		                 0,
		                 MaterialType::EMISSIVE);
		if (ImGui::IsItemEdited())
		{
			ResetAccumulator();
		}

		sphereIndex++;
	}
	if (ImGui::Button("Create 5 new Triangles"))
	{
		for (int i = 0; i < 5; i++)
			AddTriangle();
		ResetAccumulator();
	}
	if (ImGui::Button("Delete last Triangle"))
	{
		RemoveTriangle();
		ResetAccumulator();
	}
}

void Renderer::HandleImguiVoxelVolumes()
{
	if (!ImGui::CollapsingHeader("Volumes"))

		return;
	int i = 0;
	static int selectedItem = 0; // Index of the selected item in the combo box
	std::vector<const char*> cStrVoxFiles; // ImGui needs const char* array

	for (const auto& file : voxFiles)

	{
		cStrVoxFiles.push_back(file.c_str());
	}

	for (auto& scene : voxelVolumes)
	{
		if (ImGui::Button(("Generate new Sphere emissive" + to_string(i)).c_str()))

		{
			scene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);

			ResetAccumulator();
		}
		ImGui::SliderFloat(("radius volume sphere" + to_string(i)).c_str(), &radiusEmissiveSphere, 0.0f,
		                   static_cast<float>(scene.WORLDSIZE));

		if (ImGui::IsItemEdited())

		{
			scene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);
			ResetAccumulator();
		}
		ImGui::SliderFloat(("Perlin frq" + to_string(i)).c_str(), &frqGenerationPerlinNoise, 0.001f, .5f);

		if (ImGui::IsItemEdited())

		{
			scene.GenerateSomeNoise(frqGenerationPerlinNoise);
			ResetAccumulator();
		}
		ImGui::SliderFloat(("Perlin frq smoke" + to_string(i)).c_str(), &frqGenerationPerlinNoise, 0.001f, .5f);

		if (ImGui::IsItemEdited())

		{
			scene.GenerateSomeSmoke(frqGenerationPerlinNoise);
			ResetAccumulator();
		}
		std::vector<const char*> cStr; // ImGui needs const char* array


		ImGui::SliderInt(("Material Types" + to_string(i)).c_str(), &matTypeSphere, 0, MaterialType::EMISSIVE);
		if (ImGui::IsItemEdited())

		{
			scene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);
			ResetAccumulator();
		}

		//from Sven 232380


		// Dropdown for selecting .vox file


		if (ImGui::Combo(("Vox Files" + to_string(i)).c_str(), &selectedItem, cStrVoxFiles.data(),
		                 static_cast<int>(cStrVoxFiles.size())))
		{
			const std::string path = "assets/" + voxFiles[selectedItem];

			scene.LoadModel(*this, path.c_str());

			ResetAccumulator();
		}

		ImGui::SliderFloat3(("Vox model size" + to_string(i)).c_str(), scene.scaleModel.cell, 0.0f, 1.0f);

		if (ImGui::IsItemEdited())

		{
			ResetAccumulator();
		}
		float3 pos = scene.position;
		float3 rot = scene.rotation; // Rotation
		float3 scale = scene.scale; // Scale
		bool update = false;
		ImGui::SliderFloat3(("Vox position" + to_string(i)).c_str(), pos.cell, -10.0f, 10.0f, "%.1f");
		if (ImGui::IsItemEdited())
			update = true;
		ImGui::SliderFloat3(("Rotation" + to_string(i)).c_str(), rot.cell, -180, 180, "%.0f degrees");
		if (ImGui::IsItemEdited())
			update = true;
		ImGui::SliderFloat3(("Scale" + to_string(i)).c_str(), scale.cell, 0.1f, 2.0f, "%.1f");
		if (ImGui::IsItemEdited())
			update = true;
		if (update)
		{
			// Apply rotation and scaling
			scene.scale = scale;
			scene.rotation = rot;
			scene.position = pos;

			rot *= DEG2RAD;
			scene.SetTransform(rot);
			ResetAccumulator();
		}


		i++;
	}
	if (ImGui::Button("Create new Voxel Volume"))
	{
		AddVoxelVolume();
		ResetAccumulator();
	}
	if (ImGui::Button("Delete last Voxel Volume"))
	{
		RemoveVoxelVolume();
		ResetAccumulator();
	}
}

void Renderer::UI()

{
	//formatted with chatGPT
	// ImGui begin
	if (!ImGui::Begin("Debug Window"))
	{
		ImGui::End();
		return;
	}

	ImGui::BeginTabBar("##TabBar");

	// First tab for Lights
	if (ImGui::BeginTabItem("Lights"))
	{
		ImGui::BeginChild("Scrolling");


		HandleImguiAreaLights();


		HandleImguiPointLights();


		HandleImguiSpotLights();


		HandleImguiDirectionalLight();


		ImGui::EndChild();
		ImGui::EndTabItem();
	}

	// Second tab for Materials
	if (ImGui::BeginTabItem("Materials"))
	{
		ImGui::BeginChild("Scrolling");

		HandleImguiMaterials();

		ImGui::EndChild();
		ImGui::EndTabItem();
	}

	// Third tab for Entities
	if (ImGui::BeginTabItem("Entities"))
	{
		ImGui::BeginChild("Scrolling");

		HandleImguiSpheres();

		HandleImguiTriangles();

		HandleImguiVoxelVolumes();

		ImGui::EndChild();
		ImGui::EndTabItem();
	}


	if (ImGui::BeginTabItem("Camera"))
	{
		ImGui::BeginChild("Scrolling");

		HandleImguiCamera();
		ImGui::EndChild();
		ImGui::EndTabItem();
	}

	ImGui::EndTabBar();
	ImGui::End();
}
