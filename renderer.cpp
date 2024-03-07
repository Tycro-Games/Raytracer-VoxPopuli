#include "precomp.h"

#include <execution>
#include <filesystem>

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
	const auto numThreads = thread::hardware_concurrency();
	cout << "Number of threads: " << numThreads << '\n';

	JobManager::CreateJobManager(numThreads);
	JobManager* jobManager = JobManager::GetJobManager();
	InitRandomSeedThread job;
	for (uint i = 0; i < jobManager->GetNumThreads(); i++)
		jobManager->AddJob2(&job);
	jobManager->RunJobs();
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

	//reciprocal is faster than division
	const float3 dirNormalized = dir * (1 / dst);

	const float3 normal = ray.rayNormal;
	//Having a negative dot product means the light is behind the point
	const float cosTheta = dot(dirNormalized, normal);
	if (cosTheta <= 0)
		return 0;
	//the  formula for distance attenuation 
	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color * (1.0f / (dst * dst));
	//materi
	const float3 originRay = OffsetRay(intersectionPoint, normal);
	const float3 k = ray.GetAlbedo(*this);

	Ray shadowRay(originRay, dirNormalized);
	// we do not shoot the ray behind the light source
	shadowRay.t = dst;
	if (IsOccluded(shadowRay))
		return 0;


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
	if (mainScene.IsOccluded(ray))
		return true;

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

	const Ray shadowRay(OffsetRay(intersectionPoint, normal), (dir));


	if (mainScene.IsOccluded(shadowRay))
		return 0;

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
	const auto emissive = make_shared<Material>(float3(1, 0, 0));
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
	emissiveMaterials.push_back(emissive);

	for (auto& mat : nonMetalMaterials)
		materials.push_back(mat);
	for (auto& mat : metalMaterials)
		materials.push_back(mat);
	for (auto& mat : dielectricsMaterials)
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

void Renderer::ShapesSetUp()
{
	AddSphere();
}

void Renderer::Init()
{
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
	//random light type
	const auto lightType = static_cast<size_t>(Rand(MAX_LIGHT_TYPES - 1));
	//random index of the element inside a light type array
	const auto p = static_cast<size_t>(Rand(POINT_LIGHTS));
	const auto s = static_cast<size_t>(Rand(SPOT_LIGHTS));
	const auto a = static_cast<size_t>(Rand(AREA_LIGHTS));
	//choose only one random light
	switch (lightType)
	{
	case 0:
		//every method evaluates to the light
		incLight = PointLightEvaluate(ray, pointLights[p].data);
		break;
	case 1:
		incLight = AreaLightEvaluation(ray, areaLights[a].data);
		break;
	case 2:
		incLight = SpotLightEvaluate(ray, spotLights[s].data);
		break;
	case 3:
		incLight = DirectionalLightEvaluate(ray, dirLight.data);
		break;
	default: break;
	}
	//multiply by the probability of choosing a light
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

void Renderer::FindNearest(Ray& ray)
{
	mainScene.FindNearest(ray);
	ray.rayNormal = ray.GetNormalVoxel();
	//get the nearest t
	{
		Ray sphereHit{ray.O, ray.D};

		for (auto& sphere : spheres)
		{
			sphere.Hit(sphereHit);
		}
		for (auto& triangle : triangles)
		{
			triangle.Hit(sphereHit);
		}
		//change to the closest ray information
		if (ray.t > sphereHit.t)
		{
			ray.t = sphereHit.t;
			ray.indexMaterial = sphereHit.indexMaterial;
			ray.rayNormal = sphereHit.rayNormal;
			ray.isInsideGlass = sphereHit.isInsideGlass;
		}
	}
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

#pragma region FindNearest
	FindNearest(ray);
#pragma endregion

	//evaluate materials and trace again for reflections and refraction

	// Break early if no intersection
	if (ray.indexMaterial == MaterialType::NONE)
	{
		return skyDome.SampleSky(ray);
	}
	//return .5f * (normal + 1);

	const float3 intersectionPoint = ray.IntersectionPoint();
	//return intersectionPoint;
	Ray newRay;

	switch (ray.indexMaterial)
	{
	//metals
	case MaterialType::METAL_MID:
	case MaterialType::METAL_HIGH:
	case MaterialType::METAL_LOW:
		{
			float3 reflectedDirection = Reflect(ray.D, ray.rayNormal);
			newRay = Ray{
				OffsetRay(intersectionPoint, ray.rayNormal),
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
			float3 color{0};
			if (RandomFloat() > SchlickReflectanceNonMetal(dot(-ray.D, ray.rayNormal)))
			{
				float3 incLight{0};
				float3 randomDirection = DiffuseReflection(ray.rayNormal);
				Illumination(ray, incLight);
				newRay = Ray{OffsetRay(intersectionPoint, ray.rayNormal), randomDirection};
				color += incLight;
				color += Trace(newRay, depth - 1) * ray.GetAlbedo(*this);
			}
			else
			{
				float3 reflectedDirection = Reflect(ray.D, ray.rayNormal);
				newRay = Ray{
					OffsetRay(intersectionPoint, ray.rayNormal),
					reflectedDirection + ray.GetRoughness(*this) * RandomSphereSample()
				};
				color = Trace(newRay, depth - 1);
			}
			return color;
		}
	//mostly based on Ray tracing in one weekend
	case MaterialType::GLASS:
		{
			float3 color{1};
			//code for glass
			bool isInGlass = ray.isInsideGlass;
			float IORMaterial = ray.GetRefractivity(*this); //1.45
			//get the IOR
			float refractionRatio = isInGlass ? IORMaterial : 1.0f / IORMaterial;
			//we need to get to the next voxel
			if (isInGlass)
			{
				color = ray.GetAlbedo(*this);
				//outside bounds if false, no handling of other entities
				mainScene.FindMaterialExit(ray, MaterialType::GLASS);
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

			newRay = {OffsetRay(ray.IntersectionPoint(), resultingNormal), resultingDirection};
			newRay.isInsideGlass = isInGlass;

			//if (!isInsideVolume)
			//	return skyDome.SampleSky(newRay);
			return Trace(newRay, depth - 1) * color;
		}
	case MaterialType::EMISSIVE:
		return ray.GetAlbedo(*this) * ray.GetEmissive(*this);

	case MaterialType::NONE:
		return skyDome.SampleSky(ray);
	//random materials from the models
	default:
		float3 incLight{0};
		float3 randomDirection = DiffuseReflection(ray.rayNormal);
		Illumination(ray, incLight);
		newRay = Ray{OffsetRay(intersectionPoint, ray.rayNormal), randomDirection};
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

//From Remi
float Renderer::SchlickReflectanceNonMetal(const float cosine)
{
	//for diffuse
	constexpr static float r0 = 0.04f;
	return r0 + (1 - r0) * powf((1 - cosine), 5);
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick(const float deltaTime)
{
	//DOF
	if (camera.defocusJitter > 0.0f)
	{
		Ray focusRay = camera.GetPrimaryRay(SCRWIDTH / 2, SCRHEIGHT / 2);
		mainScene.FindNearest(focusRay);

		camera.focalDistance = clamp(focusRay.t, -1.0f, 1e4f);
	}


	// pixel loop
	const Timer t;

	//c++ 17 onwards parallel for loop
#ifdef PROFILE
	for (uint32_t y = 0; y < SCRHEIGHT; y++)
	{
		//do only once
		const uint32_t pitch = y * SCRWIDTH;
		for (uint32_t x = 0; x < SCRWIDTH; x++)
		{
			//Ray triangleRay = camera.GetPrimaryRay(static_cast<float>(x), static_cast<float>(y));
			float3 totalLight{0};

			for (int i = 0; i < maxRayPerPixel; i++)
			{
				//Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x), static_cast<float>(y));
				//AA
				const float randomXDir = RandomFloat() - .5f;
				const float randomYDir = RandomFloat() - .5f;

				Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x) + randomXDir,
				                                      static_cast<float>(y) + randomYDir);
				/* primaryRay.O.x += randomXOri;
				 primaryRay.O.y += randomYOri;*/

				totalLight += Trace(primaryRay, maxBounces);
			}

			//  const float4 newPixel = float4{totalLight / static_cast<float>(maxRayPerPixel), 0.0f};
			const float4 newPixel = totalLight;
			/*  bvh.IntersectBVH(triangleRay, 0);
			  if (triangleRay.t < 1e34f && triangleRay.t < primaryRay.t)
				  pixel = float4{1.0f};*/
			// translate accumulator contents to rgb32 pixels
			const float weight = 1.0f / (static_cast<float>(numRenderedFrames) + 1.0f);
			//we accumulate
			float4 pixel = accumulator[x + pitch] * (1 - weight) + newPixel * weight;
			accumulator[x + pitch] = pixel;

			pixel = ApplyReinhardJodie(pixel);
			screen->pixels[x + pitch] = RGBF32_to_RGB8(&pixel);
		}
	}

#else
	for_each(execution::par, vertIterator.begin(), vertIterator.end(),
	         [this](const uint32_t y)
	         {
		         //do only once
		         const uint32_t pitch = y * SCRWIDTH;
		         for (uint32_t x = 0; x < SCRWIDTH; x++)
		         {
			         //Ray triangleRay = camera.GetPrimaryRay(static_cast<float>(x), static_cast<float>(y));
			         float3 totalLight{0};

			         for (int i = 0; i < maxRayPerPixel; i++)
			         {
				         //Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x), static_cast<float>(y));
				         //AA
				         const float randomXDir = RandomFloat() * antiAliasingStrength;
				         const float randomYDir = RandomFloat() * antiAliasingStrength;

				         Ray primaryRay = camera.GetPrimaryRay(static_cast<float>(x) + randomXDir,
				                                               static_cast<float>(y) + randomYDir);
				         /* primaryRay.O.x += randomXOri;
				          primaryRay.O.y += randomYOri;*/

				         totalLight += Trace(primaryRay, maxBounces);
			         }

			         //  const float4 newPixel = float4{totalLight / static_cast<float>(maxRayPerPixel), 0.0f};
			         const float4 newPixel = totalLight;
			         /*  bvh.IntersectBVH(triangleRay, 0);
			           if (triangleRay.t < 1e34f && triangleRay.t < primaryRay.t)
				           pixel = float4{1.0f};*/
			         // translate accumulator contents to rgb32 pixels
			         const float weight = 1.0f / (static_cast<float>(numRenderedFrames) + 1.0f);
			         //we accumulate
			         float4 pixel = accumulator[x + pitch] * (1 - weight) + newPixel * weight;
			         accumulator[x + pitch] = pixel;

			         pixel = ApplyReinhardJodie(pixel);
			         screen->pixels[x + pitch] = RGBF32_to_RGB8(&pixel);
		         }
	         });
#endif


	// performance report - running average - ms, MRays/s
	static float avg = 10, alpha = 1;
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.5f;
	float fps = 1000.0f / avg, rps = (SCRWIDTH * SCRHEIGHT) / avg;
	printf("%5.2fms (%.1ffps) - %.1fMrays/s\n", avg, fps, rps / 1000);
	// handle user input
	numRenderedFrames++;

	if (camera.HandleInput(deltaTime))
	{
		ResetAccumulator();
	}
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
	if (!ImGui::CollapsingHeader("Camera"))

		return;

	ImGui::SliderFloat("Perlin frq", &frqGenerationPerlinNoise, 0.001f, .5f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderFloat("HDR contribution", &skyDome.HDRLightContribution, 0.1f, 10.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderInt("Max Bounces", &maxBounces, 1, 300);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

	ImGui::SliderInt("Max Rays per Pixel", &maxRayPerPixel, 1, 200);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}

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

	if (ImGui::Button("Generate new Perlin noise"))

	{
		mainScene.GenerateSomeNoise(frqGenerationPerlinNoise);

		ResetAccumulator();
	}
	if (ImGui::Button("Generate new Sphere emissive"))

	{
		mainScene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);

		ResetAccumulator();
	}
	ImGui::SliderFloat("radius sphere", &radiusEmissiveSphere, 0.0f, WORLDSIZE);

	if (ImGui::IsItemEdited())

	{
		mainScene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);
		ResetAccumulator();
	}
	std::vector<const char*> cStr; // ImGui needs const char* array


	ImGui::SliderInt("Material Types", &matTypeSphere, 0, MaterialType::EMISSIVE);
	if (ImGui::IsItemEdited())

	{
		mainScene.CreateEmmisiveSphere(static_cast<MaterialType::MatType>(matTypeSphere), radiusEmissiveSphere);
		ResetAccumulator();
	}
	//from Sven 232380


	// Dropdown for selecting .vox file

	static int selectedItem = 0; // Index of the selected item in the combo box


	std::vector<const char*> cStrVoxFiles; // ImGui needs const char* array

	for (const auto& file : voxFiles)

	{
		cStrVoxFiles.push_back(file.c_str());
	}


	ImGui::Combo("Vox Files", &selectedItem, cStrVoxFiles.data(), static_cast<int>(cStrVoxFiles.size()));

	ImGui::SliderFloat3("Vox model size", mainScene.scaleModel.cell, 0.0f, 1.0f);

	if (ImGui::IsItemEdited())

	{
		ResetAccumulator();
	}


	if (ImGui::Button("Load Vox File") && selectedItem >= 0)

	{
		// Load the selected .vox file

		const std::string path = "assets/" + voxFiles[selectedItem];

		mainScene.LoadModel(*this, path.c_str());

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
	ImGui::SliderFloat(("IOR :" + to_string(index)).c_str(), &material->IOR, 1.0f,

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

void Renderer::UI()

{
	// ray query on mouse

	/*Ray r = camera.GetPrimaryRay((float)mousePos.x, (float)mousePos.y);

	*this.FindNearest(r);

	ImGui::Text("voxel: %i", r.voxel);*/

	ImGui::Begin("Debug");


	ImGui::BeginChild("Scrolling");

	if (ImGui::CollapsingHeader("Lights"))

	{
		HandleImguiAreaLights();


		HandleImguiPointLights();


		HandleImguiSpotLights();


		HandleImguiDirectionalLight();
	}

	if (ImGui::CollapsingHeader("Materials"))

	{
		HandleImguiMaterials();
	}
	if (ImGui::CollapsingHeader("Entities"))

	{
		HandleImguiSpheres();

		HandleImguiTriangles();
	}

	ImGui::BeginChild("General");


	HandleImguiCamera();


	ImGui::EndChild();


	ImGui::End();
}
