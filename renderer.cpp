#include "precomp.h"

#include <execution>

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

float3 Renderer::PointLightEvaluate(Ray& ray, Scene& scene, PointLightData lightData)
{
	//Getting the intersection point
	const float3 intersectionPoint = ray.O + ray.t * ray.D;
	const float3 dir = lightData.position - intersectionPoint;
	const float dst = length(dir);

	//reciprocal is faster than division
	const float3 dirNormalized = dir * (1 / dst);

	const float3 normal = ray.GetNormal();
	//Having a negative dot product means the light is behind the point
	const float cosTheta = dot(dirNormalized, normal);
	if (cosTheta <= 0)
		return 0;
	//the  formula for distance attenuation 
	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color * (1.0f / (dst * dst));
	//materi
	float3 originRay = OffsetRay(intersectionPoint, normal);
	const float3 k = ray.GetAlbedo(scene);

	Ray shadowRay(originRay, dirNormalized);
	// we do not shoot the ray behind the light source
	shadowRay.t = dst;
	if (scene.IsOccluded(shadowRay))
		return 0;


	return lightIntensity * k;
}

float3 Renderer::SpotLightEvaluate(Ray& ray, Scene& scene, SpotLightData lightData)
{
	const float3 intersectionPoint = ray.O + ray.t * ray.D;
	const float3 dir = lightData.position - intersectionPoint;
	float dst = length(dir);
	float3 dirNormalized = dir / dst;

	float3 normal = ray.GetNormal();
	//light angle
	float cosTheta = dot(dirNormalized, lightData.direction);
	if (cosTheta <= lightData.angle)
		return 0;

	float alphaCutOff = 1.0f - (1.0f - cosTheta) * 1.0f / (1.0f - lightData.angle);

	float3 lightIntensity = max(0.0f, cosTheta) * lightData.color / (dst * dst);
	//material evaluation
	const float3 k = ray.GetAlbedo(scene);


	Ray shadowRay(OffsetRay(intersectionPoint, normal), dirNormalized);
	shadowRay.t = dst;
	if (scene.IsOccluded(shadowRay))
		return 0;

	return lightIntensity * k * alphaCutOff;
}

float3 Renderer::AreaLightEvaluation(Ray& ray, Scene& scene, SphereAreaLightData lightData)
{
	const float3 intersectionPoint = ray.O + ray.t * ray.D;
	const float3 normal = ray.GetNormal();
	const float3 center = lightData.position;
	const float radius = lightData.radius;
	float3 incomingLight{0};
	const float3 k = ray.GetAlbedo(scene);
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
		if (scene.IsOccluded(shadowRay))
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

float3 Renderer::DirectionalLightEvaluate(Ray& ray, Scene& scene, DirectionalLightData lightData)
{
	const float3 intersectionPoint = ray.O + ray.t * ray.D;
	const float3 dir = -lightData.direction;
	const float3 attenuation = dir;

	const float3 normal = ray.GetNormal();
	//light angle
	const float cosTheta = dot(attenuation, normal);
	if (cosTheta <= 0)
		return 0;
	const float3 lightIntensity = max(0.0f, cosTheta) * lightData.color;
	//material evaluation
	const float3 k = ray.GetAlbedo(scene);

	const Ray shadowRay(OffsetRay(intersectionPoint, normal), (dir));


	if (scene.IsOccluded(shadowRay))
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

void Renderer::Init()
{
	InitSeed(static_cast<uint>(time(nullptr)));

	// create fp32 rgb pixel buffer to render to
	accumulator = (float4*)MALLOC64(SCRWIDTH * SCRHEIGHT * 16);
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


	//Lighting set-up
	SetUpLights();
	//Material set-up
	const auto materialDifWhite = make_shared<ReflectivityMaterial>(float3(1, 1, 1));
	const auto materialDifRed = make_shared<ReflectivityMaterial>(float3(1, 0, 0));
	const auto materialDifBlue = make_shared<ReflectivityMaterial>(float3(0, 0, 1));
	const auto materialDifGreen = make_shared<ReflectivityMaterial>(float3(0, 1, 0), 0.0f);
	const auto partialMetal = make_shared<ReflectivityMaterial>(float3(1, 1, 1), 0.75f);

	//Mirror
	const auto materialDifReflectivity = make_shared<ReflectivityMaterial>(float3(1));
	const auto materialDifRefMid = make_shared<ReflectivityMaterial>(float3(0, 1, 1), 0.5f);
	const auto materialDifRefLow = make_shared<ReflectivityMaterial>(float3(1, 1, 0), 0.1f);
	//partial mirror

	nonMetalMaterials.push_back(materialDifWhite);
	nonMetalMaterials.push_back(materialDifRed);
	nonMetalMaterials.push_back(materialDifBlue);
	nonMetalMaterials.push_back(materialDifGreen);
	nonMetalMaterials.push_back(partialMetal);
	//metals
	metalMaterials.push_back(materialDifReflectivity);
	metalMaterials.push_back(materialDifRefMid);
	metalMaterials.push_back(materialDifRefLow);
	for (auto& mat : nonMetalMaterials)
		mainScene.materials.push_back(mat);
	for (auto& mat : metalMaterials)
		mainScene.materials.push_back(mat);
}

void Renderer::Illumination(Ray& ray, float3& incLight)
{
	const auto lightType = static_cast<size_t>(Rand(MAX_LIGHT_TYPES - 1));
	const auto p = static_cast<size_t>(Rand(POINT_LIGHTS));
	const auto s = static_cast<size_t>(Rand(SPOT_LIGHTS));
	const auto a = static_cast<size_t>(Rand(AREA_LIGHTS));
	switch (lightType)
	{
	case 0:
		incLight = PointLightEvaluate(ray, mainScene, pointLights[p].data);
		break;
	case 1:
		incLight = AreaLightEvaluation(ray, mainScene, areaLights[a].data);
		break;
	case 2:
		incLight = SpotLightEvaluate(ray, mainScene, spotLights[s].data);
		break;
	case 3:
		incLight = DirectionalLightEvaluate(ray, mainScene, dirLight.data);
		break;
	default: break;
	}
	incLight *= LIGHT_COUNT;
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
	mainScene.FindNearest(ray);
	// Break early if no intersection
	if (ray.indexMaterial == MaterialType::NONE)
	{
		return skyDome.SampleSky(ray);
	}

	const float3 N = ray.GetNormal();
	const float3 I = ray.IntersectionPoint();

	Ray newRay;

	switch (ray.indexMaterial)
	{
	//metals
	case MaterialType::METAL_MID:
	case MaterialType::METAL_HIGH:
	case MaterialType::METAL_LOW:
		{
			float3 reflectedDirection = ray.D - 2 * N * dot(N, ray.D);
			newRay = Ray{
				OffsetRay(I, N), reflectedDirection + ray.GetRoughness(mainScene) * RandomSphereSample()
			};
			return Trace(newRay, depth - 1) * ray.GetAlbedo(mainScene);
		}

	//non-metal
	case MaterialType::NON_METAL_WHITE:
	case MaterialType::NON_METAL_PINK:
	case MaterialType::NON_METAL_RED:
	case MaterialType::NON_METAL_BLUE:
	case MaterialType::NON_METAL_GREEN:
		{
			float3 color{0};

			if (RandomFloat() < ray.GetRoughness(mainScene))
			{
				float3 incLight{0};
				float3 randomDirection = DiffuseReflection(N);
				Illumination(ray, incLight);
				newRay = Ray{OffsetRay(I, N), randomDirection};
				color += incLight;
				color += Trace(newRay, depth - 1) * ray.GetAlbedo(mainScene);
			}
			else
			{
				float3 reflectedDirection = ray.D - 2 * N * dot(N, ray.D);
				newRay = Ray{
					OffsetRay(I, N), reflectedDirection + ray.GetRoughness(mainScene) * RandomSphereSample()
				};
				color += Trace(newRay, depth - 1);
			}
			return color;
		}
	case MaterialType::NONE:
		return skyDome.SampleSky(ray);
	}

	//fixes a warning
	return {0};
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick(float deltaTime)
{
	// pixel loop
	const Timer t;

	//c++ 17 onwards parallel for loop
	for_each(execution::par, vertIterator.begin(), vertIterator.end(),
	         [this](uint32_t y)
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
			         float weight = 1.0f / (static_cast<float>(numRenderedFrames) + 1.0f);
			         //we accumulate
			         float4 pixel = accumulator[x + pitch] * (1 - weight) + newPixel * weight;
			         screen->pixels[x + pitch] = RGBF32_to_RGB8(&pixel);
			         accumulator[x + pitch] = pixel;
		         }
	         });


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

void Renderer::HandleImguiGeneral()
{
	if (!ImGui::CollapsingHeader("General"))
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
	ImGui::SliderFloat2("DOF strength", camera.defocusJitter.cell, 0.0f, 2.0f);
	if (ImGui::IsItemEdited())
	{
		ResetAccumulator();
	}
	ImGui::DragFloat("Focal Point distance", &camera.focalDistance, .1f, -1.0f, 100.0f);
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
}

void Renderer::HandleImguiMaterials()
{
	int index = 0;

	if (ImGui::CollapsingHeader("Non-Metals"))
	{
		for (auto& material : nonMetalMaterials)
		{
			ImGui::ColorEdit3(("albedo:" + to_string(index)).c_str(), material->albedo.cell);
			if (ImGui::IsItemEdited())
			{
				ResetAccumulator();
			}
			ImGui::SliderFloat(("roughness :" + to_string(index)).c_str(), &material->roughness, 0,
			                   1.0f);
			index++;
			if (ImGui::IsItemEdited())
			{
				ResetAccumulator();
			}
		}
	}
	if (ImGui::CollapsingHeader("Metals"))
	{
		for (auto& material : metalMaterials)
		{
			ImGui::ColorEdit3(("albedo:" + to_string(index)).c_str(), material->albedo.cell);
			if (ImGui::IsItemEdited())
			{
				ResetAccumulator();
			}
			ImGui::SliderFloat(("roughness :" + to_string(index)).c_str(), &material->roughness, 0,
			                   1.0f);
			index++;
			if (ImGui::IsItemEdited())
			{
				ResetAccumulator();
			}
		}
	}
}

// -----------------------------------------------------------
// Update user interface (imgui)
// -----------------------------------------------------------
void Renderer::UI()
{
	// ray query on mouse
	/*Ray r = camera.GetPrimaryRay((float)mousePos.x, (float)mousePos.y);
	mainScene.FindNearest(r);
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
	ImGui::BeginChild("General");

	HandleImguiGeneral();

	ImGui::EndChild();

	ImGui::End();
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
