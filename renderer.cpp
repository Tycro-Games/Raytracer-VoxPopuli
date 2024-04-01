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
  pointLights.resize(0);
  spotLights.resize(5);
  areaLights.resize(0);
  CalculateLightCount();
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
  const float3 k = GetAlbedo(ray.indexMaterial);

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
  const float3 k = GetAlbedo(ray.indexMaterial);


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
  const float3 k = GetAlbedo(ray.indexMaterial);;
  const float3 point = OffsetRay(intersectionPoint, normal);

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
    const float3 lightIntensity = cosTheta * lightData.color * lightData.colorMultiplier * (radius * radius) *
      PI4 / (dst *
        dst);


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

bool Renderer::IsOccludedPlayerClimbable(Ray& ray) const
{
  bool isFirst = true;
  for (auto& scene : voxelVolumes)
  {
    if (isFirst)
    {
      isFirst = false;
      continue;
    }
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

float3 Renderer::GetAlbedo(const size_t indexMaterial) const
{
  return materials[indexMaterial]->albedo;
}

float Renderer::GetEmissive(const size_t indexMaterial) const
{
  return materials[indexMaterial]->emissiveStrength;
}

float Renderer::GetRefractivity(const size_t indexMaterial) const
{
  return materials[indexMaterial]->IOR;
}

float Renderer::GetRoughness(const size_t indexMaterial) const
{
  return materials[indexMaterial]->roughness;
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
  const float3 k = GetAlbedo(ray.indexMaterial);

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

void Renderer::RandomizeSmokeColors() const
{
  const float3 smokeColor = float3{1.0f, 0.7f, 1.0f};
  for (int i = MaterialType::SMOKE_LOW_DENSITY; i <= MaterialType::SMOKE_HIGH_DENSITY; i++)
  {
    materials[i]->albedo = smokeColor + float3{Rand(-0.2f, 0), Rand(-0.2f, 0.2f), Rand(-0.1f, 0)};
  }
}

void Renderer::MaterialSetUp()
{
  //first 5
  const auto materialDifWhite = make_shared<Material>(float3(1, 1, 1));
  const auto materialDifRed = make_shared<Material>(float3(1, 0, 0));
  const auto materialDifBlue = make_shared<Material>(float3(0, 0, 1));
  const auto materialDifGreen = make_shared<Material>(float3(0, 1, 0), 0.0f);
  const auto partialMetal = make_shared<Material>(float3(1, 1, 1), 0.75f);

  //Mirror   next3
  const auto materialDifReflectivity = make_shared<Material>(float3(1));
  const auto materialDifRefMid = make_shared<Material>(float3(0, 1, 1), 0.5f);
  const auto materialDifRefLow = make_shared<Material>(float3(0.9f), 0.01f);
  //partial mirror
  const auto glass = make_shared<Material>(float3(1, 1, 1));
  glass->IOR = 1.45f;
  float3 smokeColor = float3{1.0f, 0.7f, 1.0f};

  const auto smoke = make_shared<Material>(smokeColor);
  smoke->IOR = 1.0f;
  smoke->emissiveStrength = 3.0f;
  const auto smoke1 = make_shared<Material>(smokeColor);


  smoke1->IOR = 1.0f;
  smoke1->emissiveStrength = 8.0f;
  const auto smoke2 = make_shared<Material>(smokeColor);


  smoke2->IOR = 1.0f;
  smoke2->emissiveStrength = 12.0f;
  const auto smoke3 = make_shared<Material>(smokeColor);


  smoke3->IOR = 1.0f;
  smoke3->emissiveStrength = 15.0f;
  const auto smoke4 = make_shared<Material>(smokeColor);

  smoke4->IOR = 1.0f;
  smoke4->emissiveStrength = 16.0f;
  const auto smoke5 = make_shared<Material>(float3{0});
  smoke5->IOR = 1.0f;
  smoke5->emissiveStrength = 22.0f;

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
  smokeMaterials.push_back(smoke5);
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

void Renderer::CreateTrianglePattern()
{
  float3 triPos = {-1.75f, 0.0f, 3.0f};
  const float scale = 0.25f;
  for (int i = 0; i < 10; i++)
  {
    triangles.push_back(Triangle{static_cast<MaterialType::MatType>(Rand(MaterialType::METAL_LOW)), triPos, scale});
    triPos.x += scale * 2;
  }
}

void Renderer::RemoveTriangle()
{
  triangles.pop_back();
}

void Renderer::RemoveVoxelVolume()
{
  voxelVolumes.pop_back();
}

//5 parts
void Renderer::CreateBridge(const float3& offset, const float3& enterOffset, MaterialType::MatType doorMaterial)
{
  //7
  std::vector<Scene> bridgesParts;
  bridgesParts.emplace_back(float3{0.0f, 4.0f, -7.0f} + offset + enterOffset, 1);
  bridgesParts.emplace_back(float3{-1.0f, 0.0f, -11.0f} + offset, 1);
  //a bit to the right                                                  
  bridgesParts.emplace_back(float3{-5.0f, 1.0f, -12.0f} + offset, 1);

  bridgesParts.emplace_back(float3{-3.0f, 1.0f, -19.0f} + offset, 1);

  bridgesParts.emplace_back(float3{0.0f, -1.0f, -18.0f} + offset, 1);
  bridgesParts.emplace_back(float3{0.0f, 0.3f, -17.f} + offset, 64);

  //DOOR REUSE THIS
  bridgesParts[0].scale = {10.0f, 1.0f, 5.0f};
  bridgesParts[0].SetTransform({0});

  bridgesParts[1].scale = {3.0f, 10.0f, 1.0f};
  bridgesParts[1].ResetGrid(doorMaterial);
  bridgesParts[1].SetTransform({0});

  bridgesParts[2].scale = {2.0f, 3.0f, 10.0f};
  bridgesParts[2].SetTransform({0});
  //voxelVolumes[9].ResetGrid(MaterialType::GLASS);

  bridgesParts[3].scale = {7.0f, 1.0f, 1.0f};
  bridgesParts[3].SetTransform({0});
  bridgesParts[4].scale = {5.0f, 1.0f, 5.0f};
  bridgesParts[4].SetTransform({0});

  //add all up
  //CHECKPOINT
  bridgesParts[5].scale = {2.0f};
  //voxelVolumes[12].GenerateSomeSmoke(0.167f);
  bridgesParts[5].SetTransform({0});
  bridgesParts[5].ResetGrid(MaterialType::NONE);
  voxelVolumes.insert(voxelVolumes.end(), bridgesParts.begin(), bridgesParts.end());
}

void Renderer::CreateBridgeBlind(const float3& offset, const float3& enterOffset, MaterialType::MatType doorMaterial)
{
  //7
  std::vector<Scene> bridgesParts;
  bridgesParts.emplace_back(float3{0.0f, 4.0f, -7.0f} + offset + enterOffset, 1);
  bridgesParts.emplace_back(float3{-1.0f, 0.0f, -11.0f} + offset, 1);
  //a bit to the right                                                  
  bridgesParts.emplace_back(float3{5.0f, -41.0f, -12.0f} + offset, 1);
  bridgesParts.emplace_back(float3{-5.0f, 1.0f, -12.0f} + offset, 1);

  bridgesParts.emplace_back(float3{3.0f, 51.0f, -19.0f} + offset, 1);
  bridgesParts.emplace_back(float3{-3.0f, 1.0f, -19.0f} + offset, 1);

  bridgesParts.emplace_back(float3{0.0f, -1.0f, -18.0f} + offset, 1);
  bridgesParts.emplace_back(float3{0.0f, 0.3f, -17.f} + offset, 64);

  //DOOR REUSE THIS
  bridgesParts[0].scale = {10.0f, 1.0f, 5.0f};
  bridgesParts[0].SetTransform({0});

  bridgesParts[1].scale = {3.0f, 10.0f, 1.0f};
  bridgesParts[1].ResetGrid(doorMaterial);
  bridgesParts[1].SetTransform({0});

  bridgesParts[2].scale = {2.0f, 3.0f, 10.0f};
  bridgesParts[2].SetTransform({00,});
  bridgesParts[2].ResetGrid(MaterialType::METAL_LOW);
  bridgesParts[3].scale = {2.0f, 3.0f, 10.0f};
  bridgesParts[3].SetTransform({0});

  //bridgesParts[4].ResetGrid(MaterialType::METAL_LOW);
  bridgesParts[4].scale = {7.0f, 1.0f, 1.0f};
  bridgesParts[4].SetTransform({0});


  //voxelVolumes[9].ResetGrid(MaterialType::GLASS);

  bridgesParts[5].scale = {7.0f, 1.0f, 1.0f};
  //bridgesParts[5].ResetGrid(MaterialType::METAL_LOW);
  bridgesParts[5].SetTransform({0});
  bridgesParts[6].scale = {5.0f, 1.0f, 5.0f};
  bridgesParts[6].SetTransform({0});

  //add all up
  //CHECKPOINT
  bridgesParts[7].scale = {2.0f};
  //voxelVolumes[12].GenerateSomeSmoke(0.167f);
  bridgesParts[7].SetTransform({0});
  bridgesParts[7].ResetGrid(MaterialType::NONE);
  voxelVolumes.insert(voxelVolumes.end(), bridgesParts.begin(), bridgesParts.end());
}

void Renderer::SetUpFirstZone()
{
  CreateTrianglePattern();
  voxelVolumes.emplace_back(Scene({0}, 16));
  //environment
  voxelVolumes.emplace_back(Scene({0.0f, -1.0f, 0.0f}, 1));
  voxelVolumes.emplace_back(Scene({6.0f, 0.0f, 0.0f}, 1));
  voxelVolumes.emplace_back(Scene({-6.0f, 0.0f, 0.0f}, 1));
  voxelVolumes.emplace_back(Scene({0.0f, 4.0f, 0.0f}, 1));
  //checkpoint
  voxelVolumes.emplace_back(Scene({0.0f, 0.3f, 0.0f}, 64));
  //Text
  voxelVolumes.emplace_back(Scene({0.0f, 3.0f, -3.0f}, 32));
  //bridge
  CreateBridge({0, 0, 0.0f});
  //CreateBridge({0.0f, 0.0f, -20.0f});

  //checkpoint two


  //setup
  voxelVolumes[0].LoadModel(*this, "assets/player.vox");
  voxelVolumes[0].SetTransform(float3{0});
  voxelVolumes[1].scale = {5.0f, 1.0f, 5.0f};
  voxelVolumes[1].SetTransform({0});
  voxelVolumes[2].scale = {5.0f, 5.0f, 5.0f};
  voxelVolumes[2].SetTransform({0});
  voxelVolumes[3].scale = {5.0f, 5.0f, 5.0f};
  voxelVolumes[3].SetTransform({0});
  voxelVolumes[4].scale = {10.0f, 1.0f, 10.0f};
  voxelVolumes[4].SetTransform({0});
  voxelVolumes[5].scale = {3.0f};
  RandomizeSmokeColors();
  voxelVolumes[5].GenerateSomeSmoke(0.167f);
  voxelVolumes[5].SetTransform({0});
  voxelVolumes[6].LoadModelRandomMaterials("assets/Text.vox");
  voxelVolumes[6].scale = {5.0f};
  voxelVolumes[6].SetTransform({0});


  for (int i = 1; i < 5; i++)
  {
    voxelVolumes[i].ResetGrid(MaterialType::METAL_LOW);
  }

  for (size_t i = 0; i < spotLights.size(); i++)
  {
    if (i >= 2)
    {
      spotLights[i].data.position = {
        -3.0f, sinf((static_cast<float>(i))) + 1, -25.0f - static_cast<float>(i) * 2.0f
      };
      spotLights[i].data.direction = {1.0f, 0.0f, 0.0f};
      spotLights[i].data.angle = CosDegrees(Rand(20.0f, 45.0f));
      spotLights[i].data.color = {1.0f - RandomFloat(), RandomFloat(), RandomFloat()};
    }

    else
    {
      spotLights[i].data.position = {0.0f, 0.0f, -22.0f - static_cast<float>(i) * 3.0f};
      spotLights[i].data.direction = {0.0f, 1.0f, 0.0f};
    }
  }
  CreateBridgeBlind({0.0f, 0.0f, -17.0f}, {0.0f, -6.0f, 0.0f}, MaterialType::GLASS);
}

void Renderer::AddVoxelVolume()
{
  voxelVolumes.emplace_back(Scene({0}, 64));
}

void Renderer::ShapesSetUp()
{
  //AddSphere();
  SetUpFirstZone();
  /*constexpr int sizeX = 6;
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
  }*/
}

void Renderer::Init()
{
  CopyToPrevCamera();
  int skyBpp;
  skyPixels = stbi_loadf("assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0);
  /* for (int i = 0; i < skyWidth * skyHeight * 3; i++)
   {
     skyPixels[i] = sqrtf(skyPixels[i]);
   }*/
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
  MaterialSetUp();

  //Lighting set-up
  SetUpLights();
  //shape set-up
  ShapesSetUp();
  //Material set-up


  player.MovePlayer(voxelVolumes[0], float3{0}, float3{0, 1, 0});
  player.SetPrevios(voxelVolumes[0]);
}

void Renderer::Illumination(Ray& ray, float3& incLight)
{
  const size_t randLightIndex = static_cast<size_t>(Rand(static_cast<float>(lightCount)));
  //map every index to a certain light element				
  if (randLightIndex < pointCount)
  {
    const auto p = (randLightIndex);
    incLight = PointLightEvaluate(ray, pointLights[p].data);
  }
  else if (randLightIndex < areaCount + pointCount)
  {
    const auto a = randLightIndex - pointCount;
    incLight = AreaLightEvaluation(ray, areaLights[a].data);
  }
  else if (randLightIndex < areaCount + spotCount + pointCount)
  {
    const auto s = randLightIndex - areaCount - pointCount;
    incLight = SpotLightEvaluate(ray, spotLights[s].data);
  }
  //1 is the only directional
  else
  {
    incLight = DirectionalLightEvaluate(ray, dirLight.data);
  }

  incLight *= static_cast<float>(lightCount);
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

__m256 Renderer::SlowReciprocal(__m256& dirSSE)
{
  return _mm256_div_ps(_mm256_set1_ps(1.0f), dirSSE);
}

int32_t Renderer::FindNearest(Ray& ray)
{
  int32_t voxelIndex = -2;


  const int32_t voxelCount = static_cast<int32_t>(voxelVolumes.size());
  for (int32_t i = 0; i < voxelCount; i++)

  {
    Ray backupRay = ray;


    const mat4& invMat = voxelVolumes[i].invMatrix;
#if 1
    ray.O4 = TransformPosition_SSEM(ray.O4, invMat);

    ray.D4 = TransformVector_SSEM(ray.D4, invMat);

    //for my machine the fast reciprocal is a bit slower

#if 0
		__m128 rDSSE = SlowReciprocal(ray.D4);
#else
    __m128 rDSSE = FastReciprocal(ray.D4);

#endif

    ray.Dsign4 = ray.ComputeDsign_SSE(ray.D4);

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
      voxelIndex = -1;
    }
  }
  return voxelIndex;
}

int32_t Renderer::FindNearestPlayer(Ray& ray)
{
  int32_t voxelIndex = -2;


  const int32_t voxelCount = static_cast<int32_t>(voxelVolumes.size());
  //skip player
  for (int32_t i = 1; i < voxelCount; i++)

  {
    Ray backupRay = ray;


    mat4 invMat = voxelVolumes[i].invMatrix;
#if 1
    ray.O4 = TransformPosition_SSEM(ray.O4, invMat);

    ray.D4 = TransformVector_SSEM(ray.D4, invMat);

    //for my machine the fast reciprocal is a bit slower

#if 0
		__m128 rDSSE = SlowReciprocal(ray.D4);
#else
    __m128 rDSSE = FastReciprocal(ray.D4);

#endif

    ray.Dsign4 = ray.ComputeDsign_SSE(ray.D4);

    ray.rD4 = rDSSE;

#else
		ray.O = TransformPosition(ray.O, invMat);


		ray.D = TransformVector(ray.D, invMat);

		ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
		ray.Dsign = ray.ComputeDsign(ray.D);
#endif

    if (voxelVolumes[i].FindNearestExcept(ray, MaterialType::SMOKE_LOW_DENSITY, MaterialType::SMOKE_PLAYER))
    {
      voxelIndex = i;
    }
    backupRay.t = ray.t;
    backupRay.CopyToPrevRay(ray);
  }

  return voxelIndex;
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace(Ray& ray, int depth)
{
  //return {0};

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
        reflectedDirection + GetRoughness(ray.indexMaterial) * RandomSphereSample()
      };
      return Trace(newRay, depth - 1) * GetAlbedo(ray.indexMaterial);
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
        color += Trace(newRay, depth - 1) * GetAlbedo(ray.indexMaterial);
      }
      else
      {
        float3 reflectedDirection = Reflect(ray.D, ray.rayNormal);
        newRay = Ray{
          OffsetRay(ray.IntersectionPoint(), ray.rayNormal),
          reflectedDirection + GetRoughness(ray.indexMaterial) * RandomSphereSample()
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
      float IORMaterial = GetRefractivity(ray.indexMaterial); //1.45
      //get the IOR
      float refractionRatio = isInGlass ? IORMaterial : 1.0f / IORMaterial;
      //we need to get to the next voxel
      bool isInsideVolume = true;
      if (isInGlass)
      {
        color = GetAlbedo(ray.indexMaterial);
        //only the first one has glass
        Ray backupRay = ray;

        mat4 invMat = voxelVolumes[voxIndex].invMatrix;
        ray.O = TransformPosition(ray.O, invMat);


        ray.D = TransformVector(ray.D, invMat);

        ray.rD = float3(1 / ray.D.x, 1 / ray.D.y, 1 / ray.D.z);
        ray.Dsign = ray.ComputeDsign(ray.D);

        isInsideVolume = voxelVolumes[voxIndex].FindMaterialExit(ray, MaterialType::GLASS);
        backupRay.t = ray.t;
        backupRay.CopyToPrevRay(ray);
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
  case MaterialType::SMOKE_LOW_DENSITY:
  case MaterialType::SMOKE_LOW2_DENSITY:
  case MaterialType::SMOKE_MID_DENSITY:
  case MaterialType::SMOKE_MID2_DENSITY:
  case MaterialType::SMOKE_HIGH_DENSITY:
  case MaterialType::SMOKE_PLAYER:
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
      //player
      if (voxIndex == 0)
      {
        float3 incLight{0};

        Illumination(ray, incLight);
        float sqrThreshold = 16.0f;
        float value = sqrLength(incLight);
        if (value > sqrThreshold)
        {
          inLight = true;
          std::cout << "Player is in the light\n";
        }
      }
      if (isInGlass)
      {
        //only and only for the player
        /*if (voxIndex == 0)
        {
          float3 incLight{0};

          Illumination(ray, incLight);
          float threshold = 25.1f;
          if (sqrLength(incLight) > threshold)
          {
            inLight = true;
            std::cout << "Player is in the light\n";
          }
          else if (!inLight)
          {
            intensity = GetEmissive(ray.indexMaterial);
            color = GetAlbedo(ray.indexMaterial);
          }
        }*/

        intensity = GetEmissive(ray.indexMaterial);
        color = GetAlbedo(ray.indexMaterial);

        //only the first one has glass
        Ray backupRay = ray;

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
        ray.O = ray.O + ray.D * Rand(ray.t * .45f, ray.t);

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
    return GetAlbedo(ray.indexMaterial) * GetEmissive(ray.indexMaterial);

  //random materials from the models
  default:
    float3 incLight{0};
    float3 randomDirection = DiffuseReflection(ray.rayNormal);
    Illumination(ray, incLight);
    Ray newRay;

    newRay = Ray{OffsetRay(ray.IntersectionPoint(), ray.rayNormal), randomDirection};
    return Trace(newRay, depth - 1) * GetAlbedo(ray.indexMaterial) + incLight;
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
  const float3 exponent{
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

  static __m256 PI2SSE = _mm256_set1_ps(2 * PI);

  static __m256 antiAliasingStrengthSSE = _mm256_set1_ps(antiAliasingStrength);

#ifdef PROFILE
	for (int32_t y = 0; y < SCRHEIGHT; y++)
	{
#else

  for_each(execution::par, vertIterator.begin(), vertIterator.end(),
           [this](const int32_t y)
           {
#endif

#if 1
             const __m256 ySSE = _mm256_cvtepi32_ps(_mm256_set1_epi32(y));

             const __m256 weightSSE = _mm256_set1_ps(weight);
             const __m256 invWeightSSE = _mm256_set1_ps(1.0f - weight);
             const __m256i pitchSSE = _mm256_set1_epi32(y * SCRWIDTH);
             for (int32_t x = 0; x < SCRWIDTH; x += 8)
             {
               __m256i xIndexSSE = _mm256_set_epi32(x + 7, x + 6, x + 5,
                                                    x + 4, x + 3,
                                                    x + 2, x + 1,
                                                    x);
               const __m256 xSSE = _mm256_cvtepi32_ps(xIndexSSE);


               __m256 randomXDirSSE = _mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat(),
                                                    RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat());
               __m256 randomYDirSSE = _mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat(),
                                                    RandomFloat(), RandomFloat(), RandomFloat(), RandomFloat());


               const __m256 rSSE = _mm256_sqrt_ps(_mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(),
                                                                RandomFloat(),
                                                                RandomFloat(), RandomFloat(), RandomFloat(),
                                                                RandomFloat()));
               const __m256 thetaSSE = _mm256_mul_ps(_mm256_set_ps(RandomFloat(), RandomFloat(), RandomFloat(),
                                                                   RandomFloat(),
                                                                   RandomFloat(), RandomFloat(), RandomFloat(),
                                                                   RandomFloat()), PI2SSE);
               const __m256 xCircleSSE = _mm256_mul_ps(_mm256_cos_ps(thetaSSE), rSSE);
               const __m256 yCircleSSE = _mm256_mul_ps(_mm256_sin_ps(thetaSSE), rSSE);

               randomXDirSSE = _mm256_fmadd_ps(randomXDirSSE, antiAliasingStrengthSSE,
                                               xSSE);

               float coordinatesX[8];
               float coordinatesY[8];
               _mm256_store_ps(coordinatesX, randomXDirSSE);

               randomYDirSSE = _mm256_fmadd_ps(randomYDirSSE, antiAliasingStrengthSSE,
                                               ySSE);
               _mm256_store_ps(coordinatesY, randomYDirSSE);

               // Compute primary rays
               Ray primaryRays[8];
               float radiusX[8];
               float radiusY[8];
               _mm256_store_ps(radiusX, xCircleSSE);
               _mm256_store_ps(radiusY, yCircleSSE);

               camera.GetPrimaryRay(coordinatesX[0], coordinatesY[0], primaryRays[0].O, primaryRays[0].D,
                                    float2{radiusX[0], radiusY[0]});
               camera.GetPrimaryRay(coordinatesX[1], coordinatesY[1], primaryRays[1].O, primaryRays[1].D,
                                    float2{radiusX[1], radiusY[1]});
               //so on to 8
               camera.GetPrimaryRay(coordinatesX[2], coordinatesY[2], primaryRays[2].O, primaryRays[2].D,
                                    float2{radiusX[2], radiusY[2]});
               camera.GetPrimaryRay(coordinatesX[3], coordinatesY[3], primaryRays[3].O, primaryRays[3].D,
                                    float2{radiusX[3], radiusY[3]});
               camera.GetPrimaryRay(coordinatesX[4], coordinatesY[4], primaryRays[4].O, primaryRays[4].D,
                                    float2{radiusX[4], radiusY[4]});
               camera.GetPrimaryRay(coordinatesX[5], coordinatesY[5], primaryRays[5].O, primaryRays[5].D,
                                    float2{radiusX[5], radiusY[5]});
               camera.GetPrimaryRay(coordinatesX[6], coordinatesY[6], primaryRays[6].O, primaryRays[6].D,
                                    float2{radiusX[6], radiusY[6]});
               camera.GetPrimaryRay(coordinatesX[7], coordinatesY[7], primaryRays[7].O, primaryRays[7].D,
                                    float2{radiusX[7], radiusY[7]});

               primaryRays[0].rD4 = SlowReciprocal(primaryRays[0].D4);
               primaryRays[0].D4 = normalize(primaryRays[0].D4);
               primaryRays[0].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[0].D4);

               primaryRays[1].rD4 = SlowReciprocal(primaryRays[1].D4);
               primaryRays[1].D4 = normalize(primaryRays[1].D4);
               primaryRays[1].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[1].D4);
               //so on to 8
               primaryRays[2].rD4 = SlowReciprocal(primaryRays[2].D4);
               primaryRays[2].D4 = normalize(primaryRays[2].D4);
               primaryRays[2].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[2].D4);
               //and so on
               primaryRays[3].rD4 = SlowReciprocal(primaryRays[3].D4);
               primaryRays[3].D4 = normalize(primaryRays[3].D4);
               primaryRays[3].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[3].D4);

               primaryRays[4].rD4 = SlowReciprocal(primaryRays[4].D4);
               primaryRays[4].D4 = normalize(primaryRays[4].D4);
               primaryRays[4].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[4].D4);

               primaryRays[5].rD4 = SlowReciprocal(primaryRays[5].D4);
               primaryRays[5].D4 = normalize(primaryRays[5].D4);
               primaryRays[5].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[5].D4);

               primaryRays[6].rD4 = SlowReciprocal(primaryRays[6].D4);
               primaryRays[6].D4 = normalize(primaryRays[6].D4);
               primaryRays[6].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[6].D4);

               primaryRays[7].rD4 = SlowReciprocal(primaryRays[7].D4);
               primaryRays[7].D4 = normalize(primaryRays[7].D4);
               primaryRays[7].Dsign4 = Ray::ComputeDsign_SSE(primaryRays[7].D4);


               // Trace rays and store results
               float4 newPixels[8];

               newPixels[0] = Trace(primaryRays[0], maxBounces);
               newPixels[1] = Trace(primaryRays[1], maxBounces);
               newPixels[2] = Trace(primaryRays[2], maxBounces);
               newPixels[3] = Trace(primaryRays[3], maxBounces);
               newPixels[4] = Trace(primaryRays[4], maxBounces);
               newPixels[5] = Trace(primaryRays[5], maxBounces);
               newPixels[6] = Trace(primaryRays[6], maxBounces);
               newPixels[7] = Trace(primaryRays[7], maxBounces);


               const __m256 pixelSSE = _mm256_loadu_ps(&newPixels[6].x);
               const __m256 pixel1SSE = _mm256_loadu_ps(&newPixels[4].x);
               const __m256 pixel2SSE = _mm256_loadu_ps(&newPixels[2].x);
               const __m256 pixel3SSE = _mm256_loadu_ps(&newPixels[0].x);

               xIndexSSE = _mm256_add_epi32(xIndexSSE, pitchSSE);

               int32_t xIndex[8];
               _mm256_storeu_si256(reinterpret_cast<__m256i*>(xIndex), xIndexSSE);

               const __m256 accumulatorSSE1 = _mm256_loadu_ps(&accumulator[xIndex[6]].x);
               const __m256 accumulatorSSE2 = _mm256_loadu_ps(&accumulator[xIndex[4]].x);
               const __m256 accumulatorSSE3 = _mm256_loadu_ps(&accumulator[xIndex[2]].x);
               const __m256 accumulatorSSE4 = _mm256_loadu_ps(&accumulator[xIndex[0]].x);


               const __m256 blendedSSE1 = _mm256_fmadd_ps(invWeightSSE, accumulatorSSE1,
                                                          _mm256_mul_ps(
                                                            pixelSSE,
                                                            weightSSE));
               const __m256 blendedSSE2 = _mm256_fmadd_ps(invWeightSSE, accumulatorSSE2,
                                                          _mm256_mul_ps(
                                                            pixel1SSE,
                                                            weightSSE));
               const __m256 blendedSSE3 = _mm256_fmadd_ps(invWeightSSE, accumulatorSSE3,
                                                          _mm256_mul_ps(
                                                            pixel2SSE,
                                                            weightSSE));
               const __m256 blendedSSE4 = _mm256_fmadd_ps(invWeightSSE, accumulatorSSE4,
                                                          _mm256_mul_ps(
                                                            pixel3SSE,
                                                            weightSSE));

               //display
               float4 newPixel[8];
               _mm256_store_ps(&newPixel[6].x, blendedSSE1);
               _mm256_store_ps(&newPixel[4].x, blendedSSE2);
               _mm256_store_ps(&newPixel[2].x, blendedSSE3);
               _mm256_store_ps(&newPixel[0].x, blendedSSE4);

               accumulator[xIndex[0]] = newPixel[0];
               accumulator[xIndex[1]] = newPixel[1];
               accumulator[xIndex[2]] = newPixel[2];
               accumulator[xIndex[3]] = newPixel[3];
               accumulator[xIndex[4]] = newPixel[4];
               accumulator[xIndex[5]] = newPixel[5];
               accumulator[xIndex[6]] = newPixel[6];
               accumulator[xIndex[7]] = newPixel[7];


               newPixel[0] = ApplyReinhardJodie(newPixel[0]);
               newPixel[1] = ApplyReinhardJodie(newPixel[1]);
               newPixel[2] = ApplyReinhardJodie(newPixel[2]);
               newPixel[3] = ApplyReinhardJodie(newPixel[3]);
               newPixel[4] = ApplyReinhardJodie(newPixel[4]);
               newPixel[5] = ApplyReinhardJodie(newPixel[5]);
               newPixel[6] = ApplyReinhardJodie(newPixel[6]);
               newPixel[7] = ApplyReinhardJodie(newPixel[7]);

               screen->pixels[xIndex[0]] = RGBF32_to_RGB8(&newPixel[0]);
               screen->pixels[xIndex[1]] = RGBF32_to_RGB8(&newPixel[1]);
               screen->pixels[xIndex[2]] = RGBF32_to_RGB8(&newPixel[2]);
               screen->pixels[xIndex[3]] = RGBF32_to_RGB8(&newPixel[3]);
               screen->pixels[xIndex[4]] = RGBF32_to_RGB8(&newPixel[4]);
               screen->pixels[xIndex[5]] = RGBF32_to_RGB8(&newPixel[5]);
               screen->pixels[xIndex[6]] = RGBF32_to_RGB8(&newPixel[6]);
               screen->pixels[xIndex[7]] = RGBF32_to_RGB8(&newPixel[7]);
             }

             //avx2 
#else
		const uint32_t pitch = y * SCRWIDTH;
		const float invWeight = 1.0f - weight;

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


			pixel = BlendColor(pixel, accumulator[x + pitch], invWeight);

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

void Renderer::SetUpSecondZone()
{
  //area light dynamic voxels
  voxelVolumes[3].GenerateSomeSmoke(0.167f);
  materials[MaterialType::GLASS]->IOR = 1.0f;
  materials[MaterialType::GLASS]->albedo = {.50f};

  CreateBridgeBlind(float3{0, 0, triggerCheckpoint});
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick(const float deltaTime)
{
  const Timer t;

  if (staticCamera)
  {
    if (!IsKeyDown(GLFW_KEY_SPACE))
      if (camera.HandleInput(deltaTime))
      {
        ResetAccumulator();
      }
    camera.SetFrustumNormals();

    // pixel loop

    //DOF from Remi

    Ray focusRay = camera.GetPrimaryRay(SCRWIDTH / 2, SCRHEIGHT / 2);
    for (auto& scene : voxelVolumes)
      scene.FindNearest(focusRay);

    camera.focalDistance = clamp(focusRay.t, -1.0f, 1e4f);


    Update();


    CopyToPrevCamera();
  }
  //reproject
  else
  {
  }
  //game logic

  if (inLight || IsKeyDown(GLFW_KEY_R))
  {
    player.RevertMovePlayer(voxelVolumes[0]);
    ResetAccumulator();
  }

  player.Update(deltaTime);
  if (player.UpdateInput() && IsKeyDown(GLFW_KEY_SPACE))
  {
    Ray checkOcclusion = player.GetRay();
    if (FindNearestPlayer(checkOcclusion) > 0 && checkOcclusion.t < player.GetDistance())
    {
      //check for normal and rotate
      //move player
      constexpr float offsetPlayer = 5.0f;

      camera.camPos = float3{camera.camPos.x, camera.camPos.y, checkOcclusion.O.z + offsetPlayer};
      const float3 intersectionPoint = checkOcclusion.IntersectionPoint();
      camera.camTarget = intersectionPoint;
      camera.HandleInput(0.0f);
      //reset level state load next chunk

      player.MovePlayer(voxelVolumes[0], intersectionPoint, checkOcclusion.rayNormal);
      if (intersectionPoint.z < triggerCheckpoint && intersectionPoint.y < 0.5f)
      {
        //next chunk loaded
        RandomizeSmokeColors();
        triggerCheckpoint -= 17.0f;
        //first is always player
        voxelVolumes.erase(voxelVolumes.begin() + 1, voxelVolumes.begin() + dataChunks[currentChunk++].elementsCount);
        triangles.clear();
        switch (currentChunk)
        {
        case 1:
          SetUpSecondZone();
          break;
        default:
          break;
        }

        player.SetPrevios(voxelVolumes[0]);
      }
      ResetAccumulator();
    }
    //staticCamera = !IsOccludedPlayerClimbable(checkOcclusion);
  }

  inLight = false;

  // performance report - running average - ms, MRays/s
  static float avg = 10, alpha = 1;
  avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
  if (alpha > 0.05f) alpha *= 0.5f;
  float fps = 1000.0f / avg, rps = (SCRWIDTH * SCRHEIGHT) / avg;
  printf("%5.2fms (%.1ffps) - %.1fMrays/s\n", avg, fps, rps / 1000);
  // handle user input
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

void Renderer::AddPointLight()
{
  pointLights.push_back(PointLight{});
  CalculateLightCount();
}

void Renderer::RemovePointLight()
{
  pointLights.pop_back();
  CalculateLightCount();
}

void Renderer::AddAreaLight()
{
  areaLights.push_back(SphereAreaLight{});
  CalculateLightCount();
}

void Renderer::RemoveAreaLight()
{
  areaLights.pop_back();
  CalculateLightCount();
}

void Renderer::AddSpotLight()
{
  spotLights.push_back(SpotLight{});
  CalculateLightCount();
}

void Renderer::RemoveSpotLight()
{
  spotLights.pop_back();
  CalculateLightCount();
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

void Renderer::CalculateLightCount()
{
  pointCount = pointLights.size();
  spotCount = spotLights.size();
  areaCount = areaLights.size();
  lightCount = pointCount + spotCount + areaCount + 1; //directional light is one
}

void Renderer::MouseDown(int /*button*/)
{
  //std::cout << "Button: " << button << std::endl;

  //Ray primaryRay = {camera.GetPrimaryRay(static_cast<float>(mousePos.x),
  //                                      static_cast<float>(mousePos.y));
  //Trace(primaryRay, 5);
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
  if (ImGui::Button("Create new point light"))
  {
    AddPointLight();
    ResetAccumulator();
  }
  if (ImGui::Button("Remove last point light"))
  {
    RemovePointLight();
    ResetAccumulator();
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
  if (ImGui::Button("Create new area light"))
  {
    AddAreaLight();
    ResetAccumulator();
  }
  if (ImGui::Button("Remove last area light"))
  {
    RemoveAreaLight();
    ResetAccumulator();
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
  if (ImGui::Button("Create new spot light"))
  {
    AddSpotLight();
    ResetAccumulator();
  }
  if (ImGui::Button("Remove last spot light"))
  {
    RemoveSpotLight();
    ResetAccumulator();
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
    CreateTrianglePattern();
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
