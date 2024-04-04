#pragma once

namespace Tmpl8
{
  struct ChunkData
  {
    uint32_t elementsCount = 0;
  };

  struct AlbedoIlluminationData
  {
    float3 albedo = {1};
    float3 illumination = {0};

    float3 GetColor() const
    {
      return albedo * illumination;
    }
  };

  struct RayDataReproject
  {
    float3 intersectionPoint; //12
    float3 normal; //12
    float t; //4
    MaterialType::MatType materialIndex; //1

    uint8_t dummy; //1
    uint16_t dummy1; //2
    //32 bytes !
    static RayDataReproject GetRayInfo(const Ray& ray)
    {
      return {ray.IntersectionPoint(), ray.rayNormal, ray.t, ray.indexMaterial, 0, 0};
    }
  };

  constexpr size_t MAX_LIGHT_TYPES = 4;
  constexpr size_t CHUNK_COUNT = 3;
  //+1 for directional light


  class Renderer : public TheApp
  {
  public:
    void InitMultithreading();
    void SetUpLights();
    float3 PointLightEvaluate(Ray& ray, const PointLightData& lightData);
    float3 SpotLightEvaluate(const Ray& ray, const SpotLightData& lightData) const;
    float3 AreaLightEvaluation(Ray& ray, const SphereAreaLightData& lightData) const;
    bool IsOccluded(Ray& ray) const;
    bool IsOccludedPlayerClimbable(Ray& ray) const;
    bool IsOccludedSpheres(Ray& ray) const;
    float3 GetAlbedo(size_t indexMaterial) const;
    float GetEmissive(size_t indexMaterial) const;
    float GetRefractivity(size_t indexMaterial) const;
    float GetRoughness(size_t indexMaterial) const;
    float3 DirectionalLightEvaluate(Ray& ray, const DirectionalLightData& lightData);
    void ResetAccumulator();
    void RandomizeSmokeColors() const;
    void MaterialSetUp();
    void AddSphere();
    void RemoveLastSphere();
    void AddTriangle();
    void CreateTrianglePattern();
    void RemoveTriangle();
    void RemoveVoxelVolume();
    void CreateBridge(const float3& offset, const float3& enterOffset = {0},
                      MaterialType::MatType doorMaterial = MaterialType::GLASS);
    void CreateBridgeBlind(const float3& offset, const float3& enterOffset = {0.0f, -6.0f, 0.0f},
                           MaterialType::MatType doorMaterial = MaterialType::GLASS);
    void SetUpFirstZone();
    void AddVoxelVolume();
    void ShapesSetUp();
    // game flow methods
    void Init() override;
    void Illumination(Ray& ray, float3& incLight);
    bool IsOccludedPrevFrame(const float3& intersectionP) const;
    float3 SampleHistory(const float2& uvCoordinate);

    void ClampHistory(float3& historySample, float3 newSample, const int2& currentPixel);
    static float3 Reflect(float3 direction, float3 normal);
    static float3 Refract(float3 direction, float3 normal, float IORRatio);
    __m128 FastReciprocal(__m128& x);
    __m128 SlowReciprocal(__m128& dirSSE);
    __m256 SlowReciprocal(__m256& dirSSE);
    void TransformPositionAndDirection_SSE(__m128& oriSSE, __m128& dirSSE, const mat4& invMat, Ray& ray);
    int32_t FindNearest(Ray& ray);
    int32_t FindNearestPlayer(Ray& ray);
    float3 Trace(Ray& ray, int depth);
    AlbedoIlluminationData TraceMetal(Ray& ray, int depth);
    AlbedoIlluminationData TraceNonMetal(Ray& ray, int depth);
    AlbedoIlluminationData TraceDialectric(Ray& ray, int depth, int32_t voxIndex);
    AlbedoIlluminationData TraceSmoke(Ray& ray, int depth, int32_t voxIndex);
    AlbedoIlluminationData TraceEmmision(Ray& ray);
    AlbedoIlluminationData TraceModelMaterials(Ray& ray, int depth);
    AlbedoIlluminationData GetValue(Ray& ray, int depth, int32_t voxIndex);
    AlbedoIlluminationData TraceReproject(Ray& ray, int depth);
    static float SchlickReflectance(float cosine, float indexOfRefraction);
    float3 Absorption(const float3& color, float intensity, float distanceTraveled);
    float SchlickReflectanceNonMetal(const float cosine);
    float CalculateDistanceToPlane(const float3& point, const float3& normal, const float3& pointOnPlane);
    float3 CalculatePlaneNormal(const float3& point1, const float3& point2, const float3& point3);
    float4 SamplePreviousFrameColor(const float2& screenPosition);
    float4 BlendColor(const float4& currentColor, const float4& previousColor, float blendFactor);
    bool IsValid(const float2& uv);
    bool IsValidScreen(const float2& uv);
    void Update();
    void CopyToPrevCamera();
    void SetUpSecondZone();
    void Tick(float deltaTime) override;
    float3 ApplyReinhardJodie(const float3& color);
    float GetLuminance(const float3& color);
    void AddPointLight();
    void RemovePointLight();
    void AddAreaLight();
    void RemoveAreaLight();
    void AddSpotLight();
    void RemoveSpotLight();
    void HandleImguiPointLights();
    void HandleImguiAreaLights();
    void HandleImguiSpotLights();
    void HandleImguiDirectionalLight();
    void HandleImguiCamera();
    void MaterialEdit(int index, vector<shared_ptr<Material>>::value_type& material);
    void HandleImguiMaterials();
    void HandleImguiSpheres();
    void HandleImguiTriangles();
    void HandleImguiVoxelVolumes();
    void UI() override;
    void Shutdown() override;
    /* Input
    // input handling
    //void MouseUp(int button) override
    //{
    //	/* implement if you want to detect mouse button presses */
    //}
    void CalculateLightCount();
    size_t lightCount = 0;

    size_t pointCount = 2;
    size_t areaCount = 2;
    size_t spotCount = 2;
    void MouseDown(int button) override;

    float3 SampleSky(const float3& direction) const;
    AlbedoIlluminationData SampleSkyReproject(const float3& direction) const;

    //void MouseMove(int x, int y) override
    //{
    //	mousePos.x = x, mousePos.y = y;
    //}

    //void MouseWheel(float y) override
    //{
    //	/* implement if you want to handle the mouse wheel */
    //}

    //void KeyUp(int key) override
    //{
    //	/* implement if you want to handle keys */
    //}

    //void KeyDown(int key) override
    //{
    //	/* implement if you want to handle keys */
    //}
    //reprojection


    //reprojection
    //Cherno multithreading  https://www.youtube.com/watch?v=46ddlUImiQA
    std::vector<uint32_t> vertIterator;
    // data members
    int2 mousePos;
    int32_t maxBounces = 14;
    float weight = .10f;
    bool staticCamera = false;
    float4* accumulator;


    Camera camera;
    Camera prevCamera;
    float frqGenerationPerlinNoise = .03f;
    float antiAliasingStrength = 1.0f;
    float radiusEmissiveSphere = 1.0f;
    float colorThreshold = .1f;
    //materials
    std::vector<shared_ptr<Material>> metalMaterials; //0 10
    std::vector<shared_ptr<Material>> nonMetalMaterials; //11 20
    std::vector<shared_ptr<Material>> dielectricsMaterials; //21 30
    std::vector<shared_ptr<Material>> smokeMaterials; //31 40
    std::vector<shared_ptr<Material>> emissiveMaterials; //41 50
    //all materials
    std::vector<shared_ptr<Material>> materials;

    //std::vector<shared_ptr<DiffuseMaterial>> reflectiveMaterials;
    //lights
    std::vector<PointLight> pointLights;
    std::vector<SphereAreaLight> areaLights;
    std::vector<SpotLight> spotLights;

    DirectionalLight dirLight;

    uint64_t numRenderedFrames = 0;
    int32_t numCheckShadowsAreaLight = 3;
    // Get a list of .vox files in the assets folder
    std::vector<std::string> voxFiles;
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    //the first one is the player
    std::vector<Scene> voxelVolumes;
    //chunk data
    std::array<ChunkData, CHUNK_COUNT> dataChunks = {10, 14, 9};
    size_t currentChunk = 0;

    bool activateSky = true;

    int matTypeSphere = MaterialType::SMOKE_LOW_DENSITY;
    //BVH
    BasicBVH bvh;


    //skydome
    float HDRLightContribution = 1.0f;
    int skyWidth, skyHeight;
    float* skyPixels;


    //player
    PlayerCharacter player;
    bool inLight = false;
    float triggerCheckpoint = -17.0f;
    //modifying the environment
    std::array<std::unique_ptr<ModifyingProp>, 2> models;
    Timer timer;
    Timer staticCameraTimer;
    float timeToReactivate = 2.0f;

    //reprojection
    std::array<float3, SCRWIDTH * SCRHEIGHT> illuminationBuffer;
    std::array<float3, SCRWIDTH * SCRHEIGHT> albedoBuffer;
    std::array<float3, SCRWIDTH * SCRHEIGHT> illuminationHistoryBuffer;
    std::array<float3, SCRWIDTH * SCRHEIGHT> tempIlluminationBuffer;
    std::array<RayDataReproject, SCRWIDTH * SCRHEIGHT> rayData;
  };
} // namespace Tmpl8
