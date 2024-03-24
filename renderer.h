#pragma once

namespace Tmpl8
{
	constexpr size_t POINT_LIGHTS = 2;
	constexpr size_t AREA_LIGHTS = 2;
	constexpr size_t SPOT_LIGHTS = 2;
	constexpr size_t MAX_LIGHT_TYPES = 4;
	//+1 for directional light
	constexpr size_t LIGHT_COUNT = POINT_LIGHTS + SPOT_LIGHTS + AREA_LIGHTS + 1;

	class Renderer : public TheApp
	{
	public:
		void InitMultithreading();
		void SetUpLights();
		float3 PointLightEvaluate(Ray& ray, const PointLightData& lightData);
		float3 SpotLightEvaluate(const Ray& ray, const SpotLightData& lightData) const;
		float3 AreaLightEvaluation(Ray& ray, const SphereAreaLightData& lightData) const;
		bool IsOccluded(Ray& ray) const;
		bool IsOccludedSpheres(Ray& ray) const;
		float3 DirectionalLightEvaluate(Ray& ray, const DirectionalLightData& lightData);
		void ResetAccumulator();
		void MaterialSetUp();
		void AddSphere();
		void RemoveLastSphere();
		void AddTriangle();
		void RemoveTriangle();
		void RemoveVoxelVolume();
		void AddVoxelVolume();
		void ShapesSetUp();
		// game flow methods
		void Init() override;
		void Illumination(Ray& ray, float3& incLight);
		static float3 Reflect(float3 direction, float3 normal);
		static float3 Refract(float3 direction, float3 normal, float IORRatio);
		__m128 FastReciprocal(__m128& x);
		__m128 SlowReciprocal(__m128& dirSSE);
		__m256 SlowReciprocal(__m256& dirSSE);
		void TransformPositionAndDirection_SSE(__m128& oriSSE, __m128& dirSSE, const mat4& invMat, Ray& ray);
		int32_t FindNearest(Ray& ray);
		float3 Trace(Ray& ray, int depth);
		static float SchlickReflectance(float cosine, float indexOfRefraction);
		float3 Absorption(const float3& color, float intensity, float distanceTraveled);
		float SchlickReflectanceNonMetal(const float cosine);
		float CalculateDistanceToPlane(const float3& point, const float3& normal, const float3& pointOnPlane);
		float3 CalculatePlaneNormal(const float3& point1, const float3& point2, const float3& point3);
		float4 SamplePreviousFrameColor(const float2& screenPosition);
		float4 BlendColor(const float4& currentColor, const float4& previousColor, float blendFactor);
		bool IsValid(const float2& uv);
		void Update();
		void CopyToPrevCamera();
		void Tick(float deltaTime) override;
		float3 ApplyReinhardJodie(const float3& color);
		float GetLuminance(const float3& color);
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

		void MouseDown(int button) override;

		float3 SampleSky(const float3& direction) const;

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
		int32_t maxBounces = 5;
		float weight = .10f;
		bool staticCamera = true;
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
		PointLight pointLights[2];
		SphereAreaLight areaLights[2];
		SpotLight spotLights[2];
		SkyDome skyDome;
		DirectionalLight dirLight;

		uint64_t numRenderedFrames = 0;
		int32_t numCheckShadowsAreaLight = 3;
		// Get a list of .vox files in the assets folder
		std::vector<std::string> voxFiles;
		std::vector<Sphere> spheres;
		std::vector<Triangle> triangles;
		std::vector<Scene> voxelVolumes;
		bool activateSky = true;
		int matTypeSphere = MaterialType::SMOKE_LOW_DENSITY;
		//BVH
		BasicBVH bvh;


		//skydome
		float HDRLightContribution = 5.9f;
		int skyWidth, skyHeight;
		float* skyPixels;
	};
} // namespace Tmpl8
