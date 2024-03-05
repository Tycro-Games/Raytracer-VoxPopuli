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
		float3 PointLightEvaluate(Ray& ray, Scene& scene, const PointLightData& lightData);
		float3 SpotLightEvaluate(const Ray& ray, const Scene& scene, const SpotLightData& lightData);
		float3 AreaLightEvaluation(Ray& ray, Scene& scene, const SphereAreaLightData& lightData) const;
		bool IsOccluded(Ray& ray) const;
		static float3 DirectionalLightEvaluate(Ray& ray, Scene& scene, const DirectionalLightData& lightData);
		void ResetAccumulator();
		void MaterialSetUp();
		void AddSphere();
		void RemoveLastSphere();
		void AddTriangle();
		void RemoveTriangle();
		void ShapesSetUp();
		// game flow methods
		void Init() override;
		void Illumination(Ray& ray, float3& incLight);
		static float3 Reflect(float3 direction, float3 normal);
		static float3 Refract(float3 direction, float3 normal, float IORRatio);
		float3 Trace(Ray& ray, int depth);
		static float SchlickReflectance(float cosine, float indexOfRefraction);
		float SchlickReflectanceNonMetal(const float cosine);
		void Tick(float deltaTime) override;
		float3 ApplyReinhardJodie(const float3& color);
		float GetLuminance(const float3& color);
		void HandleImguiPointLights();
		void HandleImguiAreaLights();
		void HandleImguiSpotLights();
		void HandleImguiDirectionalLight();
		void HandleImguiCamera();
		void MaterialEdit(int index, vector<shared_ptr<ReflectivityMaterial>>::value_type& material);
		void HandleImguiMaterials();
		void HandleImguiSpheres();
		void HandleImguiTriangles();
		void UI() override;
		void Shutdown() override;
		/* Input
		// input handling
		//void MouseUp(int button) override
		//{
		//	/* implement if you want to detect mouse button presses */
		//}

		//void MouseDown(int button) override
		//{
		//	/* implement if you want to detect mouse button presses */
		//}

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

		//Cherno multithreading  https://www.youtube.com/watch?v=46ddlUImiQA
		std::vector<uint32_t> vertIterator;
		// data members
		int2 mousePos;
		int32_t maxBounces = 5;
		int32_t maxRayPerPixel = 1;

		float4* accumulator;
		Scene mainScene;
		Camera camera;
		float frqGenerationPerlinNoise = .03f;
		float HDRLightContribution = 1.5f;
		float antiAliasingStrength = 1.0f;
		//materials
		std::vector<shared_ptr<ReflectivityMaterial>> metalMaterials;
		std::vector<shared_ptr<ReflectivityMaterial>> nonMetalMaterials;
		std::vector<shared_ptr<ReflectivityMaterial>> dielectricsMaterials;
		std::vector<shared_ptr<ReflectivityMaterial>> emissiveMaterials;

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
		int matTypeSphere = MaterialType::GLASS;

		//BVH
		BasicBVH bvh;
	};
} // namespace Tmpl8
