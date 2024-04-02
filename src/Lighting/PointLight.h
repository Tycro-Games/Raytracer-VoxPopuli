#pragma once

struct PointLightData
{
  float3 position{0};
  float3 color{1};
};

//based on 
struct PointLight
{
  //float3 Evaluate(Ray& ray, Scene& scene) override;
  PointLightData data{{0.5f, 0.5f, .50f}, {1}};
};
