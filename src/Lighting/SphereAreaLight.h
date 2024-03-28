#pragma once
struct SphereAreaLightData
{
  float3 position{0};
  float3 color{0};
  float colorMultiplier{1};

  float radius = 1.0f;
};

struct SphereAreaLight
{
  SphereAreaLightData data{{-0.f, 0.5f, -3.50f}, {1.f}, 1.2f, 0.2f};
};
