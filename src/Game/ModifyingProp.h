#pragma once

class ModifyingProp
{
public:
  ModifyingProp(Scene& scene, float time, uint32_t startingIndex, uint32_t increasingRate);
  void Update(float deltaTime);
  bool GetUpdate();

private:
  std::unique_ptr<Timer> timer;
  float toChange;
  Scene& voxelVolume;
  uint32_t increaseRate = 0;
  uint32_t index = 13;
  const uint32_t maxSize = 63;
};
