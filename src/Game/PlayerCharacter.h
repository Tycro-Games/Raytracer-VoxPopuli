#pragma once
#include "scene.h"


struct Timer;

class PlayerCharacter
{
public:
  PlayerCharacter();
  Ray GetRay() const;
  float GetDistance() const;
  void SetRotation();
  bool UpdateInput();
  void SetPrevios(Scene& volume);
  void MovePlayer(Scene& volume, const float3& position, const float3& _up);

  void RevertMovePlayer(Scene& volume);

private:
  float3 up = {0, 1, 0};
  float3 direction;
  float3 origin{0};
  float distance{3};
  std::unique_ptr<Timer> movingTimer;
  float timeToMove = .10f;
  quat rotation{0, 0, 0, 0};
  quat prevRotation{0, 0, 0, 0};
  float angle;
  float3 prevOrigin{0};
  float3 prevUp{0};
  float3 prevPosition{0};
};
