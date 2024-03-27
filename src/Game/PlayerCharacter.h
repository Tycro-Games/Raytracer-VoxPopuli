#pragma once
#include "scene.h"

class PlayerCharacter
{
public:
	PlayerCharacter();
	Ray GetRay() const;
	float GetDistance() const;
	bool UpdateInput();
	void MovePlayer(Scene& volume, const float3& position);

private:
	float3 up = {0, 1, 0};
	float3 direction;
	float3 origin{0};
	float distance{3};
};
