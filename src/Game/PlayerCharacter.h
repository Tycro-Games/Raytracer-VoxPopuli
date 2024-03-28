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
	void SetRotation(float angle);
	bool UpdateInput();
	void MovePlayer(Scene& volume, const float3& position, const float3& _up);

private:
	float3 up = {0, 1, 0};
	float3 direction;
	float3 origin{0};
	float distance{3};
	std::unique_ptr<Timer> moving;
	float timeToMove = .20f;
	quat rotation{0, 0, 0, 0};
	float angle;
};
