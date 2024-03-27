#include "precomp.h"
#include "PlayerCharacter.h"

PlayerCharacter::PlayerCharacter()
{
}

Ray PlayerCharacter::GetRay() const
{
	const float3 dir = normalize(direction - up);
	const float3 rD = float3(1 / dir.x, 1 / dir.y, 1 / dir.z);

	const float3 Dsign = Ray::ComputeDsign(dir);
	return Ray{origin, dir, rD, Dsign, distance};
}

float PlayerCharacter::GetDistance() const
{
	return distance;
}

bool PlayerCharacter::UpdateInput()
{
	bool IsMoving = false;
	if (IsKeyDown(GLFW_KEY_W))
	{
		direction = {EPSILON, EPSILON, -1};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_D))
	{
		direction = {-1, EPSILON, EPSILON};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_S))
	{
		direction = {EPSILON, EPSILON, 1};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_A))
	{
		direction = {1, EPSILON, EPSILON};
		IsMoving = true;
	}
	return IsMoving;
	//also move 
}

void PlayerCharacter::MovePlayer(Scene& volume, const float3& position)
{
	volume.position = (position);
	const float3 rot = {0};

	volume.SetTransformNoPivot(rot);
	origin = volume.position;
	origin += up * .5f;
}
