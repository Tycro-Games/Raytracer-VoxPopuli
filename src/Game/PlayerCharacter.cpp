#include "precomp.h"
#include "PlayerCharacter.h"

#include <iso646.h>

PlayerCharacter::PlayerCharacter(): direction()
{
	moving = make_unique<Timer>();
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

void PlayerCharacter::SetRotation()
{
	const float3 axis = ::cross(::float3(0, 1, 0), up);
	const float dotProduct = dot(float3(0, 1, 0), up);
	const float angleOffset = ::acos(dotProduct);
	quat rotateDirectionBasedOnUp;
	rotateDirectionBasedOnUp.fromAxisAngle(axis, angleOffset);

	// Apply rotation
	const float3 rotatedDirection = rotateDirectionBasedOnUp.rotateVector(direction);
	direction = normalize(rotatedDirection);

	//rotation.fromAxisAngle(up, angle * DEG2RAD);
	// Compute rotation for the character facing direction
	quat rotateCharacterFacingDirection;
	rotateCharacterFacingDirection.fromAxisAngle(up, angle * DEG2RAD);

	rotation = rotateCharacterFacingDirection * rotateDirectionBasedOnUp;

	rotation.normalize();
}

bool PlayerCharacter::UpdateInput()
{
	if (moving->elapsed() < timeToMove)
		return false;

	bool IsMoving = false;
	angle = 0.0f;
	if (IsKeyDown(GLFW_KEY_W))
	{
		direction = {EPSILON, EPSILON, -1};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_D))
	{
		angle = 90.0f;
		direction = {-1, EPSILON, EPSILON};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_S))
	{
		angle = 180.0f;
		direction = {EPSILON, EPSILON, 1};
		IsMoving = true;
	}
	else if (IsKeyDown(GLFW_KEY_A))
	{
		angle = 270.0f;
		direction = {1, EPSILON, EPSILON};
		IsMoving = true;
	}
	if (IsMoving)
	{
		SetRotation();
	}
	return IsMoving;
	//also move 
}

float3 GetModelOffset(const float3& normal)
{
	int index = -1;
	for (int i = 0; i < 3; i++)
	{
		if (static_cast<int>(normal.cell[i]) != 0)
		{
			index = i;
		}
	}
	float3 result{0};
	for (int i = 0; i < 3; i++)
	{
		if (i != index)
		{
			result.cell[i] = normal.cell[index];
		}
	}
	if (normal.cell[index] < 0)
		result *= -1;
	return result;
}

void PlayerCharacter::MovePlayer(Scene& volume, const float3& position, const float3& _up)
{
	up = _up;
	SetRotation();
	origin = position;
	origin += up * .5f;
	//to the right direction

	float3 notUpsideOffset{0};


	if (up.y < 0.8f)
	{
		notUpsideOffset = up;
	}

	notUpsideOffset -= GetModelOffset(up) * .375f;


	volume.position = position + notUpsideOffset;
	volume.SetTransformPlayer(rotation.toMatrix());


	moving->reseting();
}
