#include "precomp.h"
#include "PlayerCharacter.h"

#include <iso646.h>

PlayerCharacter::PlayerCharacter(): direction(0, 0, -1), angle(0)
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
  float3 axis = cross(::float3(0, 1, 0), up);
  const float dotProduct = dot(float3(0, 1, 0), up);

  float angleOffset = acos(dotProduct);
  if (up.y < -.90f)
  {
    axis = float3(0, 0, -1);
    angleOffset = PI;
  }

  quat rotateDirectionBasedOnUp;
  rotateDirectionBasedOnUp.fromAxisAngle(axis, angleOffset);


  const float3 rotatedDirection = rotateDirectionBasedOnUp.rotateVector(direction);
  direction = normalize(rotatedDirection);


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


void PlayerCharacter::SetPrevios(Scene& volume)
{
  prevUp = up;
  prevOrigin = origin;
  prevPosition = volume.position;
  prevRotation = rotation;
}

void PlayerCharacter::MovePlayer(Scene& volume, const float3& position, const float3& _up)
{
  up = _up;

  SetRotation();


  origin = position;
  origin += up * .5f;
  //to the right direction

  float3 notUpsideOffset{0};

  //only for the case when the player is on ground
  if (up.y > 0.9f || up.x > 0.9f || up.z > 0.9f)
  {
    //notUpsideOffset = -up;
  }
  else
  {
    notUpsideOffset = up;
  }

  notUpsideOffset -= GetModelOffset(up) * .375f;

  volume.position = position + notUpsideOffset;
  volume.SetTransformPlayer(rotation.toMatrix());


  moving->reseting();
}


void PlayerCharacter::RevertMovePlayer(Scene& volume)
{
  up = prevUp;
  origin = prevOrigin;

  volume.position = prevPosition;
  volume.SetTransformPlayer(prevRotation.toMatrix());


  moving->reseting();
}
