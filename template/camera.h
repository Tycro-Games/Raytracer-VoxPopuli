#pragma once

// default screen resolution
#define SCRWIDTH	256
#define SCRHEIGHT	212
// #define FULLSCREEN
#define DOUBLESIZE

namespace Tmpl8
{
	class Camera
	{
	public:
		Camera()
		{
			// setup a basic view frustum
			camPos = float3(0, 0, -2);
			camTarget = float3(0, 0, -1);
			topLeft = float3(-aspect, 1, 0);
			topRight = float3(aspect, 1, 0);
			bottomLeft = float3(-aspect, -1, 0);
			ahead = normalize(camTarget - camPos);
		}

		//From Lynn
		float3 leftNormal;
		float3 rightNormal;
		float3 topNormal;
		float3 bottomNormal;

		float2 PointToUV(const float3& point) const
		{
			const float3 delta{point - camPos};

			// Get the distance between each frustum and the point
			const float leftDistance{dot(leftNormal, delta)};
			const float rightDistance{dot(rightNormal, delta)};
			const float topDistance{dot(topNormal, delta)};
			const float bottomDistance{dot(bottomNormal, delta)};

			// Get the uv-coordinate of where on the screen the point would be visible from
			const float u{leftDistance / (leftDistance + rightDistance)};
			const float v{topDistance / (topDistance + bottomDistance)};

			return {u, v};
		}


		//From Lynn
		void SetFrustumNormals()
		{
			// View pyramid directions
			const float3 leftFrustumDirection{(2 * ahead) - (aspect * right)};
			const float3 rightFrustumDirection{(2 * ahead) + (aspect * right)};
			const float3 topFrustumDirection{(2 * ahead) + up};
			const float3 bottomFrustumDirection{(2 * ahead) - up};

			// Left handed coordinate system
			leftNormal = cross(up, leftFrustumDirection);
			rightNormal = cross(rightFrustumDirection, up);
			topNormal = cross(right, topFrustumDirection);
			bottomNormal = cross(bottomFrustumDirection, right);
		}

		void GetPrimaryRay(const float x, const float y, float3& rayOrigin, float3& rayDirection, const float2& p) const
		{
			//conceptually used https://youtu.be/Qz0KTGYJtUk?si=9en1nLsgxqQyoGW2&t=2113
			const float u = x * (1.0f / SCRWIDTH);
			const float v = y * (1.0f / SCRHEIGHT);
			const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

			const float2 jitter = p * defocusJitter / SCRWIDTH;
			//Remi fixed this implementation by using the correct focal point

			//focalDistance = (focalTargetDistance + focalDistance) * .5f;
			const float3 focalPoint = camPos + focalDistance * normalize(P - camPos);
			rayOrigin = camPos + jitter.x * right + jitter.y * up;

			rayDirection = (focalPoint - rayOrigin);
		}

		Ray GetPrimaryRay(const float x, const float y) const
		{
			//conceptually used https://youtu.be/Qz0KTGYJtUk?si=9en1nLsgxqQyoGW2&t=2113
			const float u = x * (1.0f / SCRWIDTH);
			const float v = y * (1.0f / SCRHEIGHT);
			const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

			const float2 jitter = RandomPointInCircle() * defocusJitter / SCRWIDTH;
			//Remi fixed this implementation by using the correct focal point

			//focalDistance = (focalTargetDistance + focalDistance) * .5f;
			const float3 focalPoint = camPos + focalDistance * normalize(P - camPos);
			const float3 rayOrigin = camPos + jitter.x * right + jitter.y * up;

			const float3 rayDirection = (focalPoint - rayOrigin);
			return {rayOrigin, rayDirection};
		}


		bool HandleInput(const float t)
		{
			if (!WindowHasFocus()) return false;
			float speed = 0.0025f * t;
			if (IsKeyDown(GLFW_KEY_LEFT_SHIFT))
				speed /= 2;
			if (IsKeyDown(GLFW_KEY_LEFT_ALT)) speed = 0.0f;

			ahead = normalize(camTarget - camPos);
			const float3 tmpUp(0, 1, 0);
			right = normalize(cross(tmpUp, ahead));
			up = normalize(cross(ahead, right));
			bool changed = false;
			if (IsKeyDown(GLFW_KEY_UP))

			{
				if (ahead.y < stopAngle)

				{
					camTarget += speed * up;

					ahead = normalize(camTarget - camPos);

					changed = true;
				}
				else
				{
					ahead.y = stopAngle;
				}
			}

			if (IsKeyDown(GLFW_KEY_DOWN))

			{
				if (ahead.y > -stopAngle)

				{
					camTarget -= speed * up;

					ahead = normalize(camTarget - camPos);

					changed = true;
				}
				else
				{
					ahead.y = -stopAngle;
				}
			}
			if (IsKeyDown(GLFW_KEY_LEFT)) camTarget -= speed * right, changed = true;
			if (IsKeyDown(GLFW_KEY_RIGHT)) camTarget += speed * right, changed = true;
			ahead = normalize(camTarget - camPos);
			right = normalize(cross(tmpUp, ahead));
			up = normalize(cross(ahead, right));
			if (IsKeyDown(GLFW_KEY_A)) camPos -= speed * right, changed = true;
			if (IsKeyDown(GLFW_KEY_D)) camPos += speed * right, changed = true;
			if (GetAsyncKeyState('W')) camPos += speed * ahead, changed = true;
			if (IsKeyDown(GLFW_KEY_S)) camPos -= speed * ahead, changed = true;
			if (IsKeyDown(GLFW_KEY_Q)) camPos += speed * up, changed = true;
			if (IsKeyDown(GLFW_KEY_E)) camPos -= speed * up, changed = true;
			camTarget = camPos + ahead;
			ahead = normalize(camTarget - camPos);
			up = normalize(cross(ahead, right));
			right = normalize(cross(up, ahead));
			topLeft = camPos + 2 * ahead - aspect * right + up;
			topRight = camPos + 2 * ahead + aspect * right + up;
			bottomLeft = camPos + 2 * ahead - aspect * right - up;
			if (!changed) return false;
			return true;
		}

		static constexpr float aspect = static_cast<float>(SCRWIDTH) / static_cast<float>(SCRHEIGHT);
		float3 ahead, right, up;
		float3 camPos, camTarget;
		float3 topLeft, topRight, bottomLeft;

		const float stopAngle = 0.9f;
		float focalDistance{1.0f};
		float focalTargetDistance{1.0f};
		float defocusJitter{0};
	};
}
