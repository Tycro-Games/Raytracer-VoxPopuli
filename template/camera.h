#pragma once

// default screen resolution
#define SCRWIDTH	1024
#define SCRHEIGHT	640
// #define FULLSCREEN
// #define DOUBLESIZE

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

		Ray GetPrimaryRay(const float x, const float y)
		{
			//conceptually used https://youtu.be/Qz0KTGYJtUk?si=9en1nLsgxqQyoGW2&t=2113
			const float u = x * (1.0f / SCRWIDTH);
			const float v = y * (1.0f / SCRHEIGHT);
			const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

			const float jitterX = defocusJitter.x * (RandomFloat() - 0.5f);
			const float jitterY = defocusJitter.y * (RandomFloat() - 0.5f);

			const float3 focalPoint = camPos + focalDistance * normalize(P - camPos);
			const float3 rayOrigin = camPos + float3{jitterX * topRight.x, jitterY * topRight.y, 0.0f};

			const float3 rayDirection = (focalPoint - rayOrigin);

			// Return the primary ray
			return {rayOrigin, rayDirection};
		}


		bool HandleInput(const float t)
		{
			if (!WindowHasFocus()) return false;
			const float speed = 0.0025f * t;
			ahead = normalize(camTarget - camPos);
			const float3 tmpUp(0, 1, 0);
			float3 right = normalize(cross(tmpUp, ahead));
			float3 up = normalize(cross(ahead, right));
			bool changed = false;
			if (IsKeyDown(GLFW_KEY_A)) camPos -= speed * 2 * right, changed = true;
			if (IsKeyDown(GLFW_KEY_D)) camPos += speed * 2 * right, changed = true;
			if (IsKeyDown(GLFW_KEY_W)) camPos += speed * 2 * ahead, changed = true;
			if (IsKeyDown(GLFW_KEY_S)) camPos -= speed * 2 * ahead, changed = true;
			if (IsKeyDown(GLFW_KEY_Q)) camPos += speed * 2 * up, changed = true;
			if (IsKeyDown(GLFW_KEY_E)) camPos -= speed * 2 * up, changed = true;
			camTarget = camPos + ahead;
			if (IsKeyDown(GLFW_KEY_UP))
			{
				if (ahead.y < stopAngle)
				{
					camTarget += speed * up;
					ahead = normalize(camTarget - camPos);
					changed = true;
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
			}


			if (IsKeyDown(GLFW_KEY_LEFT)) camTarget -= speed * right, changed = true;
			if (IsKeyDown(GLFW_KEY_RIGHT)) camTarget += speed * right, changed = true;

			if (!changed) return false;
			ahead = normalize(camTarget - camPos);

			up = normalize(cross(ahead, right));
			right = normalize(cross(up, ahead));
			topLeft = camPos + 2 * ahead - aspect * right + up;
			topRight = camPos + 2 * ahead + aspect * right + up;
			bottomLeft = camPos + 2 * ahead - aspect * right - up;
			return true;
		}

		static constexpr float aspect = static_cast<float>(SCRWIDTH) / static_cast<float>(SCRHEIGHT);
		float3 ahead;
		float3 camPos, camTarget;
		float3 topLeft, topRight, bottomLeft;
		const float stopAngle = 0.8f;
		float focalDistance{0};
		float2 defocusJitter{0, 0};
	};
}
