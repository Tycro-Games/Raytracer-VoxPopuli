#pragma once
//based on chapter 4 of https://www.routledge.com/Fundamentals-of-Computer-Graphics/Marschner-Shirley/p/book/9780367505035


namespace Tmpl8
{
	class Ray;
	class Scene;
}

class Light
{
public:
	virtual ~Light();
	virtual float3 Evaluate(Ray& ray, Scene& scene) = 0;

protected:
};
