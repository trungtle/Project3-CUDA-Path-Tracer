#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
		intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
		normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
		return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

/**
* Test intersection between a ray and a transformed bounding box. Untransformed,
* the bounding box ranges from -0.5 to 0.5 in each axis and is centered at the origin.
* @todo: can rewrite this to only use min/max and not even transformation matrix
* 
* @return true if the ray overlaps this bounding box
*/
__host__ __device__ bool bboxIntersectionTest(Geom box, Ray r) {
	Ray q;
	q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

	bool hit = false;
	float tmin = -1e38f;
	float tmax = 1e38f;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
			float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
			float ta = glm::min(t1, t2);
			float tb = glm::max(t1, t2);
			glm::vec3 n;
			n[xyz] = t2 < t1 ? +1 : -1;
			if (ta > 0 && ta > tmin) {
				tmin = ta;
			}
			if (tb < tmax) {
				tmax = tb;
			}
		}
	}

	if (tmax >= tmin && tmax > 0) {
		hit = true;
	}
	return hit;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(const Geom& tri, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {

	glm::vec3 ro = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

	// Compute fast intersection using Muller and Trumbore, this skips computing the plane's equation.
	// See https://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

	float t = -1;

	// Find the edges that share vertice 0
	glm::vec3 edge1 = tri.vert1 - tri.vert0;
	glm::vec3 edge2 = tri.vert2 - tri.vert0;

	// Being computing determinante. Store pvec for recomputation
	glm::vec3 pvec = glm::cross(rt.direction, edge2);
	// If determinant is 0, ray lies in plane of triangle
	float det = glm::dot(pvec, edge1);
	if (det < 0.000001f && det > - 0.000001f) {
		outside = true;
		return -1;
	}
	float inv_det = 1.0f / det;
	glm::vec3 tvec = rt.origin - tri.vert0;

	// u, v are the barycentric coordinates of the intersection point in the triangle
	// t is the distance between the ray's origin and the point of intersection
	float u, v;

	// Compute u
	u = glm::dot(pvec, tvec) * inv_det;
	if (u < 0.0f || u > 1) {
		outside = true;
		return -1;
	}

	// Compute v
	glm::vec3 qvec = glm::cross(tvec, edge1);
	v = glm::dot(rt.direction, qvec) * inv_det;
	if (v < 0.0f || (u + v) > 1.0f) {
		outside = true;
		return -1;
	}

	// Compute t
	t = glm::dot(edge2, qvec) * inv_det;

	outside = false;
	glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

	intersectionPoint = multiplyMV(tri.transform, glm::vec4(objspaceIntersection, 1.f));

	// Interpolate the normal
	glm::vec3 objspaceNormal = glm::normalize(tri.norm0 * (1 - u - v) + tri.norm1 * u + tri.norm2 * v);
	normal = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(objspaceNormal, 0.f)));

	return t;

}