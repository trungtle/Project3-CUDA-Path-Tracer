#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "camera.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
	glm::vec3 vert0; // For triangle
	glm::vec3 vert1; // For triangle
	glm::vec3 vert2; // For triangle
	glm::vec3 norm0; // For triangle
	glm::vec3 norm1; // For triangle
	glm::vec3 norm2; // For triangle
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
	bool shouldTerminate;
	int time; // For motion blur
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec3 intersect_point;
};

