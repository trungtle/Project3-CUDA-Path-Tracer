#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "bvh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	int loadSceneFromObj(
		const char* filename,
		const char* basepath = nullptr,
		bool triangulate = true
		);
	int destroyBVH();
public:
    Scene(string filename);
    ~Scene();
	int initBVH();

	BVHNode* root;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
