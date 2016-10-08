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
	int initBVH();
public:
    Scene(string filename);
    ~Scene();

	BVHNode* root;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
