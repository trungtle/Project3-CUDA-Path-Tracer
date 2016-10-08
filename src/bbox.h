#pragma once

#include "sceneStructs.h"
#include "glm/glm.hpp"

typedef enum {
	X,
	Y,
	Z
} EAxis;


glm::vec3 Centroid(
	const glm::vec3& a,
	const glm::vec3& b
	);

struct BBox {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 centroid;
};

/**
 * \brief Generate a union bounding box from a and b
 * \param a : first bounding box
 * \param b : second bounding box
 * \return the union bounding box
 */
BBox BBoxUnion(const BBox& a, const BBox& b);


/**
 * \brief Generate a bounding box from a given geometry (cube or sphere)
 * \param geom 
 * \return 
 */
BBox BBoxFromGeom(const Geom& geom);


/**
 * \brief Return the maximum extent axis of the bounding box. This is used to sort the BVH tree
 * \param bbox 
 * \return 
 */
EAxis BBoxMaximumExtent(const BBox& bbox);

void initBBoxVAO();