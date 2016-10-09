#pragma once
#include "bbox.h"

typedef enum {
	FromChild,
	FromSibling,
	FromParent
} EBVHTransition;

struct BVHNode {
	int geomIdx; 
	Geom bboxGeom; // bbox geom for bounding box
	BBoxVAO bboxVao; // vao details for bounding box
	BBox:: BBox bbox;
	BVHNode* nearChild;
	BVHNode* farChild;
	BVHNode* parent;
	BBox::EAxis splitAxis;
};

struct BVHNodeDev {
	int geomIdx;
	Geom bboxGeom; // bbox geom
	BBoxVAO bboxVao; // vao details for bounding box
	BBox::BBox bbox;
	int idx;
	int nearChildIdx;
	int farChildIdx;
	int parentIdx;
	BBox::EAxis splitAxis;
};

void populateLeafBVHNode(
	BVHNode* node,
	int geomIdx,
	const Geom& geom
	);

bool isLeafBVHNode(const BVHNode* node);

// This comparator is used to sort bvh nodes based on its centroid's maximum extent
struct CompareCentroid {
	CompareCentroid(int dim) : dim(dim) {};

	int dim;
	bool operator()(const BVHNode* node1, const BVHNode* node2) const {
		return node1->bbox.centroid[dim] < node2->bbox.centroid[dim];
	}
};

void flattenBHVTree(std::vector<BVHNodeDev>& bvhNodes, BVHNode* root);
BVHNode* buildBVHTreeRecursive(std::vector<BVHNode*>& leaves, int first, int last);
void destroyBVHTreeRecursive(BVHNode* node);

/**
 * \brief Return the near child idx
 * \param idx 
 * \return 
 */
__host__ __device__
int getNearChildIdx(int idx);

/**
* \brief Return the far child idx
* \param idx
* \return
*/
__host__ __device__
int getFarChildIdx(int idx);