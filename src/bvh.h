#pragma once
#include "bbox.h"

typedef enum {
	FromChild,
	FromSibling,
	FromParent
} EBVHTransition;

struct BVHNode {
	Geom* geom; // When this isn't a leaf node, geom contains the bbox shape, which is a cube
	Geom bboxGeom; // vao details for bounding box
	BBoxVAO bboxVao; // vao details for bounding box
	BBox:: BBox bbox;
	BVHNode* nearChild;
	BVHNode* farChild;
	BVHNode* parent;
	BBox::EAxis splitAxis;
};

void populateLeafBVHNode(
	BVHNode* node,
	Geom* geom
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

BVHNode* buildBVHTreeRecursive(std::vector<BVHNode*>& leaves, int first, int last);
void destroyBVHTreeRecursive(BVHNode* node);