#include "bvh.h"
#include <algorithm>

void populateLeafBVHNode(
	BVHNode* node,
	Geom* geom
	)
{
	node->geom = geom;
	node->bbox = BBoxFromGeom(*geom);
	node->nearChild = nullptr;
	node->farChild = nullptr;
	node->parent = nullptr;
}

void populateInteriorBVHNode(
	BVHNode* node,
	Geom* geom,
	BBox& bbox,
	BVHNode* nearChild,
	BVHNode* farChild,
	BVHNode* parent,
	EAxis splitAxis
	)
{
	node->geom = geom;
	node->bbox = bbox;
	node->nearChild = nearChild;
	node->farChild = farChild;
	node->parent = parent;
	node->splitAxis = splitAxis;
}

bool isLeafBVHNode(const BVHNode* node) 
{
	return node->nearChild == nullptr && node->farChild == nullptr;
}

BVHNode* buildBVHTreeRecursive(std::vector<BVHNode*>& leaves, int first, int last) {
	if (last < first || last < 0 || first < 0) {
		return nullptr;
	}

	if (last == first) {
		// We're at a leaf node
		return leaves.at(first);
	}

	auto node = new BVHNode();

	// Choose a dimension to split
	auto dim = static_cast<int>((BBoxMaximumExtent(node->bbox)));

	// Compute the bounds of all geometries within this subtree
	for (auto i = first; i < last; ++i) {
		node->bbox = BBoxUnion(leaves.at(i)->bbox, node->bbox);
	}
		 
	// Partial sorting along the maximum extent and split at the middle
	int mid = (first + last) / 2;
	std::nth_element(&leaves[first], &leaves[mid], &leaves[last], CompareCentroid(dim));
	
	// Build near child
	node->nearChild = buildBVHTreeRecursive(leaves, first, mid);
	node->nearChild->parent = node;
	node->nearChild->splitAxis = static_cast<EAxis>(dim);

	// Build far child
	node->farChild = buildBVHTreeRecursive(leaves, mid + 1, last);
	node->farChild->parent = node;
	node->farChild->splitAxis = static_cast<EAxis>(dim);
	return node;

}