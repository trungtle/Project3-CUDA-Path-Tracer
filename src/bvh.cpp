#include "bvh.h"
#include <algorithm>

void populateLeafBVHNode(
	BVHNode* node,
	int geomIdx,
	const Geom& geom
	)
{
	node->geomIdx = geomIdx;
	node->bbox = BBox::BBoxFromGeom(geom);
	BBox::createBBoxGeom(node->bbox, node->bboxGeom);
	BBox::createBBoxVAO(node->bbox, &node->bboxVao);
	node->nearChild = nullptr;
	node->farChild = nullptr;
	node->parent = nullptr;
	node->splitAxis = BBox::EAxis::X;
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
	for (auto i = first; i <= last; ++i) {
		node->bbox = BBoxUnion(leaves.at(i)->bbox, node->bbox);
	}
	node->geomIdx = -1;
	createBBoxGeom(node->bbox, node->bboxGeom);
	BBox::createBBoxVAO(node->bbox, &node->bboxVao);

	// Partial sorting along the maximum extent and split at the middle
	int mid = (first + last) / 2;
	std::nth_element(&leaves[first], &leaves[mid], &leaves[last], CompareCentroid(dim));
	
	// Build near child
	node->nearChild = buildBVHTreeRecursive(leaves, first, mid);
	node->nearChild->parent = node;
	node->nearChild->splitAxis = static_cast<BBox::EAxis>(dim);

	// Build far child
	node->farChild = buildBVHTreeRecursive(leaves, mid + 1, last);
	node->farChild->parent = node;
	node->farChild->splitAxis = static_cast<BBox::EAxis>(dim);
	return node;

}


void destroyBVHTreeRecursive(BVHNode* node) {
	
	if (isLeafBVHNode(node)) {
		delete node;
		node = nullptr;
	}

	destroyBVHTreeRecursive(node->nearChild);
	destroyBVHTreeRecursive(node->farChild);
	delete node;
	node == nullptr;
}

void flattenBHVTreeRecursive(
	std::vector<BVHNodeDev>& bvhNodes,
	BVHNode* node,
	int idx,
	int parentIdx) 
{
	if (node == nullptr) {
		return;
	}

	BVHNodeDev bvhNodeDev;
	bvhNodeDev.splitAxis = node->splitAxis;
	bvhNodeDev.geomIdx = node->geomIdx;
	bvhNodeDev.bboxVao = node->bboxVao;
	bvhNodeDev.bboxGeom = node->bboxGeom;
	bvhNodeDev.bbox = node->bbox;
	bvhNodeDev.idx = idx;
	bvhNodeDev.nearChildIdx = idx * 2 + 1;
	bvhNodeDev.farChildIdx = idx * 2 + 2;
	bvhNodeDev.parentIdx = parentIdx;
	bvhNodes.push_back(bvhNodeDev);

	// Update pointers
	parentIdx = idx;
	flattenBHVTreeRecursive(bvhNodes, node->nearChild, getNearChildIdx(idx), parentIdx);
	flattenBHVTreeRecursive(bvhNodes, node->farChild, getFarChildIdx(idx), parentIdx);
}

void flattenBHVTree(std::vector<BVHNodeDev>& bvhNodes, BVHNode* root) {
	
	BVHNode* current = root;
	int idx = 0;
	int parentIdx = -1;
	flattenBHVTreeRecursive(bvhNodes, root, idx, parentIdx);
}

__host__ __device__
int getNearChildIdx(int idx) {
	return idx * 2 + 1;
}

__host__ __device__
int getFarChildIdx(int idx) {
	return idx * 2 + 2;
}


