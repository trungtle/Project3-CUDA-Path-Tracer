#include "bvh.h"
#include <algorithm>

void populateLeafBVHNode(
	BVHNode* node,
	int geomIdx,
	const Geom& geom
	)
{
	node->nodeIdx = -1;
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

BVHNode* buildBVHTreeRecursive(std::vector<BVHNode*>& leaves, int first, int last, size_t& nodeCount) {
	if (last < first || last < 0 || first < 0) {
		return nullptr;
	}

	if (last == first) {
		// We're at a leaf node
		return leaves.at(first);
	}

	auto node = new BVHNode();
	node->nodeIdx = nodeCount;
	++nodeCount;

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
	node->nearChild = buildBVHTreeRecursive(leaves, first, mid, nodeCount);
	node->nearChild->parent = node;
	node->nearChild->splitAxis = static_cast<BBox::EAxis>(dim);

	// Build far child
	node->farChild = buildBVHTreeRecursive(leaves, mid + 1, last, nodeCount);
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
	const BVHNode* node,
	int idx,
	int parentIdx) 
{
	if (node == nullptr) {
		return;
	}

	if (idx >= bvhNodes.size()) {
		printf("idx is wrong %d\n", idx);
	}

	BVHNodeDev bvhNodeDev;
	bvhNodeDev.splitAxis = node->splitAxis;
	bvhNodeDev.geomIdx = node->geomIdx;
	bvhNodeDev.bboxVao = node->bboxVao;
	bvhNodeDev.bboxGeom = node->bboxGeom;
	bvhNodeDev.bbox = node->bbox;
	bvhNodeDev.idx = idx;
	if (node->nearChild) {
		bvhNodeDev.nearChildIdx = node->nearChild->nodeIdx;
	} else {
		bvhNodeDev.nearChildIdx = -1;
	}
	if (node->farChild) {
		bvhNodeDev.farChildIdx = node->farChild->nodeIdx;
	} else {
		bvhNodeDev.farChildIdx = -1;
	}
	bvhNodeDev.parentIdx = parentIdx;
	assert(idx < bvhNodes.size());
	bvhNodes[idx] = bvhNodeDev;

	// Update pointers
	if (node->nearChild) {
		flattenBHVTreeRecursive(bvhNodes, node->nearChild, node->nearChild->nodeIdx, node->nodeIdx);
	}

	if (node->farChild) {
		flattenBHVTreeRecursive(bvhNodes, node->farChild, node->farChild->nodeIdx, node->nodeIdx);
	}
}

void flattenBHVTree(std::vector<BVHNodeDev>& bvhNodes, const BVHNode* root, const size_t nodeCount) {
	
	bvhNodes.resize(nodeCount);
	int parentIdx = -1;
	flattenBHVTreeRecursive(bvhNodes, root, root->nodeIdx, parentIdx);
}

