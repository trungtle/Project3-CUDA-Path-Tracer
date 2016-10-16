#pragma once
#include "src/sceneStructs.h"

namespace StreamCompaction {
namespace UnoptimizedEfficient {
	void scan(int n, int *odata, const int *idata, float* timeElapsedMs);

	int compact(int n, PathSegment *odata, const PathSegment *idata);
}

namespace OptimizedEfficient {
	int compact(int n, PathSegment *odata, const PathSegment *idata);
}
}
