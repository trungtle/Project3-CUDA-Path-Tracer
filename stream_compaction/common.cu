#include "common.h"
#include "src/sceneStructs.h"



namespace StreamCompaction {
namespace Common {

  /**
   * Convert an inclusice scan result to an exclusive scan result
   *
   */
__global__ void inclusiveToExclusiveScanResult(int n, int* odata, const int* idata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }

  if (tid == 0) {
    odata[0] = 0;
    return;
  }

  odata[tid] = idata[tid - 1];
}





}
}
