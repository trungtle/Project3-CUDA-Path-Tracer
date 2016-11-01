#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

__global__ void naiveScan(int n, int offset, int* odata, const int *idata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  if (tid >= offset) {
    odata[tid] = idata[tid - offset] + idata[tid] ;
  } else {
    odata[tid] = idata[tid];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata, float* timeElapsedMs) {
    // TODO

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int* dev_odata1;
  int* dev_odata2;

  cudaMalloc((void**)&dev_odata1, n * sizeof(int));
  cudaMalloc((void**)&dev_odata2, n * sizeof(int));

  cudaMemcpy(dev_odata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_odata2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  
  cudaEventRecord(start);
  int height = ilog2ceil(n);
  for (int level = 1; level <= height; ++level) {
    int offset = 1 << (level - 1);
    naiveScan << <BLOCK_COUNT(n), BLOCK_SIZE >> >(
      n, 
      offset, 
      (level % 2) == 0 ? dev_odata1 : dev_odata2, 
      (level % 2) == 0 ? dev_odata2 : dev_odata1
        );
  }

  if (height % 2 == 0) {
    Common::inclusiveToExclusiveScanResult << <BLOCK_COUNT(n), BLOCK_SIZE >> >(n, dev_odata2, dev_odata1);
	cudaEventRecord(stop);
    cudaMemcpy(odata, dev_odata2, n * sizeof(int), cudaMemcpyDeviceToHost);
  } else {
    Common::inclusiveToExclusiveScanResult << <BLOCK_COUNT(n), BLOCK_SIZE >> >(n, dev_odata1, dev_odata2);
	cudaEventRecord(stop);
	cudaMemcpy(odata, dev_odata1, n * sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  *timeElapsedMs = milliseconds;

  cudaFree(dev_odata1);
  cudaFree(dev_odata2);
}

}
}
