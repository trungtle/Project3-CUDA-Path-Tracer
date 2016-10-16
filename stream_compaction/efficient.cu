#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define OPTIMIZATION

/**
* Maps an array to an array of 0s and 1s for stream compaction. Elements
* which map to 0 will be removed, and elements which map to 1 will be kept.
*/
__global__ void kernMapToBoolean(int n, int *bools, const PathSegment *idata) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	bools[tid] = idata[tid].remainingBounces > 0;
}

/**
* Performs scatter on an array. That is, for each element in idata,
* if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
*/
__global__ void kernScatter(
	int n,
	PathSegment *odata,
	const PathSegment *idata,
	const int *bools,
	const int *indices
	)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= n) {
		return;
	}

	if (bools[tid] == 1) {
		odata[indices[tid]] = idata[tid];
	}
}

namespace StreamCompaction {

// Should only be launched with 1 thread?
__global__ void kernRemainingElementsCountForCompact(
	const int n,
	int* dev_indices,
	const int* dev_bools,
	int* remainingElementsCount
	)
{
	*remainingElementsCount = dev_bools[n - 1] + dev_indices[n - 1];
}

namespace UnoptimizedEfficient {

// TODO: __global__

__global__ void upsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }

  int twoToLevel = 1 << level;
  int twoToLevelPlusOne = 1 << (level + 1);
  if (tid % twoToLevelPlusOne == 0) {
    odata[tid + twoToLevelPlusOne - 1] += odata[tid + twoToLevel - 1];
  }
}

__global__ void downsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  int twoToLevel = 1 << level;
  int twoToLevelPlusOne = 1 << (level + 1);

  if (tid % twoToLevelPlusOne == 0) {
    int t = odata[tid + twoToLevel - 1];
    odata[tid + twoToLevel - 1] = odata[tid + twoToLevelPlusOne - 1];
    odata[tid + twoToLevelPlusOne - 1] += t;
  }
}


void deviceScan(int n, int* dev_odata) {

	int height = ilog2ceil(n);
	int ceilPower2 = 1 << height;

	for (int level = 0; level < height; ++level) {
		upsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
		cudaThreadSynchronize();
	}

	// Set the root to zero
	cudaMemset(dev_odata + (ceilPower2 - 1), 0, sizeof(int));

	// Downsweep
	for (int level = height - 1; level >= 0; --level) {
		downsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
		cudaThreadSynchronize();
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata, float* timeElapsedMs) {
    // TODO
  int* dev_odata;
  int height = ilog2ceil(n);
  int ceilPower2 = 1 << height;
  cudaMalloc((void**)&dev_odata, ceilPower2 * sizeof(int));
  
	// Reset to zeros
  cudaMemset(dev_odata, 0, ceilPower2 * sizeof(int));

  // Copy idata to device memory
  cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef PROFILE
  // CUDA events for profiling
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif
  deviceScan(n, dev_odata);
#ifdef  PROFILE
  cudaEventRecord(stop);
#endif

 
#ifdef PROFILE
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  *timeElapsedMs = milliseconds;
#endif
  // Transfer data back to host
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(dev_odata);

}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, PathSegment *odata, const PathSegment *idata) {
    
	int remainingElementCount = 0;

	int height = ilog2ceil(n);
	int ceilPower2 = 1 << height;
	int *dev_bools, *dev_indices;
	PathSegment* dev_odata, *dev_idata;
	cudaMalloc((void**)&dev_idata, sizeof(PathSegment) * ceilPower2);
	cudaMalloc((void**)&dev_odata, sizeof(PathSegment) * ceilPower2);
	cudaMalloc((void**)&dev_bools, sizeof(int) * ceilPower2);
	cudaMalloc((void**)&dev_indices, sizeof(int) * ceilPower2);

	// Transfer idata from host to device
	cudaMemcpy(dev_idata, idata, sizeof(PathSegment) * n, cudaMemcpyHostToDevice);

#ifdef PROFILE
	// CUDA events for profiling
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif

	// Set all non-zeros to 1s and zeros to 0s. This is our pass condition for an element to remain/discard after compaction
	kernMapToBoolean << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_bools, dev_idata);
	
	// Compute indices of the out compacted stream
	// Reset to zeros
	cudaMemset(dev_indices, 0, ceilPower2 * sizeof(int));
	// Copy dev_bools to dev_indices to device memory
	cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
	StreamCompaction::UnoptimizedEfficient::deviceScan(ceilPower2, dev_indices);

	// Move elements that are not discarded into appropriate slots based on scan result
	kernScatter << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_odata, dev_idata, dev_bools, dev_indices);

	// The max value of all the valid indices for the compacted stream is the number of remaining elements
	int* dev_remainingElementCount;
	cudaMalloc((void**)&dev_remainingElementCount, sizeof(int));
	kernRemainingElementsCountForCompact<<<1, 1>>>(ceilPower2, dev_indices, dev_bools, dev_remainingElementCount);
	cudaMemcpy(&remainingElementCount, dev_remainingElementCount, sizeof(int), cudaMemcpyDeviceToHost);
  
#ifdef PROFILE
	cudaEventRecord(stop);
#endif

	// Transfer output back to host
	cudaMemcpy(odata, dev_odata, sizeof(PathSegment) * n, cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(dev_bools);
	cudaFree(dev_indices);
	cudaFree(dev_idata);
	cudaFree(dev_odata);
  
#ifdef PROFILE
	cudaEventSynchronize(stop);
#endif

	return remainingElementCount;
}


}


namespace OptimizedEfficient {

	__global__ void scanWithSharedMemory(
		int n,
		int* odata)
	{

		int tid = threadIdx.x + (blockIdx.x * blockDim.x);
		int offset = 1;
		if (n <= offset * (2 * tid + 2) - 1) {
			return;
		}

		__shared__ int partialSum[1024];

		// Copy from device memory to shared memory
		partialSum[2 * tid] = odata[2 * tid];
		partialSum[2 * tid + 1] = odata[2 * tid + 1];

		// upsweep and building up partial sum
		for (int d = n >> 1; d > 0; d >>= 1) {
			__syncthreads();
			if (tid < d) {
				partialSum[offset * (2 * tid + 2) - 1] += partialSum[offset * (2 * tid + 1) - 1];
			}
			offset *= 2;
		}

		if (tid == 0) {
			partialSum[n - 1] = 0; // Clear the last element. Only using one thread.
		}

		// downsweep and distribute scan
		for (int d = 1; d < n; d *= 2) {
			offset >>= 1;
			__syncthreads();
			if (tid < d) {
				int leftIdx = offset * (2 * tid + 1) - 1;
				int rightIdx = offset * (2 * tid + 2) - 1;

				// Swap & partial sum right child with left child
				int t = partialSum[leftIdx];
				partialSum[leftIdx] = partialSum[rightIdx];
				partialSum[rightIdx] += t;
			}
		}

		// Copy back to device memory from shared memory
		odata[2 * tid] = partialSum[2 * tid];
		odata[2 * tid + 1] = partialSum[2 * tid + 1];
	}

	/**
	* Performs stream compaction on idata, storing the result into odata.
	* All zeroes are discarded.
	*
	* @param n      The number of elements in idata.
	* @param odata  The array into which to store elements.
	* @param idata  The array of elements to compact.
	* @returns      The number of elements remaining after compaction.
	*/
	int compact(int n, PathSegment *odata, const PathSegment *idata) {

		int remainingElementCount = 0;

		int height = ilog2ceil(n);
		int ceilPower2 = 1 << height;
		int *dev_bools, *dev_indices;
		PathSegment* dev_odata, *dev_idata;
		cudaMalloc((void**)&dev_idata, sizeof(PathSegment) * ceilPower2);
		cudaMalloc((void**)&dev_odata, sizeof(PathSegment) * ceilPower2);
		cudaMalloc((void**)&dev_bools, sizeof(int) * ceilPower2);
		cudaMalloc((void**)&dev_indices, sizeof(int) * ceilPower2);

		// Transfer idata from host to device
		cudaMemcpy(dev_idata, idata, sizeof(PathSegment) * n, cudaMemcpyHostToDevice);

#ifdef PROFILE
		// CUDA events for profiling
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
#endif

		// Set all non-zeros to 1s and zeros to 0s. This is our pass condition for an element to remain/discard after compaction
		kernMapToBoolean << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_bools, dev_idata);

		// Compute indices of the out compacted stream
		// Reset to zeros
		cudaMemset(dev_indices, 0, ceilPower2 * sizeof(int));
		// Copy dev_bools to dev_indices to device memory
		cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
		scanWithSharedMemory << < BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> > (ceilPower2, dev_indices);

		// Move elements that are not discarded into appropriate slots based on scan result
		kernScatter << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_odata, dev_idata, dev_bools, dev_indices);

		// The max value of all the valid indices for the compacted stream is the number of remaining elements
		int* dev_remainingElementCount;
		cudaMalloc((void**)&dev_remainingElementCount, sizeof(int));
		kernRemainingElementsCountForCompact << <1, 1 >> >(ceilPower2, dev_indices, dev_bools, dev_remainingElementCount);
		cudaMemcpy(&remainingElementCount, dev_remainingElementCount, sizeof(int), cudaMemcpyDeviceToHost);

#ifdef PROFILE
		cudaEventRecord(stop);
#endif

		// Transfer output back to host
		cudaMemcpy(odata, dev_odata, sizeof(PathSegment) * n, cudaMemcpyDeviceToHost);

		// Cleanup
		cudaFree(dev_bools);
		cudaFree(dev_indices);
		cudaFree(dev_idata);
		cudaFree(dev_odata);

#ifdef PROFILE
		cudaEventSynchronize(stop);
#endif

		return remainingElementCount;
	}
}
}
