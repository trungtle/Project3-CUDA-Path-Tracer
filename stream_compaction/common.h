#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define imin(a, b) (((a) < (b)) ? (a) : (b))
#define imax(a, b) (((a) > (b)) ? (a) : (b))

#define BLOCK_SIZE 128
#define BLOCK_COUNT(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Milliseconds to nanoseconds
#define MS_TO_NS(ms) ((ms) * 1000000)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}

namespace StreamCompaction {
namespace Common {
    __global__ void inclusiveToExclusiveScanResult(int n, int* odata, const int* idata);

    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);
}
}
