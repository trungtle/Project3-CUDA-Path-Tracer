#include <cstdio>
#include "cpu.h"
#include <cstring>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    odata[0] = 0;
    for (int i = 1; i < n; ++i) {
      odata[i] = odata[i - 1] + idata[i - 1];
    }
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
  int oIndex = 0;
  for (int iIndex = 0; iIndex < n; ++iIndex) {
    if (idata[iIndex] != 0) {
      odata[oIndex++] = idata[iIndex];
    }
  }
  return oIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
    memset(odata, 0, n * sizeof(int));

	for (int i = 0; i < n; ++i) {
      if (idata[i] != 0) {
        odata[i] = 1;
      }
    }

	int* scanResult = new int[n];
    scan(n, scanResult, odata);

    int remainingNumberOfElements = 0;
    for (int i = 0; i < n; ++i) {
      if (odata[i] == 1) {
        odata[scanResult[i]] = idata[i];
        remainingNumberOfElements = scanResult[i] + 1;
      }
    }

    delete[] scanResult;
    return remainingNumberOfElements;
}

}
}
