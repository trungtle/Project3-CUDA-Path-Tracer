set(SOURCE_FILES
    "profilingcommon.h"
    "efficient.h"
    "efficient.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
