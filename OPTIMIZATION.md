
I performed the following optimization:

- Use my own stream compaction with shared memory (switched out from thrust::partition compare to the previous version)

# Using shared memory in stream compaction

I followed the code listing in [GPU Gems 3 - Chapter 39](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html) to implement the efficient scan.

Here is the comparison:

Cornell box with torus - 800 x 800 resolution - 8 bounces per iteration

[insert table / graph here]