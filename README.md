CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Trung Le
* Windows 10 Home, i7-4790 CPU @ 3.60GHz 12GB, GTX 980 Ti (Person desktop)

### Stream compaction

**---- General information for CUDA device ----**
- Device name: GeForce GTX 980 Ti
- Compute capability: 5.2
- Compute mode: Default
- Clock rate: 1076000
- Integrated: 0
- Device copy overlap: Enabled
- Kernel execution timeout: Enabled
 
**---- Memory information for CUDA device ----**

- Total global memory: 6442450944
- Total constant memory: 65536
- Multiprocessor count: 22
- Shared memory per multiprocessor: 98304
- Registers per multiprocessor: 65536
- Max threads per multiprocessor: 2048
- Max grid dimensions: [2147483647, 65535, 65535]
- Max threads per block: 1024
- Max registers per block: 65536
- Max thread dimensions: [1024, 1024, 64]
- Threads per block: 512

# Description

In this project, I implemented on top of the provided CUDA started code a fully functional pathtracer with the following features:

- Stream compaction on terminated paths during bounces
- Diffuse shading
- Specular reflection
- Specular transmission
- Scene loading
- BVH acceleration data structure with stack-less BVH traversal
- Temporal AA
- Motion blur

## Analysis

## Refraction comparasion

[insert photos here]

## Comparing between using __constant__ to store materials and geoms vs. with __constant__

## Comparing with stream compaction vs. no stream compaction

## Motion blur

[insert photos here]

## Spatial data structure analysis

BVH requirements:
a) Binary BVH tree with exactly two children (also called siblings) `nearChild` and `farChild`. All primitives are stored at leaf nodes.
b) Each node has a pointer to parent
c) Each inner node has a unique traversal for a given ray from near child to far child. This order can be different for each ray but has to be the same order for the same ray.
d) Internal nodes only stores a bounding box

There are only three traversal states that a node can be entered:
1. From its parent
2. From its sibling (from nearChild to farChild)
3. From its children (out from farChild)

Uses shared memory to store BVH stack

[Stack-less BVH traversal](https://graphics.cg.uni-saarland.de/fileadmin/cguds/papers/2011/hapala_sccg2011/hapala_sccg2011.pdf)

I attempted to use glm::intersectRayTriangle, but it wasn't well documented what the output should have been. After checking https://github.com/g-truc/glm/issues/6, I realized that
` intersection (cartesian) = origin + direction * baryPosition.z `
But CUDA was also running into an issue with stack limit that failed calling this function, so I ended up writing my version for triangle intersection following the [fast, minimum storage ray/triangle intersection by Muller and Trumbore](https://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf). THe advantage of this method is that the plane equation doesn't have to be computed on the fly or be stored, which increases memory savings for triangle meshes.

#Thirdparty codes

tinyobjloader (see LICENSE page for LICENSE)

# Camera control
- Left mouse drag to pan left/righ/up/down
- Right mouse drag to zoom in/out
- Middle mouse drag to change look at point

#Note

Please update the path to the shader program properly

Added the following to CMakeList.txt
	"bbox.h"
	"bbox.cpp"
	"bvh.h"
	"bvh.cpp"
	"shaderProgram.h"
	"shaderProgram.cpp"
	"camera.h"
	"camera.cpp"
