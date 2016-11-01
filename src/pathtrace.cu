#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "stream_compaction/efficient.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static BVHNodeDev * dev_bvhNodes = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	const int samplesPerPixel = cam.samplesPerPixel;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * samplesPerPixel * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	int bvhNodesCount = hst_scene->bvhNodes.size();
	cudaMalloc(&dev_bvhNodes, bvhNodesCount * sizeof(BVHNodeDev));
	cudaMemcpy(dev_bvhNodes, hst_scene->bvhNodes.data(), bvhNodesCount * sizeof(BVHNodeDev), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void updateGeom(Scene *scene) {
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_bvhNodes);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
	Camera cam
	, int iter
	, int traceDepth
	, PathSegment* pathSegments
	)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		for (int i = 0; i < cam.samplesPerPixel; ++i) {
			int index = i + (x * cam.samplesPerPixel) + (y * cam.resolution.x);
			PathSegment & segment = pathSegments[index];

			segment.ray.origin = cam.position;
			segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

			// TODO: implement antialiasing by jittering the ray
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			segment.ray.direction = glm::normalize(
				cam.forward
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f )
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
				);

			segment.pixelIndex = x + (y * cam.resolution.x);
			segment.remainingBounces = traceDepth;
		}
	}
}

__device__ float computeIntersection(
	const PathSegment& pathSegment
	, const Geom& geom
	, glm::vec3& intersect_point
	, glm::vec3& normal
	) {
	float t = -1;
	bool outside = true;

	if (geom.type == CUBE)
	{
		t = boxIntersectionTest(geom, pathSegment.ray, intersect_point, normal, outside);
	}
	else if (geom.type == SPHERE)
	{
		t = sphereIntersectionTest(geom, pathSegment.ray, intersect_point, normal, outside);
	} 
	else if (geom.type == TRIANGLE) 
	{
		t = triangleIntersectionTest(geom, pathSegment.ray, intersect_point, normal, outside);
	}

	return t;
}

/*
* Iterative stack-less BVH traversal using state logic and pointers to nodes.
* \ref https://graphics.cg.uni-saarland.de/fileadmin/cguds/papers/2011/hapala_sccg2011/hapala_sccg2011.pdf
*/
__global__ void traverseBVH(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, int num_bvhNodes
	, BVHNodeDev* bvhNodes
	, int rootIdx
	, int geoms_size
	, Geom * geoms
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		Geom* hit_geom = nullptr;

		BVHNodeDev current = bvhNodes[rootIdx];
		EBVHTransition transition = EBVHTransition::FromParent;

		bool isIterating = true;
		while (isIterating) {
			// States (reproduced here from Stack-less BVH Traversal paper [Hapala1 el at. 2011])
			// Link: https://graphics.cg.uni-saarland.de/fileadmin/cguds/papers/2011/hapala_sccg2011/hapala_sccg2011.pdf
			switch (transition) {

				// 1. From child
				// In the fromChild case the current node was already tested when going
				// down, and does not have to bce re - tested.The next node to traverse
				// is either current’s sibling f arChild(if current is nearChild),
				// or its parent(if current was farChild).
				//
			case EBVHTransition::FromChild:
				if (current.idx == rootIdx) {
					// Current has reached root
					isIterating = false;
				}
				else if (current.idx == bvhNodes[current.parentIdx].nearChildIdx) {
					// Current is near child, so transition to far child
					current = bvhNodes[bvhNodes[current.parentIdx].farChildIdx];
					transition = EBVHTransition::FromSibling;
				}
				else {
					// Current is far child, go back to parent
					current = bvhNodes[current.parentIdx];
					transition = EBVHTransition::FromChild;
				}
				break;

				// 2. From sibling
				// In the fromSibling case, we know that we are entering farChild (it
				// cannot be reached in any other way), and that we are traversing this
				// node for the first time(i.e.a box test has to be done).If the node
				// is missed, we back - track to its parent; otherwise, the current node
				// has to be processed : if it is a leaf node, we intersect its primitives
				// against the ray, and proceed to parent. Otherwise(i.e. if the node
				// was hit but is not a leaf), we enter current’s subtree by performing
				// a fromParent step to current’s first child.

			case EBVHTransition::FromSibling:

				if (current.geomIdx != -1) {
					// Leaf node
					t = computeIntersection(pathSegment, geoms[current.geomIdx], tmp_intersect, tmp_normal);
					// Compute the minimum t from the intersection tests to determine what
					// scene geometry object was hit first.
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
						hit_geom = &(geoms[current.geomIdx]);
					}

					current = bvhNodes[current.parentIdx];
					transition = EBVHTransition::FromChild;
				}
				else {
					// When this isn't a leaf node, check bbox intersection
					bool hit = bboxIntersectionTest(current.bboxGeom, pathSegments[path_index].ray);
					if (!hit) {
						// Missed, go back up to parent

						if (current.idx == rootIdx) {
							// Current has reached root
							isIterating = false;
						} else {
							current = bvhNodes[current.parentIdx];
							transition = EBVHTransition::FromChild;
						}
					}
					else {
						// Hit, enter its subtree (near child)
						current = bvhNodes[current.nearChildIdx];
						transition = EBVHTransition::FromParent;
					}
				}
				break;

				// 3. From parent
				// Finally, in the fromParent case, we know that we are entering
				// nearChild and we do exactly the same as in the previous case,
				// except that every time we would have gone to parent we go to
				// farChild child.

			case EBVHTransition::FromParent:
				if (current.geomIdx != -1) {
					// Leaf node
					t = computeIntersection(pathSegment, geoms[current.geomIdx], tmp_intersect, tmp_normal);
					// Compute the minimum t from the intersection tests to determine what
					// scene geometry object was hit first.
					if (t > 0.0f && t_min > t)
					{
						t_min = t;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
						hit_geom = &(geoms[current.geomIdx]);
					}

					if (current.idx == rootIdx) {
						// Current has reached root
						isIterating = false;
					} else {
						// Go to far sibling
						current = bvhNodes[bvhNodes[current.parentIdx].farChildIdx];
						transition = EBVHTransition::FromSibling;
					}
				}
				else {
					// When this isn't a leaf node, check bbox intersection
					bool hit = bboxIntersectionTest(current.bboxGeom, pathSegments[path_index].ray);
					if (!hit) {
						// Missed, go to far sibling
						if (current.idx == rootIdx) {
							// Current has reached root
							isIterating = false;
						} else {
							current = bvhNodes[bvhNodes[current.parentIdx].farChildIdx];
							transition = EBVHTransition::FromSibling;
						}
					}
					else {
						// Hit, enter its subtree
						current = bvhNodes[current.nearChildIdx];
						transition = EBVHTransition::FromParent;
					}
				}
				break;
			
			default:
				// *N.B*: Should never reach here
				assert(false);
				break;
			}
		}
		
		if (hit_geom == nullptr)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_geom->materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersect_point = intersect_point;
		}
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			t = computeIntersection(pathSegments[path_index], geom, tmp_intersect, tmp_normal);

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].intersect_point = intersect_point;
		}
	}
}

__global__ void shadeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths && pathSegments[idx].remainingBounces > 0)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
			}
			else {

				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
				scatterRay(pathSegments[idx], intersection.intersect_point, intersection.surfaceNormal, materials[intersection.materialId], rng);
				pathSegments[idx].remainingBounces -= 1;
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void partialGather(
	const Camera cam
	, int nPaths
	, glm::vec3* image
	, PathSegment* iterationPaths
	)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.remainingBounces == 0) {
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(
	const Camera cam
	, int nPaths
	, glm::vec3* image
	, PathSegment* iterationPaths
	)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (iterationPath.remainingBounces >= 0) {
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}

struct shouldTerminatePath
{
	__host__ __device__
	bool operator()(const PathSegment& p)
	{
		return p.remainingBounces == 0;
	};
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 8;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a tpath to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // Perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
 		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		
		//cudaEventRecord(start);
		if (hst_scene->BVH_ENABLED) {
 			traverseBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, hst_scene->bvhNodes.size()
				, dev_bvhNodes
				, hst_scene->root->nodeIdx
				, hst_scene->geoms.size()
				, dev_geoms
				, dev_intersections
				);
		} else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
		}
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		//cudaEventRecord(stop);
		depth++;


		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);

		if (hst_scene->STREAM_COMPACTION_ENABLED) {
			// If using stream compaction, we have to use partial gather
			dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
			partialGather << <numBlocksPixels, blockSize1d >> >(cam, num_paths, dev_image, dev_paths);
#define USE_THRUST
#ifdef USE_THRUST
			PathSegment* new_dev_paths_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, shouldTerminatePath());
			num_paths = new_dev_paths_end - dev_paths;
#else
			num_paths = StreamCompaction::OptimizedEfficient::compact(num_paths, dev_paths, dev_paths);

#endif
		}

		//cudaEventSynchronize(stop);
		//float ms;
		//cudaEventElapsedTime(&ms, start, stop);
		//cout << ms << endl;

		iterationComplete = num_paths == 0 || depth > traceDepth;
	}

	if (!hst_scene->STREAM_COMPACTION_ENABLED) {
		// If not using stream compaction, apply final gather
		// Assemble this iteration and apply it to the image
		num_paths = dev_path_end - dev_paths;
		dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> >(cam, num_paths, dev_image, dev_paths);

	}


	///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
