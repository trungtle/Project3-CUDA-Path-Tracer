#include "main.h"
#include "preview.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/random.hpp>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom = 0, theta = 0, phi = 0;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 forward = cam.forward;
    glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(forward, up);
	up = glm::cross(right, forward);

    cameraPosition = cam.position;

	// Initialize CUDA and GL components
	init();

	// Initialize the BVH structure
	if (scene->isBVHEnabled) {
		scene->initBVH();
	}

    // GLFW main loop
	mainLoop(scene);

    return 0;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
		//cam.RotateAboutRight(phi);
		//cam.RotateAboutUp(theta);
		cam.TranslateAlongRight(phi);
		cam.TranslateAlongUp(theta);
		cam.Zoom(zoom);
		zoom = 0;
		theta = 0;
		phi = 0;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

	//bool toggle = true;
	//for (Geom& geo : scene->geoms) {
	//	if (geo.type == GeomType::SPHERE) {
	//		toggle = !toggle;
	//		geo.transform = utilityCore::buildTransformationMatrix(
	//			geo.translation + glm::vec3(false ? glm::sin(iteration * 10.0) * 1.5 : 0.0f, toggle ? 0.0f : glm::sin(iteration * 10.0) * 2.0, toggle ? 0.f : glm::sin(iteration * 10.0) * 3.5),
	//			glm::vec3(glm::rotate((float)iteration * 10, glm::vec3(1.0, 1.0f, 0)) * glm::vec4(geo.rotation, 1.0f)),
	//			geo.scale);
	//		geo.inverseTransform = glm::inverse(geo.transform);
	//		geo.invTranspose = glm::inverseTranspose(geo.transform);
	//	}
	//}
	//updateGeom(scene);

	if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi = (xpos - lastX) * 10.f / width;
    theta = (ypos - lastY) * 10.f / height;
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) * 10.0f / height;
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.forward;
    //forward.y = 0.0f;
    //forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    //right.y = 0.0f;
    //right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * cam.up * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
