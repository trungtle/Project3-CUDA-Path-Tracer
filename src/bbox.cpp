#include "bbox.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "shaderProgram.h"

namespace BBox {

	glm::vec3 Centroid(
		const glm::vec3& a,
		const glm::vec3& b
		)
	{
		return glm::vec3(
			(a.x + b.x) / 2.0f,
			(a.y + b.y) / 2.0f,
			(a.z + b.z) / 2.0f
			);
	}

	BBox BBoxUnion(const BBox& a, const BBox& b) {
		BBox ret;
		ret.max.x = glm::max(a.max.x, b.max.x);
		ret.max.y = glm::max(a.max.y, b.max.y);
		ret.max.z = glm::max(a.max.z, b.max.z);
		ret.min.x = glm::min(b.min.x, b.min.x);
		ret.min.y = glm::min(a.min.y, b.min.y);
		ret.min.z = glm::min(a.min.z, b.min.z);
		ret.centroid = Centroid(ret.max, ret.min);
		return ret;
	}

	BBox BBoxFromGeom(const Geom& geom) {

		BBox box;

		// Convert to world first
		glm::vec3 p0 = glm::vec3(geom.transform * glm::vec4(-.5f, -.5f, -.5f, 1.0f));
		glm::vec3 p1 = glm::vec3(geom.transform * glm::vec4(-.5f, -.5f, .5f, 1.0f));
		glm::vec3 p2 = glm::vec3(geom.transform * glm::vec4(-.5f, .5f, -.5f, 1.0f));
		glm::vec3 p3 = glm::vec3(geom.transform * glm::vec4(-.5f, .5f, .5f, 1.0f));
		glm::vec3 p4 = glm::vec3(geom.transform * glm::vec4(.5f, -.5f, -.5f, 1.0f));
		glm::vec3 p5 = glm::vec3(geom.transform * glm::vec4(.5f, -.5f, .5f, 1.0f));
		glm::vec3 p6 = glm::vec3(geom.transform * glm::vec4(.5f, .5f, -.5f, 1.0f));
		glm::vec3 p7 = glm::vec3(geom.transform * glm::vec4(.5f, .5f, .5f, 1.0f));

		box.min = glm::min(p0, glm::min(p1, glm::min(p2, glm::min(p3, glm::min(p4, glm::min(p5, glm::min(p6, p7)))))));
		box.max = glm::max(p0, glm::max(p1, glm::max(p2, glm::max(p3, glm::max(p4, glm::max(p5, glm::max(p6, p7)))))));
		box.centroid = Centroid(box.min, box.max);

		return box;
	}

	EAxis BBoxMaximumExtent(const BBox& bbox) {
		glm::vec3 diag = bbox.max - bbox.min;
		if (diag.x > diag.y && diag.x > diag.z) {
			return EAxis::X;
		}
		else if (diag.y > diag.z) {
			return EAxis::Y;
		}
		else {
			return EAxis::Z;
		}
	}

	// ============== OpenGL Specifics ===================== //

	const int BBOX_IDX_COUNT = 24;
	const int BBOX_VERT_COUNT = 8;

	void createBBoxVertexPositions(
		const BBox& bbox,
		std::vector<glm::vec3>& bbox_vert_pos
		)
	{
		bbox_vert_pos.push_back(glm::vec3(bbox.max.x, bbox.max.y, bbox.max.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.max.x, bbox.max.y, bbox.min.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.max.x, bbox.min.y, bbox.max.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.max.x, bbox.min.y, bbox.min.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.min.x, bbox.max.y, bbox.max.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.min.x, bbox.max.y, bbox.min.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.min.x, bbox.min.y, bbox.max.z));
		bbox_vert_pos.push_back(glm::vec3(bbox.min.x, bbox.min.y, bbox.min.z));
	}

	void createBBoxVertexNormals(
		std::vector<glm::vec3>& bbox_vert_normals
		)
	{
		// For bbox, we don't really care about normals, so just give it watever
		for (int i = 0; i < BBOX_VERT_COUNT; ++i) {
			bbox_vert_normals.push_back(glm::vec3());
		}
	}

	void createBBoxVertexColors(
		std::vector<glm::vec4>& bbox_vert_col
		)
	{
		for (int i = 0; i < BBOX_VERT_COUNT; i++){
			bbox_vert_col.push_back(glm::vec4(1.f, 0, 1.f, 1.0f));
		}
	}

	void createBBoxIndices(std::vector<GLushort>& bbox_idx)
	{
		int idx = 0;

		bbox_idx[idx++] = 0;
		bbox_idx[idx++] = 1;
		bbox_idx[idx++] = 1;
		bbox_idx[idx++] = 3;
		bbox_idx[idx++] = 3;
		bbox_idx[idx++] = 2;
		bbox_idx[idx++] = 2;
		bbox_idx[idx++] = 0;
		bbox_idx[idx++] = 0;
		bbox_idx[idx++] = 4;
		bbox_idx[idx++] = 4;
		bbox_idx[idx++] = 6;
		bbox_idx[idx++] = 6;
		bbox_idx[idx++] = 2;
		bbox_idx[idx++] = 3;
		bbox_idx[idx++] = 7;
		bbox_idx[idx++] = 7;
		bbox_idx[idx++] = 6;
		bbox_idx[idx++] = 1;
		bbox_idx[idx++] = 5;
		bbox_idx[idx++] = 5;
		bbox_idx[idx++] = 4;
		bbox_idx[idx++] = 5;
		bbox_idx[idx++] = 7;
	}


	void initBBoxVAO(const BBox& bbox, GeomVAO& bboxVao)
	{
		std::vector<GLushort> bbox_idx;
		std::vector<glm::vec3> bbox_vert_pos;
		std::vector<glm::vec3> bbox_vert_nor;
		std::vector<glm::vec4> bbox_vert_col;

		createBBoxVertexPositions(bbox, bbox_vert_pos);
		createBBoxVertexNormals(bbox_vert_nor);
		createBBoxVertexColors(bbox_vert_col);
		createBBoxIndices(bbox_idx);

		// Generate all the buffers that we need
		GLuint vao;
		glGenVertexArrays(1, &vao);

		GLuint vertexBufferObjID[4];
		glGenBuffers(4, vertexBufferObjID);

		updateVAO(
			vao,
			vertexBufferObjID[0],
			bbox_vert_pos,
			vertexBufferObjID[1],
			bbox_vert_nor,
			vertexBufferObjID[2],
			bbox_vert_col,
			vertexBufferObjID[3],
			bbox_idx
			);

		// Populate geom vao
		bboxVao.vao = vao;
		bboxVao.posBuf = vertexBufferObjID[0];
		bboxVao.norBuf = vertexBufferObjID[1];
		bboxVao.colBuf = vertexBufferObjID[2];
		bboxVao.idxBuf = vertexBufferObjID[3];
		bboxVao.elementCount = BBOX_IDX_COUNT;
	}

}