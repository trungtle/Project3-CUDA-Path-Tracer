#pragma once


typedef enum {
	x,
	y,
	z
} EAxis;


__host__ __device__
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

struct BBox {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 centroid;

	BBox() : min(glm::vec3(0.f)), max(glm::vec3(0.f)), centroid(glm::vec3(0.f))
	{};

	// Create a bbox for a given geometry
	BBox(const Geom* geom) {

		// Convert to world first
		glm::vec3 p0 = glm::vec3(geom->transform * glm::vec4(-.5f, -.5f, -.5f, 1.0f));
		glm::vec3 p1 = glm::vec3(geom->transform * glm::vec4(-.5f, -.5f, .5f, 1.0f));
		glm::vec3 p2 = glm::vec3(geom->transform * glm::vec4(-.5f, .5f, -.5f, 1.0f));
		glm::vec3 p3 = glm::vec3(geom->transform * glm::vec4(-.5f, .5f, .5f, 1.0f));
		glm::vec3 p4 = glm::vec3(geom->transform * glm::vec4(.5f, -.5f, -.5f, 1.0f));
		glm::vec3 p5 = glm::vec3(geom->transform * glm::vec4(.5f, -.5f, .5f, 1.0f));
		glm::vec3 p6 = glm::vec3(geom->transform * glm::vec4(.5f, .5f, -.5f, 1.0f));
		glm::vec3 p7 = glm::vec3(geom->transform * glm::vec4(.5f, .5f, .5f, 1.0f));

		this->min = glm::min(p0, glm::min(p1, glm::min(p2, glm::min(p3, glm::min(p4, glm::min(p5, glm::min(p6, p7)))))));
		this->max = glm::max(p0, glm::max(p1, glm::max(p2, glm::max(p3, glm::max(p4, glm::max(p5, glm::max(p6, p7)))))));
		this->centroid = Centroid(this->min, this->max);
	}
};

__host__ __device__
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
