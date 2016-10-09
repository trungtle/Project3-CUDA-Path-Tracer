#pragma once

/*
* Adpated from OpenGL Tutorial: http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
*/

#include "main.h"

class ShaderProgram
{

public:
	ShaderProgram(
		const char* vertFilePath,
		const char* fragFilePath
		);

	virtual void DrawBBox(
		const Camera&,
		const Geom&,
		const BBoxVAO&
		) const;

	virtual void CleanUp();

protected:
	GLuint m_programID;

	// Uniform locations
	int m_unifModel;
	int m_unifViewProj;
};

// ============== OpenGL Specifics ===================== //

void
enableVertexAttributes(
	GLuint vao,
	GLuint posBuffer,
	GLuint norBuffer,
	GLuint colBuffer,
	GLuint indexBuffer
	);

void
disableVertexAttributes();

void
updateVAO(
	GLuint vao,
	GLuint posBuffer,
	const std::vector<glm::vec3>& newPositions,
	GLuint norBuffer,
	const std::vector<glm::vec3>& newNormals,
	GLuint colBuffer,
	const std::vector<glm::vec4>& newColors,
	GLuint indexBuffer,
	const std::vector<GLushort>& newIndices
	);

