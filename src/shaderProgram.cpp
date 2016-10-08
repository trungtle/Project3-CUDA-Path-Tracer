#include "shaderProgram.h"

ShaderProgram::ShaderProgram(
	const char* vertFilePath,
	const char* fragFilePath
	)
{
	const char *attribLocations[] = { "Position", "Texcoords" };
	m_programID = glslUtility::createProgram("E:\CODES\Penn\CIS565\Project3-CUDA-Path-Tracer\src\glsl\simple.vert", "E:\CODES\Penn\CIS565\Project3-CUDA-Path-Tracer\src\glsl\simple.frag", attribLocations, 2);
	
	m_unifModel = glGetUniformLocation(m_programID, "u_model");
	m_unifViewProj = glGetUniformLocation(m_programID, "u_viewProj");

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
}

void
ShaderProgram::DrawBBox(
	const Camera& camera,
	const Geom& geom,
	const GeomVAO& geomVao
	) const
{
	glUseProgram(m_programID);

	// Enable attributes
	enableVertexAttributes(
		geomVao.vao,
		geomVao.posBuf,
		geomVao.norBuf,
		geomVao.colBuf,
		geomVao.idxBuf
		);

	// Set uniforms

	if (m_unifModel != -1) {
		glUniformMatrix4fv(
			m_unifModel,
			1,
			GL_FALSE,
			&(geom.transform[0][0])
			);
	}

	if (m_unifViewProj != -1) {

		glm::mat4 perspective = glm::perspective(camera.fov.y, static_cast<float>(camera.resolution.x) / static_cast<float>(camera.resolution.y), 0.001f, 1000.f);

		glUniformMatrix4fv(
			m_unifViewProj,
			1,
			GL_FALSE,
			&(perspective)[0][0]
			);
	}
	// Render
	glDrawElements(
		GL_LINES,
		geomVao.elementCount,
		GL_UNSIGNED_SHORT,
		nullptr
		);

	disableVertexAttributes();
}

void
ShaderProgram::CleanUp()
{
	glDeleteProgram(m_programID);
}


// ============== OpenGL Specifics ===================== //

void
enableVertexAttributes(
GLuint vao,
GLuint posBuffer,
GLuint norBuffer,
GLuint colBuffer,
GLuint indexBuffer
)
{
	glBindVertexArray(vao);

	// Enable vertex attributes
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, norBuffer);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, colBuffer);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// Bind element buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
}

void
disableVertexAttributes()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);
	glBindVertexArray(NULL);
}

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
)
{
	glBindVertexArray(vao);

	// -- Position

	glBindBuffer(GL_ARRAY_BUFFER, posBuffer);
	glBufferData(
		GL_ARRAY_BUFFER,
		newPositions.size() * sizeof(glm::vec3),
		&newPositions[0],
		GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);

	// -- Normals

	glBindBuffer(GL_ARRAY_BUFFER, norBuffer);
	glBufferData(
		GL_ARRAY_BUFFER,
		newNormals.size() * sizeof(glm::vec3),
		&newNormals[0],
		GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);

	// -- Colors

	glBindBuffer(GL_ARRAY_BUFFER, colBuffer);
	glBufferData(
		GL_ARRAY_BUFFER,
		newColors.size() * sizeof(glm::vec4),
		&newColors[0],
		GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);

	// -- Index

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		newIndices.size() * sizeof(GLushort),
		&newIndices[0],
		GL_STATIC_DRAW
		);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);

	glBindVertexArray(NULL);
}

