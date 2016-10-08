#version 150 core

precision highp float;
precision highp int;
layout(std140, column_major) uniform;

in vec4 v_color;
in vec3 v_normal;
in vec3 v_lightVec;

out vec4 color;

void main()
{
	vec4 diffuse = v_color;
	float diffuseTerm = dot(normalize(v_normal), normalize(v_lightVec));
	diffuseTerm = clamp(diffuseTerm, 0, 1);

	float ambientTerm = 0.1;
	float lightIntensity = diffuseTerm + ambientTerm;

    color = vec4(vec3(diffuse * lightIntensity), 1);
}
