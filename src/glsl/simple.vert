#version 330 core
#define POSITION_LOCATION 0
#define NORMAL_LOCATION 1
#define COLOR_LOCATION 2

precision highp float;
precision highp int;
layout(std140, column_major) uniform;

uniform mat4 u_model;
uniform mat4 u_viewProj;

layout(location = POSITION_LOCATION) in vec3 a_position;
layout(location = NORMAL_LOCATION) in vec3 a_normal;
layout(location = COLOR_LOCATION) in vec4 a_color;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_lightVec;

const vec4 lightPos = vec4(7, 5, -3, 1); // Virtual light

void main()
{
	vec4 modelPosition = u_model * vec4(a_position, 1.0);
    gl_Position = u_viewProj * modelPosition;

    // Pass through frag shader
    v_color = a_color;
    v_normal = vec3(transpose(inverse(u_model)) * vec4(a_normal, 0.0));
    v_lightVec = (lightPos - modelPosition).xyz;
}
