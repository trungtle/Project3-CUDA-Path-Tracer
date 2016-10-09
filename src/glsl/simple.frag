#version 150 core

precision highp float;
precision highp int;
layout(std140, column_major) uniform;

in vec4 v_color;
out vec4 color;

void main()
{
    color = v_color;
}
