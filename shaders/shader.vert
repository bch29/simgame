#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec4 v_Pos;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Projection;
    mat4 u_View;
    mat4 u_Model;
};

void main() {
    v_TexCoord = a_TexCoord;
    v_Pos = u_Projection * u_View * u_Model * a_Pos;
    v_Normal = (u_Model * vec4(a_Normal, 0.0)).xyz;
    gl_Position = v_Pos;
}

// vi: ft=c
