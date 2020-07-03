#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec4 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 3) in uint a_BlockType;

layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec4 v_Pos;
layout(location = 3) out uint v_BlockType;
layout(location = 4) out vec3 v_CameraPos;


layout(set = 0, binding = 0) uniform Locals {
  mat4 u_Projection;
  mat4 u_View;
  mat4 u_Model;
  vec4 u_CameraPos;
};

mat4 translation_matrix(vec3 offset) {
  return mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    offset.x, offset.y, offset.z, 1.0);
}

void main() {
  v_CameraPos = u_CameraPos.xyz;

  mat4 view = u_View * translation_matrix(-u_CameraPos.xyz);

  v_TexCoord = a_TexCoord;
  v_Pos = u_Model * a_Pos;
  v_Normal = (u_Model * a_Normal).xyz;

  gl_Position = u_Projection * view * v_Pos;
  v_BlockType = a_BlockType;
}

// vi: ft=c
