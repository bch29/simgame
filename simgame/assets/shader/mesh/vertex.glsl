#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in uint a_FaceId;
layout(location = 3) in vec2 a_TexCoord;

layout(location = 0) out vec4 v_Pos;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out uint v_TexId;
layout(location = 3) out vec2 v_TexCoord;
layout(location = 4) out vec3 v_CameraPos;

layout(set = 0, binding = 0) uniform Uniforms {
  mat4 u_Proj;
  mat4 u_View;
  vec3 u_CameraPos;
};

struct InstanceMeta {
  mat4 model;
  uint[16] faceTexIds;
};

layout(set = 0, binding = 1) readonly buffer InstanceMetaBuf {
  InstanceMeta[] b_InstanceMeta;
};

mat4 translation_matrix(vec3 offset) {
  return mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    offset.x, offset.y, offset.z, 1.0);
}

void main() {
  mat4 model = b_InstanceMeta[gl_InstanceIndex].model;
  mat4 view = u_View * translation_matrix(-u_CameraPos.xyz);

  v_Pos = model * a_Pos;
  v_Normal = (model * vec4(a_Normal, 0.0)).xyz;
  v_TexId = b_InstanceMeta[gl_InstanceIndex].faceTexIds[a_FaceId];
  v_TexCoord = a_TexCoord;
  v_CameraPos = u_CameraPos;

  gl_Position = u_Proj * view * v_Pos;
}

// vi: ft=c
