#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec4 v_Pos;
layout(location = 3) out uint v_BlockType;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_Projection;
    mat4 u_View;
    mat4 u_Model;
};

layout(set = 0, binding = 1) buffer BlockTypes {
  uint[] b_BlockTypes;
};

const uint CHUNK_SIZE_X = 16;
const uint CHUNK_SIZE_Y = 16;
const uint CHUNK_SIZE_Z = 16;

void main() {
  const float scale = 0.15;

  uint block_xy = gl_InstanceIndex  % (CHUNK_SIZE_X * CHUNK_SIZE_Y);
  uint block_x = block_xy % CHUNK_SIZE_X;
  uint block_y = block_xy / CHUNK_SIZE_Y;
  uint block_z = gl_InstanceIndex / (CHUNK_SIZE_X * CHUNK_SIZE_Y);
  vec3 offset = 
    scale * vec3(block_x, block_y, block_z) 
    + vec3(.5 - scale * CHUNK_SIZE_X * 0.5);

  mat4 translation = mat4(
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      offset.x, offset.y, offset.z, 1.0);

  mat4 rescale = mat4(
      0.5 * scale, 0.0, 0.0, 0.0,
      0.0, 0.5 * scale, 0.0, 0.0,
      0.0, 0.0, 0.5 * scale, 0.0,
      0.0, 0.0, 0.0, 1.0);

  mat4 model = u_Model * translation * rescale;

  v_TexCoord = a_TexCoord;
  v_Pos = u_Projection * u_View * model * a_Pos;
  v_Normal = (model * vec4(a_Normal, 0.0)).xyz;

  // Block types is really an array of 16-bit uints but glsl treats it as an array of 32-bit uints
  uint blockTypeBase = b_BlockTypes[gl_InstanceIndex / 2];
  if (gl_InstanceIndex % 2 == 1)
  {
    blockTypeBase >>= 2;
  }
  v_BlockType = blockTypeBase & 0xFFFF;

  if (v_BlockType == 0)
  {
    gl_Position = vec4(0);
  }
  else
  {
    gl_Position = v_Pos;
  }
}

// vi: ft=c
