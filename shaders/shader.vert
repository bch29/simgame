#version 450

layout(location = 0) in vec4 a_Pos;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;
layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec4 v_Pos;
layout(location = 3) out uint v_BlockType;
layout(location = 4) out vec3 v_CameraPos;

layout(set = 0, binding = 0) uniform Locals {
  mat4 u_Projection;
  mat4 u_View;
  mat4 u_Model;
  vec3 u_CameraPos;
};

layout(set = 0, binding = 1) buffer BlockTypes {
  uint[] b_BlockTypes;
};

const uint CHUNK_SIZE_X = 16;
const uint CHUNK_SIZE_Y = 16;
const uint CHUNK_SIZE_Z = 16;

const float scale = 1.0;
const float scale_over_2 = scale / 2.;

ivec3 decode_block_index(uint index)
{
  uint block_xy = index  % (CHUNK_SIZE_X * CHUNK_SIZE_Y);
  uint block_x = block_xy % CHUNK_SIZE_X;
  uint block_y = block_xy / CHUNK_SIZE_Y;
  uint block_z = index / (CHUNK_SIZE_X * CHUNK_SIZE_Y);
  return ivec3(block_x, block_y, block_z);
}

uint encode_block_index(uvec3 pos)
{
  return pos.x + pos.y * CHUNK_SIZE_X + pos.z * CHUNK_SIZE_X * CHUNK_SIZE_X;
}

uint block_type_at_index(uint index)
{
  // Block types is really an array of 16-bit uints but glsl treats it as an array of 32-bit uints
  uint res = b_BlockTypes[index / 2];
  if (index % 2 == 1)
  {
    res >>= 2;
  }
  return res & 0xFFFF;
}

uint block_type_at_pos(ivec3 block_pos)
{
  if (block_pos.x < 0 || block_pos.x >= CHUNK_SIZE_X)
    return 0;

  if (block_pos.y < 0 || block_pos.y >= CHUNK_SIZE_Y)
    return 0;

  if (block_pos.z < 0 || block_pos.z >= CHUNK_SIZE_Z)
    return 0;

  return block_type_at_index(encode_block_index(uvec3(block_pos)));
}

mat4 translation_matrix(vec3 offset) {
  return mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    offset.x, offset.y, offset.z, 1.0);
}

void main() {
  v_CameraPos = u_CameraPos;

  ivec3 block_pos = decode_block_index(gl_InstanceIndex);
  ivec3 inormal = ivec3(sign(a_Normal.x), sign(a_Normal.y), sign(a_Normal.z));
  ivec3 neighbour_pos = block_pos + inormal;

  uint this_block_type = block_type_at_index(gl_InstanceIndex);
  uint neighbour_block_type = block_type_at_pos(neighbour_pos);

  bool visible = this_block_type != 0 && neighbour_block_type == 0;
  /* bool visible = this_block_type != 0 && ( */
  /*     v_Normal.x > 0 || v_Normal.y > 0 || v_Normal.z > 0); */

  if (visible)
  {
    vec3 offset = 
      scale * (vec3(.5) + block_pos);
      /* - vec3(scale_over_2 * CHUNK_SIZE_X); */

    mat4 rescale = mat4(
        scale_over_2, 0.0, 0.0, 0.0,
        0.0, scale_over_2, 0.0, 0.0,
        0.0, 0.0, scale_over_2, 0.0,
        0.0, 0.0, 0.0, 1.0);

    mat4 view = u_View * translation_matrix(-u_CameraPos);
    mat4 model = u_Model * translation_matrix(offset) * rescale;

    v_TexCoord = a_TexCoord;
    v_Pos = model * a_Pos;
    v_Normal = (model * vec4(a_Normal, 0.0)).xyz;

    gl_Position = u_Projection * view * v_Pos;
    v_BlockType = this_block_type;
  }
  else
  {
    gl_Position = vec4(0);
    v_BlockType = 0;
  }
}

// vi: ft=c
