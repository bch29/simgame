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
  int[] b_BlockTypes;
};

layout(set = 0, binding = 2) buffer ChunkOffsets {
  vec4[] b_ChunkOffsets;
};

/* const uint CHUNK_SIZE_X = 16; */
/* const uint CHUNK_SIZE_Y = 16; */
/* const uint CHUNK_SIZE_Z = 16; */
const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;

const float scale = 1.0;
const float scale_over_2 = scale / 2.;

ivec3 decodeBlockIndex(uint index)
{
  uint block_xy = index  % CHUNK_SIZE_XY;
  uint block_x = block_xy % CHUNK_SIZE_X;
  uint block_y = block_xy / CHUNK_SIZE_Y;
  uint block_z = index / CHUNK_SIZE_XY;
  return ivec3(block_x, block_y, block_z);
}

uint encodeBlockIndex(uvec3 pos)
{
  return pos.x + pos.y * CHUNK_SIZE_X + pos.z * CHUNK_SIZE_XY;
}

int blockTypeAtIndex(uint index)
{
  // Block types is really an array of 16-bit uints but glsl treats it as an array of 32-bit
  // uints. This function assumes little-endian.
  int res = b_BlockTypes[index / 2];
  if (index % 2 == 0)
  {
    return res & 0xFFFF;
  }
  else
  {
    return (res >> 16) & 0xFFFF;
  }
}

int blockTypeAtPos(ivec3 blockPos)
{
  if (blockPos.x < 0 || blockPos.x >= CHUNK_SIZE_X)
    return 0;

  if (blockPos.y < 0 || blockPos.y >= CHUNK_SIZE_Y)
    return 0;

  if (blockPos.z < 0 || blockPos.z >= CHUNK_SIZE_Z)
    return 0;

  return blockTypeAtIndex(encodeBlockIndex(uvec3(blockPos)));
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

  vec3 chunkOffset = b_ChunkOffsets[gl_InstanceIndex / CHUNK_SIZE_XYZ].xyz;

  ivec3 blockPos = decodeBlockIndex(gl_InstanceIndex);
  ivec3 inormal = ivec3(sign(a_Normal.x), sign(a_Normal.y), sign(a_Normal.z));
  ivec3 neighborPos = blockPos + inormal;

  int thisBlockType = blockTypeAtIndex(gl_InstanceIndex);
  int neighborBlockType = blockTypeAtPos(neighborPos);

  bool visible = thisBlockType != 0 && neighborBlockType == 0;

  if (visible)
  {
    blockPos.x = blockPos.x % CHUNK_SIZE_X;
    blockPos.y = blockPos.y % CHUNK_SIZE_Y;
    blockPos.z = blockPos.z % CHUNK_SIZE_Z;

    vec3 offset = chunkOffset + scale * (vec3(.5) + blockPos);

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
    v_BlockType = thisBlockType;
  }
  else
  {
    gl_Position = vec4(0);
    v_BlockType = 0;
  }
}

// vi: ft=c
