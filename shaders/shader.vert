#version 450

/* #define CHUNK_SIZE_X 16 */
/* #define CHUNK_SIZE_Y 16 */
/* #define CHUNK_SIZE_Z 16 */
#define MAX_CHUNKS 1024
const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;

const float scale = 1.0;
const float half_scale = scale / 2.;

struct ChunkMetadata {
  vec4 offset;
  /*
   * Each element of neighborIndices describes the index of the neighbor chunk in one of the 6
   * axis-aligned directions. The value will be -1 if there is no neighbor chunk in that
   * direction.
   *
   * 0 is negative x
   * 1 is positive x
   * 2 is negative y
   * 3 is positive y
   * 4 is negative z
   * 5 is positive z
   */
  int[6] neighborIndices;

  // Padding ensures alignment to a multiple of 16 bytes
  int[2] padding;
};


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

layout(set = 0, binding = 2) buffer ChunkMetadataArr {
  ChunkMetadata[] b_ChunkMetadata;
};


/* 
 * w component is chunk index
 */
ivec4 decodeBlockIndex(uint index)
{
  uint insideChunk = index % CHUNK_SIZE_XYZ;
  uint chunkIndex = index / CHUNK_SIZE_XYZ;

  uint block_xy = insideChunk  % CHUNK_SIZE_XY;
  uint block_z = insideChunk / CHUNK_SIZE_XY;
  uint block_y = block_xy / CHUNK_SIZE_Y;
  uint block_x = block_xy % CHUNK_SIZE_X;

  return ivec4(block_x, block_y, block_z, chunkIndex);
}

uint encodeBlockIndex(uvec4 pos)
{
  return pos.x + pos.y * CHUNK_SIZE_X + pos.z * CHUNK_SIZE_XY + pos.w * CHUNK_SIZE_XYZ;
}

/*
 * Calculates the position of the neighboring block, given a block position and a direction.  It
 * is assumed that the direction will be axis-aligned.  If the w component of the return value is
 * -1 then there is no neighbor.
 */
ivec4 neighborBlockPos(ivec4 blockPos, vec3 direction)
{
  ivec3 idirection = ivec3(sign(direction.x), sign(direction.y), sign(direction.z));
  ivec3 posInChunk = blockPos.xyz + idirection;

  ivec3 outsideDir = ivec3(
      int(posInChunk.x >= CHUNK_SIZE_X) - int(posInChunk.x < 0),
      int(posInChunk.y >= CHUNK_SIZE_Y) - int(posInChunk.y < 0),
      int(posInChunk.z >= CHUNK_SIZE_Z) - int(posInChunk.z < 0));

  int outsideDirIndex = 
    (0 * int(outsideDir.x == -1)
     + 1 * int(outsideDir.x == 1)
     + 2 * int(outsideDir.y == -1)
     + 3 * int(outsideDir.y == 1)
     + 4 * int(outsideDir.z == -1)
     + 5 * int(outsideDir.z == 1));

  bool isOutside = outsideDir != ivec3(0, 0, 0);

  int neighborChunkIndex =
    int(isOutside) * b_ChunkMetadata[blockPos.w].neighborIndices[outsideDirIndex]
    + int(!isOutside) * blockPos.w;

  posInChunk -= ivec3(CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z) * outsideDir;

  return ivec4(posInChunk.xyz, neighborChunkIndex);
}

int blockTypeAtIndex(uint index)
{
  // Block types is really an array of 16-bit uints but glsl treats it as an array of 32-bit
  // uints. This function assumes little-endian.
  int res = b_BlockTypes[index / 2];
  uint shift = 16 * (index % 2);
  return (res >> shift) & 0xFFFF;
}

int blockTypeAtPos(ivec4 blockPos)
{
  int mult = 1;
  mult *= int(!(blockPos.x < 0 || blockPos.x >= CHUNK_SIZE_X));
  mult *= int(!(blockPos.y < 0 || blockPos.y >= CHUNK_SIZE_Y));
  mult *= int(!(blockPos.z < 0 || blockPos.z >= CHUNK_SIZE_Z));
  mult *= int(!(blockPos.w < 0 || blockPos.w >= MAX_CHUNKS));

  uvec4 blockPosU = uvec4(ivec4(
      min(max(0, blockPos.x), CHUNK_SIZE_X),
      min(max(0, blockPos.y), CHUNK_SIZE_Y),
      min(max(0, blockPos.z), CHUNK_SIZE_Z),
      min(max(0, blockPos.w), MAX_CHUNKS)));

  return mult * blockTypeAtIndex(encodeBlockIndex(blockPosU));
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

  ivec4 blockPos = decodeBlockIndex(gl_InstanceIndex);
  vec3 chunkOffset = b_ChunkMetadata[blockPos.w].offset.xyz;

  ivec4 neighborPos = neighborBlockPos(blockPos, a_Normal);

  int thisBlockType = blockTypeAtIndex(gl_InstanceIndex);
  int neighborBlockType = blockTypeAtPos(neighborPos);

  bool visible = thisBlockType != 0 && neighborBlockType == 0;

  vec3 offset = chunkOffset + scale * (vec3(.5) + blockPos.xyz);

  mat4 rescale = mat4(
      half_scale, 0.0, 0.0, 0.0,
      0.0, half_scale, 0.0, 0.0,
      0.0, 0.0, half_scale, 0.0,
      0.0, 0.0, 0.0, 1.0);

  mat4 view = u_View * translation_matrix(-u_CameraPos);
  mat4 model = u_Model * translation_matrix(offset) * rescale;

  v_TexCoord = a_TexCoord;
  v_Pos = model * a_Pos;
  v_Normal = (model * vec4(a_Normal, 0.0)).xyz;

  gl_Position = u_Projection * view * v_Pos;
  v_BlockType = int(visible) * thisBlockType;

  // If the block is invisble, hide the vertex by setting w component to 0
  gl_Position.w *= float(visible);
}

// vi: ft=c
