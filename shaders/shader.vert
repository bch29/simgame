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
  float u_Padding0;
  vec3 u_VisibleBoxOrigin;
  float u_Padding1;
  vec3 u_VisibleBoxLimit;
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
 * A block address is an ivec4 where the w component is the chunk index and the xyz components are
 * the offset relative to the chunk's origin.
 *
 * Calculates the position of the neighboring block, given a block position and a direction.  It
 * is assumed that the direction will be axis-aligned.  If the w component of the return value is
 * -1 then there is no neighbor.
 */
ivec4 neighborBlockAddr(ivec4 blockAddr, vec3 direction)
{
  ivec3 idirection = ivec3(sign(direction.x), sign(direction.y), sign(direction.z));
  ivec3 posInChunk = blockAddr.xyz + idirection;

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
    int(isOutside) * b_ChunkMetadata[blockAddr.w].neighborIndices[outsideDirIndex]
    + int(!isOutside) * blockAddr.w;

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

int getBlockType(ivec4 blockAddr)
{
  int mult = 1;
  mult *= int(!(blockAddr.x < 0 || blockAddr.x >= CHUNK_SIZE_X));
  mult *= int(!(blockAddr.y < 0 || blockAddr.y >= CHUNK_SIZE_Y));
  mult *= int(!(blockAddr.z < 0 || blockAddr.z >= CHUNK_SIZE_Z));
  mult *= int(!(blockAddr.w < 0 || blockAddr.w >= MAX_CHUNKS));

  uvec4 blockAddrU = uvec4(ivec4(
      min(max(0, blockAddr.x), CHUNK_SIZE_X),
      min(max(0, blockAddr.y), CHUNK_SIZE_Y),
      min(max(0, blockAddr.z), CHUNK_SIZE_Z),
      min(max(0, blockAddr.w), MAX_CHUNKS)));

  return mult * blockTypeAtIndex(encodeBlockIndex(blockAddrU));
}

mat4 translation_matrix(vec3 offset) {
  return mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    offset.x, offset.y, offset.z, 1.0);
}

bool isBlockVisible(ivec4 blockAddr, int blockType)
{
  vec3 blockPos = vec3(blockAddr.xyz) + b_ChunkMetadata[blockAddr.w].offset.xyz;

  bool inVisibleBox = 
    blockPos.x >= u_VisibleBoxOrigin.x && blockPos.x <= u_VisibleBoxLimit.x
    && blockPos.y >= u_VisibleBoxOrigin.y && blockPos.y < u_VisibleBoxLimit.y
    && blockPos.z >= u_VisibleBoxOrigin.z && blockPos.z < u_VisibleBoxLimit.z;

  return blockType != 0 && inVisibleBox;
}

void main() {
  v_CameraPos = u_CameraPos;

  ivec4 blockAddr = decodeBlockIndex(gl_InstanceIndex);
  vec3 chunkOffset = b_ChunkMetadata[blockAddr.w].offset.xyz;

  ivec4 neighborAddr = neighborBlockAddr(blockAddr, a_Normal);

  int thisBlockType = blockTypeAtIndex(gl_InstanceIndex);
  int neighborBlockType = getBlockType(neighborAddr);

  // No point in rendering a face if its neighbor covers it completely.
  bool visible = isBlockVisible(blockAddr, thisBlockType)
    && !isBlockVisible(neighborAddr, neighborBlockType);

  vec3 offset = chunkOffset + scale * (vec3(.5) + blockAddr.xyz);

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
