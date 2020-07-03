#version 450

/*
 * This compute shader takes chunk block data and metadata as input, and produces a buffer of
 * vertices representing the visible faces of cubes in the chunk.
 */

const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;

const float scale = 1.0;
const float half_scale = scale / 2.;

layout(local_size_x = CHUNK_SIZE_X, local_size_y = CHUNK_SIZE_Y, local_size_z = CHUNK_SIZE_Z) in;

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

  bool isActive;
  // Padding ensures alignment to a multiple of 16 bytes
  int padding;
};

struct Vertex {
  vec4 pos;
  vec4 normal;
  vec2 texCoord;
  uint blockType;
  float _padding;
};

struct CubeFace {
  vec4 normal;
  uint[8] indices;
  vec4[4] vertexLocs;
  vec2[4] vertexTexCoords;
};

struct IndirectCommand {
  uint count;
  uint instanceCount;
  uint firstIndex;
  uint baseVertex;
  uint baseInstance;
  uint[3] _padding;
};

layout(set = 0, binding = 0) buffer Locals {
  readonly vec4 u_VisibleBoxOrigin;
  readonly vec4 u_VisibleBoxLimit;
  readonly CubeFace[6] u_CubeFaces;
};

layout(set = 0, binding = 1) buffer BlockTypes {
  readonly int[] b_BlockTypes;
};

layout(set = 0, binding = 2) buffer ChunkMetadataArr {
  readonly ChunkMetadata[] b_ChunkMetadata;
};

layout(set = 0, binding = 3) buffer OutputVertices {
  writeonly Vertex[] c_OutputVertices;
};

layout(set = 0, binding = 4) buffer OutputIndices {
  writeonly uint[] c_OutputIndices;
};

layout(set = 0, binding = 5) buffer IndirectCommands {
  writeonly coherent IndirectCommand[] c_IndirectCommands;
};

// keeps track of the number of faces produced by this work group so far
shared uint s_GroupFaceCount;

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
  mult *= int(!(blockAddr.w < 0));

  uvec4 blockAddrU = uvec4(ivec4(
      min(max(0, blockAddr.x), CHUNK_SIZE_X),
      min(max(0, blockAddr.y), CHUNK_SIZE_Y),
      min(max(0, blockAddr.z), CHUNK_SIZE_Z),
      max(0, blockAddr.w)));

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

void pushVertices(ivec4 blockAddr) {
  ChunkMetadata chunkMeta = b_ChunkMetadata[blockAddr.w];

  int thisBlockType = getBlockType(blockAddr);
  vec3 chunkOffset = chunkMeta.offset.xyz;
  vec3 offset = chunkOffset + scale * (vec3(.5) + blockAddr.xyz);

  // do not push any vertices to the buffer for invisible blocks
  if (!isBlockVisible(blockAddr, thisBlockType))
    return;

  mat4 rescale = mat4(
      half_scale, 0.0, 0.0, 0.0,
      0.0, half_scale, 0.0, 0.0,
      0.0, 0.0, half_scale, 0.0,
      0.0, 0.0, 0.0, 1.0);

  mat4 model = translation_matrix(offset.xyz) * rescale;

  uint chunkIndexStart = blockAddr.w * CHUNK_SIZE_XYZ;

  for (uint faceIx = 0; faceIx < 6; faceIx++) {
    CubeFace face = u_CubeFaces[faceIx];
    ivec4 neighborAddr = neighborBlockAddr(blockAddr, face.normal.xyz);
    int neighborBlockType = getBlockType(neighborAddr);

    // No point in rendering a face if its neighbor covers it completely.
    bool faceVisible = !isBlockVisible(neighborAddr, neighborBlockType);
    if (!faceVisible) continue;

    uint groupFaceIx = atomicAdd(s_GroupFaceCount, 1);
    uint outFaceIx = chunkIndexStart + groupFaceIx;
    uint outVertexStart = 4 * outFaceIx;

    for (uint faceVertIx = 0; faceVertIx < 4; faceVertIx++) {
      vec4 vertPos = face.vertexLocs[faceVertIx];
      vec2 texCoord = face.vertexTexCoords[faceVertIx];

      Vertex vert;
      vert.pos = model * vertPos;
      vert.normal = model * face.normal;
      vert.texCoord = texCoord;
      vert.blockType = thisBlockType;
      vert._padding = 0.0;

      c_OutputVertices[outVertexStart + faceVertIx] = vert;
    }

    uint firstIndex = 4 * groupFaceIx;
    uint outIndexStart = 6 * outFaceIx;

    for (uint faceIndexIx = 0; faceIndexIx < 6; faceIndexIx++) {
      uint localIndex = face.indices[faceIndexIx];
      c_OutputIndices[outIndexStart + faceIndexIx] = firstIndex + localIndex;
    }
  }
}

void main() {
  // gl_LocalInvocationID is block pos within chunk, gl_WorkGroupID.x is chunk index
  ivec4 blockAddr = ivec4(gl_LocalInvocationID, gl_WorkGroupID.x);
  ChunkMetadata chunkMeta = b_ChunkMetadata[blockAddr.w];

  // Do not push any vertices to the buffer for inactive chunks. N.B. this applies across the
  // entire work group so we don't have to worry about hitting subsequent memory barriers.
  if (!chunkMeta.isActive)
    return;

  s_GroupFaceCount = 0;
  memoryBarrierShared();

  pushVertices(blockAddr);

  IndirectCommand command;

  memoryBarrier();
  command.count = 6 * s_GroupFaceCount;
  command.instanceCount = 1;
  command.firstIndex = 0;
  command.baseVertex = 0;
  command.baseInstance = 0;
  c_IndirectCommands[gl_WorkGroupID.x] = command;
}

// vi: ft=c
