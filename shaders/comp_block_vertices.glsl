#version 450

/*
 * This compute shader takes chunk block data and metadata as input, and produces a buffer of
 * vertices representing the visible faces of cubes in the chunk.
 */

const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;
const uint MAX_UINT = 4294967295;

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
  uint blockIndex;

  // identifies where on the cube the vertex lies:
  // faceId * 4 + faceVertexId
  uint vertexId;
};

struct IndirectCommand {
  uint count;
  uint instanceCount;
  uint first;
  uint baseInstance;
};

struct CubeFace {
  vec4 normal;
  uint[8] indices;
  vec4[4] vertexLocs;
  vec2[4] vertexTexCoords;
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
  Vertex[] c_OutputVertices;
};

layout(set = 0, binding = 5) buffer IndirectCommands {
  IndirectCommand[] c_IndirectCommands;
};

layout(set = 0, binding = 6) buffer FaceCounts {
  uint g_TotalFaceCount;
  /* uint g_WorkGroupsDone; */
};

// keeps track of the number of faces produced by this work group so far
shared uint s_LocalFaceCount;

uint encodeBlockIndex(uvec4 pos)
{
  if (pos.x >= CHUNK_SIZE_X)
    return 0;

  if (pos.y >= CHUNK_SIZE_Y)
    return 0;

  if (pos.z >= CHUNK_SIZE_Z)
    return 0;

  if (pos.w >= MAX_UINT / CHUNK_SIZE_XYZ)
    return 0;

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

bool isBlockVisible(ivec4 blockAddr, int blockType)
{
  vec3 blockPos = vec3(blockAddr.xyz) + b_ChunkMetadata[blockAddr.w].offset.xyz;

  bool inVisibleBox = 
    blockPos.x >= u_VisibleBoxOrigin.x && blockPos.x <= u_VisibleBoxLimit.x
    && blockPos.y >= u_VisibleBoxOrigin.y && blockPos.y < u_VisibleBoxLimit.y
    && blockPos.z >= u_VisibleBoxOrigin.z && blockPos.z < u_VisibleBoxLimit.z;

  bool activeChunk = b_ChunkMetadata[blockAddr.w].isActive;

  return activeChunk && blockType != 0 && inVisibleBox;
}

struct BlockInfo {
  uint visibleFaceCount;
  ivec4 addr;
  int blockType;
  bool[6] visibleFaces;
  ivec4[6] neighborAddrs;
};


BlockInfo getBlockInfo(ivec4 blockAddr) {
  BlockInfo res;
  res.addr = blockAddr;
  res.visibleFaceCount = 0;
  res.blockType = getBlockType(blockAddr);

  if (!isBlockVisible(blockAddr, res.blockType))
    return res;

  for (uint faceIx = 0; faceIx < 6; faceIx++) {
    CubeFace face = u_CubeFaces[faceIx];
    ivec4 neighborAddr = neighborBlockAddr(blockAddr, face.normal.xyz);
    int neighborBlockType = getBlockType(neighborAddr);

    // No point in rendering a face if it is invisible or its neighbor covers it completely.
    bool faceVisible = !isBlockVisible(neighborAddr, neighborBlockType);
    if (faceVisible) {
      res.neighborAddrs[faceIx] = neighborAddr;
      res.visibleFaceCount += 1;
    }

    res.visibleFaces[faceIx] = faceVisible;
  }

  return res;
}


void pushVertices(in BlockInfo info, uint startChunkIx_g, uint faceIx_l) {
  ChunkMetadata chunkMeta = b_ChunkMetadata[info.addr.w];

  for (uint faceId = 0; faceId < 6; faceId++) {
    if (!info.visibleFaces[faceId])
      continue;

    CubeFace face = u_CubeFaces[faceId];
    ivec4 neighborAddr = info.neighborAddrs[faceId];

    uint faceIx_g = startChunkIx_g + faceIx_l;
    uint outVertexStart = 6 * faceIx_g;

    for (uint faceIndexIx = 0; faceIndexIx < 6; faceIndexIx++) {
      uint faceVertexId = face.indices[faceIndexIx];
      Vertex vert;
      vert.blockIndex = encodeBlockIndex(uvec4(info.addr));
      vert.vertexId = faceId * 6 + faceVertexId;
      c_OutputVertices[outVertexStart + faceIndexIx] = vert;
    }

    faceIx_l += 1;
  }
}

void main() {
  // gl_LocalInvocationID is block pos within chunk, gl_WorkGroupID.x is chunk index
  ivec4 blockAddr = ivec4(gl_LocalInvocationID, gl_WorkGroupID.x);

  s_LocalFaceCount = 0;
  barrier(); // ensure s_LocalFaceCount is set to 0 in all invocations in the work group
  memoryBarrierShared();

  BlockInfo info = getBlockInfo(blockAddr);
  uint faceIx_l = atomicAdd(s_LocalFaceCount, info.visibleFaceCount);

  barrier(); // wait for s_LocalFaceCount to get its final value
  memoryBarrierShared();

  /* uint startChunkIx_g = atomicAdd(g_TotalFaceCount, s_LocalFaceCount); */
  uint startChunkIx_g = info.addr.w * CHUNK_SIZE_XYZ;

  IndirectCommand command;
  command.count = 6 * s_LocalFaceCount;
  command.instanceCount = 1;
  command.first = 6 * startChunkIx_g;
  command.baseInstance = 0;
  c_IndirectCommands[gl_WorkGroupID.x] = command;

  if (info.visibleFaceCount != 0)
    pushVertices(info, startChunkIx_g, faceIx_l);

  // ensure buffer writes are flushed
  memoryBarrierBuffer();
}

// vi: ft=c
