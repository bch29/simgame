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

const vec4[6] faceNormals = vec4[](
    vec4(0., 0., 1., 0.),
    vec4(0., 0., -1., 0.),
    vec4(1., 0., 0., 0.),
    vec4(-1., 0., 0., 0.),
    vec4(0., 1., 0., 0.),
    vec4(0., -1., 0., 0.)
);

layout(local_size_x = CHUNK_SIZE_X, local_size_y = CHUNK_SIZE_Y, local_size_z = CHUNK_SIZE_Z) in;

struct ChunkMetadata {
  ivec4 offset;
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

struct BlockRenderInfo {
  uint[8] faceTexIds;
  CubeFace[6] cube;
};

struct BlockInfo {
  ivec4 addr;
  uint visibleFaceCount;
  bool[6] visibleFaces;
};

layout(set = 0, binding = 0) readonly buffer Locals {
  vec4 u_VisibleBoxOrigin;
  vec4 u_VisibleBoxLimit;
};

/* layout(set = 0, binding = 1) readonly buffer BlockRenderInfoBuf { */
/*   BlockRenderInfo[] b_BlockRenderInfo; */
/* }; */

layout(set = 0, binding = 2) readonly buffer BlockTypes {
  int[] b_BlockTypes;
};

layout(set = 0, binding = 3) readonly buffer ChunkMetadataArr {
  ChunkMetadata[] b_ChunkMetadata;
};

layout(set = 0, binding = 4) buffer OutputFacePairs {
  uint[] c_OutputFacePairs;
};

layout(set = 0, binding = 5) buffer IndirectCommands {
  IndirectCommand[] c_IndirectCommands;
};

layout(set = 0, binding = 6) buffer FaceCounts {
  uint g_TotalFaceCount;
  /* uint g_WorkGroupsDone; */
};

// keeps track of the number of faces produced by this work group so far
shared uint s_LocalOutputCount;

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

uint encodeFacePair(ivec4 blockAddr, int faceId0, int faceId1) {
  // leave out chunk id as because that comes from base instance
  uint blockIndex = encodeBlockIndex(uvec4(blockAddr.xyz, 0));

  uint faceBit0;
  uint faceBit1;

  if (faceId0 == -1)
    faceBit0 = 0;
  else
    faceBit0 = (1 << faceId0);

  if (faceId1 == -1)
    faceBit1 = 0;
  else
    faceBit1 = (1 << faceId1);

  uint enabledFaces = faceBit0 | faceBit1;
  return (enabledFaces << 26) + blockIndex;
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
  ivec3 blockPos = blockAddr.xyz + b_ChunkMetadata[blockAddr.w].offset.xyz;

  bool inVisibleBox = 
    blockPos.x >= u_VisibleBoxOrigin.x && blockPos.x <= u_VisibleBoxLimit.x
    && blockPos.y >= u_VisibleBoxOrigin.y && blockPos.y < u_VisibleBoxLimit.y
    && blockPos.z >= u_VisibleBoxOrigin.z && blockPos.z < u_VisibleBoxLimit.z;

  bool activeChunk = b_ChunkMetadata[blockAddr.w].isActive;

  return activeChunk && blockType != 0 && inVisibleBox;
}

BlockInfo getBlockInfo(ivec4 blockAddr) {
  BlockInfo res;
  res.addr = blockAddr;
  res.visibleFaceCount = 0;
  int blockType = getBlockType(blockAddr);

  if (!isBlockVisible(blockAddr, blockType))
    return res;

  for (uint faceIx = 0; faceIx < 6; faceIx++) {
    vec4 faceNormal = faceNormals[faceIx];

    ivec4 neighborAddr = neighborBlockAddr(blockAddr, faceNormal.xyz);
    int neighborBlockType = getBlockType(neighborAddr);

    // do not render a face if its neighbor covers it completely.
    bool faceVisible = !isBlockVisible(neighborAddr, neighborBlockType);
    if (faceVisible) {
      res.visibleFaceCount += 1;
    }

    res.visibleFaces[faceIx] = faceVisible;
  }

  return res;
}


void pushFaces(in BlockInfo info, uint startChunkIx_g, uint pairIx_l) {
  int prevFaceId = -1;

  for (int faceId = 0; faceId < 6; faceId++) {
    if (!info.visibleFaces[faceId])
      continue;

    // wait until we have two faces to encode
    if (prevFaceId == -1) {
      prevFaceId = faceId;
      continue;
    }

    c_OutputFacePairs[startChunkIx_g + pairIx_l] =
      encodeFacePair(info.addr, prevFaceId, faceId);

    prevFaceId = -1;
    pairIx_l += 1;
  }

  // push an extra partial pair if there were an odd number of visible faces
  if (prevFaceId != -1) {
    c_OutputFacePairs[startChunkIx_g + pairIx_l] =
      encodeFacePair(info.addr, prevFaceId, -1);
  }
}

void pushDebugCommands2() {
  BlockInfo info;

  uint startChunkIx_g = 3 * CHUNK_SIZE_XYZ * info.addr.w;
  uint pairIx_l = 0;

  for (uint z = 0; z < 4; z++) {
    for (uint y = 0; y < 16; y++) {
      for (uint x = 0; x < 16; x++) {
        info = getBlockInfo(ivec4(x, y, z, gl_WorkGroupID.x));
        pushFaces(info, startChunkIx_g, pairIx_l);
        pairIx_l += (1 + info.visibleFaceCount) / 2;
      }
    }
  }

  IndirectCommand command;
  command.count = 12; // 12 vertices per pair of faces
  command.instanceCount = pairIx_l;
  command.first = 12 * gl_WorkGroupID.x; // (gl_VertexID / 12) encodes chunk index in vertex shader
  command.baseInstance = startChunkIx_g;
  c_IndirectCommands[gl_WorkGroupID.x] = command;
}

void pushDebugCommands() {
  IndirectCommand command;
  uint cmdIx = 0;
  uint pairIx;

  pairIx = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 0, 0), 0, 1);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 1, 0, 0), 2, 3);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(1, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 1, 0), 4, -1);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(2, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(3, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(4, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(4, 0, 0, 0), 1, 3);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(5, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(6, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(7, 0, 0, 0), 4, 5);
  pairIx += 1;

  command.count = 12;
  command.instanceCount = pairIx - 3 * CHUNK_SIZE_XYZ * cmdIx;
  command.first = 12 * cmdIx;
  command.baseInstance = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_IndirectCommands[cmdIx] = command;
  cmdIx += 1;

  pairIx = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 0, 0), 0, 1);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 1, 0, 0), 2, 3);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(1, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 1, 0), 4, 5);
  pairIx += 1;

  command.count = 12;
  command.instanceCount = pairIx - 3 * CHUNK_SIZE_XYZ * cmdIx;
  command.first = 12 * cmdIx;
  command.baseInstance = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_IndirectCommands[cmdIx] = command;
  cmdIx += 1;

  pairIx = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 0, 0), 0, 1);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 1, 0, 0), 2, 3);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(1, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 1, 0), 4, 5);
  pairIx += 1;

  command.count = 12;
  command.instanceCount = pairIx - 3 * CHUNK_SIZE_XYZ * cmdIx;
  command.first = 12 * cmdIx;
  command.baseInstance = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_IndirectCommands[cmdIx] = command;
  cmdIx += 1;

  pairIx = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 0, 0), 0, 1);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 1, 0, 0), 2, 3);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(1, 0, 0, 0), 4, 5);
  pairIx += 1;
  c_OutputFacePairs[pairIx] = encodeFacePair(ivec4(0, 0, 1, 0), 4, 5);
  pairIx += 1;

  command.count = 12;
  command.instanceCount = pairIx - 3 * CHUNK_SIZE_XYZ * cmdIx;
  command.first = 12 * cmdIx;
  command.baseInstance = 3 * CHUNK_SIZE_XYZ * cmdIx;
  c_IndirectCommands[cmdIx] = command;
  cmdIx += 1;
}

void main() {
  // gl_LocalInvocationID is block pos within chunk, gl_WorkGroupID.x is chunk index
  ivec4 blockAddr = ivec4(gl_LocalInvocationID, gl_WorkGroupID.x);
  BlockInfo info = getBlockInfo(blockAddr);
  // round up because e.g. 5 faces need 2 full pairs and an incomplete pair
  uint visiblePairCount = (1 + info.visibleFaceCount) / 2;

  s_LocalOutputCount = 0;
  barrier(); // ensure s_LocalOutputCount is set to 0 in all invocations in the work group
  memoryBarrierShared();
  uint pairIx_l = atomicAdd(s_LocalOutputCount, visiblePairCount);

  uint startChunkIx_g = 3 * CHUNK_SIZE_XYZ * info.addr.w;

  if (visiblePairCount != 0)
    pushFaces(info, startChunkIx_g, pairIx_l);

  barrier(); // wait for s_LocalOutputCount to get its final value
  memoryBarrierShared();

  IndirectCommand command;
  command.count = 12; // 12 vertices per pair of faces
  command.instanceCount = s_LocalOutputCount;
  command.first = 12 * info.addr.w; // (gl_VertexID / 12) encodes chunk index in vertex shader
  command.baseInstance = startChunkIx_g;
  c_IndirectCommands[info.addr.w] = command;

  // ensure buffer writes are flushed
  memoryBarrierBuffer();
}

// vi: ft=c
