#version 450

/*
 * This compute shader takes chunk voxel data and metadata as input, and produces a buffer of
 * vertices representing the visible faces of cubes in the chunk.
 */

const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;
const uint MAX_UINT = 0xFFFFFFFF;

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

struct VoxelRenderInfo {
  uint[8] faceTexIds;
  CubeFace[6] cube;
};

struct VoxelInfo {
  ivec4 addr;
  uint visibleFaceCount;
  bool[6] visibleFaces;
  int voxelType;
};

struct ComputeCommand {
  uint chunkMetaIndex;
  uint vertexDataStart;
};

layout(set = 0, binding = 0) readonly buffer Locals {
  vec4 u_VisibleBoxOrigin;
  vec4 u_VisibleBoxLimit;
};

layout(set = 0, binding = 1) readonly buffer VoxelTypes {
  int[] b_VoxelTypes;
};

layout(set = 0, binding = 2) readonly buffer ChunkMetadataArr {
  ChunkMetadata[] b_ChunkMetadata;
};

layout(set = 0, binding = 3) buffer OutputFacePairs {
  uint[] c_OutputFacePairs;
};

layout(set = 0, binding = 4) buffer IndirectCommands {
  IndirectCommand[] c_IndirectCommands;
};

layout(set = 0, binding = 5) buffer FaceCounts {
  uint g_TotalFaceCount;
  /* uint g_WorkGroupsDone; */
};

layout(set = 0, binding = 6) buffer ComputeCommands {
  ComputeCommand[] b_ComputeCommands;
};

// keeps track of the number of faces produced by this work group so far
shared uint s_LocalOutputCount;

uint encodeVoxelIndex(uvec4 pos)
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

uint encodeFacePair(int voxelType, ivec4 voxelAddr, int faceId0, int faceId1) {
  // leave out chunk id as because that comes from base instance
  uint voxelIndex = encodeVoxelIndex(uvec4(voxelAddr.xyz, 0));

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

  return ((voxelType & 0xFFFF) << 16)
    | (enabledFaces << 10)
    | (voxelIndex & 0x3FF);
}

/*
 * A voxel address is an ivec4 where the w component is the chunk index and the xyz components are
 * the offset relative to the chunk's origin.
 *
 * Calculates the position of the neighboring voxel, given a voxel position and a direction.  It
 * is assumed that the direction will be axis-aligned.  If the w component of the return value is
 * -1 then there is no neighbor.
 */
ivec4 neighborVoxelAddr(ivec4 voxelAddr, vec3 direction)
{
  ivec3 idirection = ivec3(sign(direction.x), sign(direction.y), sign(direction.z));
  ivec3 posInChunk = voxelAddr.xyz + idirection;

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
    int(isOutside) * b_ChunkMetadata[voxelAddr.w].neighborIndices[outsideDirIndex]
    + int(!isOutside) * voxelAddr.w;

  posInChunk -= ivec3(CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z) * outsideDir;

  return ivec4(posInChunk.xyz, neighborChunkIndex);
}

int voxelTypeAtIndex(uint index)
{
  // Voxel types is really an array of 16-bit uints but glsl treats it as an array of 32-bit
  // uints. This function assumes little-endian.
  int res = b_VoxelTypes[index / 2];
  uint shift = 16 * (index % 2);
  return (res >> shift) & 0xFFFF;
}

int getVoxelType(ivec4 voxelAddr)
{
  int mult = 1;
  mult *= int(!(voxelAddr.x < 0 || voxelAddr.x >= CHUNK_SIZE_X));
  mult *= int(!(voxelAddr.y < 0 || voxelAddr.y >= CHUNK_SIZE_Y));
  mult *= int(!(voxelAddr.z < 0 || voxelAddr.z >= CHUNK_SIZE_Z));
  mult *= int(!(voxelAddr.w < 0));

  uvec4 voxelAddrU = uvec4(ivec4(
      min(max(0, voxelAddr.x), CHUNK_SIZE_X),
      min(max(0, voxelAddr.y), CHUNK_SIZE_Y),
      min(max(0, voxelAddr.z), CHUNK_SIZE_Z),
      max(0, voxelAddr.w)));

  return mult * voxelTypeAtIndex(encodeVoxelIndex(voxelAddrU));
}

bool isVoxelVisible(ivec4 voxelAddr, int voxelType)
{
  ivec3 voxelPos = voxelAddr.xyz + b_ChunkMetadata[voxelAddr.w].offset.xyz;

  bool inVisibleSlice = 
    voxelPos.z >= u_VisibleBoxOrigin.z && voxelPos.z < u_VisibleBoxLimit.z;

  bool activeChunk = b_ChunkMetadata[voxelAddr.w].isActive;

  return activeChunk && voxelType != 0 && inVisibleSlice;
}

VoxelInfo getVoxelInfo(ivec4 voxelAddr, uint chunkDataIndex) {
  VoxelInfo res;
  res.addr = voxelAddr;
  res.visibleFaceCount = 0;
  res.voxelType = getVoxelType(ivec4(voxelAddr.xyz, chunkDataIndex));

  if (!isVoxelVisible(voxelAddr, res.voxelType))
    return res;

  for (uint faceIx = 0; faceIx < 6; faceIx++) {
    vec4 faceNormal = faceNormals[faceIx];

    ivec4 neighborAddr = neighborVoxelAddr(voxelAddr, faceNormal.xyz);
    int neighborVoxelType = getVoxelType(neighborAddr);

    // do not render a face if its neighbor covers it completely.
    bool faceVisible = !isVoxelVisible(neighborAddr, neighborVoxelType);
    if (faceVisible) {
      res.visibleFaceCount += 1;
    }

    res.visibleFaces[faceIx] = faceVisible;
  }

  return res;
}


void pushFaces(in VoxelInfo info, uint startChunkIx_g, uint pairIx_l) {
  int prevFaceId = -1;

  for (int faceId = 0; faceId < 6; faceId++) {
    if (!info.visibleFaces[faceId])
      continue;

    // wait until we have two faces to encode
    if (prevFaceId == -1) {
      prevFaceId = faceId;
      continue;
    }

    uint outputIx = startChunkIx_g + pairIx_l;
    c_OutputFacePairs[outputIx] =
      encodeFacePair(info.voxelType, info.addr, prevFaceId, faceId);

    prevFaceId = -1;
    pairIx_l += 1;
  }

  // push an extra partial pair if there were an odd number of visible faces
  if (prevFaceId != -1) {
    uint outputIx = startChunkIx_g + pairIx_l;
    c_OutputFacePairs[outputIx] =
      encodeFacePair(info.voxelType, info.addr, prevFaceId, -1);
  }
}

void main() {
  uint chunkDataIndex = gl_WorkGroupID.x;
  ComputeCommand computeCommand = b_ComputeCommands[chunkDataIndex];

  // gl_LocalInvocationID is voxel pos within chunk
  ivec4 voxelAddr = ivec4(gl_LocalInvocationID, computeCommand.chunkMetaIndex);
  VoxelInfo info = getVoxelInfo(voxelAddr, chunkDataIndex);
  // round up because e.g. 5 faces need 2 full pairs and an incomplete pair
  uint visiblePairCount = (1 + info.visibleFaceCount) / 2;

  s_LocalOutputCount = 0;
  barrier(); // ensure s_LocalOutputCount is set to 0 in all invocations in the work group
  memoryBarrierShared();
  uint pairIx_l = atomicAdd(s_LocalOutputCount, visiblePairCount);

  if (visiblePairCount != 0)
    pushFaces(info, computeCommand.vertexDataStart, pairIx_l);

  barrier(); // wait for s_LocalOutputCount to get its final value
  memoryBarrierShared();

  IndirectCommand command;
  command.count = 12; // 12 vertices per pair of faces
  command.instanceCount = s_LocalOutputCount;
  command.first = 12 * info.addr.w; // (gl_VertexID / 12) encodes chunk index in vertex shader
  command.baseInstance = computeCommand.vertexDataStart;
  c_IndirectCommands[info.addr.w] = command;

  // ensure buffer writes are flushed
  memoryBarrierBuffer();
}

// vi: ft=c
