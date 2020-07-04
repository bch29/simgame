#version 450

const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;

const float scale = 1.0;
const float half_scale = scale / 2.;

layout(location = 0) in uint a_BlockIndex;
layout(location = 1) in uint a_VertexId;

layout(location = 0) out vec2 v_TexCoord;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec4 v_Pos;
layout(location = 3) out uint v_BlockType;
layout(location = 4) out vec3 v_CameraPos;

struct CubeFace {
  vec4 normal;
  uint[8] indices;
  vec4[4] vertexLocs;
  vec2[4] vertexTexCoords;
};

struct ChunkMetadata {
  vec4 offset;
  int[6] neighborIndices;
  bool isActive;
  int padding;
};

layout(set = 0, binding = 0) uniform Locals {
  mat4 u_Projection;
  mat4 u_View;
  mat4 u_Model;
  vec4 u_CameraPos;
};

layout(set = 0, binding = 1) buffer BufLocals {
  readonly CubeFace[6] b_CubeFaces;
};

layout(set = 0, binding = 2) buffer BlockTypes {
  readonly int[] b_BlockTypes;
};

layout(set = 0, binding = 3) buffer ChunkMetadataArr {
  readonly ChunkMetadata[] b_ChunkMetadata;
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

int blockTypeAtIndex(uint index)
{
  // Block types is really an array of 16-bit uints but glsl treats it as an array of 32-bit
  // uints. This function assumes little-endian.
  int res = b_BlockTypes[index / 2];
  uint shift = 16 * (index % 2);
  return (res >> shift) & 0xFFFF;
}

mat4 translation_matrix(vec3 offset) {
  return mat4(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    offset.x, offset.y, offset.z, 1.0);
}

mat4 fullModelMatrix(ivec4 blockAddr, vec3 chunkOffset) {
  vec3 offset = chunkOffset + scale * (vec3(.5) + blockAddr.xyz);

  mat4 rescale = mat4(
      half_scale, 0.0, 0.0, 0.0,
      0.0, half_scale, 0.0, 0.0,
      0.0, 0.0, half_scale, 0.0,
      0.0, 0.0, 0.0, 1.0);

  return u_Model * translation_matrix(offset.xyz) * rescale;
}

void main() {
  ivec4 blockAddr = decodeBlockIndex(a_BlockIndex);
  ChunkMetadata chunkMeta = b_ChunkMetadata[blockAddr.w];

  uint faceId = a_VertexId / 6;
  uint faceVertexId = a_VertexId % 6;
  
  CubeFace face = b_CubeFaces[faceId];
  vec4 a_Normal = face.normal;
  vec4 a_Pos = face.vertexLocs[faceVertexId];
  vec2 a_TexCoord = face.vertexTexCoords[faceVertexId];

  mat4 view = u_View * translation_matrix(-u_CameraPos.xyz);
  mat4 model = fullModelMatrix(blockAddr, chunkMeta.offset.xyz);

  v_CameraPos = u_CameraPos.xyz;
  v_TexCoord = a_TexCoord;
  v_Pos = model * a_Pos;
  v_Normal = (u_Model * a_Normal).xyz;

  gl_Position = u_Projection * view * v_Pos;
  v_BlockType = blockTypeAtIndex(a_BlockIndex);
}

// vi: ft=c
