#version 450

const uint CHUNK_SIZE_XY = CHUNK_SIZE_X * CHUNK_SIZE_Y;
const uint CHUNK_SIZE_XYZ = CHUNK_SIZE_XY * CHUNK_SIZE_Z;

const float scale = 1.0;
const float half_scale = scale / 2.;

layout(location = 0) out vec4 v_Pos;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out uint v_TexId;
layout(location = 3) out vec2 v_TexCoord;
layout(location = 4) out vec3 v_CameraPos;

struct Attributes {
  // mesh data
  vec4 pos;
  vec4 normal;
  vec2 texCoord;

  // chunk/voxel data
  ivec4 voxelAddr;
  vec3 chunkOffset;
  uint voxelType;
  uint texId;
};

struct CubeFace {
  vec4 normal;
  uint[8] indices;
  vec4[4] vertexLocs;
  vec2[4] vertexTexCoords;
} cube;

struct VoxelRenderInfo {
  uint[8] faceTexIds;
  CubeFace[6] cube;
};

struct ChunkMetadata {
  ivec4 offset;
  int[6] neighborIndices;
  bool isActive;
  int padding;
};

struct VoxelTextureMetadata {
  uint x_periodicity;
  uint y_periodicity;
};

layout(set = 0, binding = 0) uniform Locals {
  mat4 u_Projection;
  mat4 u_View;
  mat4 u_Model;
  vec4 u_CameraPos;
};

layout(set = 0, binding = 1) readonly buffer VoxelRenderInfoBuf {
  VoxelRenderInfo[] b_VoxelRenderInfo;
};

layout(set = 0, binding = 2) readonly buffer VoxelTypes {
  int[] b_VoxelTypes;
};

layout(set = 0, binding = 3) readonly buffer ChunkMetadataArr {
  ChunkMetadata[] b_ChunkMetadata;
};

layout(set = 0, binding = 4) readonly buffer FacePairs {
  uint[] b_FacePairs;
};

layout(set = 0, binding = 5) readonly buffer VoxelTextureMetadataBuf {
  VoxelTextureMetadata[] b_VoxelTextureMetadata;
};

/* 
 * w component is chunk index
 */
ivec4 decodeVoxelIndex(uint index)
{
  uint insideChunk = index % CHUNK_SIZE_XYZ;
  uint chunkIndex = index / CHUNK_SIZE_XYZ;

  uint voxel_xy = insideChunk  % CHUNK_SIZE_XY;
  uint voxel_z = insideChunk / CHUNK_SIZE_XY;
  uint voxel_y = voxel_xy / CHUNK_SIZE_Y;
  uint voxel_x = voxel_xy % CHUNK_SIZE_X;

  return ivec4(voxel_x, voxel_y, voxel_z, chunkIndex);
}

int voxelTypeAtIndex(uint index)
{
  // Voxel types is really an array of 16-bit uints but glsl treats it as an array of 32-bit
  // uints. This function assumes little-endian.
  int res = b_VoxelTypes[index / 2];
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

mat4 fullModelMatrix(ivec4 voxelAddr, vec3 chunkOffset) {
  vec3 offset = chunkOffset + scale * (vec3(.5) + voxelAddr.xyz);

  mat4 rescale = mat4(
      half_scale, 0.0, 0.0, 0.0,
      0.0, half_scale, 0.0, 0.0,
      0.0, 0.0, half_scale, 0.0,
      0.0, 0.0, 0.0, 1.0);

  return u_Model * translation_matrix(offset.xyz) * rescale;
}

vec2 signedMod(vec2 v, float divisor) {
  vec2 result;
  if (v.x >= 0) {
    result.x = mod(v.x, divisor);
  } else {
    result.x = divisor - mod(-v.x, divisor);
  }
  if (v.y >= 0) {
    result.y = mod(v.y, divisor);
  } else {
    result.y = divisor - mod(-v.y, divisor);
  }

  return result;
}

/// Calculates which part of the face texture to use.
vec2 getTexOrigin(vec2 periodicity, vec3 voxelPos, vec3 faceNormal) {
  vec3 unitX = vec3(1., 0., 0.);
  vec3 unitY = vec3(0., 1., 0.);
  vec3 unitZ = vec3(0., 0., 1.);

  vec3 unitU; // texture U axis, in terms of world coordinates
  vec3 unitV; // texture V axis, in terms of world coordinates

  vec2 uvOffset = vec2(0., 0.);

  if (faceNormal.x > 0.1 || faceNormal.x < -0.1) {
    unitU = unitY;
    unitV = unitZ;
  } else if (faceNormal.y > 0.1 || faceNormal.y < -0.1) {
    unitU = -unitX;
    unitV = unitZ;
  } else {
    unitU = unitX;
    unitV = unitY;
  }

  // project world coordinates onto texture coordinates
  vec3 planeOrigin = voxelPos * abs(faceNormal);
  vec2 planeOffset = vec2(
      dot(voxelPos - planeOrigin, unitU),
      dot(voxelPos - planeOrigin, unitV));

  // wrap based on periodicity and scale down to [0, 1]
  return mod(planeOffset + uvOffset, periodicity) / periodicity;
}

// Based on shader inputs, decode and look up actual attributes for the vertex that we are to
// produce.  If this function returns false then the vertex should be dropped.
bool decodeAttributes(out Attributes attrs) {
  // gl_VertexID encodes chunk index and index of vertex within face pair
  uint chunkIndex = gl_VertexIndex / 12;

  uint pairVertexId = gl_VertexIndex % 12;

  uint pairData = b_FacePairs[gl_InstanceIndex];

  uint enabledFaces = pairData >> 26;
  uint pairBit = (pairVertexId / 6) & 1; // 0 for first face in pair, 1 for second

  uint faceVertexId = (pairVertexId - 6 * pairBit) % 6;

  // If pairBit is 0, faceId is the position of the first set bit in enabledFaces, 
  // otherwise it is the position of the second set bit.
  // If pairBit is 1 and there is no second set bit, hide the vertex.
  uint faceId;
  for (faceId = 0; faceId < 6; faceId++) {
    bool faceEnabled = (enabledFaces & (1 << faceId)) > 0;
    if (!faceEnabled)
      continue;

    if (pairBit == 1) {
      // if pairBit is 1 then we want the second enabled face
      pairBit = 0;
      continue;
    }

    break;
  }

  if (faceId == 6)
    // didn't find an enabled face, return failure
    return false;

  // look up voxel/chunk data
  uint voxelIndexMask = (1 << 26) - 1;
  uint voxelIndex = chunkIndex * CHUNK_SIZE_XYZ + (pairData & voxelIndexMask);
  ivec4 voxelAddr = decodeVoxelIndex(voxelIndex);

  attrs.voxelAddr = voxelAddr;
  attrs.chunkOffset = vec3(b_ChunkMetadata[chunkIndex].offset.xyz);
  attrs.voxelType = voxelTypeAtIndex(voxelIndex);

  // look up mesh data
  CubeFace face = b_VoxelRenderInfo[attrs.voxelType].cube[faceId];
  uint faceVertexIndex = face.indices[faceVertexId];
  attrs.normal = face.normal;
  attrs.pos = face.vertexLocs[faceVertexIndex];
  vec2 faceTexCoord = face.vertexTexCoords[faceVertexIndex];

  attrs.texId = b_VoxelRenderInfo[attrs.voxelType].faceTexIds[faceId];
  vec2 periodicity = vec2(
      float(b_VoxelTextureMetadata[attrs.texId].x_periodicity),
      float(b_VoxelTextureMetadata[attrs.texId].y_periodicity));
  vec3 voxelOffset = attrs.chunkOffset + vec3(voxelAddr.xyz);
  vec2 texOrigin = getTexOrigin(periodicity, voxelOffset, attrs.normal.xyz);
  attrs.texCoord = texOrigin + faceTexCoord / periodicity;

  return true;
}

mat4 identity() {
  return mat4(
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0);
}

void main() {
  Attributes attrs;
  if (!decodeAttributes(attrs)) {
    return;
  }

  mat4 view = u_View * translation_matrix(-u_CameraPos.xyz);
  mat4 model = fullModelMatrix(attrs.voxelAddr, attrs.chunkOffset);
  mat4 proj = u_Projection;

  v_CameraPos = u_CameraPos.xyz;
  v_TexCoord = attrs.texCoord;
  v_Pos = model * attrs.pos;
  v_Normal = (model * attrs.normal).xyz;
  v_TexId = attrs.texId;

  gl_Position = proj * view * v_Pos;
}

// vi: ft=c
