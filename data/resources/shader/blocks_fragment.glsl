#version 450

#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec4 v_Pos;
layout(location = 3) flat in uint v_BlockType;
layout(location = 4) in vec3 v_CameraPos;
layout(location = 5) flat in uint v_TexId;
layout(location = 0) out vec4 o_Target;

layout(set = 0, binding = 5) uniform texture2D[] t_Textures;
layout(set = 0, binding = 6) uniform sampler s_Textures;

const vec4 ambientColor = vec4(1.0, 1.0, 1.0, 1.0);
const float ambientStrength = 0.1;
/* const vec3 lightPos = vec3(0., 0., 0.); */
vec3 lightPos = v_CameraPos;
const vec4 lightColor = vec4(1.0, 1.0, 1.0, 1.0);

float spot(vec2 p) {
  float max_dist = max(
      max(
        length(p - vec2(0., 0.)),
        length(p - vec2(1., 0.))),
      max(
        length(p - vec2(0., 1.)),
        length(p - vec2(1., 1.))));

  float dist = length(p - v_TexCoord);
  float val = (max_dist - dist);

  return val * val * val * val;
}

void main() {
  vec4 ambient = ambientStrength * ambientColor;

  vec3 norm = normalize(v_Normal);
  vec3 lightDir = normalize(lightPos - v_Pos.xyz); 
  float diff = max(dot(norm, lightDir), 0.0);
  vec4 diffuse = diff * lightColor;

  float colorScale = 0.3 + (float(v_BlockType) / 8.0) * 0.7;
  vec4 colorScaleV = vec4(colorScale * 0.7, colorScale * 0.5, colorScale, 1.);

  vec4 texColor = texture(sampler2D(t_Textures[v_TexId], s_Textures), v_TexCoord);

  o_Target = (ambient + diffuse) * texColor;
}

// vi: ft=c