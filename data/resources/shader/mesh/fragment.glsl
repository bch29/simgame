#version 450

layout(location = 0) in vec3 v_Normal;
layout(location = 1) in vec2 v_TexCoord;
layout(location = 2) in vec4 v_Pos;
layout(location = 4) in vec3 v_CameraPos;
layout(location = 5) flat in uint v_InstanceIndex;

layout(location = 0) out vec4 o_Target;

/* layout(set = 0, binding = 1) uniform texture2D t_Texture; */
/* layout(set = 0, binding = 2) uniform sampler s_Texture; */

const vec4 ambientColor = vec4(1.0, 1.0, 1.0, 1.0);
const float ambientStrength = 0.1;
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
  vec4[6] colors = {
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(0.5, 0.5, 0.0, 1.0),
    vec4(0.0, 0.5, 0.5, 1.0),
    vec4(0.5, 0.0, 0.5, 1.0)
  };

  vec4 texColor = colors[v_InstanceIndex % 6];

  vec4 ambient = ambientStrength * ambientColor;
  vec3 norm = normalize(v_Normal);
  vec3 lightDir = normalize(lightPos - v_Pos.xyz); 
  float diff = max(dot(norm, lightDir), 0.0);
  vec4 diffuse = diff * lightColor;

  o_Target = (ambient + diffuse) * texColor;
  /* o_Target = texColor; */
}

// vi: ft=c
