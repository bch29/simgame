#version 450

layout(location = 0) in vec2 v_TexCoord;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec4 v_Pos;
layout(location = 3) flat in uint v_BlockType;
layout(location = 0) out vec4 o_Target;

const vec4 ambientColor = vec4(1.0, 1.0, 1.0, 1.0);
const float ambientStrength = 0.1;
const vec3 lightPos = vec3(3., -10., 6.);
const vec4 lightColor = vec4(1.0, 0.9, 0.8, 1.0);

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
  if (v_BlockType == 0)
  {
    // Air blocks are completely transparent
    o_Target = vec4(0.0);
  }
  else
  {
    vec4 ambient = ambientStrength * ambientColor;

    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(lightPos - v_Pos.xyz); 
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = diff * lightColor;

    float dist = length(vec2(0.75, 0.75));
    float hi = 1.;
    float lo = 0.2;
    vec4 texColor = vec4(.2, .2, 1., 1.);
    /* vec4 texColor = */
    /*   0.25 * vec4(hi, lo, lo, 1.) * spot(vec2(0.25, 0.25)) + */
    /*   0.25 * vec4(lo, hi, lo, 1.) * spot(vec2(0.25, 0.75)) + */
    /*   0.25 * vec4(lo, lo, hi, 1.) * spot(vec2(0.75, 0.25)) + */
    /*   0.25 * vec4(hi, lo, lo, 1.) * spot(vec2(0.75, 0.75)); */

    o_Target = (ambient + diffuse) * texColor;
  }
}

// vi: ft=c
