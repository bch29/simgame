#version 450

layout(location = 0) out vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform Uniforms {
  mat4 u_Model;
};

void main() {
  vec2[4] vertices = vec2[](
      vec2(-1.0, -1.0), // bottom left
      vec2(-1.0, 1.0), // top left
      vec2(1.0, -1.0), // bottom right
      vec2(1.0, 1.0) // top right
  );

  vec2[4] texCoords = vec2[](
      vec2(0., 0.), // bottom left
      vec2(0., 1.), // top left
      vec2(1., 0.), // bottom right
      vec2(1., 1.) // top right
  );

  gl_Position = u_Model * vec4(vertices[gl_VertexIndex], 0., 1.);
  v_TexCoord = texCoords[gl_VertexIndex];
}

// vi: ft=c
