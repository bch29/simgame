use crate::mesh::{Mesh, Vertex};

use zerocopy::{AsBytes, FromBytes};

#[derive(Clone, Copy, Debug, AsBytes, FromBytes, Default)]
#[repr(C)]
pub struct Face {
    normal: [f32; 4],
    indices: [u32; 6],
    _padding: [u32; 2],
    vertex_locs: [[f32; 4]; 4],
    vertex_tex_coords: [[f32; 2]; 4],
}

#[derive(Clone, Copy, Debug, AsBytes, FromBytes, Default)]
#[repr(C)]
pub struct Cube {
    pub faces: [Face; 6],
}

impl Cube {
    pub fn mesh(&self) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for face in &self.faces {
            let first_index = vertices.len() as u16;

            for i in 0..4 {
                vertices.push(Vertex {
                    pos: face.vertex_locs[i],
                    normal: [face.normal[0], face.normal[1], face.normal[2]],
                    tex_coord: face.vertex_tex_coords[i],
                });
            }

            indices.extend(face.indices.iter().map(|&x| first_index + x as u16));
        }

        Mesh { vertices, indices }
    }

    pub fn new() -> Self {
        let tex_coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        let normals = [
            [0.0, 0.0, 1.0],  // top
            [0.0, 0.0, -1.0], // bottom
            [1.0, 0.0, 0.0],  // right
            [-1.0, 0.0, 0.0], // left
            [0.0, 1.0, 0.0],  // front
            [0.0, -1.0, 0.0], // back
        ];

        let vertices = [
            // top (0, 0, 1)
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            // bottom (0, 0, -1)
            [
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            // right (1, 0, 0)
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
            // left (-1, 0, 0)
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            // front (0, 1, 0)
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
            // back (0, -1, 0)
            [
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
        ];

        let indices = [
            [0, 1, 2, 2, 3, 0], // top
            [0, 1, 2, 2, 3, 0], // bottom
            [0, 1, 2, 2, 3, 0], // right
            [0, 1, 2, 2, 3, 0], // left
            [0, 1, 2, 2, 3, 0], // front
            [0, 1, 2, 2, 3, 0], // back
        ];

        fn fill_4<T: Copy>(inp: [T; 3], w: T) -> [T; 4] {
            [inp[0], inp[1], inp[2], w]
        }

        fn fill_4_4<T: Copy>(inp: [[T; 3]; 4], w: T) -> [[T; 4]; 4] {
            [
                fill_4(inp[0], w),
                fill_4(inp[1], w),
                fill_4(inp[2], w),
                fill_4(inp[3], w),
            ]
        }

        let face = |ix: usize| -> Face {
            Face {
                normal: fill_4(normals[ix], 0.0),
                indices: indices[ix],
                _padding: [0, 0],
                vertex_locs: fill_4_4(vertices[ix], 1.0),
                vertex_tex_coords: tex_coords,
            }
        };

        Cube {
            faces: [face(0), face(1), face(2), face(3), face(4), face(5)],
        }
    }
}
