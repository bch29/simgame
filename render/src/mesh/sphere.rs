use crate::mesh::{Mesh, Vertex};

use zerocopy::{AsBytes, FromBytes};

/// A "face" of a unit sphere which has been triangulated by projecting a subdivided cube onto the
/// sphere.
#[derive(Clone, Copy, Debug, AsBytes, FromBytes, Default)]
#[repr(C)]
pub struct UnitSphereFace {
    pub detail: u16,
}

impl UnitSphereFace {
    /// Triangle the sphere
    pub fn mesh(&self) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let n = 1 + self.detail;

        for i in 0..n {
            for j in 0..n {
                // (square_x, square_y) are coordinates within the square face that we are
                // projecting from
                let square_x = 2.0 * i as f32 / self.detail as f32 - 1.0;
                let square_y = 2.0 * j as f32 / self.detail as f32 - 1.0;

                let x = square_x
                    * (0.5 - square_y * square_y / 6.0).sqrt();
                let y = square_y
                    * (0.5 - square_x * square_x / 6.0).sqrt();
                let z = -(1.0 - square_x * square_x / 2.0 - square_y * square_y / 2.0
                    + square_x * square_x * square_y * square_y / 3.0)
                    .sqrt();

                vertices.push(Vertex {
                    pos: [x, y, z, 1.],
                    normal: [x, y, z],
                    tex_coord: [square_x, square_y],
                });

                // each vertex adds indices for the two triangles forming the square below and to
                // its right
                if i < n - 1 && j < n - 1 {
                    indices.push((1 + j) + (1 + i) * n);
                    indices.push((1 + j) + i * n);
                    indices.push(j + i * n);

                    indices.push(j + i * n);
                    indices.push(j + (1 + i) * n);
                    indices.push((1 + j) + (1 + i) * n);
                }
            }
        }

        Mesh { vertices, indices }
    }

    pub fn new() -> Self {
        UnitSphereFace { detail: 6 }
    }
}
