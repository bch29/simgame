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

        let quarter_pi = std::f32::consts::PI / 4.;
        let half_pi = std::f32::consts::PI / 2.;

        let n = 1 + self.detail;

        for i in 0..n {
            for j in 0..n {
                // (square_x, square_y) are coordinates within the square face that we are
                // projecting from
                let square_x = i as f32 / self.detail as f32;
                let square_y = j as f32 / self.detail as f32;

                // (theta, phi) are the angular coordinates within the sphere
                let theta = -quarter_pi + half_pi * square_x;
                let phi = -quarter_pi + half_pi * square_y;

                // (x, y, z) are the coordinates of the point on the sphere in 3D Euclidean space
                let x = theta.cos() * phi.cos();
                let y = theta.sin() * phi.cos();
                let z = phi.sin();

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
