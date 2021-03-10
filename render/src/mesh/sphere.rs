use cgmath::Matrix4;
use zerocopy::{AsBytes, FromBytes};

use crate::mesh::{Mesh, Vertex};

/// A "face" of a unit sphere which has been triangulated by projecting a subdivided cube onto the
/// sphere.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct UnitSphereFace {
    pub detail: u16,
    pub tex_index: u32,
}

#[derive(Clone, Copy, Debug, AsBytes, FromBytes, Default)]
#[repr(C)]
pub struct UnitSphere {
    pub detail: u16,
}

impl UnitSphereFace {
    /// Triangulate the sphere face. The default orientation produces the "front" face.
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

                let x = square_x * (0.5 - square_y * square_y / 6.0).sqrt();
                let z = square_y * (0.5 - square_x * square_x / 6.0).sqrt();
                let y = -(1.0 - square_x * square_x / 2.0 - square_y * square_y / 2.0
                    + square_x * square_x * square_y * square_y / 3.0)
                    .sqrt();

                vertices.push(Vertex {
                    pos: [x, y, z, 1.],
                    normal: [x, y, z],
                    tex_index: self.tex_index,
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
        UnitSphereFace {
            detail: 6,
            tex_index: 0,
        }
    }
}

impl UnitSphere {
    /// Triangulate the sphere face. The default orientation produces the "front" face.
    pub fn mesh(&self) -> Mesh {
        let transforms = vec![
            Matrix4::from_angle_x(cgmath::Deg(270.)), // top
            Matrix4::from_angle_x(cgmath::Deg(90.)),  // bottom
            Matrix4::from_angle_z(cgmath::Deg(90.)),  // right
            Matrix4::from_angle_z(cgmath::Deg(270.)), // left
            Matrix4::from_angle_z(cgmath::Deg(180.)), // back
            Matrix4::from_angle_z(cgmath::Deg(0.)),   // front
        ];

        let mut mesh = Mesh::default();

        for (face_index, transform) in transforms.into_iter().enumerate() {
            mesh.transformed_union_from(
                transform,
                &UnitSphereFace {
                    detail: self.detail,
                    tex_index: face_index as u32,
                }
                .mesh(),
            );
        }

        mesh
    }

    pub fn new() -> Self {
        UnitSphere { detail: 6 }
    }
}
