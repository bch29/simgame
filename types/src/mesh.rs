use cgmath::{Matrix4, Transform, Vector3, Vector4};
use serde::{Deserialize, Serialize};
use zerocopy::AsBytes;

use crate::config::ModelKind;

pub mod cube;
pub mod sphere;

pub type Index = u16;

#[derive(Clone, Copy, Debug, AsBytes, Serialize, Deserialize)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub normal: [f32; 3],
    pub face_id: u32,
    pub tex_coord: [f32; 2],
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
}

pub fn vertex(pos: [f64; 3], tex_coord: [f64; 2], face_id: u32, normal: [f64; 3]) -> Vertex {
    Vertex {
        pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
        face_id,
        tex_coord: [tex_coord[0] as f32, tex_coord[1] as f32],
    }
}

impl Mesh {
    /// Combine another mesh into this one, resulting in a mesh that is the union of the two.
    pub fn union_from(&mut self, other: &Mesh) {
        let start_index = self.vertices.len() as Index;
        self.vertices.extend(other.vertices.iter().cloned());
        self.indices
            .extend(other.indices.iter().cloned().map(|ix| start_index + ix));
    }

    pub fn transformed_union_from(&mut self, transform: Matrix4<f32>, other: &Mesh) {
        let start_index = self.vertices.len() as Index;
        self.vertices.extend(other.vertices.iter().map(|vtx| {
            let pos: Vector4<f32> = vtx.pos.into();
            let normal: Vector3<f32> = vtx.normal.into();

            Vertex {
                pos: (transform * pos).into(),
                normal: (transform.transform_vector(normal)).into(),
                face_id: vtx.face_id,
                tex_coord: vtx.tex_coord,
            }
        }));
        self.indices
            .extend(other.indices.iter().cloned().map(|ix| start_index + ix));
    }

    /// Combine two meshes, creating a mesh that is the union of the two.
    pub fn union(lhs: &Mesh, rhs: &Mesh) -> Mesh {
        let mut result = lhs.clone();
        result.union_from(rhs);
        result
    }

    pub fn from_model_kind(kind: &ModelKind) -> Mesh {
        match kind {
            ModelKind::Cube => cube::Cube::new().mesh(),
            ModelKind::Sphere => sphere::UnitSphere { detail: 12 }.mesh(),
        }
    }
}
