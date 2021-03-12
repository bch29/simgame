#![allow(dead_code)]

use cgmath::{Matrix4, Transform, Vector3, Vector4};

pub mod cube;
pub mod sphere;

use zerocopy::AsBytes;

use simgame_types::entity::config::ModelKind;

pub type Index = u16;

#[derive(Clone, Copy, Debug, AsBytes)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub normal: [f32; 3],
    pub face_id: u32,
    pub tex_coord: [f32; 2],
}

#[derive(Clone, Debug, Default)]
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
    pub fn vertex_buffer_size(&self) -> wgpu::BufferAddress {
        (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64
    }

    pub fn index_buffer_size(&self) -> wgpu::BufferAddress {
        (std::mem::size_of::<Index>() * self.indices.len()) as u64
    }

    pub fn index_format() -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }

    pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                // pos
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float4,
                    offset: 0,
                    shader_location: 0,
                },
                // normal
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float3,
                    offset: 4 * 4,
                    shader_location: 1,
                },
                // face_id
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint,
                    offset: 4 * 4 + 3 * 4,
                    shader_location: 2,
                },
                // tex_coord
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float2,
                    offset: 4 * 4 + 4 + 3 * 4,
                    shader_location: 3,
                },
            ],
        }
    }

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
