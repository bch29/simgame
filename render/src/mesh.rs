#![allow(dead_code)]

pub mod cube;

use zerocopy::AsBytes;

#[derive(Clone, Copy, Debug, AsBytes)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

pub fn vertex(pos: [f64; 3], tex_coord: [f64; 2], normal: [f64; 3]) -> Vertex {
    Vertex {
        pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
        tex_coord: [tex_coord[0] as f32, tex_coord[1] as f32],
    }
}

impl Mesh {
    pub fn vertex_buffer_size(&self) -> wgpu::BufferAddress {
        (std::mem::size_of::<Vertex>() * self.vertices.len()) as u64
    }

    pub fn index_buffer_size(&self) -> wgpu::BufferAddress {
        (std::mem::size_of::<u16>() * self.indices.len()) as u64
    }

    pub fn index_format(&self) -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }

    /// Combine another mesh into this one, resulting in a mesh that is the union of the two.
    pub fn union_from(&mut self, other: &Mesh) {
        let start_index = self.vertices.len() as u16;
        self.vertices.extend(other.vertices.iter().cloned());
        self.indices.extend(other.indices.iter().cloned().map(|ix| start_index + ix));
    }

    /// Combine two meshes, creating a mesh that is the union of the two.
    pub fn union(lhs: &Mesh, rhs: &Mesh) -> Mesh {
        let mut result = lhs.clone();
        result.union_from(rhs);
        result
    }
}
