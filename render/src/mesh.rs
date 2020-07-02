pub mod cube;
pub mod half_open_cube;

use zerocopy::AsBytes;

#[derive(Clone, Copy, Debug, AsBytes)]
#[repr(C)]
pub struct Vertex {
    pos: [f32; 4],
    normal: [f32; 3],
    tex_coord: [f32; 2],
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

    pub fn vertex_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mapped_buffer = device
            .create_buffer_mapped(&wgpu::BufferDescriptor {
                label: None,
                size: self.vertex_buffer_size(),
                usage: wgpu::BufferUsage::VERTEX,
            });

        mapped_buffer.data.copy_from_slice(self.vertices.as_bytes());
        mapped_buffer.finish()
    }

    pub fn index_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mapped_buffer = device
            .create_buffer_mapped(&wgpu::BufferDescriptor {
                label: None,
                size: self.index_buffer_size(),
                usage: wgpu::BufferUsage::INDEX,
            });

        mapped_buffer.data.copy_from_slice(self.indices.as_bytes());
        mapped_buffer.finish()
    }

    pub fn vertex_buffer_descriptor(&self) -> wgpu::VertexBufferDescriptor {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float3,
                    offset: 4 * 4,
                    shader_location: 1,
                },
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float2,
                    offset: (4 + 3) * 4,
                    shader_location: 2,
                },
            ],
        }
    }

    pub fn index_format(&self) -> wgpu::IndexFormat {
        wgpu::IndexFormat::Uint16
    }
}
