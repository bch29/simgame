pub mod cube;

#[derive(Clone, Copy, Debug)]
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
    pub fn vertex_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device
            .create_buffer_mapped(self.vertices.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(&self.vertices)
    }

    pub fn index_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device
            .create_buffer_mapped(self.indices.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(&self.indices)
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
