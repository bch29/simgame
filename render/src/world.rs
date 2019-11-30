use std::collections::HashMap;

use anyhow::Result;
use cgmath::{Deg, ElementWise, Matrix4, Point3, SquareMatrix, Vector3};
use log::info;

use simgame_core::block;
use simgame_core::block::index_utils;
use simgame_core::world::{UpdatedWorldState, World};

use crate::mesh;

const BLOCK_TYPE_SIZE_BYTES: u32 = std::mem::size_of::<block::Block>() as u32;
const CHUNK_BUFFER_SIZE_BYTES: wgpu::BufferAddress = index_utils::chunk_size_total()
    as wgpu::BufferAddress
    * BLOCK_TYPE_SIZE_BYTES as wgpu::BufferAddress;
const MATRIX4_SIZE: wgpu::BufferAddress = std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress;
const POINT3_SIZE: wgpu::BufferAddress = std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress;
const UNIFORM_BUF_LEN: wgpu::BufferAddress =
    MATRIX4_SIZE + MATRIX4_SIZE + MATRIX4_SIZE + POINT3_SIZE;

pub struct WorldRenderInit<RV, RF> {
    pub vert_shader_spirv_bytes: RV,
    pub frag_shader_spirv_bytes: RF,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
}

pub(crate) struct WorldRenderState {
    render_pipeline: wgpu::RenderPipeline,
    cube_vertex_buf: wgpu::Buffer,
    cube_index_buf: wgpu::Buffer,
    cube_index_count: usize,
    base_uniform_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    rotation: Matrix4<f32>,
    depth_texture: wgpu::TextureView,
    /// Each point is a u8, represents block type at that point
    /// Dimensions are 16x16x16
    per_chunk: HashMap<Point3<usize>, PerChunkRenderState>,
    camera_pos: Point3<f32>,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView,
}

struct PerChunkRenderState {
    block_type_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
}

impl WorldRenderState {
    pub fn new<RV, RF>(init: WorldRenderInit<RV, RF>, device: &wgpu::Device) -> Result<Self>
    where
        RV: std::io::Seek + std::io::Read,
        RF: std::io::Seek + std::io::Read,
    {
        let vs_module =
            device.create_shader_module(&wgpu::read_spirv(init.vert_shader_spirv_bytes)?);
        let fs_module =
            device.create_shader_module(&wgpu::read_spirv(init.frag_shader_spirv_bytes)?);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    // Uniforms
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                    },
                    // Block type buffer
                    wgpu::BindGroupLayoutBinding {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        },
                    },
                    // wgpu::BindGroupLayoutBinding {
                    //     binding: 1,
                    //     visibility: wgpu::ShaderStage::FRAGMENT,
                    //     ty: wgpu::BindingType::SampledTexture {
                    //         multisampled: false,
                    //         dimension: wgpu::TextureViewDimension::D2,
                    //     },
                    // },
                    // wgpu::BindGroupLayoutBinding {
                    //     binding: 2,
                    //     visibility: wgpu::ShaderStage::FRAGMENT,
                    //     ty: wgpu::BindingType::Sampler,
                    // },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let cube_mesh = mesh::cube::create();
        let cube_vertex_buf = cube_mesh.vertex_buffer(device);
        let cube_index_buf = cube_mesh.index_buffer(device);

        let proj_matrix =
            OPENGL_TO_WGPU_MATRIX * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 100.0);
        let view_matrix = Matrix4::look_at_dir(
            Point3::new(0., 0., 0.),
            Vector3::new(1., 1., -2.),
            Vector3::unit_z(),
        );
        let model_matrix = Matrix4::<f32>::identity();
        let mut uniform_data: Vec<f32> = Vec::new();
        uniform_data.extend::<&[f32; 16]>(proj_matrix.as_ref());
        uniform_data.extend::<&[f32; 16]>(view_matrix.as_ref());
        uniform_data.extend::<&[f32; 16]>(model_matrix.as_ref());
        uniform_data.extend::<&[f32; 3]>(Point3::new(0f32, 0f32, 0f32).as_ref());

        let base_uniform_buf = device
            .create_buffer_mapped(uniform_data.len(), wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(uniform_data.as_ref());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: init.width,
                height: init.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0u32,
                stencil_write_mask: 0u32,
            }),
            index_format: cube_mesh.index_format(),
            vertex_buffers: &[cube_mesh.vertex_buffer_descriptor()],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Ok(WorldRenderState {
            render_pipeline,
            cube_vertex_buf,
            cube_index_buf,
            cube_index_count: cube_mesh.indices.len(),
            base_uniform_buf,
            bind_group_layout,
            rotation: Matrix4::identity(),
            depth_texture: depth_texture.create_default_view(),
            per_chunk: HashMap::new(),
            camera_pos: Point3::new(-20f32, -20f32, 20f32),
        })
    }

    pub fn update_camera_pos(&mut self, delta: Vector3<f32>) {
        self.camera_pos += delta;
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let background_color = wgpu::Color::BLACK;

        // Clear texture and viewport on first pass (i.e. first chunk), then store it on later
        // passes to preserve existing rendering.
        let mut load_op = wgpu::LoadOp::Clear;

        for (chunk_loc, per_chunk) in self.per_chunk.iter() {
            let chunk_offset_int =
                (chunk_loc - Point3::new(0, 0, 0)).mul_element_wise(index_utils::chunk_size());
            let chunk_offset = Vector3 {
                x: chunk_offset_int.x as f32,
                y: chunk_offset_int.y as f32,
                z: chunk_offset_int.z as f32,
            };
            let translation = Matrix4::from_translation(chunk_offset);
            let model_matrix = self.rotation * translation;

            let model_slice: &[f32; 16] = model_matrix.as_ref();
            let vec_slice: &[f32; 3] = self.camera_pos.as_ref();
            let extra_uniform_buf = {
                let buffer_mapped =
                    device.create_buffer_mapped(16 + 3, wgpu::BufferUsage::COPY_SRC);
                buffer_mapped.data[..16].copy_from_slice(model_slice);
                buffer_mapped.data[16..19].copy_from_slice(vec_slice);
                buffer_mapped.finish()
            };

            // Copy view/projection matrix from the base buffer
            encoder.copy_buffer_to_buffer(
                &self.base_uniform_buf,
                0,
                &per_chunk.uniform_buf,
                0,
                2 * MATRIX4_SIZE,
            );

            // Copy model matrix from the in-memory buffer
            encoder.copy_buffer_to_buffer(
                &extra_uniform_buf,
                0,
                &per_chunk.uniform_buf,
                2 * MATRIX4_SIZE,
                MATRIX4_SIZE + POINT3_SIZE,
            );

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &per_chunk.uniform_buf,
                            range: 0..UNIFORM_BUF_LEN,
                        },
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &per_chunk.block_type_buf,
                            range: 0..CHUNK_BUFFER_SIZE_BYTES,
                        },
                    },
                    // wgpu::Binding {
                    //     binding: 2,
                    //     resource: wgpu::BindingResource::Sampler(&sampler),
                    // },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: background_color,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_load_op: load_op,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_index_buffer(&self.cube_index_buf, 0);
            rpass.set_vertex_buffers(0, &[(&self.cube_vertex_buf, 0)]);
            rpass.draw_indexed(
                0..self.cube_index_count as u32,
                0,
                0..(index_utils::chunk_size_total() as u32),
            );

            load_op = wgpu::LoadOp::Load;
        }
    }

    pub fn init(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        world: &World,
    ) {
        for (p, chunk) in world.blocks.iter_chunks() {
            let all_empty = chunk.blocks.iter().all(|b| b.is_empty());
            if all_empty {
                continue;
            }

            self.insert_per_chunk(encoder, device, p, chunk);

            info!("Inserted chunk at point {:?}", p);
        }
    }

    fn insert_per_chunk(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        loc: Point3<usize>,
        chunk: &block::Chunk,
    ) {
        let per_chunk = PerChunkRenderState::new(&device);
        per_chunk.update(encoder, device, chunk);
        self.per_chunk.insert(loc, per_chunk);
    }

    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        world: &World,
        diff: &UpdatedWorldState,
    ) {
        // self.rotation = self.rotation * Matrix4::<f32>::from_angle_z(Rad::full_turn() / 1000.);

        for &chunk_loc in &diff.modified_chunks {
            let chunk = world.blocks.get_chunk(chunk_loc);
            let chunk_empty = chunk.blocks.iter().all(|b| b.is_empty());
            if self.per_chunk.contains_key(&chunk_loc) {
                if chunk_empty {
                    self.per_chunk.remove(&chunk_loc);
                } else {
                    self.per_chunk[&chunk_loc].update(encoder, device, &chunk);
                }
            } else if !chunk_empty {
                self.insert_per_chunk(encoder, device, chunk_loc, chunk);
            }
        }
    }
}

impl PerChunkRenderState {
    fn new(device: &wgpu::Device) -> Self {
        let block_type_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: CHUNK_BUFFER_SIZE_BYTES,
            usage: wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::STORAGE_READ,
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: UNIFORM_BUF_LEN,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        PerChunkRenderState {
            block_type_buf,
            uniform_buf,
        }
    }

    #[allow(clippy::cast_lossless)]
    fn update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        chunk: &block::Chunk,
    ) {
        let blocks_slice = simgame_core::block::blocks_to_u16(&chunk.blocks);
        let blocks_buf = device
            .create_buffer_mapped(chunk.blocks.len(), wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(blocks_slice);
        encoder.copy_buffer_to_buffer(
            &blocks_buf,
            0,
            &self.block_type_buf,
            0,
            index_utils::chunk_size_total() as wgpu::BufferAddress
                * BLOCK_TYPE_SIZE_BYTES as wgpu::BufferAddress,
        );
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
