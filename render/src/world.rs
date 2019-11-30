use std::collections::HashMap;

use anyhow::Result;
use cgmath::{Deg, ElementWise, EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use log::info;

use simgame_core::block;
use simgame_core::block::index_utils;
use simgame_core::world::{UpdatedWorldState, World};

use crate::buffer_util::{BufferSyncHelperDesc, BufferSyncedData, IntoBufferSynced};
use crate::mesh;

const BLOCK_TYPE_SIZE_BYTES: u32 = std::mem::size_of::<block::Block>() as u32;
const CHUNK_BUFFER_SIZE_BYTES: wgpu::BufferAddress = index_utils::chunk_size_total()
    as wgpu::BufferAddress
    * BLOCK_TYPE_SIZE_BYTES as wgpu::BufferAddress;

pub struct WorldRenderInit<RV, RF> {
    pub vert_shader_spirv_bytes: RV,
    pub frag_shader_spirv_bytes: RF,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
}

pub struct WorldRenderState {
    render_pipeline: wgpu::RenderPipeline,
    cube_vertex_buf: wgpu::Buffer,
    cube_index_buf: wgpu::Buffer,
    cube_index_count: usize,

    uniforms: BufferSyncedData<Uniforms, f32>,

    bind_group_layout: wgpu::BindGroupLayout,
    rotation: Matrix4<f32>,
    depth_texture: wgpu::TextureView,

    per_chunk: HashMap<Point3<usize>, PerChunkRenderState>,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView,
}

struct Uniforms {
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    model: Matrix4<f32>,
    camera_pos: Point3<f32>,
}

struct PerChunkRenderState {
    block_type_buf: wgpu::Buffer,
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

        let uniforms = Uniforms {
            proj: OPENGL_TO_WGPU_MATRIX
                * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 100.0),
            view: Matrix4::look_at_dir(
                Point3::new(0., 0., 0.),
                Vector3::new(1., 1., -2.),
                Vector3::unit_z(),
            ),
            model: Matrix4::<f32>::identity(),
            camera_pos: Point3::new(-20f32, -20f32, 20f32),
        }
        .buffer_synced(device);

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
            uniforms,
            bind_group_layout,
            rotation: Matrix4::identity(),
            depth_texture: depth_texture.create_default_view(),
            per_chunk: HashMap::new(),
        })
    }

    pub fn update_camera_pos(&mut self, delta: Vector3<f32>) {
        self.uniforms.camera_pos += delta;
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
            {
                let chunk_offset_int =
                    (chunk_loc - Point3::origin()).mul_element_wise(index_utils::chunk_size());
                let chunk_offset = Vector3 {
                    x: chunk_offset_int.x as f32,
                    y: chunk_offset_int.y as f32,
                    z: chunk_offset_int.z as f32,
                };

                let translation = Matrix4::from_translation(chunk_offset);
                self.uniforms.model = self.rotation * translation;
                self.uniforms
                    .sync_with(device, encoder, |data| data.as_slices());
            }

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                bindings: &[
                    self.uniforms.as_binding(0),
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

        PerChunkRenderState { block_type_buf }
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

impl Uniforms {
    #[inline]
    fn as_slices(&self) -> impl Iterator<Item = &[f32]> {
        let proj: &[f32; 16] = self.proj.as_ref();
        let view: &[f32; 16] = self.view.as_ref();
        let model: &[f32; 16] = self.model.as_ref();
        let camera_pos: &[f32; 3] = self.camera_pos.as_ref();
        std::iter::once(proj as &[f32])
            .chain(std::iter::once(view as &[f32]))
            .chain(std::iter::once(model as &[f32]))
            .chain(std::iter::once(camera_pos as &[f32]))
    }
}

impl IntoBufferSynced for Uniforms {
    type Item = f32;

    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            buffer_len: 16 + 16 + 16 + 3,
            max_chunk_len: 64,
            gpu_usage: wgpu::BufferUsage::UNIFORM,
        }
    }
}

impl BufferSyncedData<Uniforms, f32> {
    fn as_binding(&self, index: u32) -> wgpu::Binding {
        wgpu::Binding {
            binding: index,
            resource: wgpu::BindingResource::Buffer {
                buffer: &self.buffer(),
                range: 0..self.sync_helper().buffer_byte_len(),
            },
        }
    }
}
