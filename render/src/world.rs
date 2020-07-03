use anyhow::Result;
// use cgmath::{Angle, Rad};
use cgmath::{Deg, ElementWise, EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use std::iter;
use zerocopy::AsBytes;

use simgame_core::{
    block::{self, index_utils},
    convert_point, convert_vec,
    util::{Bounds, DivUp},
    world::{UpdatedWorldState, World},
};

use crate::buffer_util::{
    BufferSyncHelper, BufferSyncHelperDesc, BufferSyncedData, IntoBufferSynced, OpaqueBuffer,
};
use crate::mesh;

const LOOK_AT_DIR: Vector3<f32> = Vector3::new(1., 1., -3.);
const CUBE_VERTEX_STRIDE: u64 = 16 + 16 + 8 + 4 + 4;

#[derive(Debug, Clone)]
pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
    pub visible_size: Vector3<i32>,
}

pub struct Shaders<R> {
    pub vert: R,
    pub frag: R,
    pub comp: R,
}

pub struct WorldRenderInit<R> {
    pub shaders: Shaders<R>,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
}

pub struct ComputeData {
    pipeline: wgpu::ComputePipeline,
    uniforms: BufferSyncedData<ComputeUniforms, u8>,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub struct RenderData {
    pipeline: wgpu::RenderPipeline,
    uniforms: BufferSyncedData<RenderUniforms, f32>,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: wgpu::TextureView,
}

pub struct WorldRenderState {
    compute_data: ComputeData,
    render_data: RenderData,

    view_params: ViewParams,

    rotation: Matrix4<f32>,

    chunk_batch: ChunkBatchRenderState,
    needs_compute_pass: bool,

    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView,
}

#[repr(C)]
struct RenderUniforms {
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    model: Matrix4<f32>,
    camera_pos: Point3<f32>,
}

#[repr(C)]
struct ComputeUniforms {
    visible_box_origin: Point3<f32>,
    visible_box_limit: Point3<f32>,
    cube: mesh::cube::Cube,
}

type ActiveChunks = crate::stable_map::StableMap<Point3<i32>, ()>;

struct ChunkBatchRenderState {
    active_chunks: ActiveChunks,
    chunk_metadata_helper: BufferSyncHelper<u8>,
    chunk_metadata_buf: wgpu::Buffer,
    block_type_helper: BufferSyncHelper<u16>,
    block_type_buf: wgpu::Buffer,
    active_view_box: Option<Bounds<i32>>,

    cube_vertex_buf: OpaqueBuffer,
    cube_index_buf: OpaqueBuffer,
    cube_indirect_buf: OpaqueBuffer,
}

#[repr(C)]
#[derive(Debug, Clone)]
struct ChunkMeta {
    offset: Point3<f32>,
    // padding to align offset and neighbor indices to 16 bytes
    _padding0: [f32; 1],
    neighbor_indices: [i32; 6],
    active: bool,
    // padding to align struct to 16 bytes
    _padding1: i32,
}

impl WorldRenderState {
    pub fn set_view(&mut self, params: ViewParams) {
        if params.visible_size != self.view_params.visible_size {
            self.chunk_batch.set_visible_size(params.visible_size);
        }
        self.render_data
            .uniforms
            .update_view_params(&self.view_params);
        self.view_params = params;
    }

    pub fn new<R>(init: WorldRenderInit<R>, device: &wgpu::Device) -> Result<Self>
    where
        R: std::io::Seek + std::io::Read,
    {
        let shaders = init.shaders.map_result::<std::io::Error, _, _>(|stream| {
            Ok(device.create_shader_module(&wgpu::read_spirv(stream)?))
        })?;

        let view_params = ViewParams::default();

        let compute_data = {
            let uniforms = ComputeUniforms {
                visible_box_origin: Point3::origin(),
                visible_box_limit: Point3::origin(),
                cube: mesh::cube::Cube::new(),
            }
            .buffer_synced(device);

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simgame_render::world::WorldRenderState/compute/layout"),
                    bindings: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                        },
                        // Block type buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                            },
                        },
                        // Chunk metadata buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                            },
                        },
                        // Output vertex buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                        },
                        // Output index buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                        },
                        // Output indirect buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: &pipeline_layout,
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &shaders.comp,
                    entry_point: "main",
                },
            });

            ComputeData {
                pipeline,
                uniforms,
                bind_group_layout,
            }
        };

        let render_data = {
            let mut uniforms = RenderUniforms {
                proj: OPENGL_TO_WGPU_MATRIX
                    * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 1000.0),
                view: Matrix4::look_at_dir(
                    Point3::new(0., 0., 0.),
                    LOOK_AT_DIR,
                    Vector3::unit_z(),
                ),
                model: Matrix4::<f32>::identity(),
                camera_pos: Point3::origin(),
            }
            .buffer_synced(device);
            uniforms.update_view_params(&view_params);

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("simgame_render::world::WorldRenderState/vertex/layout"),
                    bindings: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX,
                            ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("simgame_render::world::WorldRenderState/depth"),
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

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &pipeline_layout,
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &shaders.vert,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &shaders.frag,
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
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint32,
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: CUBE_VERTEX_STRIDE,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float4,
                                offset: 4 * 4,
                                shader_location: 1,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Float2,
                                offset: (4 + 4) * 4,
                                shader_location: 2,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Uint,
                                offset: (4 + 4 + 2) * 4,
                                shader_location: 3,
                            },
                        ],
                    }],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

            RenderData {
                pipeline,
                uniforms,
                bind_group_layout,
                depth_texture: depth_texture.create_default_view(),
            }
        };

        Ok(WorldRenderState {
            compute_data,
            render_data,
            rotation: Matrix4::identity(),
            chunk_batch: ChunkBatchRenderState::new(device, view_params.visible_size),
            needs_compute_pass: false,
            view_params,
        })
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.needs_compute_pass {
            self.compute_pass(device, encoder);
        }
        self.render_pass(device, frame, encoder);
    }

    fn compute_pass(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let uniform_slices = self.compute_data.uniforms.as_slices();

        self.compute_data
            .uniforms
            .sync_with(device, encoder, |_| &uniform_slices);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute_data.bind_group_layout,
            bindings: &[
                self.compute_data.uniforms.as_binding(0),
                self.chunk_batch.block_type_binding(1),
                self.chunk_batch.chunk_metadata_binding(2),
                self.chunk_batch.cube_vertex_buf.as_binding(3),
                self.chunk_batch.cube_index_buf.as_binding(4),
                self.chunk_batch.cube_indirect_buf.as_binding(5),
            ],
        });

        let mut cpass = encoder.begin_compute_pass();

        cpass.set_pipeline(&self.compute_data.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        cpass.dispatch(self.chunk_batch.count_chunks() as u32, 1, 1);
    }

    fn render_pass(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let background_color = wgpu::Color::BLACK;

        self.render_data.uniforms.model = self.rotation;
        self.render_data
            .uniforms
            .sync_with(device, encoder, |data| data.as_slices());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.render_data.bind_group_layout,
            bindings: &[self.render_data.uniforms.as_binding(0)],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                load_op: wgpu::LoadOp::Clear,
                store_op: wgpu::StoreOp::Store,
                clear_color: background_color,
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: &self.render_data.depth_texture,
                depth_load_op: wgpu::LoadOp::Clear,
                depth_store_op: wgpu::StoreOp::Store,
                stencil_load_op: wgpu::LoadOp::Clear,
                stencil_store_op: wgpu::StoreOp::Store,
                clear_depth: 1.0,
                clear_stencil: 0,
            }),
        });
        rpass.set_pipeline(&self.render_data.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);

        // One draw call per visible chunk. The compute stage has populated our index, vertex
        // and indirect buffers.
        for (_, chunk_index, _) in self.chunk_batch.active_chunks.iter() {
            let chunk_index = chunk_index as u64;

            let chunk_index_size = 4 * index_utils::chunk_size_total() as u64;
            let chunk_vert_size = CUBE_VERTEX_STRIDE * index_utils::chunk_size_total() as u64;

            rpass.set_index_buffer(
                &self.chunk_batch.cube_index_buf.buffer(),
                chunk_index * chunk_index_size,
                (1 + chunk_index) * chunk_index_size,
            );

            rpass.set_vertex_buffer(
                0,
                &self.chunk_batch.cube_vertex_buf.buffer(),
                chunk_index * chunk_vert_size,
                (1 + chunk_index) * chunk_vert_size,
            );

            rpass.draw_indexed_indirect(
                &self.chunk_batch.cube_indirect_buf.buffer(),
                chunk_index * 4 * 8,
            );
        }
    }

    /// Calculates the box containing chunks that will be rendered according to current view.
    fn calculate_view_box(&self, world: &World) -> Option<Bounds<i32>> {
        let mut center = self.render_data.uniforms.camera_pos + 60. * LOOK_AT_DIR;
        center.z = self.view_params.z_level as f32 - 0.5;
        let size = convert_vec!(self.view_params.visible_size, f32);
        let float_bounds = Bounds::new(center - 0.5 * size, size);
        let world_bounds_limit = world.blocks.bounds().limit();
        let positive_box =
            Bounds::from_limit(Point3::origin(), convert_point!(world_bounds_limit, f32));
        float_bounds.intersection(positive_box).map(|bounds| {
            Bounds::new(
                convert_point!(bounds.origin(), i32),
                convert_vec!(bounds.size(), i32),
            )
        })
    }

    pub fn init(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
    ) {
        let active_view_box = self.calculate_view_box(world);
        if let Some(active_view_box) = active_view_box {
            self.compute_data.uniforms.update_view_box(active_view_box);
            self.chunk_batch.update_view_box(active_view_box, world);
        }

        self.needs_compute_pass = self.chunk_batch.update_buffers(device, encoder, world);
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        diff: &UpdatedWorldState,
    ) {
        let active_view_box = self.calculate_view_box(world);
        if let Some(active_view_box) = active_view_box {
            self.compute_data.uniforms.update_view_box(active_view_box);
            self.chunk_batch.update_view_box(active_view_box, world);
        } else {
            self.chunk_batch.clear_view_box();
        }

        self.chunk_batch.apply_chunk_diff(world, diff);
        self.needs_compute_pass = self.chunk_batch.update_buffers(device, encoder, world);
    }
}

impl ChunkBatchRenderState {
    const fn max_batch_chunks() -> usize {
        // 16 MB of video memory holds a batch
        (1024 * 1024 * 8) / index_utils::chunk_size_total()
    }

    const fn max_batch_blocks() -> usize {
        Self::max_batch_chunks() * index_utils::chunk_size_total()
    }

    const fn metadata_size() -> usize {
        std::mem::size_of::<ChunkMeta>()
    }

    fn count_chunks(&self) -> usize {
        self.active_chunks.capacity()
    }

    pub fn new(device: &wgpu::Device, visible_size: Vector3<i32>) -> Self {
        let visible_chunk_size = visible_size
            .div_up(&convert_vec!(index_utils::chunk_size(), i32))
            + Vector3::new(1, 1, 1);
        let max_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;

        assert!(max_visible_chunks <= Self::max_batch_chunks() as i32);

        let active_chunks = ActiveChunks::new(max_visible_chunks as usize);

        let block_type_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: Self::max_batch_blocks(),
            max_chunk_len: index_utils::chunk_size_total(),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::STORAGE_READ
                | wgpu::BufferUsage::COPY_DST,
        });

        let chunk_metadata_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: Self::metadata_size() * Self::max_batch_chunks(),
            max_chunk_len: index_utils::chunk_size_total(),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::STORAGE_READ
                | wgpu::BufferUsage::COPY_DST,
        });

        let cube_index_buf = OpaqueBuffer::new(
            device,
            BufferSyncHelperDesc {
                buffer_len: 4 * Self::max_batch_blocks(),
                max_chunk_len: 0,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDEX,
            },
        );

        let cube_vertex_buf = OpaqueBuffer::new(
            device,
            BufferSyncHelperDesc {
                buffer_len: CUBE_VERTEX_STRIDE as usize * Self::max_batch_blocks(),
                max_chunk_len: 0,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::VERTEX,
            },
        );

        let cube_indirect_buf = OpaqueBuffer::new(
            device,
            BufferSyncHelperDesc {
                buffer_len: CUBE_VERTEX_STRIDE as usize * Self::max_batch_blocks(),
                max_chunk_len: 0,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
            },
        );

        ChunkBatchRenderState {
            active_chunks,
            block_type_buf: block_type_helper.make_buffer(device),
            block_type_helper,
            chunk_metadata_buf: chunk_metadata_helper.make_buffer(device),
            chunk_metadata_helper,
            active_view_box: None,
            cube_index_buf,
            cube_vertex_buf,
            cube_indirect_buf,
        }
    }

    pub fn set_visible_size(&mut self, visible_size: Vector3<i32>) {
        let visible_chunk_size = visible_size
            .div_up(&convert_vec!(index_utils::chunk_size(), i32))
            + Vector3::new(1, 1, 1);
        let max_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;

        assert!(max_visible_chunks <= Self::max_batch_chunks() as i32);
        self.active_chunks.set_capacity(max_visible_chunks as usize);
    }

    fn update_box_chunks(&mut self, view_box: Bounds<i32>, world: &World) {
        assert!(self.active_chunks.len() == 0);
        let bounds = Bounds::new(
            convert_point!(view_box.origin(), usize),
            convert_vec!(view_box.size(), usize),
        );
        for (p, _chunk) in world
            .blocks
            .iter_chunks_in_bounds(bounds)
            .map(|(p, chunk)| (convert_point!(p, i32), chunk))
        {
            self.active_chunks.update(p, ());
        }
    }

    pub fn clear_view_box(&mut self) {
        // no chunks in view; clear all
        self.active_view_box = None;
        self.active_chunks.clear();
    }

    pub fn update_view_box(&mut self, active_view_box: Bounds<i32>, world: &World) {
        let old_view_box = self.active_view_box;
        self.active_view_box = Some(active_view_box);

        let old_view_box = match old_view_box {
            Some(x) => x,
            None => {
                // no chunks in previous view; insert all
                self.update_box_chunks(active_view_box, world);
                return;
            }
        };

        let new_chunk_box =
            active_view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32));
        let box_size =
            (new_chunk_box.size().x * new_chunk_box.size().y * new_chunk_box.size().z) as usize;
        assert!(
            box_size <= self.active_chunks.capacity(),
            "{} <= {}",
            box_size,
            self.active_chunks.capacity()
        );

        let old_chunk_box =
            old_view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32));
        if new_chunk_box != old_chunk_box {
            // 1. delete chunks that have left the view
            for pos in old_chunk_box.iter_diff(new_chunk_box) {
                self.active_chunks.remove(&pos);
            }

            // 2. insert chunks that are newly in the view
            for pos in new_chunk_box.iter_diff(old_chunk_box) {
                if pos.x < 0 || pos.y < 0 || pos.z < 0 {
                    continue;
                }

                if let Some(_chunk) = world.blocks.chunks().get(convert_point!(pos, usize)) {
                    self.active_chunks.update(pos, ());
                }
            }

            for &pos in self.active_chunks.keys() {
                assert!(
                    new_chunk_box.contains_point(pos),
                    "Position {:?} is out of active view box: new={:?}/{:?} old={:?}/{:?}",
                    pos,
                    new_chunk_box,
                    active_view_box,
                    old_chunk_box,
                    old_view_box
                );
            }
        }
    }

    pub fn apply_chunk_diff(&mut self, world: &World, diff: &UpdatedWorldState) {
        let active_chunk_box = match self.active_view_box {
            Some(active_view_box) => {
                active_view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32))
            }
            None => return,
        };

        for &pos in &diff.modified_chunks {
            let pos_i32 = convert_point!(pos, i32);
            if !active_chunk_box.contains_point(pos_i32) {
                continue;
            }

            if let Some(_chunk) = world.blocks.chunks().get(pos) {
                self.active_chunks.update(pos_i32, ());
            } else {
                self.active_chunks.remove(&pos_i32);
            }
        }
    }

    fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
    ) -> bool {
        let mut any_updates = false;

        let mut fill_block_types =
            self.block_type_helper
                .begin_fill_buffer(device, &self.block_type_buf, 0);

        let mut fill_chunk_metadatas =
            self.chunk_metadata_helper
                .begin_fill_buffer(device, &self.chunk_metadata_buf, 0);

        // Copy chunk data to GPU buffers for only the chunks that have changed since last time
        // buffers were updated.
        for (index, opt_point) in self.active_chunks.take_diff().changed_entries().into_iter() {
            any_updates = true;

            if let Some((&p, _)) = opt_point {
                let chunk = world.blocks.chunks().get(convert_point!(p, usize)).unwrap();
                let chunk_data = block::blocks_to_u16(&chunk.blocks);
                fill_block_types.seek(encoder, index * index_utils::chunk_size_total());
                fill_block_types.advance(encoder, chunk_data);
            } else {
                // if a chunk was deleted, write inactive metadata
                let meta = ChunkMeta::empty();
                fill_chunk_metadatas.seek(encoder, index * Self::metadata_size());
                for slice in &meta.as_slices() {
                    fill_chunk_metadatas.advance(encoder, slice);
                }
            }
        }

        fill_block_types.finish(encoder);

        if any_updates {
            // If any chunks have been updated, update metadata for every chunk. In theory we only
            // need to update new/deleted chunks and those with neighbours that are new/deleted.
            // Updating everything doesn't cost too much though.

            for (&point, index, _) in self.active_chunks.iter() {
                let meta = Self::make_chunk_meta(&self.active_chunks, point);
                fill_chunk_metadatas.seek(encoder, index * Self::metadata_size());
                for slice in &meta.as_slices() {
                    fill_chunk_metadatas.advance(encoder, slice);
                }
            }

            fill_chunk_metadatas.finish(encoder);
        }

        any_updates
    }

    fn block_type_binding(&self, index: u32) -> wgpu::Binding {
        self.block_type_helper
            .as_binding(index, &self.block_type_buf, 0)
    }

    fn chunk_metadata_binding(&self, index: u32) -> wgpu::Binding {
        self.chunk_metadata_helper
            .as_binding(index, &self.chunk_metadata_buf, 0)
    }

    fn make_chunk_meta(active_chunks: &ActiveChunks, p: Point3<i32>) -> ChunkMeta {
        fn make_neighbor_indices(map: &ActiveChunks, chunk_loc: Point3<i32>) -> [i32; 6] {
            let directions = [
                Vector3::new(-1, 0, 0),
                Vector3::new(1, 0, 0),
                Vector3::new(0, -1, 0),
                Vector3::new(0, 1, 0),
                Vector3::new(0, 0, -1),
                Vector3::new(0, 0, 1),
            ];

            let mut result = [0i32; 6];
            for (src, dst) in directions
                .iter()
                .map(|dir| {
                    let neighbor_loc = chunk_loc + dir;
                    map.get(&neighbor_loc).map_or(-1, |(index, _)| index as i32)
                })
                .zip(&mut result)
            {
                *dst = src;
            }

            result
        }

        let offset = convert_point!(p, f32)
            .mul_element_wise(Point3::origin() + convert_vec!(index_utils::chunk_size(), f32));

        let neighbor_indices = make_neighbor_indices(&active_chunks, p);

        ChunkMeta {
            offset,
            _padding0: [0f32],
            neighbor_indices,
            active: true,
            _padding1: 0,
        }
    }

    #[allow(dead_code)]
    fn debug_get_active_key_ixes(&self) -> Vec<(usize, Point3<i32>)> {
        let mut active_keys: Vec<_> = self
            .active_chunks
            .keys_ixes()
            .map(|(&p, i)| (i, p))
            .collect();
        active_keys.sort_by_key(|&(i, p)| (i, p.x, p.y, p.z));
        active_keys
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl RenderUniforms {
    #[inline]
    fn as_slices(&self) -> impl Iterator<Item = &[f32]> {
        let proj: &[f32; 16] = self.proj.as_ref();
        let view: &[f32; 16] = self.view.as_ref();
        let model: &[f32; 16] = self.model.as_ref();
        let camera_pos: &[f32; 3] = self.camera_pos.as_ref();

        iter::once(proj as &[f32])
            .chain(iter::once(view as &[f32]))
            .chain(iter::once(model as &[f32]))
            .chain(iter::once(camera_pos as &[f32]))
    }

    fn update_view_params(&mut self, params: &ViewParams) {
        self.camera_pos = params.camera_pos;
    }
}

impl IntoBufferSynced for RenderUniforms {
    type Item = f32;

    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            buffer_len: 16 + 16 + 16 + 4,
            max_chunk_len: std::mem::size_of::<Matrix4<f32>>(),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ComputeUniforms {
    #[inline]
    fn as_slices(&self) -> [&[u8]; 5] {
        let visible_box_origin = self.visible_box_origin.as_ref() as &[f32; 3];
        let visible_box_limit = self.visible_box_limit.as_ref() as &[f32; 3];
        [
            visible_box_origin.as_bytes(),
            &[0f32].as_bytes(),
            visible_box_limit.as_bytes(),
            &[0f32].as_bytes(),
            self.cube.faces.as_bytes(),
        ]
    }

    fn update_view_box(&mut self, view_box: Bounds<i32>) {
        self.visible_box_origin = convert_point!(view_box.origin(), f32);
        self.visible_box_limit = convert_point!(view_box.limit(), f32);
    }
}

impl IntoBufferSynced for ComputeUniforms {
    type Item = u8;

    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        let face_len = 6 * std::mem::size_of::<mesh::cube::Face>();
        BufferSyncHelperDesc {
            buffer_len: (4 + 4) * 4 + face_len,
            max_chunk_len: face_len,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ChunkMeta {
    #[inline]
    fn as_slices(&self) -> [&[u8]; 5] {
        let offset_vec = self.offset.as_ref() as &[f32; 3];
        [
            offset_vec.as_bytes(),
            self._padding0.as_bytes(),
            self.neighbor_indices.as_bytes(),
            self.active.as_bytes(),
            self._padding1.as_bytes(),
        ]
    }

    fn empty() -> Self {
        Self {
            offset: Point3::new(0.0, 0.0, 0.0),
            _padding0: [0.0],
            neighbor_indices: [0, 0, 0, 0, 0, 0],
            active: false,
            _padding1: 0,
        }
    }
}

impl Default for ViewParams {
    fn default() -> Self {
        ViewParams {
            camera_pos: Point3::new(0., 0., 0.),
            z_level: 0,
            visible_size: Vector3::new(1, 1, 1),
        }
    }
}

impl<R> Shaders<R> {
    pub fn map<R2, F>(self, mut f: F) -> Shaders<R2>
    where
        F: FnMut(R) -> R2,
    {
        Shaders {
            vert: f(self.vert),
            frag: f(self.frag),
            comp: f(self.comp),
        }
    }

    pub fn map_result<E, R2, F>(self, mut f: F) -> std::result::Result<Shaders<R2>, E>
    where
        F: FnMut(R) -> std::result::Result<R2, E>,
    {
        Ok(Shaders {
            vert: f(self.vert)?,
            frag: f(self.frag)?,
            comp: f(self.comp)?,
        })
    }
}
