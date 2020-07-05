#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use cgmath::{ElementWise, EuclideanSpace, Matrix4, Point3, Vector3};
use zerocopy::{AsBytes, FromBytes};

use simgame_core::{
    block::{self, index_utils},
    convert_point, convert_vec,
    util::{Bounds, DivUp},
    world::{UpdatedWorldState, World},
};

use crate::buffer_util::{
    BufferSyncHelper, BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer,
    InstancedBuffer, InstancedBufferDesc, IntoBufferSynced,
};
use crate::mesh;
use crate::world::{self, Shaders, ViewParams};

/// ### GLSL definition:
/// ```
/// struct Vertex {
///   uint localBlockIndex;
///   uint vertexId;
/// };
/// ```
const VERTEX_STRIDE: u64 = 4 + 4;

pub struct BlocksRenderInit<'a> {
    pub shaders: &'a Shaders<wgpu::ShaderModule>,
    pub view_params: &'a ViewParams,
    pub aspect_ratio: f32,
    pub depth_texture: &'a wgpu::Texture,
    pub uniforms: &'a world::Uniforms,
    pub world: &'a World,
    pub active_view_box: Option<Bounds<i32>>
}

pub struct BlocksRenderState {
    needs_compute_pass: bool,
    chunk_batch: ChunkBatchRenderState,
    compute_stage: ComputeStage,
    render_stage: RenderStage,
}

pub struct ComputeStage {
    pipeline: wgpu::ComputePipeline,
    uniforms: BufferSyncedData<ComputeUniforms, u8>,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub struct RenderStage {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: wgpu::TextureView,

    uniforms: BufferSyncedData<RenderUniforms, f32>,
    locals: BufferSyncedData<RenderLocals, u8>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
struct RenderLocals {
    cube_faces: [mesh::cube::Face; 6],
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

struct ComputeShaderBuffers {
    vertex: InstancedBuffer,
    indirect: InstancedBuffer,
    globals: InstancedBuffer,
}

struct ChunkBatchRenderState {
    active_chunks: ActiveChunks,
    chunk_metadata_helper: BufferSyncHelper<ChunkMeta>,
    chunk_metadata_buf: wgpu::Buffer,
    block_type_helper: BufferSyncHelper<u16>,
    block_type_buf: wgpu::Buffer,
    active_view_box: Option<Bounds<i32>>,

    compute_shader_buffers: ComputeShaderBuffers,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
struct ChunkMeta {
    offset: [f32; 3],
    _padding0: [f32; 1],
    neighbor_indices: [i32; 6],
    active: u32,
    _padding1: i32,
}

impl BlocksRenderState {
    pub fn set_view(&mut self, params: &ViewParams) {
        self.chunk_batch.set_visible_size(params.visible_size);
    }

    pub fn new(init: BlocksRenderInit, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cube = mesh::cube::Cube::new();

        let mut compute_stage = {
            let uniforms = ComputeUniforms {
                visible_box_origin: Point3::origin(),
                visible_box_limit: Point3::origin(),
                cube,
            }
            .into_buffer_synced(device);

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blocks compute layout"),
                    bindings: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry::new(
                            0,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Block type buffer
                        wgpu::BindGroupLayoutEntry::new(
                            1,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Chunk metadata buffer
                        wgpu::BindGroupLayoutEntry::new(
                            2,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Output vertex buffer
                        wgpu::BindGroupLayoutEntry::new(
                            3,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                                min_binding_size: None,
                            },
                        ),
                        // Output indirect buffer
                        wgpu::BindGroupLayoutEntry::new(
                            5,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                                min_binding_size: None,
                            },
                        ),
                        // Globals within a compute invocation
                        wgpu::BindGroupLayoutEntry::new(
                            6,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                                min_binding_size: None,
                            },
                        ),
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                layout: &pipeline_layout,
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &init.shaders.comp,
                    entry_point: "main",
                },
            });

            ComputeStage {
                pipeline,
                uniforms,
                bind_group_layout,
            }
        };

        let render_stage = {
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blocks vertex layout"),
                    bindings: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry::new(
                            0,
                            wgpu::ShaderStage::VERTEX,
                            wgpu::BindingType::UniformBuffer {
                                dynamic: false,
                                min_binding_size: None,
                            },
                        ),
                        // Locals
                        wgpu::BindGroupLayoutEntry::new(
                            1,
                            wgpu::ShaderStage::VERTEX,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Block types
                        wgpu::BindGroupLayoutEntry::new(
                            2,
                            wgpu::ShaderStage::VERTEX,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Chunk metadata
                        wgpu::BindGroupLayoutEntry::new(
                            3,
                            wgpu::ShaderStage::VERTEX,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &pipeline_layout,
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &init.shaders.vert,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &init.shaders.frag,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::None,
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
                    index_format: wgpu::IndexFormat::Uint16,
                    vertex_buffers: &[wgpu::VertexBufferDescriptor {
                        stride: VERTEX_STRIDE,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &[
                            // localBlockIndex
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Uint,
                                offset: 0,
                                shader_location: 0,
                            },
                            // vertexId
                            wgpu::VertexAttributeDescriptor {
                                format: wgpu::VertexFormat::Uint,
                                offset: 4,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

            let uniforms = RenderUniforms {
                proj: init.uniforms.proj,
                view: init.uniforms.view,
                model: init.uniforms.view,
                camera_pos: init.uniforms.camera_pos,
            }
            .into_buffer_synced(device);

            let locals = RenderLocals {
                cube_faces: cube.faces,
            }
            .into_buffer_synced(device);

            RenderStage {
                pipeline,
                uniforms,
                bind_group_layout,
                depth_texture: init.depth_texture.create_default_view(),
                locals,
            }
        };

        let mut chunk_batch = ChunkBatchRenderState::new(device, init.view_params.visible_size);
        let mut needs_compute_pass = false;

        if let Some(active_view_box) = init.active_view_box {
            compute_stage.uniforms.update_view_box(active_view_box);
            if chunk_batch.update_view_box(active_view_box, init.world) {
                needs_compute_pass = true;
            }
        }

        if chunk_batch.update_buffers(queue, init.world) {
            needs_compute_pass = true;
        }

        Self {
            needs_compute_pass,
            chunk_batch, 
            compute_stage,
            render_stage,
        }
    }

    pub fn render_frame(
        &mut self,
        frame_render: &world::FrameRender,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.render_stage.uniforms.proj = frame_render.uniforms.proj;
        self.render_stage.uniforms.model = frame_render.uniforms.model;
        self.render_stage.uniforms.view = frame_render.uniforms.view;
        self.render_stage.uniforms.camera_pos = frame_render.uniforms.camera_pos;

        if self.needs_compute_pass {
            log::debug!("Running compute pass");
            self.compute_pass(frame_render, encoder);
            self.needs_compute_pass = false;
        }
        self.render_pass(frame_render, encoder);
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        world: &World,
        diff: &UpdatedWorldState,
        active_view_box: Option<Bounds<i32>>,
    ) {
        if let Some(active_view_box) = active_view_box {
            self.compute_stage.uniforms.update_view_box(active_view_box);
            if self.chunk_batch.update_view_box(active_view_box, world) {
                self.needs_compute_pass = true;
            }
        } else if self.chunk_batch.clear_view_box() {
            self.needs_compute_pass = true;
        }

        self.chunk_batch.apply_chunk_diff(world, diff);
        if self.chunk_batch.update_buffers(queue, world) {
            self.needs_compute_pass = true;
        }
    }

    fn compute_pass(&mut self, frame_render: &world::FrameRender, encoder: &mut wgpu::CommandEncoder) {
        self.compute_stage.uniforms.sync(frame_render.queue);

        let bufs = &self.chunk_batch.compute_shader_buffers;

        // reset compute shader globals to 0
        bufs.globals.clear(frame_render.queue);

        let bind_group = frame_render.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute_stage.bind_group_layout,
            bindings: &[
                self.compute_stage.uniforms.as_binding(0),
                self.chunk_batch.block_type_binding(1),
                self.chunk_batch.chunk_metadata_binding(2),
                bufs.vertex.as_binding(3),
                bufs.indirect.as_binding(5),
                bufs.globals.as_binding(6),
            ],
        });

        let mut cpass = encoder.begin_compute_pass();

        cpass.set_pipeline(&self.compute_stage.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(self.chunk_batch.count_chunks() as u32, 1, 1);
    }

    fn render_pass(
        &mut self,
        frame_render: &world::FrameRender,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let background_color = wgpu::Color::BLACK;

        self.render_stage.uniforms.sync(frame_render.queue);
        self.render_stage.locals.sync(frame_render.queue);

        let bind_group = frame_render.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.render_stage.bind_group_layout,
            bindings: &[
                self.render_stage.uniforms.as_binding(0),
                self.render_stage.locals.as_binding(1),
                self.chunk_batch.block_type_binding(2),
                self.chunk_batch.chunk_metadata_binding(3),
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame_render.frame.output.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_color),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                attachment: &self.render_stage.depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        rpass.set_pipeline(&self.render_stage.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);

        let bufs = &self.chunk_batch.compute_shader_buffers;
        let vertex_buf = &bufs.vertex;
        let indirect_buf = &bufs.indirect;

        rpass.set_vertex_buffer(0, vertex_buf.buffer().slice(..));

        for (_, chunk_index, _) in self.chunk_batch.active_chunks.iter() {
            rpass.draw_indirect(
                &indirect_buf.buffer(),
                indirect_buf.instance_offset(chunk_index),
            );
        }
    }
}

impl ChunkBatchRenderState {
    const fn max_batch_chunks() -> usize {
        let block_types_mb = 32;
        (1024 * 1024 * block_types_mb / 2) / index_utils::chunk_size_total()
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
            label: "block types",
            buffer_len: Self::max_batch_blocks(),
            max_chunk_len: index_utils::chunk_size_total(),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        let chunk_metadata_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            label: "chunk metadata",
            buffer_len: Self::max_batch_chunks(),
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        let compute_shader_buffers = {
            let vertex = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block vertices",
                    instance_len: 6 * VERTEX_STRIDE as usize * index_utils::chunk_size_total(),
                    n_instances: Self::max_batch_chunks(),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::VERTEX,
                },
            );

            let indirect = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block indirect",
                    instance_len: 4 * 4,
                    n_instances: Self::max_batch_chunks(),
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
                },
            );

            let globals = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "compute globals",
                    instance_len: 8,
                    n_instances: 1,
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST,
                },
            );

            ComputeShaderBuffers {
                vertex,
                indirect,
                globals,
            }
        };

        ChunkBatchRenderState {
            active_chunks,
            block_type_buf: block_type_helper.make_buffer(device),
            block_type_helper,
            chunk_metadata_buf: chunk_metadata_helper.make_buffer(device),
            chunk_metadata_helper,
            active_view_box: None,
            compute_shader_buffers,
        }
    }

    pub fn set_visible_size(&mut self, visible_size: Vector3<i32>) {
        let visible_chunk_size = visible_size
            .div_up(&convert_vec!(index_utils::chunk_size(), i32))
            + Vector3::new(1, 1, 1);
        let max_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;

        assert!(max_visible_chunks <= Self::max_batch_chunks() as i32);
        if max_visible_chunks != self.active_chunks.capacity() as i32 {
            self.active_chunks.set_capacity(max_visible_chunks as usize);
        }
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

    pub fn clear_view_box(&mut self) -> bool {
        if self.active_view_box.is_none() {
            return false;
        }

        // no chunks in view; clear all
        self.active_view_box = None;
        self.active_chunks.clear();

        true
    }

    pub fn update_view_box(&mut self, active_view_box: Bounds<i32>, world: &World) -> bool {
        let old_view_box = self.active_view_box;
        self.active_view_box = Some(active_view_box);

        let old_view_box = match old_view_box {
            Some(x) => x,
            None => {
                // no chunks in previous view; insert all
                self.update_box_chunks(active_view_box, world);
                return true;
            }
        };

        if active_view_box == old_view_box {
            return false; // nothing changed at all
        }

        let new_chunk_box =
            active_view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32));

        let old_chunk_box =
            old_view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32));

        if new_chunk_box == old_chunk_box {
            return true; // set of active chunks didn't change but visible blocks at edges did
        }

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

        true
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
        queue: &wgpu::Queue,
        world: &World,
    ) -> bool {
        let mut any_updates = false;

        let mut fill_block_types =
            self.block_type_helper
                .begin_fill_buffer(queue, &self.block_type_buf, 0);

        let mut fill_chunk_metadatas =
            self.chunk_metadata_helper
                .begin_fill_buffer(queue, &self.chunk_metadata_buf, 0);

        // Copy chunk data to GPU buffers for only the chunks that have changed since last time
        // buffers were updated.
        for (index, opt_point) in self.active_chunks.take_diff().changed_entries().into_iter() {
            any_updates = true;

            if let Some((&p, _)) = opt_point {
                let chunk = world.blocks.chunks().get(convert_point!(p, usize)).unwrap();
                let chunk_data = block::blocks_to_u16(&chunk.blocks);
                fill_block_types.seek(index * index_utils::chunk_size_total());
                fill_block_types.advance(chunk_data);
            } else {
                // if a chunk was deleted, write inactive metadata
                let meta = ChunkMeta::empty();
                fill_chunk_metadatas.seek(index);
                fill_chunk_metadatas.advance(&[meta]);
            }
        }

        fill_block_types.finish();

        if any_updates {
            // If any chunks have been updated, update metadata for every chunk. In theory we only
            // need to update new/deleted chunks and those with neighbours that are new/deleted.
            // Updating everything doesn't cost too much though.

            for (&point, index, _) in self.active_chunks.iter() {
                let meta = Self::make_chunk_meta(&self.active_chunks, point);
                fill_chunk_metadatas.seek(index);
                fill_chunk_metadatas.advance(&[meta]);
            }
        }

        fill_chunk_metadatas.finish();

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
            offset: offset.into(),
            _padding0: [0f32],
            neighbor_indices,
            active: 1,
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

impl BufferSyncable for RenderUniforms {
    type Item = f32;

    fn sync<'a>(
        &self,
        fill_buffer: &mut FillBuffer<'a, Self::Item>,
    ) {
        let proj: &[f32; 16] = self.proj.as_ref();
        let view: &[f32; 16] = self.view.as_ref();
        let model: &[f32; 16] = self.model.as_ref();
        let camera_pos: &[f32; 3] = self.camera_pos.as_ref();

        fill_buffer.advance(proj);
        fill_buffer.advance(view);
        fill_buffer.advance(model);
        fill_buffer.advance(camera_pos);
    }
}

impl IntoBufferSynced for RenderUniforms {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            label: "render uniforms",
            buffer_len: 16 + 16 + 16 + 4,
            max_chunk_len: std::mem::size_of::<Matrix4<f32>>(),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl BufferSyncable for RenderLocals {
    type Item = u8;

    fn sync<'a>(
        &self,
        fill_buffer: &mut FillBuffer<'a, Self::Item>,
    ) {
        fill_buffer.advance(self.as_bytes());
    }
}

impl IntoBufferSynced for RenderLocals {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            label: "render locals",
            buffer_len: std::mem::size_of::<Self>(),
            max_chunk_len: std::mem::size_of::<Self>(),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ComputeUniforms {
    fn update_view_box(&mut self, view_box: Bounds<i32>) {
        self.visible_box_origin = convert_point!(view_box.origin(), f32);
        self.visible_box_limit = convert_point!(view_box.limit(), f32);
    }
}

impl BufferSyncable for ComputeUniforms {
    type Item = u8;

    fn sync<'a>(
        &self,
        fill_buffer: &mut FillBuffer<'a, Self::Item>,
    ) {
        let visible_box_origin = self.visible_box_origin.as_ref() as &[f32; 3];
        let visible_box_limit = self.visible_box_limit.as_ref() as &[f32; 3];

        fill_buffer.advance(visible_box_origin.as_bytes());
        fill_buffer.advance(0f32.as_bytes());
        fill_buffer.advance(visible_box_limit.as_bytes());
        fill_buffer.advance(0f32.as_bytes());
        fill_buffer.advance(self.cube.faces.as_bytes());
    }
}

impl IntoBufferSynced for ComputeUniforms {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        let face_len = 6 * std::mem::size_of::<mesh::cube::Face>();
        let full_len = (4 + 4) * 4 + face_len;

        BufferSyncHelperDesc {
            label: "compute uniforms",
            buffer_len: full_len,
            max_chunk_len: full_len,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ChunkMeta {
    fn empty() -> Self {
        Self {
            offset: Point3::new(0.0, 0.0, 0.0).into(),
            _padding0: [0.0],
            neighbor_indices: [0, 0, 0, 0, 0, 0],
            active: 0,
            _padding1: 0,
        }
    }
}
