use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
use zerocopy::{AsBytes, FromBytes};

use simgame_core::{
    block::{self, index_utils, BlockConfigHelper, UpdatedBlocksState, WorldBlockData},
    convert_point, convert_vec,
    util::{Bounds, DivUp},
};

use crate::buffer_util::{
    BufferSyncHelper, BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer,
    InstancedBuffer, InstancedBufferDesc, IntoBufferSynced,
};
use crate::mesh::cube::Cube;
use crate::resource::ResourceLoader;
use crate::stable_map::StableMap;
use crate::world::{self, ViewParams};

pub(crate) struct BlocksRenderStateBuilder<'a> {
    pub depth_texture: &'a wgpu::Texture,
    pub view_state: &'a world::ViewState,
    pub blocks: &'a WorldBlockData,
    pub max_visible_chunks: usize,
    pub multi_draw_enabled: bool,
    pub resource_loader: &'a ResourceLoader,
    pub block_config: &'a BlockConfigHelper,
}

pub(crate) struct BlocksRenderState {
    multi_draw_enabled: bool,
    needs_compute_pass: bool,
    chunk_state: ChunkState,
    compute_stage: ComputeStage,
    render_stage: RenderStage,

    block_info_handler: BlockInfoHandler,
    geometry_buffers: GeometryBuffers,
}

struct ComputeStage {
    pipeline: wgpu::ComputePipeline,
    uniforms: BufferSyncedData<ComputeUniforms, ComputeUniforms>,
    bind_group_layout: wgpu::BindGroupLayout,
}

struct RenderStage {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture: wgpu::TextureView,
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct BlockRenderInfo {
    // index into texture list, by face
    face_tex_ids: [u32; 6],
    _padding: [u32; 2],
    vertex_data: Cube,
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct RenderUniforms {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_pos: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct ComputeUniforms {
    visible_box_origin: [f32; 3],
    _padding0: f32,
    visible_box_limit: [f32; 3],
    _padding1: f32,
}

type ActiveChunks = StableMap<Point3<i32>, ()>;

struct GeometryBuffers {
    faces: InstancedBuffer,
    indirect: InstancedBuffer,
    globals: InstancedBuffer,
}

struct ChunkState {
    active_chunks: ActiveChunks,
    meta_tracker: ChunkMetaTracker,

    chunk_metadata_helper: BufferSyncHelper<ChunkMeta>,
    chunk_metadata_buf: wgpu::Buffer,
    block_type_helper: BufferSyncHelper<u16>,
    block_type_buf: wgpu::Buffer,
    active_view_box: Option<Bounds<i32>>,
    max_visible_chunks: usize,
}

struct ChunkMetaTracker {
    touched: HashSet<usize>,
    chunk_metas: EndlessVec<ChunkMeta>,
    new_metas: HashMap<usize, ChunkMeta>,
}

struct EndlessVec<T> {
    data: Vec<T>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default, PartialEq)]
struct ChunkMeta {
    offset: [i32; 3],
    _padding0: [i32; 1],
    neighbor_indices: [i32; 6],
    active: u32,
    _padding1: i32,
}

#[allow(unused)]
struct BlockInfoHandler {
    index_map: HashMap<block::FaceTexture, usize>,
    dimensions: (u32, u32),

    texture_arr: wgpu::Texture,
    texture_arr_views: Vec<wgpu::TextureView>,
    sampler: wgpu::Sampler,

    block_info_buf: InstancedBuffer,
}

impl<'a> BlocksRenderStateBuilder<'a> {
    pub fn build(self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<BlocksRenderState> {
        let block_info_handler =
            BlockInfoHandler::new(self.block_config, self.resource_loader, device, queue)?;

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
            self.resource_loader
                .load_shader("shader/blocks/compute")?
                .as_slice(),
        ));

        let vert_shader = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
            self.resource_loader
                .load_shader("shader/blocks/vertex")?
                .as_slice(),
        ));

        let frag_shader = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
            self.resource_loader
                .load_shader("shader/blocks/fragment")?
                .as_slice(),
        ));

        let compute_stage = {
            let uniforms = ComputeUniforms::new(self.view_state.params.calculate_view_box())
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
                        // // Block render info
                        // wgpu::BindGroupLayoutEntry::new(
                        //     1,
                        //     wgpu::ShaderStage::COMPUTE,
                        //     wgpu::BindingType::StorageBuffer {
                        //         dynamic: false,
                        //         readonly: true,
                        //         min_binding_size: None,
                        //     },
                        // ),
                        // Block type buffer
                        wgpu::BindGroupLayoutEntry::new(
                            2,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Chunk metadata buffer
                        wgpu::BindGroupLayoutEntry::new(
                            3,
                            wgpu::ShaderStage::COMPUTE,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Output vertex buffer
                        wgpu::BindGroupLayoutEntry::new(
                            4,
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
                    module: &compute_shader,
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
                        // Block info
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
                        // Faces
                        wgpu::BindGroupLayoutEntry::new(
                            4,
                            wgpu::ShaderStage::VERTEX,
                            wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: true,
                                min_binding_size: None,
                            },
                        ),
                        // Textures
                        wgpu::BindGroupLayoutEntry {
                            count: Some(block_info_handler.index_map.len() as u32),
                            ..wgpu::BindGroupLayoutEntry::new(
                                5,
                                wgpu::ShaderStage::FRAGMENT,
                                wgpu::BindingType::SampledTexture {
                                    dimension: wgpu::TextureViewDimension::D2Array,
                                    component_type: wgpu::TextureComponentType::Float,
                                    multisampled: false,
                                },
                            )
                        },
                        // Sampler
                        wgpu::BindGroupLayoutEntry::new(
                            6,
                            wgpu::ShaderStage::FRAGMENT,
                            wgpu::BindingType::Sampler { comparison: false },
                        ),
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &pipeline_layout,
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vert_shader,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &frag_shader,
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
                    index_format: wgpu::IndexFormat::Uint16,
                    vertex_buffers: &[],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            });

            let uniforms = RenderUniforms::new(&self.view_state).into_buffer_synced(device);

            RenderStage {
                pipeline,
                uniforms,
                bind_group_layout,
                depth_texture: self.depth_texture.create_default_view(),
            }
        };

        let geometry_buffers = {
            let faces = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block faces",
                    instance_len: (4 * 3 * index_utils::chunk_size_total()) as usize,
                    n_instances: self.max_visible_chunks,
                    usage: wgpu::BufferUsage::STORAGE,
                },
            );

            let indirect = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block indirect",
                    instance_len: 4 * 4,
                    n_instances: self.max_visible_chunks,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
                },
            );

            let globals = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "compute globals",
                    instance_len: 8,
                    n_instances: 1,
                    usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
                },
            );

            GeometryBuffers {
                faces,
                indirect,
                globals,
            }
        };

        let mut chunk_state = ChunkState::new(
            device,
            self.max_visible_chunks,
            self.view_state.params.visible_size,
        );
        let mut needs_compute_pass = false;

        if let Some(active_view_box) = self.view_state.params.calculate_view_box() {
            if chunk_state.update_view_box(active_view_box, self.blocks) {
                needs_compute_pass = true;
            }
        }

        if chunk_state.update_buffers(queue, self.blocks) {
            needs_compute_pass = true;
        }

        Ok(BlocksRenderState {
            multi_draw_enabled: self.multi_draw_enabled,
            needs_compute_pass,
            chunk_state,
            compute_stage,
            render_stage,
            geometry_buffers,
            block_info_handler,
        })
    }
}

impl BlocksRenderState {
    pub fn set_view(&mut self, params: &ViewParams) {
        self.chunk_state.set_visible_size(params.visible_size);
    }

    pub fn set_depth_texture(&mut self, depth_texture: &wgpu::Texture) {
        self.render_stage.depth_texture = depth_texture.create_default_view();
    }

    pub fn render_frame(
        &mut self,
        frame_render: &world::FrameRender,
        _view_state: &world::ViewState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
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
        blocks: &WorldBlockData,
        diff: &UpdatedBlocksState,
        view_state: &world::ViewState,
    ) {
        if let Some(active_view_box) = view_state.params.calculate_view_box() {
            self.compute_stage.uniforms.update_view_box(active_view_box);

            if self.chunk_state.update_view_box(active_view_box, blocks) {
                self.needs_compute_pass = true;
            }
        } else if self.chunk_state.clear_view_box() {
            self.needs_compute_pass = true;
        }

        self.chunk_state.apply_chunk_diff(blocks, diff);
        if self.chunk_state.update_buffers(queue, blocks) {
            self.needs_compute_pass = true;
        }

        *self.render_stage.uniforms = RenderUniforms::new(view_state);
    }

    fn compute_pass(
        &mut self,
        frame_render: &world::FrameRender,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let bufs = &self.geometry_buffers;

        self.compute_stage.uniforms.sync(frame_render.queue);
        // reset compute shader globals to 0
        bufs.globals.clear(frame_render.queue);

        let bind_group = frame_render
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.compute_stage.bind_group_layout,
                bindings: &[
                    self.compute_stage.uniforms.as_binding(0),
                    // self.block_info_handler.block_info_buf.as_binding(1),
                    self.chunk_state.block_type_binding(2),
                    self.chunk_state.chunk_metadata_binding(3),
                    bufs.faces.as_binding(4),
                    bufs.indirect.as_binding(5),
                    bufs.globals.as_binding(6),
                ],
            });

        let mut cpass = encoder.begin_compute_pass();

        cpass.set_pipeline(&self.compute_stage.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(self.chunk_state.count_chunks() as u32, 1, 1);
    }

    fn render_pass(
        &mut self,
        frame_render: &world::FrameRender,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let background_color = wgpu::Color::BLACK;

        self.render_stage.uniforms.sync(frame_render.queue);

        let bind_group = frame_render
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.render_stage.bind_group_layout,
                bindings: &[
                    self.render_stage.uniforms.as_binding(0),
                    self.block_info_handler.block_info_buf.as_binding(1),
                    self.chunk_state.block_type_binding(2),
                    self.chunk_state.chunk_metadata_binding(3),
                    self.geometry_buffers.faces.as_binding(4),
                    wgpu::Binding {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureViewArray(
                            &self.block_info_handler.texture_arr_views[..],
                        ),
                    },
                    wgpu::Binding {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.block_info_handler.sampler),
                    },
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

        let bufs = &self.geometry_buffers;
        let indirect_buf = &bufs.indirect;

        if self.multi_draw_enabled {
            rpass.multi_draw_indirect(
                &indirect_buf.buffer(),
                0,
                self.chunk_state.active_chunks.capacity() as u32,
            );
        } else {
            for (_, chunk_index, _) in self.chunk_state.active_chunks.iter() {
                rpass.draw_indirect(
                    &indirect_buf.buffer(),
                    indirect_buf.instance_offset(chunk_index),
                );
            }
        }
    }
}

impl ChunkState {
    fn count_chunks(&self) -> usize {
        self.active_chunks.capacity()
    }

    pub fn new(
        device: &wgpu::Device,
        max_visible_chunks: usize,
        visible_size: Vector3<i32>,
    ) -> Self {
        let visible_chunk_size = visible_size_to_chunks(visible_size);

        let active_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;
        assert!(active_visible_chunks <= max_visible_chunks as i32);

        let max_visible_blocks = max_visible_chunks * index_utils::chunk_size_total() as usize;
        let active_chunks = ActiveChunks::new(max_visible_chunks as usize);

        let block_type_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            label: "block types",
            buffer_len: max_visible_blocks,
            max_chunk_len: index_utils::chunk_size_total() as usize,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        let chunk_metadata_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            label: "chunk metadata",
            buffer_len: max_visible_chunks,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

        ChunkState {
            meta_tracker: ChunkMetaTracker::with_capacity(active_chunks.capacity()),
            active_chunks,
            block_type_buf: block_type_helper.make_buffer(device),
            block_type_helper,
            chunk_metadata_buf: chunk_metadata_helper.make_buffer(device),
            chunk_metadata_helper,
            active_view_box: None,
            max_visible_chunks,
        }
    }

    pub fn set_visible_size(&mut self, visible_size: Vector3<i32>) {
        let visible_chunk_size = visible_size_to_chunks(visible_size);
        let max_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;

        assert!(max_visible_chunks <= self.max_visible_chunks as i32);
        if max_visible_chunks != self.active_chunks.capacity() as i32 {
            self.active_chunks.set_capacity(max_visible_chunks as usize);
        }
    }

    fn update_box_chunks(&mut self, view_box: Bounds<i32>, blocks: &WorldBlockData) {
        assert!(self.active_chunks.len() == 0);
        let bounds = Bounds::new(
            convert_point!(view_box.origin(), i64),
            convert_vec!(view_box.size(), i64),
        );
        for (p, _chunk) in blocks
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

    pub fn update_view_box(
        &mut self,
        active_view_box: Bounds<i32>,
        blocks: &WorldBlockData,
    ) -> bool {
        let old_view_box = self.active_view_box;
        self.active_view_box = Some(active_view_box);

        let old_view_box = match old_view_box {
            Some(x) => x,
            None => {
                // no chunks in previous view; insert all
                self.update_box_chunks(active_view_box, blocks);
                return true;
            }
        };

        if active_view_box == old_view_box {
            return false; // nothing changed at all
        }

        let new_chunk_box = view_box_to_chunks(active_view_box);
        let old_chunk_box = view_box_to_chunks(old_view_box);

        if new_chunk_box == old_chunk_box {
            return true; // set of active chunks didn't change but visible blocks at edges did
        }

        // if we get here we need to update the set of active chunks

        // 1. delete chunks that have left the view
        for pos in old_chunk_box.iter_diff(new_chunk_box) {
            self.active_chunks.remove(&pos);
        }

        // 2. insert chunks that are newly in the view
        for pos in new_chunk_box.iter_diff(old_chunk_box) {
            if let Some(_chunk) = blocks.chunks().get(convert_point!(pos, i64)) {
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

    pub fn apply_chunk_diff(&mut self, blocks: &WorldBlockData, diff: &UpdatedBlocksState) {
        let active_chunk_box = match self.active_view_box {
            Some(active_view_box) => view_box_to_chunks(active_view_box),
            None => return,
        };

        for &pos in &diff.modified_chunks {
            let pos_i32 = convert_point!(pos, i32);
            if !active_chunk_box.contains_point(pos_i32) {
                continue;
            }

            if let Some(_chunk) = blocks.chunks().get(pos) {
                self.active_chunks.update(pos_i32, ());
            } else {
                self.active_chunks.remove(&pos_i32);
            }
        }
    }

    fn update_buffers(&mut self, queue: &wgpu::Queue, blocks: &WorldBlockData) -> bool {
        let mut any_updates = false;

        let mut fill_block_types =
            self.block_type_helper
                .begin_fill_buffer(queue, &self.block_type_buf, 0);

        let mut fill_chunk_metadatas =
            self.chunk_metadata_helper
                .begin_fill_buffer(queue, &self.chunk_metadata_buf, 0);

        self.meta_tracker.reset();

        let diff = self.active_chunks.take_diff();
        let active_chunks = diff.inner();

        // Copy chunk data to GPU buffers for only the chunks that have changed since last time
        // buffers were updated.
        for (index, opt_point) in diff.changed_entries().into_iter() {
            any_updates = true;

            if let Some((&point, _)) = opt_point {
                self.meta_tracker.modify(point, index, active_chunks);

                let chunk = blocks.chunks().get(convert_point!(point, i64)).unwrap();
                let chunk_data = block::blocks_to_u16(&chunk.blocks);
                fill_block_types.seek(index * index_utils::chunk_size_total() as usize);
                fill_block_types.advance(chunk_data);
            } else {
                self.meta_tracker.remove(index)
            }
        }

        for (index, meta) in self.meta_tracker.update(&active_chunks) {
            fill_chunk_metadatas.seek(index);
            fill_chunk_metadatas.advance(&[meta]);
        }

        fill_block_types.finish();
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

impl BlockInfoHandler {
    fn new(
        config: &BlockConfigHelper,
        resource_loader: &ResourceLoader,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Self> {
        let index_map = config.texture_index_map();
        log::info!("Got {} block textures total", index_map.len());
        let dimensions = Self::find_max_dimensions(&index_map, resource_loader)?;
        log::info!(
            "Block texture dimensions are {}x{}",
            dimensions.0,
            dimensions.1
        );

        let mip_level_count = Self::log2_exact(dimensions.0.min(dimensions.1))
            .expect("expected dimensions to be powers of 2");

        let texture_arr = Self::make_texture(
            dimensions,
            mip_level_count,
            &index_map,
            resource_loader,
            device,
            queue,
        )?;

        let texture_arr_views = (0..index_map.len())
            .map(|index| {
                texture_arr.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("block textures"),
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    dimension: wgpu::TextureViewDimension::D2,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    level_count: mip_level_count,
                    base_array_layer: index as u32,
                    array_layer_count: 1,
                })
            })
            .collect();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: None,
            ..Default::default()
        });

        let block_info_buf = InstancedBuffer::new(
            device,
            InstancedBufferDesc {
                label: "block info",
                instance_len: std::mem::size_of::<BlockRenderInfo>(),
                n_instances: config.blocks().len(),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            },
        );

        for (index, block) in config.blocks().iter().enumerate() {
            let render_info = BlockRenderInfo {
                face_tex_ids: Self::get_face_textures(&index_map, block)
                    .ok_or_else(|| anyhow!("Unable to get face textures for block {:?}", block))?,
                _padding: [0; 2],
                vertex_data: Cube::new(),
            };

            block_info_buf.write(queue, index, render_info.as_bytes());
        }

        Ok(Self {
            index_map,
            dimensions,

            texture_arr_views,
            texture_arr,
            sampler,

            block_info_buf,
        })
    }

    fn make_texture(
        dimensions: (u32, u32),
        mip_level_count: u32,
        index_map: &HashMap<block::FaceTexture, usize>,
        resource_loader: &ResourceLoader,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<wgpu::Texture> {
        let texture_arr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block texture array"),
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth: index_map.len() as u32,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
        });

        let face_texes = {
            let mut face_texes: Vec<_> = index_map
                .iter()
                .map(|(face_tex, &index)| (index, face_tex))
                .collect();
            face_texes.sort_by_key(|(index, _)| *index);
            face_texes
        };

        for (index, face_tex) in face_texes {
            let sample_generator = face_tex::to_sample_generator(resource_loader, face_tex)?;

            for mip_level in 0..mip_level_count {
                let mip_dimensions = (dimensions.0 >> mip_level, dimensions.1 >> mip_level);

                let copy_view = wgpu::TextureCopyView {
                    texture: &texture_arr,
                    mip_level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: index as u32,
                    },
                };

                let (samples, bytes_per_row) =
                    sample_generator.generate(mip_dimensions.0, mip_dimensions.1);

                queue.write_texture(
                    copy_view,
                    samples.as_slice(),
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: 0,
                    },
                    wgpu::Extent3d {
                        width: mip_dimensions.0,
                        height: mip_dimensions.1,
                        depth: 1,
                    },
                );
            }
        }

        Ok(texture_arr)
    }

    fn find_max_dimensions(
        index_map: &HashMap<block::FaceTexture, usize>,
        resource_loader: &ResourceLoader,
    ) -> Result<(u32, u32)> {
        // result will be 1x1 if every face texture is a solid color
        let mut res: (u32, u32) = (1, 1);

        for (face_tex, _) in index_map.iter() {
            match face_tex {
                block::FaceTexture::Texture { resource } => {
                    let reader = resource_loader.open_image(&resource[..])?;
                    let (width, height) = reader.into_dimensions()?;
                    if Self::log2_exact(width).is_none() || Self::log2_exact(height).is_none() {
                        bail!("Block texture resource {:?} has dimensions {}x{}. Expected powers of 2.",
                              resource, width, height);
                    }

                    let (old_width, old_height) = &mut res;
                    if width > *old_width {
                        *old_width = width;
                    }
                    if height > *old_height {
                        *old_height = height;
                    }
                }
                _ => {}
            }
        }

        Ok(res)
    }

    fn get_face_textures(
        index_map: &HashMap<block::FaceTexture, usize>,
        block: &block::BlockInfo,
    ) -> Option<[u32; 6]> {
        // Some([0, 1, 2, 3, 0, 1])

        match &block.texture {
            block::BlockTexture::Uniform(face_tex) => {
                let index = *index_map.get(face_tex)? as u32;
                Some([index; 6])
            }
            block::BlockTexture::Nonuniform { top, bottom, side } => {
                let index_top = *index_map.get(top)? as u32;
                let index_bottom = *index_map.get(bottom)? as u32;
                let index_sides = *index_map.get(side)? as u32;
                Some([
                    index_top,
                    index_bottom,
                    index_sides,
                    index_sides,
                    index_sides,
                    index_sides,
                ])
            }
        }
    }

    /// If the value is a power of 2, returns its log base 2. Otherwise, returns `None`.
    fn log2_exact(value: u32) -> Option<u32> {
        if value == 0 {
            return None;
        }

        let mut log2 = 0;
        let mut shifted = value;
        while shifted > 1 {
            shifted = shifted >> 1;
            log2 += 1;
        }

        if 1 << log2 == value {
            Some(log2)
        } else {
            None
        }
    }
}

mod face_tex {
    use anyhow::Result;

    use simgame_core::block;

    use crate::resource::ResourceLoader;

    pub trait SampleGenerator {
        /// Returns the buffer of samples and the number of bytes per row in the result.
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32);
    }

    pub fn to_sample_generator(
        resource_loader: &ResourceLoader,
        face_tex: &block::FaceTexture,
    ) -> Result<Box<dyn SampleGenerator>> {
        match face_tex {
            block::FaceTexture::SolidColor { red, green, blue } => {
                log::info!(
                    "Creating solid color texture with R/G/B {}/{}/{}",
                    red,
                    green,
                    blue
                );
                Ok(Box::new(SolidColorSampleGenerator {
                    red: *red,
                    green: *green,
                    blue: *blue,
                }))
            }
            block::FaceTexture::Texture { resource } => {
                log::info!("Loading block texture {:?}", resource);
                let image = resource_loader.load_image(&resource[..])?.into_rgba();

                Ok(Box::new(ImageSampleGenerator { image }))
            }
        }
    }

    struct ImageSampleGenerator {
        image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    }

    struct SolidColorSampleGenerator {
        red: u8,
        green: u8,
        blue: u8,
    }

    impl SampleGenerator for ImageSampleGenerator {
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32) {
            let (img_width, img_height) = self.image.dimensions();

            let resized;
            let image = if img_width != width || img_height != height {
                resized = image::imageops::resize(
                    &self.image,
                    width,
                    height,
                    image::imageops::FilterType::CatmullRom,
                );
                &resized
            } else {
                &self.image
            };

            let samples = image.as_flat_samples();
            let bytes_per_row = samples.layout.height_stride as u32;
            (samples.as_slice().to_vec(), bytes_per_row)
        }
    }

    impl SampleGenerator for SolidColorSampleGenerator {
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32) {
            let mut buf = Vec::with_capacity(width as usize * height as usize * 4);
            for _row_ix in 0..height {
                for _col_ix in 0..width {
                    buf.push(self.red);
                    buf.push(self.green);
                    buf.push(self.blue);
                    buf.push(255); // alpha
                }
            }
            (buf, 4 * width)
        }
    }
}

impl<T> EndlessVec<T>
where
    T: Default + Clone,
{
    pub fn new(initial_len: usize) -> Self {
        Self {
            data: Vec::with_capacity(initial_len),
        }
    }

    pub fn get(&self, index: usize) -> T {
        if index < self.data.len() {
            self.data[index].clone()
        } else {
            T::default()
        }
    }

    pub fn set(&mut self, index: usize, value: T) {
        if index >= self.data.len() {
            self.data
                .extend(std::iter::repeat(T::default()).take(1 + index - self.data.len()));
        }

        self.data[index] = value;
    }
}

impl ChunkMetaTracker {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            touched: HashSet::with_capacity(capacity),
            chunk_metas: EndlessVec::new(capacity),
            new_metas: HashMap::with_capacity(capacity),
        }
    }

    fn reset(&mut self) {
        self.touched.clear();
        self.new_metas.clear();
    }

    fn touch_neighbors(touched: &mut HashSet<usize>, meta: &ChunkMeta) {
        if meta.active == 0 {
            return;
        }

        for &neighbor_index in &meta.neighbor_indices {
            if neighbor_index != -1 {
                touched.insert(neighbor_index as usize);
            }
        }
    }

    fn modify(&mut self, point: Point3<i32>, index: usize, active_chunks: &ActiveChunks) {
        let old_meta = self.chunk_metas.get(index);
        if old_meta.active == 1 {
            let old_point: Point3<i32> = old_meta.offset.into();

            if old_point == point {
                // an updated chunk whose location has not changed does no trigger any metadata
                // updates
                return;
            }

            Self::touch_neighbors(&mut self.touched, &old_meta);
        } else {
            let new_meta = Self::make_chunk_meta(active_chunks, point);
            Self::touch_neighbors(&mut self.touched, &new_meta);
            self.new_metas.insert(index, new_meta);
        }
    }

    fn remove(&mut self, index: usize) {
        let old_meta = self.chunk_metas.get(index);
        Self::touch_neighbors(&mut self.touched, &old_meta);
        self.new_metas.insert(index, ChunkMeta::default());
    }

    fn update<'a>(
        &'a mut self,
        active_chunks: &ActiveChunks,
    ) -> impl Iterator<Item = (usize, ChunkMeta)> + 'a {
        for &index in &self.touched {
            if index > active_chunks.capacity() || self.new_metas.contains_key(&index) {
                continue;
            }

            let (&point, _) = match active_chunks.index(index) {
                Some(x) => x,
                None => continue,
            };
            self.new_metas
                .insert(index, Self::make_chunk_meta(active_chunks, point));
        }

        for (&index, &chunk_meta) in &self.new_metas {
            self.chunk_metas.set(index, chunk_meta);
        }

        self.new_metas.iter().map(|(index, meta)| (*index, *meta))
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

        let offset =
            p.mul_element_wise(Point3::origin() + convert_vec!(index_utils::chunk_size(), i32));

        let neighbor_indices = make_neighbor_indices(&active_chunks, p);

        ChunkMeta {
            offset: offset.into(),
            _padding0: [0],
            neighbor_indices,
            active: 1,
            _padding1: 0,
        }
    }
}

impl RenderUniforms {
    fn new(view_state: &world::ViewState) -> Self {
        Self {
            proj: view_state.proj().into(),
            model: view_state.model().into(),
            view: view_state.view().into(),
            camera_pos: view_state.camera_pos().into(),
        }
    }
}

impl BufferSyncable for RenderUniforms {
    type Item = RenderUniforms;

    fn sync<'a>(&self, fill_buffer: &mut FillBuffer<'a, Self::Item>) {
        fill_buffer.advance(&[*self]);
    }
}

impl IntoBufferSynced for RenderUniforms {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            label: "world blocks render uniforms",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ComputeUniforms {
    fn new(view_box: Option<Bounds<i32>>) -> Self {
        let view_box = match view_box {
            Some(x) => x,
            None => Bounds::new(Point3::new(0, 0, 0), Vector3::new(0, 0, 0)),
        };

        Self {
            visible_box_origin: convert_point!(view_box.origin(), f32).into(),
            _padding0: 0f32,
            visible_box_limit: convert_point!(view_box.limit(), f32).into(),
            _padding1: 0f32,
        }
    }

    fn update_view_box(&mut self, view_box: Bounds<i32>) {
        self.visible_box_origin = convert_point!(view_box.origin(), f32).into();
        self.visible_box_limit = convert_point!(view_box.limit(), f32).into();
    }
}

impl BufferSyncable for ComputeUniforms {
    type Item = ComputeUniforms;

    fn sync<'a>(&self, fill_buffer: &mut FillBuffer<'a, Self::Item>) {
        fill_buffer.advance(&[*self]);
    }
}

impl IntoBufferSynced for ComputeUniforms {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            label: "compute uniforms",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        }
    }
}

pub fn visible_size_to_chunks(visible_size: Vector3<i32>) -> Vector3<i32> {
    visible_size.div_up(convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}

pub fn view_box_to_chunks(view_box: Bounds<i32>) -> Bounds<i32> {
    view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32))
}
