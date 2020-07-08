use std::collections::{HashMap, HashSet};

use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
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
use crate::mesh::cube::Cube;
use crate::stable_map::StableMap;
use crate::world::{self, ViewParams};
use crate::LoadedWorldShaders;

pub struct BlocksRenderInit<'a> {
    pub shaders: &'a LoadedWorldShaders,
    pub depth_texture: &'a wgpu::Texture,
    pub view_state: &'a world::ViewState,
    pub world: &'a World,
    pub max_visible_chunks: usize,
    pub multi_draw_enabled: bool,
}

pub struct BlocksRenderState {
    multi_draw_enabled: bool,
    needs_compute_pass: bool,
    chunk_state: ChunkState,
    compute_stage: ComputeStage,
    render_stage: RenderStage,

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
    locals: BufferSyncedData<RenderLocals, RenderLocals>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct RenderLocals {
    cube: Cube,
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
    cube: Cube,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default, PartialEq)]
struct ChunkMeta {
    offset: [i32; 3],
    _padding0: [i32; 1],
    neighbor_indices: [i32; 6],
    active: u32,
    _padding1: i32,
}

impl BlocksRenderState {
    pub fn set_view(&mut self, params: &ViewParams) {
        self.chunk_state.set_visible_size(params.visible_size);
    }

    pub fn new(init: BlocksRenderInit, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let cube = Cube::new();

        let compute_stage = {
            let uniforms =
                ComputeUniforms::new(init.view_state.params.calculate_view_box(init.world), cube)
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

            let locals = RenderLocals { cube }.into_buffer_synced(device);

            let uniforms = RenderUniforms::new(&init.view_state).into_buffer_synced(device);

            RenderStage {
                pipeline,
                uniforms,
                bind_group_layout,
                depth_texture: init.depth_texture.create_default_view(),
                locals,
            }
        };

        let geometry_buffers = {
            let faces = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block faces",
                    instance_len: 4 * 3 as usize * index_utils::chunk_size_total(),
                    n_instances: init.max_visible_chunks,
                    usage: wgpu::BufferUsage::STORAGE,
                },
            );

            let indirect = InstancedBuffer::new(
                device,
                InstancedBufferDesc {
                    label: "block indirect",
                    instance_len: 4 * 4,
                    n_instances: init.max_visible_chunks,
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
            init.max_visible_chunks,
            init.view_state.params.visible_size,
        );
        let mut needs_compute_pass = false;

        if let Some(active_view_box) = init.view_state.params.calculate_view_box(init.world) {
            if chunk_state.update_view_box(active_view_box, init.world) {
                needs_compute_pass = true;
            }
        }

        if chunk_state.update_buffers(queue, init.world) {
            needs_compute_pass = true;
        }

        Self {
            multi_draw_enabled: init.multi_draw_enabled,
            needs_compute_pass,
            chunk_state,
            compute_stage,
            render_stage,
            geometry_buffers,
        }
    }

    pub fn set_depth_texture(&mut self, depth_texture: &wgpu::Texture) {
        self.render_stage.depth_texture = depth_texture.create_default_view();
    }

    pub fn render_frame(
        &mut self,
        frame_render: &world::FrameRender,
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
        world: &World,
        diff: &UpdatedWorldState,
        view_state: &world::ViewState,
    ) {
        if let Some(active_view_box) = view_state.params.calculate_view_box(world) {
            self.compute_stage.uniforms.update_view_box(active_view_box);

            if self.chunk_state.update_view_box(active_view_box, world) {
                self.needs_compute_pass = true;
            }
        } else if self.chunk_state.clear_view_box() {
            self.needs_compute_pass = true;
        }

        self.chunk_state.apply_chunk_diff(world, diff);
        if self.chunk_state.update_buffers(queue, world) {
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
                    self.chunk_state.block_type_binding(1),
                    self.chunk_state.chunk_metadata_binding(2),
                    bufs.faces.as_binding(3),
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
        self.render_stage.locals.sync(frame_render.queue);

        let bind_group = frame_render
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.render_stage.bind_group_layout,
                bindings: &[
                    self.render_stage.uniforms.as_binding(0),
                    self.render_stage.locals.as_binding(1),
                    self.chunk_state.block_type_binding(2),
                    self.chunk_state.chunk_metadata_binding(3),
                    self.geometry_buffers.faces.as_binding(4),
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

        let max_visible_blocks = max_visible_chunks * index_utils::chunk_size_total();
        let active_chunks = ActiveChunks::new(max_visible_chunks as usize);

        let block_type_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            label: "block types",
            buffer_len: max_visible_blocks,
            max_chunk_len: index_utils::chunk_size_total(),
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

        let new_chunk_box = view_box_to_chunks(active_view_box);
        let old_chunk_box = view_box_to_chunks(old_view_box);

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
            Some(active_view_box) => view_box_to_chunks(active_view_box),
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

    fn update_buffers(&mut self, queue: &wgpu::Queue, world: &World) -> bool {
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

                let chunk = world
                    .blocks
                    .chunks()
                    .get(convert_point!(point, usize))
                    .unwrap();
                let chunk_data = block::blocks_to_u16(&chunk.blocks);
                fill_block_types.seek(index * index_utils::chunk_size_total());
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

struct ChunkMetaTracker {
    touched: HashSet<usize>,
    chunk_metas: EndlessVec<ChunkMeta>,
    new_metas: HashMap<usize, ChunkMeta>,
}

struct EndlessVec<T> {
    data: Vec<T>,
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
            if self.new_metas.contains_key(&index) {
                continue;
            }

            let (&point, _) = active_chunks
                .index(index)
                .expect("touched chunk without new_metas entry was deleted");
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
            camera_pos: view_state.params().camera_pos.into(),
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
            label: "render uniforms",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl BufferSyncable for RenderLocals {
    type Item = RenderLocals;

    fn sync<'a>(&self, fill_buffer: &mut FillBuffer<'a, Self::Item>) {
        fill_buffer.advance(&[*self]);
    }
}

impl IntoBufferSynced for RenderLocals {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            label: "render locals",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ComputeUniforms {
    fn new(view_box: Option<Bounds<i32>>, cube: Cube) -> Self {
        let view_box = match view_box {
            Some(x) => x,
            None => Bounds::new(Point3::new(0, 0, 0), Vector3::new(0, 0, 0)),
        };

        Self {
            visible_box_origin: convert_point!(view_box.origin(), f32).into(),
            _padding0: 0f32,
            visible_box_limit: convert_point!(view_box.limit(), f32).into(),
            _padding1: 0f32,
            cube,
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
    visible_size.div_up(&convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}

pub fn view_box_to_chunks(view_box: Bounds<i32>) -> Bounds<i32> {
    view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32))
}
