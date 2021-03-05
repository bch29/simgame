mod block_info;
mod chunk_state;

use std::convert::TryInto;
use std::time::Instant;

use anyhow::Result;
use cgmath::{Point3, Vector3};
use zerocopy::{AsBytes, FromBytes};

use simgame_blocks::{index_utils, BlockConfigHelper, BlockData, UpdatedBlocksState};
use simgame_util::{convert_point, convert_vec, Bounds, DivUp};

use crate::buffer_util::{
    BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, InstancedBuffer,
    InstancedBufferDesc, IntoBufferSynced,
};
use crate::mesh::cube::Cube;
use crate::world::{self, ViewParams};

use block_info::BlockInfoHandler;
use chunk_state::ChunkState;

pub(crate) struct BlocksRenderStateBuilder<'a> {
    pub depth_texture: &'a wgpu::Texture,
    pub view_state: &'a world::ViewState,
    pub blocks: &'a BlockData,
    pub max_visible_chunks: usize,
    pub block_config: &'a BlockConfigHelper,
}

pub(crate) struct BlocksRenderState {
    chunk_state: ChunkState,
    compute_stage: ComputeStage,
    render_stage: RenderStage,

    #[allow(dead_code)]
    block_info_handler: BlockInfoHandler,
    geometry_buffers: GeometryBuffers,
}

struct ComputeStage {
    pipeline: wgpu::ComputePipeline,
    uniforms: BufferSyncedData<ComputeUniforms, ComputeUniforms>,

    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

struct RenderStage {
    pipeline: wgpu::RenderPipeline,
    depth_texture: wgpu::TextureView,
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,

    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct BlockRenderInfo {
    // index into texture list, by face
    face_tex_ids: [u32; 6],
    _padding: [u32; 2],
    vertex_data: Cube,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct BlockTextureMetadata {
    periodicity: u32,
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

struct GeometryBuffers {
    faces: InstancedBuffer,
    indirect: InstancedBuffer,
    globals: InstancedBuffer,
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

impl<'a> BlocksRenderStateBuilder<'a> {
    pub fn build(self, ctx: &crate::GraphicsContext) -> Result<BlocksRenderState> {
        let block_info_handler =
            BlockInfoHandler::new(self.block_config, &ctx.resource_loader, ctx)?;

        let compute_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    ctx.resource_loader
                        .load_shader("shader/blocks/compute")?
                        .into(),
                ),
            });

        let vert_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    ctx.resource_loader
                        .load_shader("shader/blocks/vertex")?
                        .into(),
                ),
            });

        let frag_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    ctx.resource_loader
                        .load_shader("shader/blocks/fragment")?
                        .into(),
                ),
            });

        let geometry_buffers = self.build_geometry_buffers(ctx);

        let mut chunk_state = ChunkState::new(
            &ctx.device,
            self.max_visible_chunks,
            self.view_state.params.visible_size,
        );

        if let Some(active_view_box) = self.view_state.params.calculate_view_box() {
            chunk_state.update_view_box(active_view_box, self.blocks);
        }

        log::info!("initializing blocks compute stage");
        let compute_stage =
            self.build_compute_stage(ctx, &chunk_state, &geometry_buffers, compute_shader);

        log::info!("initializing blocks render stage");
        let render_stage = self.build_render_stage(
            ctx,
            &block_info_handler,
            &chunk_state,
            &geometry_buffers,
            vert_shader,
            frag_shader,
        );

        Ok(BlocksRenderState {
            chunk_state,
            compute_stage,
            render_stage,
            geometry_buffers,
            block_info_handler,
        })
    }

    fn build_compute_stage(
        &self,
        ctx: &crate::GraphicsContext,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
        compute_shader: wgpu::ShaderModule,
    ) -> ComputeStage {
        let uniforms = ComputeUniforms::new(self.view_state.params.calculate_view_box())
            .into_buffer_synced(&ctx.device);

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blocks compute layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Block type buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Chunk metadata buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Output vertex buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: None,
                            },
                        },
                        // Output indirect buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: None,
                            },
                        },
                        // Globals within a compute invocation
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                min_binding_size: None,
                            },
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &compute_shader,
                entry_point: "main",
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                uniforms.as_binding(0),
                // self.block_info_handler.block_info_buf.as_binding(1),
                chunk_state.block_type_binding(1),
                chunk_state.chunk_metadata_binding(2),
                geometry_buffers.faces.as_binding(3),
                geometry_buffers.indirect.as_binding(4),
                geometry_buffers.globals.as_binding(5),
            ],
        });

        ComputeStage {
            pipeline,
            uniforms,
            bind_group_layout,
            bind_group,
        }
    }

    fn build_render_stage(
        &self,
        ctx: &crate::GraphicsContext,
        block_info_handler: &BlockInfoHandler,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
        vert_shader: wgpu::ShaderModule,
        frag_shader: wgpu::ShaderModule,
    ) -> RenderStage {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blocks vertex layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                min_binding_size: None,
                                ty: wgpu::BufferBindingType::Uniform,
                            },
                        },
                        // Block info
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Block types
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Chunk metadata
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Faces
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Texture metadata
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStage::VERTEX,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                        // Textures
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            count: Some(
                                (block_info_handler.index_map.len() as u32)
                                    .try_into()
                                    .expect("index map len must be nonzero"),
                            ),
                            ty: wgpu::BindingType::Texture {
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                        },
                        // Sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            count: None,
                            ty: wgpu::BindingType::Sampler {
                                filtering: true,
                                comparison: false,
                            },
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vert_shader,
                    entry_point: "main",
                    buffers: &[], // index_format: wgpu::IndexFormat::Uint16,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &frag_shader,
                    entry_point: "main",
                    targets: &[wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        color_blend: wgpu::BlendState::REPLACE,
                        alpha_blend: wgpu::BlendState::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::Back,
                    strip_index_format: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState {
                        front: wgpu::StencilFaceState::IGNORE,
                        back: wgpu::StencilFaceState::IGNORE,
                        read_mask: 0u32,
                        write_mask: 0u32,
                    },
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                    clamp_depth: false,
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
            });

        let uniforms = RenderUniforms::new(&self.view_state).into_buffer_synced(&ctx.device);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                uniforms.as_binding(0),
                block_info_handler.block_info_buf.as_binding(1),
                chunk_state.block_type_binding(2),
                chunk_state.chunk_metadata_binding(3),
                geometry_buffers.faces.as_binding(4),
                block_info_handler.texture_metadata_buf.as_binding(5),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureViewArray(
                        block_info_handler
                            .texture_arr_views
                            .iter()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&block_info_handler.sampler),
                },
            ],
        });

        RenderStage {
            pipeline,
            uniforms,
            depth_texture: self.depth_texture.create_view(&Default::default()),

            bind_group_layout,
            bind_group,
        }
    }

    fn build_geometry_buffers(&self, ctx: &crate::GraphicsContext) -> GeometryBuffers {
        let faces = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "block faces",
                instance_len: (4 * 3 * index_utils::chunk_size_total()) as usize,
                n_instances: self.max_visible_chunks,
                usage: wgpu::BufferUsage::STORAGE,
            },
        );

        let indirect = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "block indirect",
                instance_len: 4 * 4,
                n_instances: self.max_visible_chunks,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
            },
        );

        let globals = InstancedBuffer::new(
            &ctx.device,
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
    }
}

impl BlocksRenderState {
    pub fn set_view(&mut self, params: &ViewParams) {
        self.chunk_state.set_visible_size(params.visible_size);
    }

    pub fn set_depth_texture(&mut self, depth_texture: &wgpu::Texture) {
        self.render_stage.depth_texture = depth_texture.create_view(&Default::default());
    }

    pub fn render_frame(
        &mut self,
        ctx: &crate::GraphicsContext,
        frame_render: &mut crate::FrameRenderContext,
        _view_state: &world::ViewState,
    ) {
        let ts_begin = Instant::now();
        self.chunk_state.update_buffers(&ctx.queue);
        let ts_update = Instant::now();
        self.compute_pass(ctx, frame_render);
        self.render_pass(ctx, frame_render);

        metrics::timing!(
            "render.world.blocks.update_buffers",
            ts_update.duration_since(ts_begin)
        );
    }

    pub fn update(
        &mut self,
        blocks: &BlockData,
        diff: &UpdatedBlocksState,
        view_state: &world::ViewState,
    ) {
        if let Some(active_view_box) = view_state.params.calculate_view_box() {
            self.compute_stage.uniforms.update_view_box(active_view_box);

            self.chunk_state.update_view_box(active_view_box, blocks);
        } else {
            self.chunk_state.clear_view_box();
        }

        self.chunk_state.apply_chunk_diff(blocks, diff);
        *self.render_stage.uniforms = RenderUniforms::new(view_state);
    }

    fn compute_pass(
        &mut self,
        ctx: &crate::GraphicsContext,
        frame_render: &mut crate::FrameRenderContext,
    ) {
        let bufs = &self.geometry_buffers;

        self.compute_stage.uniforms.sync(&ctx.queue);
        // reset compute shader globals to 0
        bufs.globals.clear(&ctx.queue);

        let mut cpass = frame_render
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        cpass.set_pipeline(&self.compute_stage.pipeline);
        cpass.set_bind_group(0, &self.compute_stage.bind_group, &[]);
        cpass.dispatch(self.chunk_state.count_chunks() as u32, 1, 1);
    }

    fn render_pass(
        &mut self,
        ctx: &crate::GraphicsContext,
        frame_render: &mut crate::FrameRenderContext,
    ) {
        let background_color = wgpu::Color::BLACK;

        self.render_stage.uniforms.sync(&ctx.queue);

        let mut rpass = frame_render
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
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
        rpass.set_bind_group(0, &self.render_stage.bind_group, &[]);

        let bufs = &self.geometry_buffers;
        let indirect_buf = &bufs.indirect;

        if ctx.multi_draw_enabled {
            rpass.multi_draw_indirect(
                &indirect_buf.buffer(),
                0,
                self.chunk_state.count_chunks() as u32,
            );
        } else {
            for chunk_index in self.chunk_state.iter_chunk_indices() {
                rpass.draw_indirect(
                    &indirect_buf.buffer(),
                    indirect_buf.instance_offset(chunk_index),
                );
            }
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

fn view_box_to_chunks(view_box: Bounds<i32>) -> Bounds<i32> {
    view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32))
}
