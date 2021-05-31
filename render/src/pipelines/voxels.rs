mod chunk_state;
mod voxel_info;

use std::{convert::TryInto};

use anyhow::Result;
use cgmath::Matrix4;
use zerocopy::{AsBytes, FromBytes};

use simgame_types::Directory;
use simgame_util::{convert_point, Bounds};
use simgame_voxels::{index_utils, VoxelData, VoxelDelta};

use crate::{
    buffer_util::{
        BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, InstancedBuffer,
        InstancedBufferDesc, IntoBufferSynced,
    },
    pipelines::{self, GraphicsContext},
    resource::ResourceLoader,
    ViewState,
};

use chunk_state::ChunkState;
use voxel_info::VoxelInfoManager;

/// Holds the static pipeline for rendering voxels. Can be used to render multiple different pieces
/// of voxel geometry simultaneously.
pub(crate) struct VoxelRenderPipeline {
    voxel_info: VoxelInfoManager,

    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,

    render_pipeline: wgpu::RenderPipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Clone, Copy)]
pub(crate) struct VoxelRenderInput<'a> {
    pub model: Matrix4<f32>,
    pub voxels: &'a VoxelData,
    pub max_visible_chunks: usize,
    pub view_state: &'a ViewState,
}

#[derive(Clone, Copy)]
pub(crate) struct VoxelRenderInputDelta<'a> {
    pub model: Matrix4<f32>,
    pub view_state: &'a ViewState,
    pub voxels: &'a VoxelData,
    pub voxel_diff: &'a VoxelDelta,
}

/// Holds the current state (including GPU buffers) of rendering a particular piece of voxel
/// geometry.
pub(crate) struct VoxelRenderState {
    chunk_state: ChunkState,
    compute_stage: ComputeStage,
    render_stage: RenderStage,
    geometry_buffers: GeometryBuffers,
}

struct ComputeStage {
    uniforms: BufferSyncedData<ComputeUniforms, ComputeUniforms>,
    bind_group: wgpu::BindGroup,
}

struct RenderStage {
    depth_texture: wgpu::TextureView,
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct RenderUniforms {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding0: f32,
    visible_box_origin: [f32; 3],
    _padding1: f32,
    visible_box_limit: [f32; 3],
    _padding2: f32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default, PartialEq)]
struct IndirectCommand {
    count: u32,         // number of vertices per instance
    instance_count: u32, // number of instances to draw
    first: u32,         // index of first vertex
    base_instance: u32,  // index of first instance
}

impl pipelines::Pipeline for VoxelRenderPipeline {
    type State = VoxelRenderState;
    fn create_pipeline(
        ctx: &GraphicsContext,
        directory: &Directory,
        resource_loader: &ResourceLoader,
    ) -> Result<Self> {
        let voxel_info = pipelines::voxels::VoxelInfoManager::new(directory, ctx)?;

        let compute_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    resource_loader.load_shader("shader/voxels/compute")?.into(),
                ),
            });

        let vert_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    resource_loader.load_shader("shader/voxels/vertex")?.into(),
                ),
            });

        let frag_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: wgpu::ShaderFlags::default(),
                source: wgpu::ShaderSource::SpirV(
                    resource_loader
                        .load_shader("shader/voxels/fragment")?
                        .into(),
                ),
            });

        let compute_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxels compute layout"),
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
                        // Voxel type buffer
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
                        // Compute commands
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                min_binding_size: None,
                            },
                        },
                    ],
                });

        let compute_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_pipeline_layout),
                    module: &compute_shader,
                    entry_point: "main",
                });

        let render_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxels vertex layout"),
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
                        // Voxel info
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
                                (voxel_info.texture_array().len() as u32)
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

        let render_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&render_pipeline_layout),
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
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    strip_index_format: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    clamp_depth: false,
                    conservative: false,
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
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
            });

        Ok(VoxelRenderPipeline {
            voxel_info,

            compute_pipeline,
            compute_bind_group_layout,

            render_pipeline,
            render_bind_group_layout,
        })
    }

    fn render_frame(
        &self,
        ctx: &crate::GraphicsContext,
        load_action: crate::pipelines::LoadAction,
        frame_render: &mut crate::FrameRenderContext,
        state: &mut VoxelRenderState,
    ) {
        frame_render.encoder.write_timestamp(&ctx.timestamp_query_set, 1);
        let count_work_groups = state.chunk_state.update_buffers(&ctx.queue);
        frame_render.encoder.write_timestamp(&ctx.timestamp_query_set, 2);
        if count_work_groups != 0 {
            self.compute_pass(ctx, frame_render, state, count_work_groups);
        }
        frame_render.encoder.write_timestamp(&ctx.timestamp_query_set, 3);
        self.render_pass(ctx, load_action, frame_render, state);
        frame_render.encoder.write_timestamp(&ctx.timestamp_query_set, 4);
    }

    fn create_state(
        &self,
        ctx: &crate::GraphicsContext,
        params: pipelines::Params,
        input: VoxelRenderInput,
    ) -> Result<Self::State> {
        let geometry_buffers = self.build_geometry_buffers(ctx, input);

        let mut chunk_state = ChunkState::new(
            &ctx.device,
            &ctx.queue,
            input.max_visible_chunks,
            input.view_state.params.visible_size,
        );

        let active_view_box = input.view_state.params.calculate_view_box();
        chunk_state.update_view_box(active_view_box, input.voxels);

        log::info!("initializing voxels compute stage");
        let compute_stage = self.build_compute_stage(ctx, input, &chunk_state, &geometry_buffers);

        log::info!("initializing voxels render stage");
        let render_stage =
            self.build_render_stage(ctx, params, input, &chunk_state, &geometry_buffers);

        Ok(VoxelRenderState {
            chunk_state,
            compute_stage,
            render_stage,
            geometry_buffers,
        })
    }
}

impl<'a> pipelines::State<'a> for VoxelRenderState {
    type Input = VoxelRenderInput<'a>;
    type InputDelta = VoxelRenderInputDelta<'a>;

    fn update(&mut self, _ctx: &crate::GraphicsContext, input: VoxelRenderInputDelta) {
        let active_view_box = input.view_state.params.calculate_view_box();
        self.compute_stage.uniforms.update_view_box(active_view_box);

        self.chunk_state
            .update_view_box(active_view_box, input.voxels);

        self.chunk_state
            .apply_chunk_diff(input.voxels, input.voxel_diff);

        *self.render_stage.uniforms =
            RenderUniforms::new(input.view_state, input.model, active_view_box);
    }

    fn update_window(&mut self, _ctx: &crate::GraphicsContext, params: pipelines::Params) {
        self.render_stage.depth_texture = params.depth_texture.create_view(&Default::default());
    }
}

impl VoxelRenderPipeline {
    fn compute_pass(
        &self,
        ctx: &crate::GraphicsContext,
        frame_render: &mut crate::FrameRenderContext,
        state: &VoxelRenderState,
        count_work_groups: u32,
    ) {
        state.compute_stage.uniforms.sync(&ctx.queue);

        let mut cpass = frame_render
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        log::info!("Dispatching {} work groups", count_work_groups);

        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &state.compute_stage.bind_group, &[]);
        cpass.dispatch(count_work_groups, 1, 1);
    }

    fn render_pass(
        &self,
        ctx: &crate::GraphicsContext,
        load_action: crate::pipelines::LoadAction,
        frame_render: &mut crate::FrameRenderContext,
        state: &VoxelRenderState,
    ) {
        state.render_stage.uniforms.sync(&ctx.queue);

        let mut rpass = frame_render
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame_render.frame.output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_action.into_load_op(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.render_stage.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: load_action.into_load_op(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_bind_group(0, &state.render_stage.bind_group, &[]);

        let bufs = &state.geometry_buffers;
        let indirect_buf = &bufs.indirect;

        if ctx.multi_draw_enabled {
            rpass.multi_draw_indirect(
                &indirect_buf.buffer(),
                0,
                state.chunk_state.count_chunks() as u32,
            );
        } else {
            for chunk_index in state.chunk_state.iter_chunk_indices() {
                rpass.draw_indirect(
                    &indirect_buf.buffer(),
                    indirect_buf.instance_offset(chunk_index),
                );
            }
        }
    }

    fn build_compute_stage(
        &self,
        ctx: &crate::GraphicsContext,
        input: VoxelRenderInput,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
    ) -> ComputeStage {
        let uniforms = ComputeUniforms::new(input.view_state.params.calculate_view_box())
            .into_buffer_synced(&ctx.device);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.compute_bind_group_layout,
            entries: &[
                uniforms.as_binding(0),
                chunk_state.voxel_type_binding(1),
                chunk_state.chunk_metadata_binding(2),
                geometry_buffers.faces.as_binding(3),
                geometry_buffers.indirect.as_binding(4),
                chunk_state.compute_commands_binding(6),
            ],
        });

        ComputeStage {
            uniforms,
            bind_group,
        }
    }

    fn build_render_stage(
        &self,
        ctx: &crate::GraphicsContext,
        params: pipelines::Params,
        input: VoxelRenderInput,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
    ) -> RenderStage {
        let uniforms = RenderUniforms::new(
            &input.view_state,
            input.model,
            input.view_state.params.calculate_view_box(),
        )
        .into_buffer_synced(&ctx.device);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.render_bind_group_layout,
            entries: &[
                uniforms.as_binding(0),
                self.voxel_info.voxel_info_buf.as_binding(1),
                chunk_state.chunk_metadata_binding(3),
                geometry_buffers.faces.as_binding(4),
                self.voxel_info.texture_metadata_buf.as_binding(5),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureViewArray(
                        self.voxel_info
                            .texture_array()
                            .iter()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.voxel_info.sampler),
                },
            ],
        });

        RenderStage {
            uniforms,
            depth_texture: params.depth_texture.create_view(&Default::default()),
            bind_group,
        }
    }

    fn build_geometry_buffers(
        &self,
        ctx: &crate::GraphicsContext,
        input: VoxelRenderInput,
    ) -> GeometryBuffers {
        let faces = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "voxel faces",
                instance_len: (4 * 3 * index_utils::chunk_size_total()) as usize,
                n_instances: input.max_visible_chunks,
                usage: wgpu::BufferUsage::STORAGE,
            },
        );

        let indirect = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "voxel indirect",
                instance_len: std::mem::size_of::<IndirectCommand>(),
                n_instances: input.max_visible_chunks,
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::INDIRECT,
            },
        );

        GeometryBuffers {
            faces,
            indirect,
        }
    }
}

impl RenderUniforms {
    fn new(view_state: &ViewState, model: Matrix4<f32>, view_box: Bounds<i32>) -> Self {
        Self {
            proj: view_state.proj().into(),
            model: model.into(),
            view: view_state.view().into(),
            camera_pos: view_state.camera_pos().into(),
            _padding0: 0.0,
            visible_box_origin: convert_point!(view_box.origin(), f32).into(),
            _padding1: 0.0,
            visible_box_limit: convert_point!(view_box.limit(), f32).into(),
            _padding2: 0.0,
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
            label: "voxel render uniforms",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl ComputeUniforms {
    fn new(view_box: Bounds<i32>) -> Self {
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
