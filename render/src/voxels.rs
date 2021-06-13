mod chunk_state;
mod voxel_info;

use anyhow::Result;
use bevy::{
    asset::{AddAsset, AssetServer, Assets, Handle},
    ecs::{
        system::{Commands, IntoSystem, Res, ResMut},
        world::World,
    },
    render2::{
        camera::{CameraPlugin, ExtractedCamera, ExtractedCameraNames},
        core_pipeline::{CorePipelinePlugin, ViewDepthTexture},
        pass::{
            LoadOp, Operations, PassDescriptor, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, TextureAttachment,
        },
        pipeline::{ComputePipelineDescriptor, PipelineLayout, RenderPipelineDescriptor},
        render_graph::{
            NodeRunError, RenderGraph, RenderGraphContext, SlotInfo, SlotType, SlotValue,
        },
        render_resource::{BindGroup, BindGroupBuilder, BufferUsage},
        renderer::{RenderContext, RenderResourceContext, RenderResources},
        shader::{ComputeShaderStages, Shader, ShaderStages},
        view::ExtractedWindows,
    },
    wgpu2::WgpuRenderResourceContext,
    window::{WindowId, Windows},
};
use cgmath::{Matrix4, SquareMatrix};
use zerocopy::{AsBytes, FromBytes};

use simgame_types::VoxelDirectory;
use simgame_util::{convert_point, Bounds};
use simgame_voxels::{index_utils, SharedVoxelData, VoxelData, VoxelDelta};

use crate::{
    assets::TextureAssets,
    buffer_util::{
        BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, InstancedBuffer,
        InstancedBufferDesc, IntoBufferSynced,
    },
    ViewParams, ViewState,
};

use chunk_state::ChunkState;
use voxel_info::VoxelInfoManager;

pub struct VoxelRenderPlugin;

#[derive(Clone, Copy)]
pub struct Params {
    pub max_visible_chunks: usize,
}

impl bevy::app::Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_asset::<Shader>();
        app.add_plugin(CorePipelinePlugin);

        let shaders: ShaderHandles;

        let render_app = app.sub_app_mut(0);
        render_app
            .add_system_to_stage(bevy::render2::RenderStage::Extract, extract_assets.system())
            .add_system_to_stage(bevy::render2::RenderStage::Prepare, prepare_assets.system());

        render_app
            .add_system_to_stage(bevy::render2::RenderStage::Extract, extract_chunks.system())
            .add_system_to_stage(bevy::render2::RenderStage::Prepare, prepare_chunks.system())
            .add_system_to_stage(bevy::render2::RenderStage::Queue, queue_chunks.system());

        {
            let world = app.world.cell();
            let asset_server = world
                .get_resource::<AssetServer>()
                .expect("got no asset server resource");

            shaders = ShaderHandles {
                vertex: asset_server.load("shader/voxels/vertex.vert_glsl"),
                compute: asset_server.load("shader/voxels/compute.compute_glsl"),
                fragment: asset_server.load("shader/voxels/fragment.frag_glsl"),
            };
        }

        app.insert_resource(LoadingVoxelAssets {
            loaded: false,
            shaders,
        });
    }
}

struct LoadingVoxelAssets {
    loaded: bool,
    shaders: ShaderHandles,
}

struct Shaders<T> {
    compute: T,
    vertex: T,
    fragment: T,
}

type ShaderHandles = Shaders<Handle<Shader>>;
type LoadedShaders = Shaders<Shader>;

struct ExtractedChunks {
    voxel_delta: VoxelDelta,
}

#[derive(Clone, Copy)]
struct CountWorkGroups {
    count_work_groups: u32,
}

/// Holds the static pipeline for rendering voxels. Can be used to render multiple different pieces
/// of voxel geometry simultaneously.
pub(crate) struct VoxelRenderNode {
    pipeline: bevy::render2::pipeline::PipelineId,
    pipeline_descriptor: RenderPipelineDescriptor,
}

struct VoxelComputeNode {
    pipeline: bevy::render2::pipeline::PipelineId,
    pipeline_descriptor: ComputePipelineDescriptor,
}

/// Holds the current state (including GPU buffers) of rendering a particular piece of voxel
/// geometry.
pub(crate) struct VoxelRenderState {
    chunk_state: ChunkState,
    compute_stage: ComputeStageState,
    render_stage: RenderStageState,
    geometry_buffers: GeometryBuffers,
}

struct ComputeStageState {
    uniforms: BufferSyncedData<ComputeUniforms, ComputeUniforms>,
    bind_group: BindGroup,
}

struct RenderStageState {
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,
    bind_group: BindGroup,
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
    count: u32,          // number of vertices per instance
    instance_count: u32, // number of instances to draw
    first: u32,          // index of first vertex
    base_instance: u32,  // index of first instance
}

fn extract_assets(
    mut commands: Commands,
    directory: Res<VoxelDirectory>,
    voxels: Res<SharedVoxelData>,
    shaders: Res<Assets<Shader>>,
    textures: Option<Res<TextureAssets>>,
    load_state: Option<ResMut<LoadingVoxelAssets>>,
    view_params: Res<ViewParams>,
    params: Res<Params>,
) {
    commands.insert_resource(voxels.clone());
    commands.insert_resource(params.clone());
    commands.insert_resource(ViewState::new(
        view_params.clone(),
        cgmath::Vector2 { x: 800, y: 600 },
    ));

    let mut load_state = match load_state {
        Some(load_state) => load_state,
        None => return,
    };

    if load_state.loaded {
        return;
    }

    let textures = match textures {
        Some(textures) => textures,
        None => return,
    };

    if !(shaders.get(&load_state.shaders.vertex).is_some()
        && shaders.get(&load_state.shaders.fragment).is_some()
        && shaders.get(&load_state.shaders.compute).is_some())
    {
        return;
    }

    let shaders: Shaders<Shader> = Shaders {
        vertex: shaders
            .get(&load_state.shaders.vertex)
            .unwrap()
            .get_spirv_shader(None)
            .unwrap(),
        fragment: shaders
            .get(&load_state.shaders.fragment)
            .unwrap()
            .get_spirv_shader(None)
            .unwrap(),
        compute: shaders
            .get(&load_state.shaders.compute)
            .unwrap()
            .get_spirv_shader(None)
            .unwrap(),
    };

    // have to set a flag instead of removing the resource because an extract stage system is
    // unable to modify the main app world
    load_state.loaded = true;

    log::info!("Finished loading assets");

    commands.insert_resource(shaders);
    commands.insert_resource(directory.clone());
    commands.insert_resource(textures.clone());
}

fn prepare_assets(
    mut commands: Commands,
    shaders: Option<Res<LoadedShaders>>,
    textures: Option<Res<TextureAssets>>,
    directory: Option<Res<VoxelDirectory>>,
    render_resources: Res<RenderResources>,
    voxels: Res<SharedVoxelData>,
    params: Res<Params>,
    view_state: Res<ViewState>,
    mut render_graph: ResMut<bevy::render2::render_graph::RenderGraph>,
) {
    let shaders = match shaders {
        Some(shaders) => shaders,
        None => return,
    };

    let textures = match textures {
        Some(textures) => textures,
        None => return,
    };

    let directory = directory.unwrap();
    let ctx = render_resources.downcast_ref().unwrap();

    let voxel_info = crate::voxels::VoxelInfoManager::new(&*directory, &*textures, ctx).unwrap();

    let compute_node = VoxelComputeNode::new(&*ctx, &*shaders).unwrap();
    let render_node = VoxelRenderNode::new(&*ctx, &*shaders).unwrap();

    let voxels = voxels.data.lock();
    let render_state = VoxelRenderState::new(
        &render_node,
        &compute_node,
        &*ctx,
        &*params,
        &voxel_info,
        &*voxels,
        &*view_state,
    )
    .expect("Failed to create voxel render state");

    {
        let mut draw_voxels_graph = RenderGraph::default();
        draw_voxels_graph.add_node("compute_pass", compute_node);
        draw_voxels_graph.add_node("render_pass", render_node);
        let input_node_id = draw_voxels_graph.set_input(vec![
            SlotInfo::new("render_target", SlotType::TextureView),
            SlotInfo::new("depth", SlotType::TextureView),
        ]);
        draw_voxels_graph
            .add_slot_edge(
                input_node_id,
                "render_target",
                "render_pass",
                "color_attachment",
            )
            .unwrap();
        draw_voxels_graph
            .add_slot_edge(input_node_id, "depth", "render_pass", "depth")
            .unwrap();
        draw_voxels_graph
            .add_node_edge("compute_pass", "render_pass")
            .unwrap();
        render_graph.add_sub_graph("voxels", draw_voxels_graph);
    }

    render_graph.add_node("voxel_driver", VoxelDriverNode);
    render_graph
        .add_node_edge("main_pass_driver", "voxel_driver")
        .unwrap();

    commands.remove_resource::<LoadedShaders>();
    commands.remove_resource::<TextureAssets>();
    commands.insert_resource(voxel_info);
    commands.insert_resource(render_state);

    log::info!("Finished preparing assets");
}

fn extract_chunks(
    mut commands: Commands,
    view_params: Res<ViewParams>,
    voxel_delta: Res<VoxelDelta>,
    windows: Res<Windows>,
) {
    let window = windows.get(WindowId::primary()).unwrap();

    commands.insert_resource(ExtractedChunks {
        voxel_delta: voxel_delta.clone(),
    });
    commands.insert_resource(ViewState::new(
        view_params.clone(),
        cgmath::Vector2 {
            x: window.physical_width(),
            y: window.physical_height(),
        },
    ));
}

fn prepare_chunks(
    extracted_chunks: Res<ExtractedChunks>,
    view_state: Res<ViewState>,
    voxels: Res<SharedVoxelData>,
    state: Option<ResMut<VoxelRenderState>>,
) {
    let mut state = match state {
        Some(state) => state,
        None => return,
    };

    let model = Matrix4::identity();
    let voxels = voxels.data.lock();
    let voxel_delta = &extracted_chunks.voxel_delta;

    let active_view_box = view_state.params.calculate_view_box();
    state
        .compute_stage
        .uniforms
        .update_view_box(active_view_box);

    state.chunk_state.update_view_box(active_view_box, &*voxels);

    state.chunk_state.apply_chunk_diff(&*voxels, &*voxel_delta);

    *state.render_stage.uniforms = RenderUniforms::new(&*view_state, model, active_view_box);
}

fn queue_chunks(
    mut commands: Commands,
    state: Option<ResMut<VoxelRenderState>>,
    render_resources: Res<RenderResources>,
) {
    let mut state = match state {
        Some(state) => state,
        None => return,
    };

    let ctx = render_resources.downcast_ref().unwrap();
    let count_work_groups = state.chunk_state.update_buffers(ctx);
    state.compute_stage.uniforms.sync(ctx);

    commands.insert_resource(CountWorkGroups { count_work_groups });

    state.render_stage.uniforms.sync(&ctx);
}

pub struct VoxelDriverNode;

impl bevy::render2::render_graph::Node for VoxelDriverNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        _render_context: &mut dyn RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let extracted_cameras = world.get_resource::<ExtractedCameraNames>().unwrap();
        let extracted_windows = world.get_resource::<ExtractedWindows>().unwrap();

        if let Some(camera_3d) = extracted_cameras.entities.get(CameraPlugin::CAMERA_3D) {
            let extracted_camera = world.entity(*camera_3d).get::<ExtractedCamera>().unwrap();
            let depth_texture = world.entity(*camera_3d).get::<ViewDepthTexture>().unwrap();
            let extracted_window = extracted_windows.get(&extracted_camera.window_id).unwrap();
            let swap_chain_texture = extracted_window.swap_chain_texture.unwrap();
            graph.run_sub_graph(
                "voxels",
                vec![
                    SlotValue::TextureView(swap_chain_texture),
                    SlotValue::TextureView(depth_texture.view),
                ],
            )?;
        }

        Ok(())
    }
}

impl bevy::render2::render_graph::Node for VoxelComputeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut dyn RenderContext,
        world: &World,
    ) -> std::result::Result<(), NodeRunError> {
        let count_work_groups =
            if let Some(CountWorkGroups { count_work_groups }) = world.get_resource() {
                *count_work_groups
            } else {
                return Ok(());
            };

        if count_work_groups == 0 {
            return Ok(());
        }

        let state: &VoxelRenderState = world
            .get_resource()
            .expect("got no VoxelRenderState resource");

        let layout = &self.pipeline_descriptor.layout;

        render_context.begin_compute_pass(&mut |pass| {
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(
                0,
                layout.bind_groups[0].id,
                state.compute_stage.bind_group.id,
                None,
            );
            log::debug!("Dispatching {} work groups", count_work_groups);
            pass.dispatch(count_work_groups, 1, 1);
        });

        Ok(())
    }
}

impl bevy::render2::render_graph::Node for VoxelRenderNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![
            SlotInfo::new("color_attachment", SlotType::TextureView),
            SlotInfo::new("depth", SlotType::TextureView),
        ]
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut dyn RenderContext,
        world: &World,
    ) -> std::result::Result<(), NodeRunError> {
        let state: &VoxelRenderState = match world.get_resource() {
            Some(state) => state,
            None => return Ok(()),
        };

        let layout = &self.pipeline_descriptor.layout;

        let color_attachment_texture = graph.get_input_texture("color_attachment")?;
        let depth_texture = graph.get_input_texture("depth")?;

        let pass_descriptor = PassDescriptor {
            color_attachments: vec![RenderPassColorAttachment {
                attachment: TextureAttachment::Id(color_attachment_texture),
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                attachment: TextureAttachment::Id(depth_texture),
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: true,
                }),
                stencil_ops: None,
            }),
            sample_count: 1,
        };

        let bufs = &state.geometry_buffers;
        let indirect_buf = &bufs.indirect;

        render_context.begin_render_pass(&pass_descriptor, &mut |pass| {
            pass.set_pipeline(self.pipeline);
            pass.set_bind_group(
                0,
                layout.bind_groups[0].id,
                state.render_stage.bind_group.id,
                None,
            );
            pass.multi_draw_indirect(
                indirect_buf.buffer(),
                0,
                state.chunk_state.count_chunks() as u32,
            );
        });

        Ok(())
    }
}

impl VoxelComputeNode {
    fn new(ctx: &WgpuRenderResourceContext, shaders: &LoadedShaders) -> Result<Self> {
        let compute_layout = shaders
            .compute
            .reflect_layout(true)
            .expect("failed to reflect compute shader layout");

        let pipeline_layout = PipelineLayout::from_shader_layouts(&mut [compute_layout]);

        let pipeline_descriptor = ComputePipelineDescriptor {
            name: None,
            layout: pipeline_layout,
            shader_stages: ComputeShaderStages {
                compute: ctx.create_shader_module(&shaders.compute),
            },
        };

        let pipeline = ctx.create_compute_pipeline(&pipeline_descriptor);

        Ok(VoxelComputeNode {
            pipeline,
            pipeline_descriptor,
        })
    }
}

impl VoxelRenderNode {
    fn new(ctx: &WgpuRenderResourceContext, shaders: &LoadedShaders) -> Result<Self> {
        let vertex_layout = shaders
            .vertex
            .reflect_layout(true)
            .expect("failed to reflect vertex shader layout");
        let fragment_layout = shaders
            .fragment
            .reflect_layout(true)
            .expect("failed to reflect fragment shader layout");

        let pipeline_descriptor = RenderPipelineDescriptor {
            name: None,
            layout: PipelineLayout::from_shader_layouts(&mut [vertex_layout, fragment_layout]),
            shader_stages: ShaderStages {
                vertex: ctx.create_shader_module(&shaders.vertex),
                fragment: Some(ctx.create_shader_module(&shaders.fragment)),
            },
            primitive: bevy::render2::pipeline::PrimitiveState {
                topology: bevy::render2::pipeline::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: bevy::render2::pipeline::FrontFace::Ccw,
                cull_mode: Some(bevy::render2::pipeline::Face::Back),
                polygon_mode: bevy::render2::pipeline::PolygonMode::Fill,
                clamp_depth: false,
                conservative: false,
            },
            depth_stencil: Some(bevy::render2::pipeline::DepthStencilState {
                format: bevy::render2::texture::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: bevy::render2::pipeline::CompareFunction::Less,
                stencil: bevy::render2::pipeline::StencilState {
                    front: bevy::render2::pipeline::StencilFaceState::IGNORE,
                    back: bevy::render2::pipeline::StencilFaceState::IGNORE,
                    read_mask: 0u32,
                    write_mask: 0u32,
                },
                bias: bevy::render2::pipeline::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: bevy::render2::pipeline::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            color_target_states: vec![bevy::render2::pipeline::ColorTargetState {
                format: bevy::render2::texture::TextureFormat::default(),
                blend: Some(bevy::render2::pipeline::BlendState {
                    color: bevy::render2::pipeline::BlendComponent {
                        src_factor: bevy::render2::pipeline::BlendFactor::SrcAlpha,
                        dst_factor: bevy::render2::pipeline::BlendFactor::OneMinusSrcAlpha,
                        operation: bevy::render2::pipeline::BlendOperation::Add,
                    },
                    alpha: bevy::render2::pipeline::BlendComponent {
                        src_factor: bevy::render2::pipeline::BlendFactor::One,
                        dst_factor: bevy::render2::pipeline::BlendFactor::One,
                        operation: bevy::render2::pipeline::BlendOperation::Add,
                    },
                }),
                write_mask: bevy::render2::pipeline::ColorWrite::ALL,
            }],
        };

        let pipeline = ctx.create_render_pipeline(&pipeline_descriptor);

        Ok(VoxelRenderNode {
            pipeline,
            pipeline_descriptor,
        })
    }
}

impl VoxelRenderState {
    fn new(
        render_node: &VoxelRenderNode,
        compute_node: &VoxelComputeNode,
        ctx: &WgpuRenderResourceContext,
        params: &Params,
        voxel_info: &VoxelInfoManager,
        voxels: &VoxelData,
        view_state: &ViewState,
    ) -> Result<VoxelRenderState> {
        let geometry_buffers = GeometryBuffers::new(ctx, params.max_visible_chunks);

        let mut chunk_state = ChunkState::new(
            ctx,
            params.max_visible_chunks,
            view_state.params.visible_size,
        );

        let active_view_box = view_state.params.calculate_view_box();
        chunk_state.update_view_box(active_view_box, voxels);

        log::info!("initializing voxel compute stage");
        let compute_stage =
            ComputeStageState::new(ctx, view_state, &chunk_state, &geometry_buffers);
        ctx.create_bind_group(
            compute_node.pipeline_descriptor.layout.bind_groups[0].id,
            &compute_stage.bind_group,
        );

        log::info!("initializing voxel render stage");
        let render_stage = RenderStageState::new(
            ctx,
            voxel_info,
            view_state,
            &chunk_state,
            &geometry_buffers,
        );
        ctx.create_bind_group(
            render_node.pipeline_descriptor.layout.bind_groups[0].id,
            &render_stage.bind_group,
        );

        Ok(VoxelRenderState {
            chunk_state,
            compute_stage,
            render_stage,
            geometry_buffers,
        })
    }
}

impl ComputeStageState {
    fn new(
        ctx: &WgpuRenderResourceContext,
        view_state: &ViewState,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
    ) -> ComputeStageState {
        let uniforms =
            ComputeUniforms::new(view_state.params.calculate_view_box()).into_buffer_synced(&ctx);

        let bind_group = BindGroupBuilder::default()
            .add_binding(0, uniforms.as_binding())
            .add_binding(1, chunk_state.voxel_type_binding())
            .add_binding(2, chunk_state.chunk_metadata_binding())
            .add_binding(3, geometry_buffers.faces.as_binding())
            .add_binding(4, geometry_buffers.indirect.as_binding())
            .add_binding(6, chunk_state.compute_commands_binding())
            .finish();

        ComputeStageState {
            uniforms,
            bind_group,
        }
    }
}

impl RenderStageState {
    fn new(
        ctx: &WgpuRenderResourceContext,
        voxel_info: &VoxelInfoManager,
        view_state: &ViewState,
        chunk_state: &ChunkState,
        geometry_buffers: &GeometryBuffers,
    ) -> RenderStageState {
        let uniforms = RenderUniforms::new(
            view_state,
            Matrix4::identity(),
            view_state.params.calculate_view_box(),
        )
        .into_buffer_synced(ctx);

        let bind_group = BindGroupBuilder::default()
            .add_binding(0, uniforms.as_binding())
            .add_binding(1, voxel_info.voxel_info_buf.as_binding())
            .add_binding(3, chunk_state.chunk_metadata_binding())
            .add_binding(4, geometry_buffers.faces.as_binding())
            .add_binding(5, voxel_info.texture_metadata_buf.as_binding())
            //         wgpu::BindGroupEntry {
            //             binding: 6,
            //             resource: wgpu::BindingResource::TextureViewArray(
            //                 self.voxel_info
            //                     .texture_array()
            //                     .iter()
            //                     .collect::<Vec<_>>()
            //                     .as_slice(),
            //             ),
            //         },
            //         wgpu::BindGroupEntry {
            //             binding: 7,
            //             resource: wgpu::BindingResource::Sampler(&self.voxel_info.sampler),
            //         },
            .finish();

        RenderStageState {
            uniforms,
            bind_group,
        }
    }
}

impl GeometryBuffers {
    fn new(ctx: &WgpuRenderResourceContext, max_visible_chunks: usize) -> GeometryBuffers {
        let faces = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "voxel faces",
                instance_len: (4 * 3 * index_utils::chunk_size_total()) as usize,
                n_instances: max_visible_chunks,
                usage: BufferUsage::STORAGE,
            },
        );

        let indirect = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "voxel indirect",
                instance_len: std::mem::size_of::<IndirectCommand>(),
                n_instances: max_visible_chunks,
                usage: BufferUsage::STORAGE | BufferUsage::INDIRECT,
            },
        );

        GeometryBuffers { faces, indirect }
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
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
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
            usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
        }
    }
}
