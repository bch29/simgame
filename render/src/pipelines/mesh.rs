use anyhow::Result;
use cgmath::Matrix4;
use zerocopy::{AsBytes, FromBytes};

use crate::buffer_util::{
    BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, InstancedBuffer,
    InstancedBufferDesc, IntoBufferSynced,
};
use crate::pipelines;

const INSTANCES_PER_PASS: usize = 1024;

pub(crate) struct MeshRenderPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub(crate) struct MeshRenderState {
    depth_texture: wgpu::TextureView,
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,
    geometry_buffers: GeometryBuffers,
    instances: Vec<InstanceMeta>,
}

pub(crate) struct MeshRenderInput<'a> {
    pub mesh: &'a crate::mesh::Mesh,
    pub instances: &'a [Matrix4<f32>],
    pub view_state: &'a crate::ViewState,
}

pub(crate) struct MeshRenderInputDelta<'a> {
    pub instances: &'a [Matrix4<f32>],
    pub view_state: &'a crate::ViewState,
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct RenderUniforms {
    proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    camera_pos: [f32; 3],
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct InstanceMeta {
    model_matrix: [[f32; 4]; 4],
}

struct GeometryBuffers {
    instances: InstancedBuffer,
    vertices: InstancedBuffer,
    indexes: InstancedBuffer,
}

impl MeshRenderPipeline {
    pub fn new(ctx: &crate::GraphicsContext) -> Result<MeshRenderPipeline> {
        let vert_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: Default::default(),
                source: wgpu::ShaderSource::SpirV(
                    ctx.resource_loader
                        .load_shader("shader/mesh/vertex")?
                        .into(),
                ),
            });

        let frag_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: Default::default(),
                source: wgpu::ShaderSource::SpirV(
                    ctx.resource_loader
                        .load_shader("shader/mesh/fragment")?
                        .into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mesh vertex layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Instances
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
                    buffers: &[crate::mesh::Mesh::vertex_buffer_layout()],
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

        Ok(MeshRenderPipeline {
            pipeline,
            bind_group_layout,
        })
    }
}

impl<'a> pipelines::State<'a> for MeshRenderState {
    type Input = MeshRenderInput<'a>;
    type InputDelta = MeshRenderInputDelta<'a>;

    fn update(&mut self, input: MeshRenderInputDelta<'a>) {
        self.instances.clear();
        self.instances.extend(
            input
                .instances
                .iter()
                .copied()
                .map(InstanceMeta::from_matrix4),
        );

        *self.uniforms = RenderUniforms::new(input.view_state);
    }

    fn update_window(&mut self, _ctx: &crate::GraphicsContext, params: pipelines::Params) {
        self.depth_texture = params.depth_texture.create_view(&Default::default());
    }
}

impl pipelines::Pipeline for MeshRenderPipeline {
    type State = MeshRenderState;

    fn create_state<'a>(
        &self,
        ctx: &crate::GraphicsContext,
        params: pipelines::Params<'a>,
        input: MeshRenderInput<'a>,
    ) -> Result<Self::State> {
        let depth_texture = params.depth_texture.create_view(&Default::default());

        let uniforms = RenderUniforms::new(input.view_state).into_buffer_synced(&ctx.device);
        uniforms.sync(&ctx.queue);

        let geometry_buffers = {
            let instances = InstancedBuffer::new(
                &ctx.device,
                InstancedBufferDesc {
                    label: "mesh instances",
                    instance_len: std::mem::size_of::<InstanceMeta>() * INSTANCES_PER_PASS,
                    n_instances: 1,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::STORAGE,
                },
            );

            let vertices = InstancedBuffer::new(
                &ctx.device,
                InstancedBufferDesc {
                    label: "mesh vertices",
                    instance_len: std::mem::size_of::<crate::mesh::Vertex>()
                        * input.mesh.vertices.len(),
                    n_instances: 1,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
                },
            );
            vertices.write(&ctx.queue, 0, input.mesh.vertices.as_bytes());

            let indexes = InstancedBuffer::new(
                &ctx.device,
                InstancedBufferDesc {
                    label: "mesh indexes",
                    instance_len: std::mem::size_of::<crate::mesh::Index>()
                        * input.mesh.indices.len(),
                    n_instances: 1,
                    usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::INDEX,
                },
            );
            indexes.write(&ctx.queue, 0, input.mesh.indices.as_bytes());

            GeometryBuffers {
                instances,
                vertices,
                indexes,
            }
        };

        Ok(MeshRenderState {
            uniforms,
            geometry_buffers,
            depth_texture,
            instances: input
                .instances
                .iter()
                .copied()
                .map(InstanceMeta::from_matrix4)
                .collect(),
        })
    }

    fn render_frame(
        &self,
        ctx: &crate::GraphicsContext,
        load_action: crate::pipelines::LoadAction,
        frame_render: &mut crate::FrameRenderContext,
        state: &mut MeshRenderState,
    ) {
        state.uniforms.sync(&ctx.queue);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gui render"),
            layout: &self.bind_group_layout,
            entries: &[
                // uniforms
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: state.uniforms.buffer(),
                        offset: 0,
                        size: None,
                    },
                },
                // instances
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: state.geometry_buffers.instances.buffer(),
                        offset: 0,
                        size: None,
                    },
                },
            ],
        });

        for instance_chunk in state.instances.chunks(INSTANCES_PER_PASS) {
            let n_indexes = state.geometry_buffers.indexes.size() as u32
                / std::mem::size_of::<crate::mesh::Index>() as u32;

            state
                .geometry_buffers
                .instances
                .write(&ctx.queue, 0, instance_chunk.as_bytes());

            let mut rpass = frame_render
                .encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame_render.frame.output.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: load_action.into_load_op(wgpu::Color::BLACK),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: Some(
                        wgpu::RenderPassDepthStencilAttachmentDescriptor {
                            attachment: &state.depth_texture,
                            depth_ops: Some(wgpu::Operations {
                                load: load_action.into_load_op(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        },
                    ),
                });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_vertex_buffer(0, state.geometry_buffers.vertices.buffer().slice(..));
            rpass.set_index_buffer(
                state.geometry_buffers.indexes.buffer().slice(..),
                crate::mesh::Mesh::index_format(),
            );

            rpass.draw_indexed(0..n_indexes, 0, 0..instance_chunk.len() as u32);
        }
    }
}

impl RenderUniforms {
    fn new(view_state: &crate::ViewState) -> Self {
        Self {
            proj: view_state.proj().into(),
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
            label: "gui render uniforms",
            buffer_len: 1,
            max_chunk_len: 1,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        }
    }
}

impl InstanceMeta {
    fn from_matrix4(model_matrix: Matrix4<f32>) -> Self {
        InstanceMeta {
            model_matrix: model_matrix.into(),
        }
    }
}
