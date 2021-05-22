use std::convert::TryInto;

use anyhow::Result;
use cgmath::Matrix4;
use zerocopy::{AsBytes, FromBytes};

use simgame_types::Directory;

use crate::{
    buffer_util::{
        BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, IntoBufferSynced,
    },
    pipelines::{self, GraphicsContext},
    resource::ResourceLoader,
};

pub(crate) struct GuiRenderPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    crosshair_texture: wgpu::TextureView,
    // multisampled_framebuffer: wgpu::TextureView,
}

pub(crate) struct GuiRenderState {
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct RenderUniforms {
    model: [[f32; 4]; 4],
}

impl<'a> pipelines::State<'a> for GuiRenderState {
    type Input = ();
    type InputDelta = ();

    fn update(&mut self, _ctx: &crate::GraphicsContext, _input: Self::InputDelta) {}

    fn update_window(&mut self, ctx: &crate::GraphicsContext, params: pipelines::Params) {
        // self.multisampled_framebuffer =
        //     Self::create_multisampled_framebuffer(device, swapchain, SAMPLE_COUNT);
        //
        let aspect_ratio = params.physical_win_size.x as f32 / params.physical_win_size.y as f32;

        *self.uniforms = RenderUniforms::new(aspect_ratio);
        self.uniforms.sync(&ctx.queue);
    }
}

impl pipelines::Pipeline for GuiRenderPipeline {
    type State = GuiRenderState;

    fn create_pipeline(
        ctx: &GraphicsContext,
        directory: &Directory,
        resource_loader: &ResourceLoader,
    ) -> Result<GuiRenderPipeline> {
        let vert_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: Default::default(),
                source: wgpu::ShaderSource::SpirV(
                    resource_loader.load_shader("shader/gui/vertex")?.into(),
                ),
            });

        let frag_shader = ctx
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                flags: Default::default(),
                source: wgpu::ShaderSource::SpirV(
                    resource_loader.load_shader("shader/gui/fragment")?.into(),
                ),
            });

        let crosshair_texture = ctx
            .textures
            .from_resource(&directory.texture, "tex/gui/crosshair")?
            .texture
            .create_view(&Default::default());

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gui vertex layout"),
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
                        // Texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            ty: wgpu::BindingType::Sampler {
                                filtering: true,
                                comparison: false,
                            },
                            count: None,
                        },
                    ],
                });

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            anisotropy_clamp: Some(16.try_into().expect("nonzero")),
            ..Default::default()
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
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &frag_shader,
                    entry_point: "main",
                    targets: &[wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Max,
                            },
                        }),
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: None,
                    strip_index_format: Some(wgpu::IndexFormat::Uint16),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    clamp_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
            });

        // let multisampled_framebuffer =
        //     GuiRenderState::create_multisampled_framebuffer(device, self.swapchain, SAMPLE_COUNT);

        Ok(GuiRenderPipeline {
            pipeline,
            bind_group_layout,
            // multisampled_framebuffer,
            sampler,
            crosshair_texture,
        })
    }

    fn create_state<'a>(
        &self,
        ctx: &crate::GraphicsContext,
        params: pipelines::Params<'a>,
        _input: (),
    ) -> Result<Self::State> {
        let aspect_ratio = params.physical_win_size.x as f32 / params.physical_win_size.y as f32;
        let uniforms = RenderUniforms::new(aspect_ratio).into_buffer_synced(&ctx.device);
        uniforms.sync(&ctx.queue);

        Ok(GuiRenderState { uniforms })
    }

    fn render_frame(
        &self,
        ctx: &crate::GraphicsContext,
        load_action: crate::pipelines::LoadAction,
        frame_render: &mut crate::FrameRenderContext,
        state: &mut GuiRenderState,
    ) {
        state.uniforms.sync(&ctx.queue);

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gui render"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: state.uniforms.buffer(),
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.crosshair_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

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
                depth_stencil_attachment: None,
            });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);

        rpass.draw(0..4, 0..1);
    }
}

impl GuiRenderState {
    #[allow(dead_code)]
    fn create_multisampled_framebuffer(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let multisampled_texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth_or_array_layers: 1,
        };
        let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
            size: multisampled_texture_extent,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            label: None,
        };

        device
            .create_texture(multisampled_frame_descriptor)
            .create_view(&Default::default())
    }
}

impl RenderUniforms {
    fn new(aspect_ratio: f32) -> Self {
        let width = 0.08;
        let height = width * aspect_ratio;

        Self {
            model: Matrix4::from_nonuniform_scale(width, height, 1.0).into(),
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
