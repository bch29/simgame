use anyhow::Result;
use zerocopy::{AsBytes, FromBytes};
use cgmath::Matrix4;

use crate::buffer_util::{
    BufferSyncHelperDesc, BufferSyncable, BufferSyncedData, FillBuffer, IntoBufferSynced,
};
use crate::resource::ResourceLoader;
use crate::DeviceResult;
use crate::FrameRender;

pub(crate) struct GuiRenderInit<'a> {
    pub resource_loader: &'a ResourceLoader,
    pub swapchain: &'a wgpu::SwapChainDescriptor,
}

pub(crate) struct GuiRenderState {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: BufferSyncedData<RenderUniforms, RenderUniforms>,

    // multisampled_framebuffer: wgpu::TextureView,
    sampler: wgpu::Sampler,
    crosshair_texture: wgpu::TextureView,
}

#[derive(Debug, Clone, Copy, AsBytes, FromBytes)]
#[repr(C)]
struct RenderUniforms {
    model: [[f32; 4]; 4]
}

impl GuiRenderState {
    pub fn new(init: GuiRenderInit, device_result: &DeviceResult) -> Result<Self> {
        let device = &device_result.device;

        let vert_shader = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
            init.resource_loader.load_shader("shader/gui/vertex")?.as_slice(),
        ));

        let frag_shader = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(
            init.resource_loader.load_shader("shader/gui/fragment")?.as_slice(),
        ));

        let crosshair_texture: wgpu::TextureView = {
            let image = init.resource_loader.load_image("tex/gui/crosshair")?.into_rgba();

            let (width, height) = image.dimensions();

            let size = wgpu::Extent3d {
                width,
                height,
                depth: 1,
            };

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("crosshair"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
            });

            let copy_view = wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            };

            let samples = image.as_flat_samples();

            device_result.queue.write_texture(
                copy_view,
                samples.as_slice(),
                wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: samples.layout.height_stride as u32,
                    rows_per_image: 0,
                },
                size,
            );

            texture.create_default_view()
        };

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
                    // Texture
                    wgpu::BindGroupLayoutEntry::new(
                        1,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            dimension: wgpu::TextureViewDimension::D2,
                            component_type: wgpu::TextureComponentType::Float,
                            multisampled: false,
                        },
                    ),
                    wgpu::BindGroupLayoutEntry::new(
                        2,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::Sampler { comparison: false },
                    ),
                ],
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            anisotropy_clamp: Some(16),
            ..Default::default()
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
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: init.swapchain.format,
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let aspect_ratio = init.swapchain.width as f32 / init.swapchain.height as f32;

        let uniforms = RenderUniforms::new(aspect_ratio).into_buffer_synced(device);
        uniforms.sync(&device_result.queue);

        // let multisampled_framebuffer =
        //     Self::create_multisampled_framebuffer(device, init.swapchain, SAMPLE_COUNT);

        Ok(Self {
            pipeline,
            uniforms,
            bind_group_layout,

            //             multisampled_framebuffer,
            sampler,
            crosshair_texture,
        })
    }

    pub fn update_swapchain(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        swapchain: &wgpu::SwapChainDescriptor,
    ) -> Result<()> {
        // self.multisampled_framebuffer =
        //     Self::create_multisampled_framebuffer(device, swapchain, SAMPLE_COUNT);
        //
        let aspect_ratio = swapchain.width as f32 / swapchain.height as f32;

        *self.uniforms = RenderUniforms::new(aspect_ratio);
        self.uniforms.sync(queue);

        Ok(())
    }

    pub fn render_frame(
        &mut self,
        frame_render: &FrameRender,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.uniforms.sync(frame_render.queue);

        let bind_group = frame_render
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gui render"),
                layout: &self.bind_group_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(self.uniforms.buffer().slice(..)),
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.crosshair_texture),
                    },
                    wgpu::Binding {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame_render.frame.output.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);

        rpass.draw(0..4, 0..1);
    }

    #[allow(dead_code)]
    fn create_multisampled_framebuffer(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let multisampled_texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };
        let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
            size: multisampled_texture_extent,
            mip_level_count: 1,
            sample_count: sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            label: None,
        };

        device
            .create_texture(multisampled_frame_descriptor)
            .create_default_view()
    }
}

impl RenderUniforms {
    fn new(aspect_ratio: f32) -> Self {
        let width = 0.08;
        let height = width * aspect_ratio;

        Self {
            model: Matrix4::from_nonuniform_scale(width, height, 1.0).into()
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
