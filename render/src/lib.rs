use anyhow::{anyhow, Result};
use raw_window_handle::HasRawWindowHandle;
// use simgame_core::world::{UpdatedWorldState, World};

pub mod test;

// TODO: UI render pipeline

pub struct RenderInit<'a, RV, RF, W> {
    pub window: &'a W,
    pub world: WorldRenderInit<RV, RF>,
    pub physical_win_size: (u32, u32),
}

pub struct WorldRenderInit<RV, RF> {
    pub vert_shader_spirv_bytes: RV,
    pub frag_shader_spirv_bytes: RF,
}

pub struct RenderState {
    device: wgpu::Device,
    swap_chain: wgpu::SwapChain,
    queue: wgpu::Queue,
    world: WorldRenderState,
}

struct WorldRenderState {
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl RenderState {
    pub fn new<RV, RF, W>(init: RenderInit<RV, RF, W>) -> Result<Self>
    where
        RV: std::io::Seek + std::io::Read,
        RF: std::io::Seek + std::io::Read,
        W: HasRawWindowHandle,
    {
        let surface = wgpu::Surface::create(init.window);

        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            backends: wgpu::BackendBit::PRIMARY,
        })
        .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        });

        let swap_chain = device.create_swap_chain(
            &surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: init.physical_win_size.0,
                height: init.physical_win_size.1,
                present_mode: wgpu::PresentMode::Vsync,
            },
        );

        let world_render_state = WorldRenderState::new(init.world, &device)?;

        Ok(RenderState {
            swap_chain,
            world: world_render_state,
            queue,
            device,
        })
    }

    pub fn render_frame(&mut self) {
        let frame = self.swap_chain.get_next_texture();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        self.world.render_frame(&frame, &mut encoder);
        self.queue.submit(&[encoder.finish()]);
    }
}

impl WorldRenderState {
    fn new<RV, RF>(
        init: WorldRenderInit<RV, RF>,
        device: &wgpu::Device,
    ) -> Result<Self>
    where
        RV: std::io::Seek + std::io::Read,
        RF: std::io::Seek + std::io::Read,
    {
        let vs_module =
            device.create_shader_module(&wgpu::read_spirv(init.vert_shader_spirv_bytes)?);
        let fs_module =
            device.create_shader_module(&wgpu::read_spirv(init.frag_shader_spirv_bytes)?);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Front,
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
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Ok(WorldRenderState {
            render_pipeline,
            bind_group,
        })
    }

    fn render_frame(&mut self, frame: &wgpu::SwapChainOutput, encoder: &mut wgpu::CommandEncoder) {
        let background_color = wgpu::Color::BLACK;
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: background_color,
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }


    // pub fn update(&mut self, world: &World, updated_state: &UpdatedWorldState) {
    //     unimplemented!();
    // }
}
