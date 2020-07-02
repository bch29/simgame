use anyhow::{anyhow, Result};
use raw_window_handle::HasRawWindowHandle;

use simgame_core::world::{UpdatedWorldState, World};

pub mod buffer_util;
pub mod mesh;
pub mod stable_map;
pub mod test;
mod triangulate;
mod world;

pub use world::WorldRenderInit;

// TODO: UI rendering pipeline

pub struct RenderInit<'a, RV, RF, W> {
    pub window: &'a W,
    pub world: WorldRenderInit<RV, RF>,
    pub physical_win_size: (u32, u32),
}

pub struct RenderState {
    device: wgpu::Device,
    swap_chain: wgpu::SwapChain,
    queue: wgpu::Queue,
    world: world::WorldRenderState,
}

impl RenderState {
    pub fn set_world_view(&mut self, params: world::ViewParams) {
        self.world.set_view(params);
    }

    pub async fn new<'a, RV, RF, W>(init: RenderInit<'a, RV, RF, W>) -> Result<Self>
    where
        RV: std::io::Seek + std::io::Read,
        RF: std::io::Seek + std::io::Read,
        W: HasRawWindowHandle,
    {
        let surface = wgpu::Surface::create(init.window);

        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            },
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        }).await;

        let swap_chain = device.create_swap_chain(
            &surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: init.physical_win_size.0,
                height: init.physical_win_size.1,
                present_mode: wgpu::PresentMode::Fifo,
            },
        );

        let world_render_state = world::WorldRenderState::new(init.world, &device)?;

        Ok(RenderState {
            swap_chain,
            world: world_render_state,
            queue,
            device,
        })
    }

    pub fn render_frame(&mut self) {
        let frame = self.swap_chain.get_next_texture().expect("failed to get next texture");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.world.render_frame(&self.device, &frame, &mut encoder);
        self.queue.submit(&[encoder.finish()]);
    }

    pub fn init(&mut self, world: &World) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.world.init(&self.device, &mut encoder, world);
        self.queue.submit(&[encoder.finish()]);
    }

    pub fn update(&mut self, world: &World, diff: &UpdatedWorldState) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.world.update(&self.device, &mut encoder, world, diff);
        self.queue.submit(&[encoder.finish()]);
    }
}
