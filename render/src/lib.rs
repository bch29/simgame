use anyhow::{anyhow, Result};
use raw_window_handle::HasRawWindowHandle;

use simgame_core::world::{UpdatedWorldState, World};

pub mod buffer_util;
pub mod mesh;
pub mod stable_map;
pub mod test;
mod triangulate;
mod world;

pub use world::Shaders as WorldShaders;
pub use world::WorldRenderInit;

// TODO: UI rendering pipeline

pub struct RenderInit<'a, W> {
    pub window: &'a W,
    pub world: WorldRenderInit<'a>,
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

    pub async fn new<'a, W>(init: RenderInit<'a, W>) -> Result<Self>
    where
        W: HasRawWindowHandle,
    {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        let surface = unsafe { instance.create_surface(init.window) };

        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                },
                wgpu::UnsafeFeatures::disallow(),
            )
            .await
            .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::MULTI_DRAW_INDIRECT,
                    shader_validation: false,
                    limits: wgpu::Limits::default()
                },
                None,
            )
            .await
            .map_err(|err| anyhow!("{:?}", err))?;

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

        let world_render_state = world::WorldRenderState::new(init.world, &device, &queue)?;

        Ok(RenderState {
            swap_chain,
            world: world_render_state,
            queue,
            device,
        })
    }

    pub fn render_frame(&mut self) {
        let frame = self
            .swap_chain
            .get_next_frame()
            .expect("failed to get next frame");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.world.render_frame(&self.queue, &self.device, &frame, &mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn update(&mut self, world: &World, diff: &UpdatedWorldState) {
        self.world.update(&self.queue, world, diff);
    }
}
