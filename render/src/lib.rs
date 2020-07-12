use anyhow::{anyhow, Result};
use cgmath::Vector2;
use raw_window_handle::HasRawWindowHandle;

use simgame_core::world::{UpdatedWorldState, World};

pub mod resource;
pub mod shaders;

mod buffer_util;
mod gui;
mod mesh;
mod stable_map;
mod world;

pub use world::{visible_size_to_chunks, ViewParams};

use resource::ResourceLoader;

// TODO: UI rendering pipeline

pub struct RenderInit<'a, W> {
    pub window: &'a W,
    pub physical_win_size: Vector2<u32>,
    pub resource_loader: ResourceLoader,

    pub display_size: Vector2<u32>,
    pub view_params: ViewParams,
    pub world: &'a World,
    pub max_visible_chunks: usize,
}

#[derive(Debug, Clone)]
pub struct RenderParams<'a> {
    pub trace_path: Option<&'a std::path::Path>,
}

pub struct RenderState {
    device: wgpu::Device,
    surface: wgpu::Surface,
    swap_chain: wgpu::SwapChain,
    queue: wgpu::Queue,
    world: world::WorldRenderState,
    _resource_loader: ResourceLoader,
}

pub(crate) struct DeviceResult {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub multi_draw_enabled: bool,
}

impl RenderState {
    pub async fn new<'a, W>(params: RenderParams<'a>, init: RenderInit<'a, W>) -> Result<Self>
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

        let device_result = Self::request_device(&params, &adapter).await?;
        let device = &device_result.device;

        let swap_chain = Self::create_swap_chain(&device, &surface, init.physical_win_size)?;

        let world_render_state = world::WorldRenderState::new(
            world::WorldRenderInit {
                display_size: init.display_size,
                view_params: init.view_params,
                world: init.world,
                max_visible_chunks: init.max_visible_chunks,
                resource_loader: &init.resource_loader,
            },
            &device_result,
        )?;

        Ok(RenderState {
            surface,
            swap_chain,
            world: world_render_state,
            queue: device_result.queue,
            device: device_result.device,
            _resource_loader: init.resource_loader,
        })
    }

    pub fn update_win_size(&mut self, physical_win_size: Vector2<u32>) -> Result<()> {
        let swap_chain = Self::create_swap_chain(&self.device, &self.surface, physical_win_size)?;
        self.swap_chain = swap_chain;
        self.world
            .update_win_size(&self.device, physical_win_size)?;
        Ok(())
    }

    async fn request_device<'a>(
        params: &RenderParams<'a>,
        adapter: &wgpu::Adapter,
    ) -> Result<DeviceResult> {
        let result = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::MULTI_DRAW_INDIRECT,
                    shader_validation: false,
                    limits: wgpu::Limits::default(),
                },
                params.trace_path,
            )
            .await;

        match result {
            Ok((device, queue)) => {
                log::info!("MULTI_DRAW_INDIRECT is enabled");
                return Ok(DeviceResult {
                    device,
                    queue,
                    multi_draw_enabled: true,
                });
            }
            Err(e) => {
                log::warn!(
                    "Failed to get device with MULTI_DRAW_INDIRECT support: {:?}",
                    e
                );
            }
        };

        let result = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    shader_validation: false,
                    limits: wgpu::Limits::default(),
                },
                params.trace_path,
            )
            .await;

        match result {
            Ok((device, queue)) => {
                return Ok(DeviceResult {
                    device,
                    queue,
                    multi_draw_enabled: true,
                })
            }
            Err(e) => {
                log::warn!("Failed to get fallback device {:?}", e);
            }
        }

        Err(anyhow!("Failed to request wgpu::Adaptor"))
    }

    pub fn set_world_view(&mut self, params: world::ViewParams) {
        self.world.set_view(params);
    }

    pub fn render_frame(&mut self) {
        let frame = self
            .swap_chain
            .get_next_frame()
            .expect("failed to get next frame");
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.world
            .render_frame(&self.queue, &self.device, &frame, &mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn update(&mut self, world: &World, diff: &UpdatedWorldState) {
        self.world.update(&self.queue, world, diff);
    }

    fn create_swap_chain(
        device: &wgpu::Device,
        surface: &wgpu::Surface,
        win_size: Vector2<u32>,
    ) -> Result<wgpu::SwapChain> {
        Ok(device.create_swap_chain(
            &surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: win_size.x,
                height: win_size.y,
                present_mode: wgpu::PresentMode::Fifo,
            },
        ))
    }
}
