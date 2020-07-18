use anyhow::{anyhow, bail, Result};
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

use gui::{GuiRenderState, GuiRenderStateBuilder};
use resource::ResourceLoader;
use world::{WorldRenderState, WorldRenderStateBuilder};

// TODO: UI rendering pipeline

pub struct RenderStateBuilder<'a, W> {
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
    ctx: GraphicsContext,
    surface: wgpu::Surface,
    swapchain: wgpu::SwapChain,
    world_render_state: WorldRenderState,
    gui_render_state: GuiRenderState,
}

pub(crate) struct GraphicsContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub resource_loader: ResourceLoader,
    pub multi_draw_enabled: bool,
}

struct DeviceResult {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub multi_draw_enabled: bool,
}

pub struct FrameRenderContext {
    pub frame: wgpu::SwapChainFrame,
    pub encoder: wgpu::CommandEncoder,
}

impl<'a, W> RenderStateBuilder<'a, W>
where
    W: HasRawWindowHandle,
{
    pub async fn build(self, params: RenderParams<'a>) -> Result<RenderState> {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        let surface = unsafe { instance.create_surface(self.window) };

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

        let device_result: DeviceResult = RenderState::request_device(&params, &adapter).await?;
        let ctx = GraphicsContext {
            device: device_result.device,
            queue: device_result.queue,
            multi_draw_enabled: device_result.multi_draw_enabled,
            resource_loader: self.resource_loader,
        };

        let swapchain_descriptor = RenderState::swapchain_descriptor(self.physical_win_size);
        let swapchain = ctx
            .device
            .create_swap_chain(&surface, &swapchain_descriptor);

        let world_render_state = WorldRenderStateBuilder {
            view_params: self.view_params,
            world: self.world,
            max_visible_chunks: self.max_visible_chunks,
            swapchain: &swapchain_descriptor,
        }
        .build(&ctx)?;

        let gui_render_state = GuiRenderStateBuilder {
            swapchain: &swapchain_descriptor,
        }
        .build(&ctx)?;

        Ok(RenderState {
            ctx,
            surface,
            swapchain,
            world_render_state,
            gui_render_state,
        })
    }
}

impl RenderState {
    pub fn update_win_size(&mut self, physical_win_size: Vector2<u32>) -> Result<()> {
        let swapchain_descriptor = Self::swapchain_descriptor(physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&self.surface, &swapchain_descriptor);

        self.swapchain = swapchain;
        self.world_render_state
            .update_swapchain(&self.ctx, &swapchain_descriptor)?;
        self.gui_render_state
            .update_swapchain(&self.ctx, &swapchain_descriptor)?;
        Ok(())
    }

    async fn request_device<'a>(
        params: &RenderParams<'a>,
        adapter: &wgpu::Adapter,
    ) -> Result<DeviceResult> {
        let required_features = wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
            | wgpu::Features::UNSIZED_BINDING_ARRAY;

        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::MULTI_DRAW_INDIRECT | required_features,
                    shader_validation: false,
                    limits: wgpu::Limits::default(),
                },
                params.trace_path,
            )
            .await
        {
            Ok(res) => res,
            Err(err) => {
                bail!("Failed to request graphics device: {:?}", err);
            }
        };

        if !device.features().contains(required_features) {
            bail!("Graphics device does not support required features");
        }

        let multi_draw_enabled = device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT);
        if !multi_draw_enabled {
            log::warn!("Graphics device does not support MULTI_DRAW_INDIRECT");
        }

        Ok(DeviceResult {
            device,
            queue,
            multi_draw_enabled,
        })
    }

    pub fn set_world_view(&mut self, params: ViewParams) {
        self.world_render_state.set_view(params);
    }

    pub fn render_frame(&mut self) {
        let frame = self
            .swapchain
            .get_next_frame()
            .expect("failed to get next frame");
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render = FrameRenderContext { encoder, frame };

        self.world_render_state.render_frame(&self.ctx, &mut render);
        self.gui_render_state.render_frame(&self.ctx, &mut render);

        self.ctx
            .queue
            .submit(std::iter::once(render.encoder.finish()));
    }

    pub fn update(&mut self, world: &World, diff: &UpdatedWorldState) {
        self.world_render_state.update(&self.ctx.queue, world, diff);
    }

    fn swapchain_descriptor(win_size: Vector2<u32>) -> wgpu::SwapChainDescriptor {
        wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: win_size.x,
            height: win_size.y,
            present_mode: wgpu::PresentMode::Fifo,
        }
    }
}
