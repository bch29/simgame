mod buffer_util;
mod mesh;
mod pipelines;
pub mod resource;
pub mod shaders;
mod view;

use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use cgmath::{SquareMatrix, Vector2, Vector3};
use raw_window_handle::HasRawWindowHandle;

use simgame_types::{UpdatedWorldState, World};
use simgame_util::{convert_vec, DivUp};
use simgame_voxels::{index_utils, VoxelConfigHelper};

pub(crate) use pipelines::{FrameRenderContext, GraphicsContext};
use pipelines::{Pipeline, State as PipelineState};
use resource::ResourceLoader;
pub use view::ViewParams;
pub(crate) use view::ViewState;

pub struct RendererBuilder<'a, W> {
    pub window: &'a W,
    pub resource_loader: ResourceLoader,
    pub voxel_helper: &'a VoxelConfigHelper,
    pub max_visible_chunks: usize,
}

pub struct RenderStateInputs<'a> {
    pub physical_win_size: Vector2<u32>,
    pub voxel_helper: &'a VoxelConfigHelper,
    pub view_params: ViewParams,
    pub world: &'a World,
}

#[derive(Debug, Clone)]
pub struct RenderParams<'a> {
    pub trace_path: Option<&'a std::path::Path>,
}

/// Holds all state involved in rendering the game.
pub struct RenderState {
    swapchain: wgpu::SwapChain,
    gui: pipelines::gui::GuiRenderState,
    world_voxels: pipelines::voxels::VoxelRenderState,
    view: ViewState,
}

/// Object responsible for rendering the game.
pub struct Renderer {
    ctx: GraphicsContext,
    surface: wgpu::Surface,
    max_visible_chunks: usize,
    pipelines: Pipelines,
}

pub fn visible_size_to_chunks(visible_size: Vector3<i32>) -> Vector3<i32> {
    visible_size.div_up(convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}

struct Pipelines {
    gui: pipelines::gui::GuiRenderPipeline,
    voxel: pipelines::voxels::VoxelRenderPipeline,
}

struct DeviceResult {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub multi_draw_enabled: bool,
}
impl<'a, W> RendererBuilder<'a, W>
where
    W: HasRawWindowHandle,
{
    pub async fn build(self, params: RenderParams<'a>) -> Result<Renderer> {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        let surface = unsafe { instance.create_surface(self.window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let device_result: DeviceResult = request_device(&params, &adapter).await?;
        let ctx = GraphicsContext {
            device: device_result.device,
            queue: device_result.queue,
            multi_draw_enabled: device_result.multi_draw_enabled,
            resource_loader: self.resource_loader,
        };

        let max_visible_chunks = self.max_visible_chunks;

        let pipelines = {
            let gui = pipelines::gui::GuiRenderPipeline::new(&ctx)?;
            let voxel = {
                let voxel_info_manager =
                    pipelines::voxels::VoxelInfoManager::new(self.voxel_helper, &ctx.resource_loader, &ctx)?;
                pipelines::voxels::VoxelRenderPipeline::new(&ctx, voxel_info_manager)?
            };
            Pipelines { gui, voxel }
        };

        Ok(Renderer {
            ctx,
            surface,
            max_visible_chunks,
            pipelines,
        })
    }
}

impl Renderer {
    pub fn create_state(&self, input: RenderStateInputs) -> Result<RenderState> {
        let swapchain_descriptor = swapchain_descriptor(input.physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&self.surface, &swapchain_descriptor);

        let view = ViewState::new(input.view_params, input.physical_win_size);
        let depth_texture = make_depth_texture(&self.ctx.device, input.physical_win_size);
        let pipeline_params = pipelines::Params {
            physical_win_size: input.physical_win_size,
            depth_texture: &depth_texture,
        };

        let world_voxels = self.pipelines.voxel.create_state(
            &self.ctx,
            pipeline_params,
            pipelines::voxels::VoxelRenderInput {
                model: SquareMatrix::identity(),
                voxels: &input.world.voxels,
                max_visible_chunks: self.max_visible_chunks,
                view_state: &view,
            },
        )?;

        let gui = self
            .pipelines
            .gui
            .create_state(&self.ctx, pipeline_params, ())?;
        Ok(RenderState {
            swapchain,
            gui,
            world_voxels,
            view,
        })
    }

    pub fn update_win_size(
        &self,
        state: &mut RenderState,
        physical_win_size: Vector2<u32>,
    ) -> Result<()> {
        let swapchain_descriptor = swapchain_descriptor(physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&self.surface, &swapchain_descriptor);

        state.swapchain = swapchain;

        state.view.update_display_size(physical_win_size);

        let depth_texture = make_depth_texture(&self.ctx.device, physical_win_size);
        let pipeline_params = pipelines::Params {
            physical_win_size,
            depth_texture: &depth_texture,
        };
        state.world_voxels.update_window(&self.ctx, pipeline_params);
        state.gui.update_window(&self.ctx, pipeline_params);
        Ok(())
    }

    pub fn set_world_view(&self, state: &mut RenderState, params: ViewParams) {
        state.view.params = params;
    }

    pub fn render_frame(&self, state: &mut RenderState) -> Result<()> {
        let frame = state.swapchain.get_current_frame()?;
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render = FrameRenderContext { encoder, frame };

        self.pipelines
            .voxel
            .render_frame(&self.ctx, &mut render, &mut state.world_voxels);
        self.pipelines
            .gui
            .render_frame(&self.ctx, &mut render, &mut state.gui);

        let ts_begin = Instant::now();
        self.ctx
            .queue
            .submit(std::iter::once(render.encoder.finish()));
        let ts_submit = Instant::now();

        metrics::timing!("render.submit", ts_submit.duration_since(ts_begin));

        Ok(())
    }

    pub fn update(&self, state: &mut RenderState, world: &World, diff: &UpdatedWorldState) {
        state.world_voxels.update(
            pipelines::voxels::VoxelRenderInput {
                model: SquareMatrix::identity(),
                voxels: &world.voxels,
                max_visible_chunks: self.max_visible_chunks,
                view_state: &state.view,
            },
            pipelines::voxels::VoxelRenderInputDelta {
                voxels: &diff.voxels,
            },
        );
        state.gui.update((), ());
    }
}

async fn request_device(
    params: &RenderParams<'_>,
    adapter: &wgpu::Adapter,
) -> Result<DeviceResult> {
    let required_features = wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
        | wgpu::Features::UNSIZED_BINDING_ARRAY;

    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MULTI_DRAW_INDIRECT | required_features,
                limits: wgpu::Limits {
                    max_bind_groups: 6,
                    max_storage_buffers_per_shader_stage: 6,
                    max_storage_textures_per_shader_stage: 6,
                    ..Default::default()
                },
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

fn make_depth_texture(device: &wgpu::Device, size: Vector2<u32>) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world depth texture"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
    })
}

fn swapchain_descriptor(win_size: Vector2<u32>) -> wgpu::SwapChainDescriptor {
    wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: win_size.x,
        height: win_size.y,
        present_mode: wgpu::PresentMode::Fifo,
    }
}
