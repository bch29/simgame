mod buffer_util;
mod gui;
mod mesh;
pub mod resource;
pub mod shaders;
mod voxels;

use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use cgmath::{Deg, InnerSpace, Matrix4, Point3, SquareMatrix, Vector2, Vector3};
use raw_window_handle::HasRawWindowHandle;

use simgame_types::{UpdatedWorldState, World};
use simgame_util::{convert_point, convert_vec, Bounds, DivUp};
use simgame_voxels::{index_utils, VoxelConfigHelper};

use gui::{GuiRenderState, GuiRenderStateBuilder};
use resource::ResourceLoader;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
    pub visible_size: Vector3<i32>,
    pub look_at_dir: Vector3<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct ViewState {
    pub params: ViewParams,
    pub proj: Matrix4<f32>,
}

pub struct RenderStateBuilder<'a, W> {
    pub window: &'a W,
    pub physical_win_size: Vector2<u32>,
    pub resource_loader: ResourceLoader,
    pub view_params: ViewParams,
    pub world: &'a World,
    pub voxel_helper: &'a VoxelConfigHelper,
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
    gui_render_state: GuiRenderState,
    voxel_state: voxels::VoxelRenderState,
    view_state: ViewState,
    voxel_pipeline: voxels::VoxelRenderPipeline,
}

struct GraphicsContext {
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

struct FrameRenderContext {
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
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let device_result: DeviceResult = RenderState::request_device(&params, &adapter).await?;
        let ctx = GraphicsContext {
            device: device_result.device,
            queue: device_result.queue,
            multi_draw_enabled: device_result.multi_draw_enabled,
            resource_loader: self.resource_loader,
        };

        let swapchain_descriptor = swapchain_descriptor(self.physical_win_size);
        let swapchain = ctx
            .device
            .create_swap_chain(&surface, &swapchain_descriptor);

        let view_state = ViewState::new(self.view_params, self.physical_win_size);

        let voxel_info_manager =
            voxels::VoxelInfoManager::new(self.voxel_helper, &ctx.resource_loader, &ctx)?;

        let voxel_pipeline = voxels::VoxelRenderPipeline::new(&ctx, voxel_info_manager)?;

        let depth_texture = make_depth_texture(&ctx.device, self.physical_win_size);
        let voxel_state = voxels::VoxelRenderStateBuilder {
            view_state: &view_state,
            model: SquareMatrix::identity(),
            depth_texture: &depth_texture,
            voxels: &self.world.voxels,
            max_visible_chunks: self.max_visible_chunks,
            pipeline: &voxel_pipeline,
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
            view_state,
            voxel_pipeline,
            voxel_state,
            gui_render_state,
        })
    }
}

impl RenderState {
    pub fn update_win_size(&mut self, physical_win_size: Vector2<u32>) -> Result<()> {
        let swapchain_descriptor = swapchain_descriptor(physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&self.surface, &swapchain_descriptor);

        self.swapchain = swapchain;

        let depth_texture = make_depth_texture(&self.ctx.device, physical_win_size);
        self.voxel_state.set_depth_texture(&depth_texture);

        self.view_state.update_display_size(physical_win_size);

        self.gui_render_state
            .update_swapchain(&self.ctx, &swapchain_descriptor)?;
        Ok(())
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

    pub fn set_world_view(&mut self, params: ViewParams) {
        self.view_state.params = params;
    }

    pub fn render_frame(&mut self) -> Result<()> {
        let frame = self.swapchain.get_current_frame()?;
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render = FrameRenderContext { encoder, frame };

        self.voxel_pipeline.render_frame(
            &self.ctx,
            &mut render,
            &mut self.voxel_state,
            &self.view_state,
        );
        self.gui_render_state.render_frame(&self.ctx, &mut render);

        let ts_begin = Instant::now();
        self.ctx
            .queue
            .submit(std::iter::once(render.encoder.finish()));
        let ts_submit = Instant::now();

        metrics::timing!("render.submit", ts_submit.duration_since(ts_begin));

        Ok(())
    }

    pub fn update(&mut self, world: &World, diff: &UpdatedWorldState) {
        self.voxel_state.update(
            &world.voxels,
            &diff.voxels,
            &self.view_state,
            SquareMatrix::identity(),
        );
    }
}

pub fn visible_size_to_chunks(visible_size: Vector3<i32>) -> Vector3<i32> {
    visible_size.div_up(convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}

fn create_projection_matrix(aspect_ratio: f32) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * cgmath::perspective(Deg(70f32), aspect_ratio, 1.0, 1000.0)
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

impl ViewParams {
    /// Calculates the box containing voxels that will be rendered according to current view.
    pub fn calculate_view_box(&self) -> Option<Bounds<i32>> {
        let visible_distance = Vector2 {
            x: self.visible_size.x as f32,
            y: self.visible_size.y as f32,
        }
        .magnitude()
            / 3.;

        let mut center = self.camera_pos + visible_distance * self.look_at_dir.normalize();

        // z_level is the topmost visible level
        center.z = self.z_level as f32 + 1.0 - self.visible_size.z as f32 / 2.0;

        let size = convert_vec!(self.visible_size, f32);
        let float_bounds = Bounds::new(center - 0.5 * size, size);

        Some(Bounds::new(
            convert_point!(float_bounds.origin(), i32),
            convert_vec!(float_bounds.size(), i32),
        ))
    }

    pub fn effective_camera_pos(&self) -> Point3<f32> {
        self.camera_pos + Vector3::unit_z() * self.z_level as f32
    }
}

impl Default for ViewParams {
    fn default() -> Self {
        ViewParams {
            camera_pos: Point3::new(0., 0., 0.),
            z_level: 0,
            visible_size: Vector3::new(1, 1, 1),
            look_at_dir: Vector3::new(1.0, 1.0, -6.0),
        }
    }
}

impl ViewState {
    fn new(params: ViewParams, display_size: Vector2<u32>) -> Self {
        let aspect_ratio = display_size.x as f32 / display_size.y as f32;

        ViewState {
            params,
            proj: create_projection_matrix(aspect_ratio),
        }
    }

    pub fn params(&self) -> ViewParams {
        self.params
    }

    pub fn proj(&self) -> Matrix4<f32> {
        self.proj
    }

    pub fn view(&self) -> Matrix4<f32> {
        Matrix4::look_at_dir(
            Point3::new(0., 0., 0.),
            self.params().look_at_dir,
            Vector3::unit_z(),
        )
    }

    pub fn camera_pos(&self) -> Point3<f32> {
        self.params.effective_camera_pos()
    }

    pub fn update_display_size(&mut self, display_size: Vector2<u32>) {
        let aspect_ratio = display_size.x as f32 / display_size.y as f32;
        self.proj = create_projection_matrix(aspect_ratio);
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
