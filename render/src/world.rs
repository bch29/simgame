use anyhow::Result;
use cgmath::{Deg, InnerSpace, Matrix4, Point3, SquareMatrix, Vector2, Vector3};

use simgame_core::{
    convert_point, convert_vec,
    util::Bounds,
    world::{UpdatedWorldState, World},
};

mod blocks;

pub use blocks::visible_size_to_chunks;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
    pub visible_size: Vector3<i32>,
    pub look_at_dir: Vector3<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewState {
    params: ViewParams,
    proj: Matrix4<f32>,
    rotation: Matrix4<f32>,
}

pub(crate) struct WorldRenderStateBuilder<'a> {
    pub view_params: ViewParams,
    pub world: &'a World,
    pub max_visible_chunks: usize,
    pub swapchain: &'a wgpu::SwapChainDescriptor,
}

pub(crate) struct WorldRenderState {
    render_blocks: blocks::BlocksRenderState,

    view_state: ViewState,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView
}

impl<'a> WorldRenderStateBuilder<'a> {
    pub fn build(self, ctx: &crate::GraphicsContext) -> Result<WorldRenderState> {
        let display_size = Vector2::new(self.swapchain.width, self.swapchain.height);
        let depth_texture = WorldRenderState::make_depth_texture(&ctx.device, display_size);

        let aspect_ratio = display_size.x as f32 / display_size.y as f32;

        let view_state = ViewState {
            params: self.view_params,
            proj: WorldRenderState::create_projection_matrix(aspect_ratio),
            rotation: Matrix4::identity(),
        };

        let render_blocks = blocks::BlocksRenderStateBuilder {
            view_state: &view_state,
            depth_texture: &depth_texture,
            blocks: &self.world.blocks,
            max_visible_chunks: self.max_visible_chunks,
            block_config: &self.world.block_helper,
        }
        .build(ctx)?;

        Ok(WorldRenderState {
            render_blocks,
            view_state,
        })
    }
}

impl WorldRenderState {
    pub fn set_view(&mut self, params: ViewParams) {
        self.view_state.params = params;
        self.render_blocks.set_view(&params);
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
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        })
    }

    pub fn update_swapchain(
        &mut self,
        ctx: &crate::GraphicsContext,
        swapchain: &wgpu::SwapChainDescriptor,
    ) -> Result<()> {
        let display_size = Vector2::new(swapchain.width, swapchain.height);

        let depth_texture = Self::make_depth_texture(&ctx.device, display_size);
        self.render_blocks.set_depth_texture(&depth_texture);

        let aspect_ratio = display_size.x as f32 / display_size.y as f32;
        self.view_state.proj = Self::create_projection_matrix(aspect_ratio);

        Ok(())
    }

    fn create_projection_matrix(aspect_ratio: f32) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * cgmath::perspective(Deg(70f32), aspect_ratio, 1.0, 1000.0)
    }

    pub fn render_frame(
        &mut self,
        ctx: &crate::GraphicsContext,
        frame_render: &mut crate::FrameRenderContext,
    ) {
        self.render_blocks
            .render_frame(ctx, frame_render, &self.view_state);
    }

    pub fn update(&mut self, queue: &wgpu::Queue, world: &World, diff: &UpdatedWorldState) {
        self.render_blocks
            .update(queue, &world.blocks, &diff.blocks, &self.view_state);
    }
}

impl ViewParams {
    /// Calculates the box containing blocks that will be rendered according to current view.
    pub fn calculate_view_box(&self) -> Option<Bounds<i32>> {
        let visible_distance = Vector2 {
            x: self.visible_size.x as f32,
            y: self.visible_size.y as f32
        }.magnitude() / 3.;

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

    pub fn model(&self) -> Matrix4<f32> {
        // let translation =
        //     Matrix4::from_translation(Vector3::new(0.0, 0.0, -self.params.z_level as f32));
        // translation * self.rotation
        self.rotation
    }

    pub fn camera_pos(&self) -> Point3<f32> {
        self.params.effective_camera_pos()
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
