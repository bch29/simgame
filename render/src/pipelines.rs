use anyhow::Result;
use cgmath::Vector2;

use crate::resource::ResourceLoader;
use simgame_types::Directory;

pub mod gui;
pub mod mesh;
pub mod voxels;

pub(crate) struct GraphicsContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub textures: crate::resource::Textures,
    pub multi_draw_enabled: bool,
}

pub(crate) struct FrameRenderContext {
    pub frame: wgpu::SwapChainFrame,
    pub encoder: wgpu::CommandEncoder,
}

#[derive(Clone, Copy)]
pub(crate) struct Params<'a> {
    pub physical_win_size: Vector2<u32>,
    pub depth_texture: &'a wgpu::Texture,
    pub resource_loader: &'a ResourceLoader,
}

pub(crate) trait State<'a> {
    type Input;
    type InputDelta;

    fn update(&mut self, ctx: &GraphicsContext, input: Self::InputDelta);

    fn update_window(&mut self, ctx: &GraphicsContext, params: Params);
}

pub(crate) trait Pipeline: Sized {
    type State: for<'a> State<'a>;

    fn create_pipeline(
        ctx: &GraphicsContext,
        directory: &Directory,
        resource_loader: &ResourceLoader,
    ) -> Result<Self>;

    fn create_state<'a>(
        &self,
        ctx: &GraphicsContext,
        params: Params<'a>,
        input: <Self::State as State<'a>>::Input,
    ) -> Result<Self::State>;

    fn render_frame(
        &self,
        ctx: &GraphicsContext,
        load_action: LoadAction,
        frame_render: &mut FrameRenderContext,
        state: &mut Self::State,
    );
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LoadAction {
    Load,
    Clear,
}

impl LoadAction {
    pub fn into_load_op<T>(self, clear_value: T) -> wgpu::LoadOp<T> {
        match self {
            LoadAction::Load => wgpu::LoadOp::Load,
            LoadAction::Clear => wgpu::LoadOp::Clear(clear_value),
        }
    }
}
