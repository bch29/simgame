use anyhow::Result;
use cgmath::Vector2;

pub mod gui;
pub mod voxels;

pub(crate) struct GraphicsContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub resource_loader: crate::resource::ResourceLoader,
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
}

pub(crate) trait State<'a> {
    type Input;
    type InputDelta;

    fn update(&mut self, input: Self::Input, delta: Self::InputDelta);

    fn update_window(&mut self, ctx: &GraphicsContext, params: Params);
}

pub(crate) trait Pipeline {
    type State: for<'a> State<'a>;

    fn create_state<'a>(
        &self,
        ctx: &GraphicsContext,
        params: Params<'a>,
        input: <Self::State as State<'a>>::Input,
    ) -> Result<Self::State>;

    fn render_frame(
        &self,
        ctx: &GraphicsContext,
        frame_render: &mut FrameRenderContext,
        state: &mut Self::State,
    );
}
