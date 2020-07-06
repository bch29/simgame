use anyhow::Result;
use cgmath::{Deg, EuclideanSpace, InnerSpace, Matrix4, Point3, SquareMatrix, Vector3};

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

#[derive(Debug, Clone)]
pub struct Shaders<R> {
    pub vert: R,
    pub frag: R,
    pub comp: R,
}

#[derive(Debug)]
pub struct WorldRenderInit<'a> {
    pub shaders: Shaders<&'a [u32]>,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
    pub view_params: ViewParams,
    pub world: &'a World,
    pub max_visible_chunks: usize,
}

pub struct WorldRenderState {
    render_blocks: blocks::BlocksRenderState,

    view_state: ViewState,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView
}

pub struct FrameRender<'a> {
    pub queue: &'a wgpu::Queue,
    pub device: &'a wgpu::Device,
    pub frame: &'a wgpu::SwapChainFrame,
    pub view_state: &'a ViewState,
}

impl WorldRenderState {
    pub fn set_view(&mut self, params: ViewParams) {
        self.view_state.params = params;
        self.render_blocks.set_view(&params);
    }

    pub fn new(init: WorldRenderInit, device_result: &crate::DeviceResult) -> Result<Self> {
        let crate::DeviceResult {
            device,
            queue,
            multi_draw_enabled,
        } = device_result;

        let shaders = init.shaders.map_result::<std::io::Error, _, _>(|stream| {
            Ok(device.create_shader_module(wgpu::ShaderModuleSource::SpirV(stream)))
        })?;

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world depth texture"),
            size: wgpu::Extent3d {
                width: init.width,
                height: init.height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let view_state = ViewState {
            params: init.view_params,
            proj: OPENGL_TO_WGPU_MATRIX
                * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 1000.0),
            rotation: Matrix4::identity(),
        };

        let render_blocks = blocks::BlocksRenderState::new(
            blocks::BlocksRenderInit {
                shaders: &shaders,
                view_state: &view_state,
                aspect_ratio: init.aspect_ratio,
                depth_texture: &depth_texture,
                world: init.world,
                max_visible_chunks: init.max_visible_chunks,
                multi_draw_enabled: *multi_draw_enabled,
            },
            device,
            queue,
        );

        Ok(WorldRenderState {
            render_blocks,
            view_state,
        })
    }

    pub fn render_frame(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainFrame,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let render = FrameRender {
            queue,
            device,
            frame,
            view_state: &self.view_state,
        };
        self.render_blocks.render_frame(&render, encoder);
    }

    pub fn update(&mut self, queue: &wgpu::Queue, world: &World, diff: &UpdatedWorldState) {
        self.render_blocks
            .update(queue, world, diff, &self.view_state);
    }
}

impl ViewParams {
    /// Calculates the box containing blocks that will be rendered according to current view.
    pub fn calculate_view_box(&self, world: &World) -> Option<Bounds<i32>> {
        // center x and y 60 blocks in front of the camera
        let mut center = self.camera_pos + 60. * self.look_at_dir.normalize();

        // z_level is the topmost visible level
        center.z = self.z_level as f32 + 1.0 - self.visible_size.z as f32 / 2.0;

        let size = convert_vec!(self.visible_size, f32);
        let float_bounds = Bounds::new(center - 0.5 * size, size);

        let world_bounds_limit = world.blocks.bounds().limit();
        let positive_box =
            Bounds::from_limit(Point3::origin(), convert_point!(world_bounds_limit, f32));
        float_bounds.intersection(positive_box).map(|bounds| {
            Bounds::new(
                convert_point!(bounds.origin(), i32),
                convert_vec!(bounds.size(), i32),
            )
        })
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
        let translation =
            Matrix4::from_translation(Vector3::new(0.0, 0.0, -self.params.z_level as f32));
        translation * self.rotation
    }
}

impl<R> Shaders<R> {
    pub fn map<'a, R2, F>(&'a self, mut f: F) -> Shaders<R2>
    where
        F: FnMut(&'a R) -> R2,
    {
        Shaders {
            vert: f(&self.vert),
            frag: f(&self.frag),
            comp: f(&self.comp),
        }
    }

    pub fn map_result<'a, E, R2, F>(&'a self, mut f: F) -> std::result::Result<Shaders<R2>, E>
    where
        F: FnMut(&'a R) -> std::result::Result<R2, E>,
    {
        Ok(Shaders {
            vert: f(&self.vert)?,
            frag: f(&self.frag)?,
            comp: f(&self.comp)?,
        })
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
