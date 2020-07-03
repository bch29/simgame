use anyhow::Result;
use cgmath::{Deg, EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};

use simgame_core::{
    convert_point, convert_vec,
    util::Bounds,
    world::{UpdatedWorldState, World},
};

mod blocks;

const LOOK_AT_DIR: Vector3<f32> = Vector3::new(1., 1., -3.);

#[derive(Debug, Clone, PartialEq)]
pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
    pub visible_size: Vector3<i32>,
}

pub struct Shaders<R> {
    pub vert: R,
    pub frag: R,
    pub comp: R,
}

pub struct WorldRenderInit<R> {
    pub shaders: Shaders<R>,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
}

pub struct WorldRenderState {
    render_blocks: blocks::BlocksRenderState,

    view_params: ViewParams,
    rotation: Matrix4<f32>,
    uniforms: Uniforms,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView
}

pub struct Uniforms {
    pub proj: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub model: Matrix4<f32>,
    pub camera_pos: Point3<f32>,
}

pub struct FrameRender<'a> {
    device: &'a wgpu::Device,
    frame: &'a wgpu::SwapChainOutput,
    uniforms: &'a Uniforms,
}

impl WorldRenderState {
    pub fn set_view(&mut self, params: ViewParams) {
        if params == self.view_params {
            return;
        }

        self.render_blocks.set_view(&params);
        self.uniforms.update_view_params(&params);
        self.view_params = params;
    }

    pub fn new<R>(init: WorldRenderInit<R>, device: &wgpu::Device) -> Result<Self>
    where
        R: std::io::Seek + std::io::Read,
    {
        let shaders = init.shaders.map_result::<std::io::Error, _, _>(|stream| {
            Ok(device.create_shader_module(&wgpu::read_spirv(stream)?))
        })?;

        let view_params = ViewParams::default();

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("simgame_render::world::WorldRenderState/depth"),
            size: wgpu::Extent3d {
                width: init.width,
                height: init.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let mut uniforms = Uniforms {
            proj: OPENGL_TO_WGPU_MATRIX
                * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 1000.0),
            view: Matrix4::look_at_dir(Point3::new(0., 0., 0.), LOOK_AT_DIR, Vector3::unit_z()),
            model: Matrix4::<f32>::identity(),
            camera_pos: Point3::origin(),
        };
        uniforms.update_view_params(&view_params);

        let render_blocks = blocks::BlocksRenderState::new(
            blocks::BlocksRenderInit {
                shaders: &shaders,
                view_params: &view_params,
                aspect_ratio: init.aspect_ratio,
                depth_texture: &depth_texture,
                uniforms: &uniforms,
            },
            device,
        );

        Ok(WorldRenderState {
            render_blocks,
            rotation: Matrix4::identity(),
            view_params,
            uniforms,
        })
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.uniforms.model = self.rotation;

        let render = FrameRender {
            device,
            frame,
            uniforms: &self.uniforms,
        };
        self.render_blocks.render_frame(&render, encoder);
    }

    /// Calculates the box containing chunks that will be rendered according to current view.
    fn calculate_view_box(&self, world: &World) -> Option<Bounds<i32>> {
        let mut center = self.uniforms.camera_pos + 60. * LOOK_AT_DIR;
        center.z = self.view_params.z_level as f32 - 0.5;
        let size = convert_vec!(self.view_params.visible_size, f32);
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

    pub fn init(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
    ) {
        let active_view_box = self.calculate_view_box(world);
        self.render_blocks
            .init(device, encoder, world, active_view_box);
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        diff: &UpdatedWorldState,
    ) {
        let active_view_box = self.calculate_view_box(world);
        self.render_blocks
            .update(device, encoder, world, diff, active_view_box);
    }
}

impl Default for ViewParams {
    fn default() -> Self {
        ViewParams {
            camera_pos: Point3::new(0., 0., 0.),
            z_level: 0,
            visible_size: Vector3::new(1, 1, 1),
        }
    }
}

impl Uniforms {
    fn update_view_params(&mut self, params: &ViewParams) {
        self.camera_pos = params.camera_pos;
    }
}

impl<R> Shaders<R> {
    pub fn map<R2, F>(self, mut f: F) -> Shaders<R2>
    where
        F: FnMut(R) -> R2,
    {
        Shaders {
            vert: f(self.vert),
            frag: f(self.frag),
            comp: f(self.comp),
        }
    }

    pub fn map_result<E, R2, F>(self, mut f: F) -> std::result::Result<Shaders<R2>, E>
    where
        F: FnMut(R) -> std::result::Result<R2, E>,
    {
        Ok(Shaders {
            vert: f(self.vert)?,
            frag: f(self.frag)?,
            comp: f(self.comp)?,
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
