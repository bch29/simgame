use cgmath::{Deg, InnerSpace, Matrix4, Point3, Vector2, Vector3};
use simgame_util::{convert_point, convert_vec, Bounds};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
    pub visible_size: Vector3<i32>,
    pub look_at_dir: Vector3<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewState {
    pub params: ViewParams,
    pub proj: Matrix4<f32>,
}

impl ViewParams {
    /// Calculates the box containing voxels that will be rendered according to current view.
    pub fn calculate_view_box(&self) -> Bounds<i32> {
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

        Bounds::new(
            convert_point!(float_bounds.origin(), i32),
            convert_vec!(float_bounds.size(), i32),
        )
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
    pub fn new(params: ViewParams, display_size: Vector2<u32>) -> Self {
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
        Matrix4::look_to_rh(
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

pub fn create_projection_matrix(aspect_ratio: f32) -> Matrix4<f32> {
    OPENGL_TO_WGPU_MATRIX * cgmath::perspective(Deg(70f32), aspect_ratio, 1.0, 1000.0)
}
