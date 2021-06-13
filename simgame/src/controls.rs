use bevy::{
    app::{App, EventReader, Plugin},
    core::Time,
    ecs::system::{IntoSystem, Res, ResMut},
    input::{
        keyboard::{KeyCode, KeyboardInput},
        mouse::MouseMotion,
        ElementState,
    },
};
use cgmath::{
    Angle, Basis3, Deg, InnerSpace, Point3, Rad, Rotation, Rotation3, Vector2, Vector3, Zero,
};

use simgame_util::{convert_point, convert_vec, DivUp};
use simgame_voxels::index_utils;

use crate::settings::RenderTestParams;

pub struct ControlsPlugin(pub ControlState);

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.0.clone())
            .add_system(controls_system.system());
    }
}

fn controls_system(
    mut control_state: ResMut<ControlState>,
    mut view_params: ResMut<simgame_render::ViewParams>,
    time: Res<Time>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut keyboard_input_events: EventReader<KeyboardInput>,
) {
    for event in mouse_motion_events.iter() {
        control_state.move_cursor(Vector2 {
            x: event.delta.x as f64 / 50.0,
            y: event.delta.y as f64 / 50.0,
        });
    }

    for event in keyboard_input_events.iter() {
        control_state.handle_keyboard_input(event);
    }

    control_state.tick(time.delta_seconds_f64(), &mut *view_params);
}

#[derive(Clone)]
pub struct ControlState {
    look_at_dir: Vector3<f64>,
    camera_dir: Vector3<i32>,

    camera_pan: AccelControlState,
    camera_height: AccelControlState,

    z_level: AccelControlState,
    visible_height: AccelControlState,

    max_visible_chunks: usize,
}

#[derive(Debug, Clone)]
struct AccelControlParams {
    pub initial_step: f64,
    pub delay_seconds: f64,
    pub acceleration: f64,
    pub decay: f64,
    pub max_speed: Option<f64>,
    pub max_value: Option<Point3<f64>>,
    pub min_value: Option<Point3<f64>>,
}

#[derive(Clone)]
struct AccelControlState {
    // parameters
    params: AccelControlParams,

    // state
    direction: Option<Vector3<f64>>,
    initial_step_required: bool,
    delay_seconds_remaining: f64,
    current_vel: Vector3<f64>,
    current_value: Point3<f64>,
}

impl ControlState {
    pub fn new(params: &RenderTestParams) -> Self {
        let base_accel_params = AccelControlParams {
            initial_step: 0.0,
            delay_seconds: 0.0,
            acceleration: 1.0,
            decay: 0.0,
            max_speed: None,
            max_value: None,
            min_value: None,
        };

        ControlState {
            look_at_dir: convert_vec!(params.look_at_dir, f64).normalize(),
            max_visible_chunks: params.max_visible_chunks,
            camera_dir: Vector3::zero(),
            camera_pan: AccelControlState::new(
                convert_point!(params.initial_camera_pos, f64),
                AccelControlParams {
                    acceleration: 8.0,
                    decay: 5.0,
                    ..base_accel_params
                },
            ),
            camera_height: AccelControlState::new(
                convert_point!(params.initial_camera_pos, f64),
                AccelControlParams {
                    acceleration: 8.0,
                    decay: 5.0,
                    ..base_accel_params
                },
            ),

            z_level: AccelControlState::new(
                Point3::new(0.0, 0.0, params.initial_z_level as f64),
                AccelControlParams {
                    min_value: Some(Point3::new(0.0, 0.0, 0.0)),
                    delay_seconds: 0.5,
                    initial_step: 1.0,
                    max_speed: Some(2.0),
                    ..base_accel_params
                },
            ),
            visible_height: AccelControlState::new(
                Point3::new(0.0, 0.0, params.initial_visible_size.z as f64),
                AccelControlParams {
                    min_value: Some(Point3::new(0.0, 0.0, 1.0)),
                    delay_seconds: 0.5,
                    initial_step: 1.0,
                    max_speed: Some(2.0),
                    ..base_accel_params
                },
            ),
        }
    }

    fn handle_keyboard_input(&mut self, event: &KeyboardInput) {
        if let KeyboardInput {
            key_code: Some(key),
            state,
            ..
        } = event
        {
            let mag = self.state_to_mag(state);
            let mag_z = if mag == 0 {
                None
            } else {
                Some(Vector3::new(0.0, 0.0, mag as f64))
            };
            let neg_mag_z = mag_z.map(|v| Vector3::zero() - v);

            match key {
                KeyCode::D => self.camera_dir.x = mag,
                KeyCode::A => self.camera_dir.x = -mag,
                KeyCode::W => self.camera_dir.y = mag,
                KeyCode::S => self.camera_dir.y = -mag,
                KeyCode::K => self.camera_dir.z = mag,
                KeyCode::J => self.camera_dir.z = -mag,
                KeyCode::I => self.z_level.set_direction(mag_z),
                KeyCode::U => self.z_level.set_direction(neg_mag_z),
                KeyCode::M => self.visible_height.set_direction(mag_z),
                KeyCode::N => self.visible_height.set_direction(neg_mag_z),
                _ => {}
            }
        }

        let normalize_camera_axis = |value: &mut i32| {
            if *value > 1 {
                *value = 1
            } else if *value < -1 {
                *value = -1
            }
        };
        normalize_camera_axis(&mut self.camera_dir.x);
        normalize_camera_axis(&mut self.camera_dir.y);
        normalize_camera_axis(&mut self.camera_dir.z);

        self.camera_pan
            .set_direction(Some(Self::pan_direction(self.look_at_dir, self.camera_dir)));

        let height_dir = Vector3::new(0.0, 0.0, self.camera_dir.z as f64);
        self.camera_height.set_direction(Some(height_dir));
    }

    // fn clear_key_states(&mut self) {
    //     self.z_level.set_direction(None);
    //     self.visible_height.set_direction(None);
    // }

    fn move_cursor(&mut self, offset: Vector2<f64>) {
        if offset.x == 0. && offset.y == 0. {
            return;
        }

        let speed = 0.5;

        let up = Vector3::new(0., 0., 1.);
        let right = {
            let mut right = self.look_at_dir.cross(up);
            right.z = 0.;
            right = right.normalize();
            right
        };

        let pitch = {
            let pitch = -Rad::atan(offset.y * speed);

            // Prevent camera from turning too close to straight up or down to avoid Gimbal lock.
            let max_angle_from_vertical: Rad<f64> = Deg(15.).into();

            let angle_from_up = Rad::acos(self.look_at_dir.dot(up));
            let angle_from_down = Rad::acos(self.look_at_dir.dot(-up));

            let max_pitch: Rad<f64> = (angle_from_up - max_angle_from_vertical).normalize_signed();
            let min_pitch: Rad<f64> =
                -(angle_from_down - max_angle_from_vertical).normalize_signed();

            if pitch < min_pitch {
                min_pitch
            } else if pitch > max_pitch {
                max_pitch
            } else {
                pitch
            }
        };

        let yaw = -Rad::atan(offset.x * speed);

        let rotation: Basis3<f64> =
            Basis3::from_axis_angle(up, yaw) * Basis3::from_axis_angle(right, pitch);

        self.look_at_dir = rotation.rotate_vector(self.look_at_dir);
    }

    fn tick(&mut self, elapsed: f64, view_state: &mut simgame_render::ViewParams) {
        self.camera_pan.tick(elapsed);
        self.camera_height.tick(elapsed);
        self.z_level.tick(elapsed);
        self.visible_height.tick(elapsed);

        view_state.look_at_dir = convert_vec!(self.look_at_dir, f32);

        view_state.camera_pos.x = self.camera_pan.value().x as f32;
        view_state.camera_pos.y = self.camera_pan.value().y as f32;
        view_state.camera_pos.z = self.camera_height.value().z as f32;

        view_state.z_level = self.z_level.value().z.round() as i32;
        view_state.visible_size.z = self.visible_height.value().z.round() as i32;

        let visible_size = restrict_visible_size(self.max_visible_chunks, view_state.visible_size);
        if visible_size != view_state.visible_size {
            log::warn!(
                "Tried to increase visible size to {:?} which would exceed max_visible_chunks setting of {}. Decreasing to {:?}.",
                view_state.visible_size,
                self.max_visible_chunks,
                visible_size
            );
            view_state.visible_size = visible_size;
            self.visible_height.value_mut().z = visible_size.z as f64;
        }
    }

    fn pan_direction(look_at_dir: Vector3<f64>, direction: Vector3<i32>) -> Vector3<f64> {
        let mut forward_step = look_at_dir;
        forward_step.z = 0.0;
        forward_step = forward_step.normalize();

        let mut right_step = Vector3::new(0., 0., -1.).cross(look_at_dir);
        right_step.z = 0.0;
        right_step = right_step.normalize();

        let mut dir = Vector3::zero();
        dir += forward_step * direction.y as f64;
        dir += right_step * direction.x as f64;
        dir
    }

    fn state_to_mag(&self, state: &ElementState) -> i32 {
        match state {
            ElementState::Pressed => 1,
            ElementState::Released => 0,
        }
    }
}

impl AccelControlState {
    fn new(initial_value: Point3<f64>, params: AccelControlParams) -> Self {
        Self {
            params,

            direction: None,
            initial_step_required: false,
            delay_seconds_remaining: 0.0,
            current_vel: Vector3::zero(),
            current_value: initial_value,
        }
    }

    fn tick(&mut self, interval: f64) {
        let direction = match self.direction {
            Some(x) => x,
            None => Vector3::zero(),
        };

        let mut offset = Vector3::zero();
        if self.initial_step_required {
            offset += self.params.initial_step * direction;
            self.initial_step_required = false;
        }

        if self.delay_seconds_remaining > 0.0 {
            self.delay_seconds_remaining -= interval;
        } else {
            self.current_vel += self.params.acceleration * interval * direction;
        }

        if let Some(max_speed) = self.params.max_speed {
            if self.current_vel.magnitude() > max_speed {
                self.current_vel = self.current_vel.normalize_to(max_speed)
            }
        }

        self.current_value += self.current_vel + offset;
        self.current_vel -= self.current_vel * self.params.decay * interval;

        if let Some(max_value) = self.params.max_value {
            if self.current_value.x > max_value.x {
                self.current_value.x = max_value.x;
            }
            if self.current_value.y > max_value.y {
                self.current_value.y = max_value.y;
            }
            if self.current_value.z > max_value.z {
                self.current_value.z = max_value.z;
            }
        }

        if let Some(min_value) = self.params.min_value {
            if self.current_value.x < min_value.x {
                self.current_value.x = min_value.x;
            }
            if self.current_value.y < min_value.y {
                self.current_value.y = min_value.y;
            }
            if self.current_value.z < min_value.z {
                self.current_value.z = min_value.z;
            }
        }
    }

    fn value(&self) -> Point3<f64> {
        self.current_value
    }

    fn value_mut(&mut self) -> &mut Point3<f64> {
        &mut self.current_value
    }

    fn set_direction(&mut self, direction: Option<Vector3<f64>>) {
        if direction == self.direction {
            return;
        }

        if direction.is_none() {
            self.current_vel = Vector3::zero();
        }

        self.direction = direction.map(|d| {
            if d.magnitude() > 0.01 {
                d.normalize()
            } else {
                Vector3::zero()
            }
        });

        self.initial_step_required = true;
        self.delay_seconds_remaining = self.params.delay_seconds;
    }
}

pub fn restrict_visible_size(max_chunks: usize, mut visible_size: Vector3<i32>) -> Vector3<i32> {
    let chunk_size = visible_size_to_chunks(visible_size);
    let max_z_chunks = max_chunks as i32 / (chunk_size.x * chunk_size.y) - 1;
    let max_z_voxels = simgame_voxels::index_utils::chunk_size().z as i32 * max_z_chunks;

    if visible_size.z > max_z_voxels {
        visible_size.z = max_z_voxels;
    }

    visible_size
}

fn visible_size_to_chunks(visible_size: Vector3<i32>) -> Vector3<i32> {
    visible_size.div_up(convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}
