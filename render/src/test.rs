use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use cgmath::{InnerSpace, Point3, Vector3, Zero};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

pub use simgame_core::settings::RenderTestParams;
use simgame_core::world::{UpdatedWorldState, World};
use simgame_core::{convert_point, convert_vec};

use crate::world::visible_size_to_chunks;

use crate::world;
use crate::{RenderInit, RenderParams, RenderState, WorldRenderInit};

struct TimingParams {
    window_len: usize,
    tick_size: Duration,
    render_interval: Option<Duration>,
}

struct TimeTracker {
    params: TimingParams,
    samples: VecDeque<Duration>,
    start_time: Instant,
    prev_sample_time: Instant,
    total_ticks: i64,
    prev_render_time: Instant,
}

struct ControlState {
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

fn build_window(event_loop: &EventLoop<()>) -> Result<winit::window::Window> {
    let builder = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 1920.0,
            height: 1080.0,
        })
        .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

fn restrict_visible_size(max_chunks: usize, mut visible_size: Vector3<i32>) -> Vector3<i32> {
    let chunk_size = visible_size_to_chunks(visible_size);
    let max_z_chunks = max_chunks as i32 / (chunk_size.x * chunk_size.y) - 1;
    let max_z_blocks = simgame_core::block::index_utils::chunk_size().z as i32 * max_z_chunks;

    if visible_size.z > max_z_blocks {
        visible_size.z = max_z_blocks;
    }

    visible_size
}

pub async fn test_render<'a>(
    mut world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'a>,
    shaders: crate::WorldShaders<&'a [u32]>,
) -> Result<()> {
    let event_loop = EventLoop::new();

    let window = build_window(&event_loop)?;

    let physical_win_size = window.inner_size();
    let win_size: (u32, u32) = physical_win_size.into();

    let visible_size = restrict_visible_size(
        test_params.max_visible_chunks,
        test_params.initial_visible_size,
    );

    if visible_size != test_params.initial_visible_size {
        let new_visible_chunks = visible_size_to_chunks(visible_size);
        log::warn!(
            "Initial visible size of {:?} would exceed max_visible_chunks setting of {}. Decreasing to {:?}. This will use {} out of {} chunks.",
            test_params.initial_visible_size,
            test_params.max_visible_chunks,
            visible_size,
            new_visible_chunks.x * new_visible_chunks.y * new_visible_chunks.z,
            test_params.max_visible_chunks
        );
    }

    let mut view_state = world::ViewParams {
        camera_pos: test_params.initial_camera_pos,
        z_level: test_params.initial_z_level,
        visible_size,
        look_at_dir: test_params.look_at_dir,
    };

    let render_init = RenderInit {
        window: &window,
        physical_win_size: win_size,
        world: WorldRenderInit {
            shaders,
            aspect_ratio: (physical_win_size.width / physical_win_size.height) as f32,
            width: win_size.0,
            height: win_size.1,
            world: &world,
            view_params: view_state.clone(),
            max_visible_chunks: test_params.max_visible_chunks,
        },
    };

    let mut render_state = RenderState::new(render_params.clone(), render_init).await?;

    use event::{Event, VirtualKeyCode, WindowEvent};

    let mut control_state = ControlState::new(&test_params);

    let mut time_tracker = TimeTracker::new(
        TimingParams {
            window_len: 30,
            tick_size: Duration::from_millis(test_params.game_step_millis),
            render_interval: test_params
                .fixed_refresh_rate
                .map(|rate| Duration::from_millis(1000 / rate)),
        },
        Instant::now(),
    );

    let mut world_diff = UpdatedWorldState::empty();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    control_state.update_from_keyboard_event(input);
                    match input {
                        event::KeyboardInput {
                            virtual_keycode: Some(key),
                            state: event::ElementState::Pressed,
                            ..
                        } => match key {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Space => world.toggle_updates(),
                            VirtualKeyCode::E => world.modify_filled_blocks(1, &mut world_diff),
                            VirtualKeyCode::Q => world.modify_filled_blocks(-1, &mut world_diff),
                            _ => {}
                        },
                        _ => {}
                    }
                }
                WindowEvent::Focused(false) => control_state.clear_key_states(),
                _ => {}
            },
            Event::MainEventsCleared => {
                if time_tracker.check_render(Instant::now()) {
                    render_state.update(&world, &world_diff);
                    render_state.render_frame();
                    world_diff.clear();
                    time_tracker.sample(Instant::now());

                    log::debug!(
                        "Frame rate: {}/{}",
                        time_tracker.min_fps(),
                        time_tracker.mean_fps()
                    );
                }
            }
            _ => (),
        };

        for _tick_time in time_tracker.tick(Instant::now()) {
            world.tick(&mut world_diff);
            control_state.tick(
                test_params.game_step_millis as f64 / 1000.0,
                &mut view_state,
            );
        }

        render_state.set_world_view(view_state.clone());
    });
}

impl ControlState {
    fn new(params: &RenderTestParams) -> Self {
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
            look_at_dir: convert_vec!(params.look_at_dir, f64),
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

    fn update_from_keyboard_event(&mut self, event: event::KeyboardInput) {
        use event::VirtualKeyCode;
        match event {
            event::KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            } => {
                let mag = self.state_to_mag(state);
                let mag_z = if mag == 0 {
                    None
                } else {
                    Some(Vector3::new(0.0, 0.0, mag as f64))
                };
                let neg_mag_z = mag_z.map(|v| Vector3::zero() - v);

                match key {
                    VirtualKeyCode::D => self.camera_dir.x = mag,
                    VirtualKeyCode::A => self.camera_dir.x = -mag,
                    VirtualKeyCode::W => self.camera_dir.y = mag,
                    VirtualKeyCode::S => self.camera_dir.y = -mag,
                    VirtualKeyCode::K => self.camera_dir.z = mag,
                    VirtualKeyCode::J => self.camera_dir.z = -mag,
                    VirtualKeyCode::I => self.z_level.set_direction(mag_z),
                    VirtualKeyCode::U => self.z_level.set_direction(neg_mag_z),
                    VirtualKeyCode::M => self.visible_height.set_direction(mag_z),
                    VirtualKeyCode::N => self.visible_height.set_direction(neg_mag_z),
                    _ => {}
                }
            }
            _ => {}
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

    fn pan_direction(look_at_dir: Vector3<f64>, direction: Vector3<i32>) -> Vector3<f64> {
        let mut forward_step = look_at_dir;
        forward_step.z = 0.0;
        forward_step = forward_step.normalize();

        let right_step = Vector3::new(0., 0., -1.).cross(look_at_dir);
        forward_step.z = 0.0;
        forward_step = forward_step.normalize();

        let mut dir = Vector3::zero();
        dir += forward_step * direction.y as f64;
        dir += right_step * direction.x as f64;
        dir
    }

    fn clear_key_states(&mut self) {
        self.z_level.set_direction(None);
        self.visible_height.set_direction(None);
    }

    fn state_to_mag(&self, state: event::ElementState) -> i32 {
        match state {
            event::ElementState::Pressed => 1,
            event::ElementState::Released => 0,
        }
    }

    pub fn tick(&mut self, elapsed: f64, view_state: &mut world::ViewParams) {
        self.camera_pan.tick(elapsed);
        self.camera_height.tick(elapsed);
        self.z_level.tick(elapsed);
        self.visible_height.tick(elapsed);

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

impl TimeTracker {
    pub fn new(params: TimingParams, now: Instant) -> Self {
        Self {
            samples: VecDeque::with_capacity(params.window_len),
            start_time: now,
            prev_sample_time: now,
            total_ticks: 0,
            prev_render_time: now,
            params,
        }
    }

    /// Advance the TimeTracker forward to a new instant. Returns an iterator over each tick since
    /// the previous call to tick().
    pub fn tick(&mut self, now: Instant) -> impl Iterator<Item = Instant> {
        let old_total_ticks = self.total_ticks;
        self.total_ticks = (now.duration_since(self.start_time).as_secs_f64()
            / self.params.tick_size.as_secs_f64())
        .floor() as i64;

        let start_time = self.start_time;
        let tick_size = self.params.tick_size;
        (old_total_ticks..self.total_ticks)
            .map(move |tick_ix| start_time + tick_size * (tick_ix as u32))
    }

    pub fn sample(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.prev_sample_time);
        while self.samples.len() >= self.params.window_len {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed);
        self.prev_sample_time = now;
    }

    pub fn check_render(&mut self, now: Instant) -> bool {
        if let Some(render_interval) = self.params.render_interval {
            if now.duration_since(self.prev_render_time) > render_interval {
                self.prev_render_time = now;
                return true;
            }
        } else {
            self.prev_render_time = now;
            return true;
        }

        false
    }

    pub fn mean_fps(&self) -> f64 {
        let elapsed_total: f64 = self.samples.iter().map(|d| d.as_secs_f64()).sum();
        let elapsed_mean = elapsed_total / self.samples.len() as f64;
        1. / elapsed_mean
    }

    pub fn min_fps(&self) -> f64 {
        let elapsed_max: f64 = self.samples.iter().max().unwrap().as_secs_f64();
        1. / elapsed_max
    }
}
