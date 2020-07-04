use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use cgmath::{InnerSpace, Point3, Vector3, Zero};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

use simgame_core::world::{UpdatedWorldState, World};

use crate::world;
use crate::{RenderInit, RenderState, WorldRenderInit};

const GAME_STEP_MILLIS: u64 = 10;
const RENDER_INTERVAL_MILLIS: u64 = 1000 / 120;

const INITIAL_VISIBLE_SIZE: Vector3<i32> = Vector3::new(128, 128, 32);

pub fn build_window(
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window> {
    let builder = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 1920.0,
            height: 1080.0,
        })
        .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

#[derive(Debug, Clone)]
pub struct AccelControlParams {
    pub initial_step: f64,
    pub delay_ticks: u64,
    pub acceleration_per_tick: f64,
    pub max_speed: Option<f64>,
    pub max_value: Option<f64>,
    pub min_value: Option<f64>,
}

struct AccelControlState {
    // parameters
    params: AccelControlParams,

    // state
    direction: i32,
    initial_step_required: bool,
    delay_ticks_remaining: u64,
    current_speed: f64,
    current_value: f64,
}

impl AccelControlState {
    fn new(initial_value: f64, params: AccelControlParams) -> Self {
        Self {
            params,

            direction: 0,
            initial_step_required: false,
            delay_ticks_remaining: 0,
            current_speed: 0.0,
            current_value: initial_value,
        }
    }

    fn tick(&mut self) {
        let mut offset = 0.0;
        if self.initial_step_required {
            offset += self.params.initial_step;
            self.initial_step_required = false;
        }

        if self.delay_ticks_remaining > 0 {
            self.delay_ticks_remaining -= 1;
        } else {
            self.current_speed += self.params.acceleration_per_tick;
        }

        if let Some(max_speed) = self.params.max_speed {
            if self.current_speed > max_speed {
                self.current_speed = max_speed
            }
        }

        offset += self.current_speed;
        offset *= self.direction as f64;
        self.current_value += offset;

        if let Some(max_value) = self.params.max_value {
            if self.current_value > max_value {
                self.current_value = max_value;
            }
        }

        if let Some(min_value) = self.params.min_value {
            if self.current_value < min_value {
                self.current_value = min_value;
            }
        }
    }

    fn value(&self) -> f64 {
        self.current_value
    }

    fn set_direction(&mut self, direction: i32) {
        if direction == self.direction {
            return;
        }

        self.direction = direction;
        self.initial_step_required = true;
        self.delay_ticks_remaining = self.params.delay_ticks;
        self.current_speed = 0.0;
    }
}

struct ControlState {
    forward: i32,
    right: i32,
    up: i32,

    z_level_state: AccelControlState,
    visible_height_state: AccelControlState,
}

impl ControlState {
    fn new() -> Self {
        let base_accel_params = AccelControlParams {
            initial_step: 1.0,
            delay_ticks: 50,
            acceleration_per_tick: 0.01,
            max_speed: Some(2.0),
            max_value: None,
            min_value: None,
        };

        ControlState {
            forward: 0,
            right: 0,
            up: 0,
            z_level_state: AccelControlState::new(
                0.0,
                AccelControlParams {
                    min_value: Some(0.0),
                    ..base_accel_params
                },
            ),
            visible_height_state: AccelControlState::new(
                INITIAL_VISIBLE_SIZE.z as f64,
                AccelControlParams {
                    min_value: Some(1.0),
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
                match key {
                    VirtualKeyCode::W => self.forward = mag,
                    VirtualKeyCode::S => self.forward = -mag,
                    VirtualKeyCode::D => self.right = mag,
                    VirtualKeyCode::A => self.right = -mag,
                    VirtualKeyCode::K => self.up = mag,
                    VirtualKeyCode::J => self.up = -mag,
                    VirtualKeyCode::I => self.z_level_state.set_direction(mag),
                    VirtualKeyCode::U => self.z_level_state.set_direction(-mag),
                    VirtualKeyCode::M => self.visible_height_state.set_direction(mag),
                    VirtualKeyCode::N => self.visible_height_state.set_direction(-mag),
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn clear_key_states(&mut self) {
        self.z_level_state.set_direction(0);
        self.visible_height_state.set_direction(0);
    }

    fn state_to_mag(&self, state: event::ElementState) -> i32 {
        match state {
            event::ElementState::Pressed => 1,
            event::ElementState::Released => 0,
        }
    }

    fn camera_delta(&self) -> Vector3<f32> {
        let speed = 0.4;
        let forward_step = Vector3::new(1., 1., 0.).normalize();
        let right_step = Vector3::new(1., -1., 0.).normalize();
        let up_step = Vector3::new(0., 0., 1.).normalize();

        let mut dir = Vector3::zero();

        dir += forward_step * self.forward as f32;
        dir += right_step * self.right as f32;
        dir += up_step * self.up as f32;

        if dir.magnitude() > 0.01 {
            dir.normalize() * speed
        } else {
            Vector3::zero()
        }
    }

    pub fn tick(&mut self, view_state: &mut world::ViewParams) {
        self.z_level_state.tick();
        self.visible_height_state.tick();

        view_state.camera_pos += self.camera_delta();
        view_state.z_level = self.z_level_state.value().round() as i32;
        view_state.visible_size.z = self.visible_height_state.value().round() as i32;
    }
}

pub async fn test_render(mut world: World, shaders: crate::WorldShaders<&[u8]>) -> Result<()> {
    let event_loop = EventLoop::new();

    let window = build_window(&event_loop)?;

    // let physical_win_size = window.inner_size().to_physical(window.hidpi_factor());
    let physical_win_size = window.inner_size();
    let win_size: (u32, u32) = physical_win_size.into();

    let render_init = RenderInit {
        window: &window,
        physical_win_size: win_size,
        world: WorldRenderInit {
            shaders: shaders.map(|bytes| std::io::Cursor::new(bytes)),
            aspect_ratio: (physical_win_size.width / physical_win_size.height) as f32,
            width: win_size.0,
            height: win_size.1,
        },
    };

    let mut render_state = RenderState::new(render_init).await?;
    let mut view_state = world::ViewParams {
        camera_pos: Point3::new(0f32, 0f32, 20f32),
        z_level: 1,
        visible_size: INITIAL_VISIBLE_SIZE,
    };
    render_state.set_world_view(view_state.clone());
    render_state.init(&world);

    use event::{Event, VirtualKeyCode, WindowEvent};

    let mut control_state = ControlState::new();

    let mut time_tracker = TimeTracker::new(
        TimingParams {
            window_len: 30,
            tick_size: Duration::from_millis(GAME_STEP_MILLIS),
            render_interval: Duration::from_millis(RENDER_INTERVAL_MILLIS),
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
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Space),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => world.toggle_updates(),
                WindowEvent::KeyboardInput { input, .. } => {
                    control_state.update_from_keyboard_event(input)
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

                    log::info!(
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
            control_state.tick(&mut view_state);
        }

        render_state.set_world_view(view_state.clone());
    });
}

struct TimingParams {
    window_len: usize,
    tick_size: Duration,
    render_interval: Duration,
}

struct TimeTracker {
    params: TimingParams,
    samples: VecDeque<Duration>,
    start_time: Instant,
    prev_sample_time: Instant,
    total_ticks: i64,
    prev_render_time: Instant,
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

    fn sample(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.prev_sample_time);
        while self.samples.len() >= self.params.window_len {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed);
        self.prev_sample_time = now;
    }

    pub fn check_render(&mut self, now: Instant) -> bool {
        if now.duration_since(self.prev_render_time) > self.params.render_interval {
            self.prev_render_time = now;
            return true;
        }

        return false;
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
