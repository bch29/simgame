use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use cgmath::{InnerSpace, Point3, Vector3, Zero};
use log::info;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

use simgame_core::world::{UpdatedWorldState, World};

use crate::world;
use crate::{RenderInit, RenderState, WorldRenderInit};

const GAME_STEP_MILLIS: u64 = 10;

pub fn build_window(
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window> {
    let builder = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize { width: 1920.0, height: 1080.0 })
        .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

struct ControlState {
    forward: i32,
    right: i32,
    up: i32,

    z_level_delta: i32,
    visible_height_delta: i32,
}

impl ControlState {
    fn new() -> Self {
        ControlState {
            forward: 0,
            right: 0,
            up: 0,
            z_level_delta: 0,
            visible_height_delta: 0,
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

    fn update_from_keyboard_event(&mut self, event: event::KeyboardInput) {
        use event::VirtualKeyCode;
        match event {
            event::KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            } => match key {
                VirtualKeyCode::W => self.forward = self.state_to_mag(state),
                VirtualKeyCode::S => self.forward = -self.state_to_mag(state),
                VirtualKeyCode::D => self.right = self.state_to_mag(state),
                VirtualKeyCode::A => self.right = -self.state_to_mag(state),
                VirtualKeyCode::K => self.up = self.state_to_mag(state),
                VirtualKeyCode::J => self.up = -self.state_to_mag(state),
                VirtualKeyCode::I => self.z_level_delta = self.state_to_mag(state),
                VirtualKeyCode::U => self.z_level_delta = -self.state_to_mag(state),
                VirtualKeyCode::M => self.visible_height_delta = self.state_to_mag(state),
                VirtualKeyCode::N => self.visible_height_delta = -self.state_to_mag(state),
                _ => {}
            },
            _ => {}
        }
    }

    fn state_to_mag(&self, state: event::ElementState) -> i32 {
        match state {
            event::ElementState::Pressed => 1,
            event::ElementState::Released => 0,
        }
    }

    fn update_view(&mut self, view_state: &mut world::ViewParams) {
        view_state.camera_pos += self.camera_delta();
        view_state.z_level += self.z_level_delta;
        self.z_level_delta = 0;
        view_state.visible_size.z += self.visible_height_delta;
        self.visible_height_delta = 0;

        if view_state.z_level < 0 {
            view_state.z_level = 0;
        }
    }
}

pub fn test_render(mut world: World, vert_shader: &[u8], frag_shader: &[u8]) -> Result<()> {
    let event_loop = EventLoop::new();

    let window = build_window(&event_loop)?;

    let physical_win_size = window.inner_size().to_physical(window.hidpi_factor());
    let win_size: (u32, u32) = physical_win_size.into();

    let render_init = RenderInit {
        window: &window,
        physical_win_size: win_size,
        world: WorldRenderInit {
            vert_shader_spirv_bytes: std::io::Cursor::new(vert_shader),
            frag_shader_spirv_bytes: std::io::Cursor::new(frag_shader),
            aspect_ratio: (physical_win_size.width / physical_win_size.height) as f32,
            width: win_size.0,
            height: win_size.1,
        },
    };

    let mut render_state = RenderState::new(render_init)?;
    let mut view_state = world::ViewParams {
        camera_pos: Point3::new(0f32, 0f32, 20f32),
        z_level: 1,
        visible_size: Vector3::new(128, 128, 32),
    };
    render_state.set_world_view(view_state.clone());
    render_state.init(&world);

    use event::{Event, VirtualKeyCode, WindowEvent};

    let mut control_state = ControlState::new();

    let mut time_tracker =
        TimeTracker::new(60, Duration::from_millis(GAME_STEP_MILLIS), Instant::now());

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
                WindowEvent::KeyboardInput { input, .. } => {
                    control_state.update_from_keyboard_event(input)
                }
                _ => {}
            },
            Event::EventsCleared => {
                render_state.update(&world, &world_diff);
                render_state.render_frame();
                world_diff.clear();
            }
            _ => (),
        };

        for _tick_time in time_tracker.tick(Instant::now()) {
            world.tick(&mut world_diff);
            control_state.update_view(&mut view_state);
        }

        render_state.set_world_view(view_state.clone());

        info!(
            "Frame rate: {}/{}",
            time_tracker.min_fps(),
            time_tracker.mean_fps()
        );
    });
}

struct TimeTracker {
    window_len: usize,
    samples: VecDeque<Duration>,
    start_time: Instant,
    prev_sample_time: Instant,
    tick_size: Duration,
    total_ticks: i64,
}

impl TimeTracker {
    pub fn new(window_len: usize, tick_size: Duration, now: Instant) -> Self {
        Self {
            window_len,
            samples: VecDeque::with_capacity(window_len),
            start_time: now,
            prev_sample_time: now,
            tick_size,
            total_ticks: 0,
        }
    }

    /// Advance the TimeTracker forward to a new instant. Returns an iterator over each tick since
    /// the previous call to tick().
    pub fn tick(&mut self, now: Instant) -> impl Iterator<Item = Instant> {
        let elapsed = now.duration_since(self.prev_sample_time);
        while self.samples.len() >= self.window_len {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed);
        self.prev_sample_time = now;

        let old_total_ticks = self.total_ticks;
        self.total_ticks = (now.duration_since(self.start_time).as_secs_f64()
            / self.tick_size.as_secs_f64())
        .floor() as i64;

        let start_time = self.start_time;
        let tick_size = self.tick_size;
        (old_total_ticks..self.total_ticks)
            .map(move |tick_ix| start_time + tick_size * (tick_ix as u32))
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
