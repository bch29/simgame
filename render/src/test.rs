use std::time::{Duration, Instant};
use std::collections::VecDeque;

use anyhow::Result;
use cgmath::{InnerSpace, Point3, Vector3, Zero};
use log::info;
use rand::{self, Rng};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

use simgame_core::block::Block;
use simgame_core::util::Bounds;
use simgame_core::world::{UpdatedWorldState, World};

use crate::world;
use crate::{RenderInit, RenderState, WorldRenderInit};

pub fn build_window(
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window> {
    // let monitor = event_loop.primary_monitor();
    let builder = winit::window::WindowBuilder::new();
    // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
    // .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

struct ControlState {
    forward: i32,
    right: i32,
    up: i32,
}

impl ControlState {
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
        camera_pos: Point3::new(-20f32, -20f32, 20f32),
        z_level: 1,
    };
    render_state.set_world_view(&view_state);
    render_state.init(&world);

    use event::{Event, VirtualKeyCode, WindowEvent};

    let mut control_state = ControlState {
        forward: 0,
        right: 0,
        up: 0,
    };

    fn state_to_mag(state: event::ElementState) -> i32 {
        match state {
            event::ElementState::Pressed => 1,
            event::ElementState::Released => 0,
        }
    }

    let mut update_world = {
        let mut rng = rand::thread_rng();
        let bounds = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        move |world: &mut World| {
            let mut diff = UpdatedWorldState::empty();
            for _ in 0..1024 {
                let point = bounds.origin()
                    + Vector3 {
                        x: rng.gen::<usize>() % bounds.size().x,
                        y: rng.gen::<usize>() % bounds.size().y,
                        z: rng.gen::<usize>() % bounds.size().z,
                    };
                world.blocks.set_block(point, Block::from_u16(1));
                let (chunk_pos, _) = simgame_core::block::index_utils::to_chunk_pos(point);
                diff.modified_chunks.insert(chunk_pos);
            }
            diff
        }
    };

    let mut fps_counter = FpsCounter::new(60, Instant::now());

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
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                } => match (state, key) {
                    (_, VirtualKeyCode::W) => control_state.forward = state_to_mag(state),
                    (_, VirtualKeyCode::S) => control_state.forward = -state_to_mag(state),
                    (_, VirtualKeyCode::D) => control_state.right = state_to_mag(state),
                    (_, VirtualKeyCode::A) => control_state.right = -state_to_mag(state),
                    (_, VirtualKeyCode::K) => control_state.up = state_to_mag(state),
                    (_, VirtualKeyCode::J) => control_state.up = -state_to_mag(state),
                    (event::ElementState::Pressed, VirtualKeyCode::I) => view_state.z_level += 1,
                    (event::ElementState::Pressed, VirtualKeyCode::U) => view_state.z_level -= 1,
                    _ => {}
                },
                _ => {}
            },
            Event::EventsCleared => {
                let diff = update_world(&mut world);
                render_state.update(&world, &diff);
                render_state.render_frame();
            }
            _ => (),
        };

        if view_state.z_level < 0 {
            view_state.z_level = 0;
        }

        view_state.camera_pos += control_state.camera_delta();
        render_state.set_world_view(&view_state);

        fps_counter.sample(Instant::now());
        info!("Frame rate: {}/{}", fps_counter.min(), fps_counter.mean());
    });
}

struct FpsCounter {
    window_len: usize,
    samples: VecDeque<Duration>,
    prev_sample_time: Instant
}

impl FpsCounter {
    pub fn new(window_len: usize, now: Instant) -> Self {
        Self {
            window_len,
            samples: VecDeque::with_capacity(window_len),
            prev_sample_time: now
        }
    }

    pub fn sample(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.prev_sample_time);
        while self.samples.len() >= self.window_len {
            self.samples.pop_front();
        }
        self.samples.push_back(elapsed);
        self.prev_sample_time = now;
    }

    pub fn mean(&self) -> f64 {
        let elapsed_total: f64 = self.samples.iter().map(|d| d.as_secs_f64()).sum();
        let elapsed_mean = elapsed_total / self.samples.len() as f64;
        1. / elapsed_mean
    }

    pub fn min(&self) -> f64 {
        let elapsed_max: f64 = self.samples.iter().max().unwrap().as_secs_f64();
        1. / elapsed_max
    }
}
