use anyhow::Result;
use cgmath::{InnerSpace, Point3, Vector3, Zero};
use rand::{self, Rng};
use simgame_core::util::Bounds;
use simgame_core::block::Block;
use simgame_core::world::{UpdatedWorldState, World};
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

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
        let bounds = Bounds::new(
            Point3::new(32, 32, 0),
            Vector3::new(16, 16, 128));

        move |world: &mut World| {
            for _ in 0..32 {
                let point = bounds.origin() + Vector3 {
                    x: rng.gen::<usize>() % bounds.size().x,
                    y: rng.gen::<usize>() % bounds.size().y,
                    z: rng.gen::<usize>() % bounds.size().z,
                };
                world.blocks.set_block(point, Block::from_u16(1));
            }
        }
    };

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
                } => match key {
                    VirtualKeyCode::W => control_state.forward = state_to_mag(state),
                    VirtualKeyCode::S => control_state.forward = -state_to_mag(state),
                    VirtualKeyCode::D => control_state.right = state_to_mag(state),
                    VirtualKeyCode::A => control_state.right = -state_to_mag(state),
                    VirtualKeyCode::K => control_state.up = state_to_mag(state),
                    VirtualKeyCode::J => control_state.up = -state_to_mag(state),
                    _ => {}
                },
                _ => {}
            },
            Event::EventsCleared => {
                update_world(&mut world);
                render_state.update(&world, &UpdatedWorldState::empty());
                render_state.render_frame();
            }
            _ => (),
        };

        render_state.update_camera_pos(control_state.camera_delta());
    });
}
