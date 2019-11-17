use crate::{RenderInit, RenderState, WorldRenderInit};
use anyhow::Result;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};
use simgame_core::world::{World, UpdatedWorldState};
use cgmath::{Vector3, InnerSpace};

pub fn build_window(
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window> {
    // let monitor = event_loop.primary_monitor();
    let builder = winit::window::WindowBuilder::new();
    // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
    // .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

pub fn test_render(world: World, vert_shader: &[u8], frag_shader: &[u8]) -> Result<()> {
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

    let forward_step = Vector3::new(1., 1., 0.).normalize();
    let right_step = Vector3::new(1., -1., 0.).normalize();
    let up_step = Vector3::new(0., 0., 1.).normalize();

    let mut render_state = RenderState::new(render_init)?;
    render_state.init(&world);

    use event::{Event, WindowEvent, VirtualKeyCode};

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
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => match key {
                    VirtualKeyCode::W => render_state.update_camera_pos(forward_step),
                    VirtualKeyCode::S => render_state.update_camera_pos(-forward_step),
                    VirtualKeyCode::D => render_state.update_camera_pos(right_step),
                    VirtualKeyCode::A => render_state.update_camera_pos(-right_step),
                    VirtualKeyCode::J => render_state.update_camera_pos(-up_step),
                    VirtualKeyCode::K => render_state.update_camera_pos(up_step),
                    _ => {}
                }
                _ => {}
            },
            Event::EventsCleared => {
                render_state.update(&world, &UpdatedWorldState::empty());
                render_state.render_frame();
            },
            _ => (),
        }
    });
}
