use crate::{RenderInit, RenderState, WorldRenderInit};
use anyhow::Result;
use winit::{
    event,
    event_loop::{ControlFlow, EventLoop},
};

pub fn build_window(
    event_loop: &winit::event_loop::EventLoop<()>,
) -> Result<winit::window::Window> {
    // let monitor = event_loop.primary_monitor();
    let builder = winit::window::WindowBuilder::new();
    // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
    // .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

pub fn test_render(vert_shader: &[u8], frag_shader: &[u8]) -> Result<()> {
    let event_loop = EventLoop::new();

    let window = build_window(&event_loop)?;

    let physical_win_size = window.inner_size().to_physical(window.hidpi_factor());

    let render_init = RenderInit {
        window: &window,
        physical_win_size: physical_win_size.into(),
        world: WorldRenderInit {
            vert_shader_spirv_bytes: std::io::Cursor::new(vert_shader),
            frag_shader_spirv_bytes: std::io::Cursor::new(frag_shader),
        },
    };

    let mut render_state = RenderState::new(render_init)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::WindowEvent { event, .. } => match event {
                event::WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            event::Event::EventsCleared => render_state.render_frame(),
            _ => (),
        }
    });
}
