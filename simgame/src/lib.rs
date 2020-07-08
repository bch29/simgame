use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use cgmath::{EuclideanSpace, Point2, Vector2};
use winit::{
    event::{self, Event},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::{Window, WindowBuilder},
};

use simgame_core::settings::RenderTestParams;
use simgame_core::world::{UpdatedWorldState, World};

use simgame_render::world;
use simgame_render::{RenderInit, RenderState, WorldRenderInit};
pub use simgame_render::{RenderParams, WorldShaderData, WorldShaders};

mod controls;

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

fn build_window(event_loop: &EventLoop<()>, dimensions: Vector2<f64>) -> Result<Window> {
    let builder = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: dimensions.x,
            height: dimensions.y,
        })
        .with_decorations(true);
    Ok(builder.build(event_loop)?)
}

pub struct TestRender {
    world: World,
    win_dimensions: Vector2<f64>,
    window: Window,
    control_state: controls::ControlState,
    cursor_reset_position: Point2<f64>,
    time_tracker: TimeTracker,
    render_state: RenderState,
    view_state: world::ViewParams,
    world_diff: UpdatedWorldState,
    has_cursor_control: bool,
}

impl TestRender {
    pub async fn new<'a>(
        event_loop: &EventLoop<()>,
        world: World,
        test_params: RenderTestParams,
        render_params: RenderParams<'a>,
        shaders: WorldShaders<&'a [u32]>,
    ) -> Result<Self> {
        let win_dimensions = Vector2::new(1920., 1080.);

        let window = build_window(&event_loop, win_dimensions)?;

        let physical_win_size = window.inner_size();
        let win_size: (u32, u32) = physical_win_size.into();

        let visible_size = controls::restrict_visible_size(
            test_params.max_visible_chunks,
            test_params.initial_visible_size,
        );

        if visible_size != test_params.initial_visible_size {
            let new_visible_chunks = world::visible_size_to_chunks(visible_size);
            log::warn!(
                "Initial visible size of {:?} would exceed max_visible_chunks setting of {}. Decreasing to {:?}. This will use {} out of {} chunks.",
                test_params.initial_visible_size,
                test_params.max_visible_chunks,
                visible_size,
                new_visible_chunks.x * new_visible_chunks.y * new_visible_chunks.z,
                test_params.max_visible_chunks
                );
        }

        let view_state = world::ViewParams {
            camera_pos: test_params.initial_camera_pos,
            z_level: test_params.initial_z_level,
            visible_size,
            look_at_dir: test_params.look_at_dir,
        };

        let render_state = RenderState::new(
            render_params.clone(),
            RenderInit {
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
            },
        )
        .await?;

        let control_state = controls::ControlState::new(&test_params);

        let time_tracker = TimeTracker::new(
            TimingParams {
                window_len: 30,
                tick_size: Duration::from_millis(test_params.game_step_millis),
                render_interval: test_params
                    .fixed_refresh_rate
                    .map(|rate| Duration::from_millis(1000 / rate)),
            },
            Instant::now(),
        );

        let cursor_reset_position = Point2::origin() + win_dimensions / 2.;

        let world_diff = UpdatedWorldState::empty();

        Ok(TestRender {
            world,
            win_dimensions,
            window,
            control_state,
            cursor_reset_position,
            time_tracker,
            render_state,
            view_state,
            world_diff,
            has_cursor_control: false,
        })
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, window_target, control_flow| {
            match self.handle_event(event, window_target, control_flow) {
                Ok(()) => {}
                Err(end_error) => {
                    for error in end_error.chain() {
                        log::error!("{}", error);
                        log::error!("========");
                    }
                    *control_flow = ControlFlow::Exit;
                }
            }
        });
    }

    fn handle_event(
        &mut self,
        event: Event<()>,
        _window_target: &EventLoopWindowTarget<()>,
        control_flow: &mut ControlFlow,
    ) -> Result<()> {
        use event::{VirtualKeyCode, WindowEvent};

        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => {
                    self.control_state.handle_keyboad_input(input);
                    match input {
                        event::KeyboardInput {
                            virtual_keycode: Some(key),
                            state: event::ElementState::Pressed,
                            ..
                        } => match key {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Space => self.world.toggle_updates(),
                            VirtualKeyCode::E => {
                                self.world.modify_filled_blocks(1, &mut self.world_diff)
                            }
                            VirtualKeyCode::Q => {
                                self.world.modify_filled_blocks(-1, &mut self.world_diff)
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.has_cursor_control {
                        self.handle_cursor_move(position)?;
                    } else {
                        self.try_grab_cursor()?;
                    }
                }
                WindowEvent::Focused(focused) => {
                    if focused {
                        self.try_grab_cursor()?;
                    } else {
                        let _ = self.window.set_cursor_grab(false);
                        self.control_state.clear_key_states();
                        self.has_cursor_control = false;
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                if self.time_tracker.check_render(Instant::now()) {
                    self.redraw()?;
                }
            }
            _ => (),
        };

        for (now, elapsed) in self.time_tracker.tick(Instant::now()) {
            self.tick(now, elapsed)?;
        }

        Ok(())
    }

    fn try_grab_cursor(&mut self) -> Result<()> {
        if self.window.set_cursor_grab(true).is_ok() {
            self.has_cursor_control = true;
            reset_cursor(&self.window, self.cursor_reset_position)?;
        }

        Ok(())
    }

    fn redraw(&mut self) -> Result<()> {
        self.render_state.set_world_view(self.view_state.clone());
        self.render_state.update(&self.world, &self.world_diff);
        self.render_state.render_frame();
        self.world_diff.clear();
        self.time_tracker.sample(Instant::now());

        log::debug!(
            "Frame rate: {}/{}",
            self.time_tracker.min_fps(),
            self.time_tracker.mean_fps()
        );

        Ok(())
    }

    fn handle_cursor_move(&mut self, position: winit::dpi::PhysicalPosition<f64>) -> Result<()> {
        let pos_logical = position.to_logical(self.window.scale_factor());
        let pos: Point2<f64> = Point2::new(pos_logical.x, pos_logical.y);
        let offset = (pos - self.cursor_reset_position) / self.win_dimensions.x;
        self.control_state.move_cursor(offset);
        reset_cursor(&self.window, self.cursor_reset_position)?;
        Ok(())
    }

    fn tick(&mut self, _now: Instant, elapsed: Duration) -> Result<()> {
        let elapsed = elapsed.as_secs_f64();

        self.world.tick(elapsed, &mut self.world_diff);
        self.control_state.tick(elapsed, &mut self.view_state);
        Ok(())
    }
}

pub async fn test_render<'a>(
    world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'a>,
    shaders: WorldShaders<&'a [u32]>,
) -> Result<()> {
    let event_loop = EventLoop::new();
    let game = TestRender::new(&event_loop, world, test_params, render_params, shaders).await?;
    Ok(game.run(event_loop))
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
    pub fn tick(&mut self, now: Instant) -> impl Iterator<Item = (Instant, Duration)> {
        let old_total_ticks = self.total_ticks;
        self.total_ticks = (now.duration_since(self.start_time).as_secs_f64()
            / self.params.tick_size.as_secs_f64())
        .floor() as i64;

        let start_time = self.start_time;
        let tick_size = self.params.tick_size;
        (old_total_ticks..self.total_ticks).map(move |tick_ix| {
            let tick_time = start_time + tick_size * (tick_ix as u32);
            let elapsed = tick_size;
            (tick_time, elapsed)
        })
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

fn reset_cursor(window: &Window, pos: Point2<f64>) -> Result<(), winit::error::ExternalError> {
    window.set_cursor_position(winit::dpi::Position::Logical(winit::dpi::LogicalPosition {
        x: pos.x,
        y: pos.y,
    }))
}
