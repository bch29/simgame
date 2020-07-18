use std::collections::VecDeque;
use std::sync::{Mutex, Arc};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Point2, Vector2};
use winit::{
    event::{self, Event},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::{Window, WindowBuilder},
};

use simgame_core::block::BlockConfigHelper;
use simgame_core::world::World;
use simgame_core::{convert_point, convert_vec};

use simgame_render::resource::ResourceLoader;
pub use simgame_render::RenderParams;
use simgame_render::{visible_size_to_chunks, RenderState, RenderStateBuilder, ViewParams};

mod controls;
pub mod files;
pub mod settings;
mod world_state;

use settings::RenderTestParams;
use world_state::WorldState;

pub async fn test_render<'a>(
    world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'a>,
    resource_loader: ResourceLoader,
    block_helper: BlockConfigHelper,
) -> Result<()> {
    let event_loop = EventLoop::new();

    let game = TestRenderBuilder {
        event_loop: &event_loop,
        world,
        test_params,
        render_params,
        resource_loader,
        block_helper,
    }
    .build()
    .await?;
    Ok(game.run(event_loop))
}

pub struct TestRenderBuilder<'a> {
    event_loop: &'a EventLoop<()>,
    world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'a>,
    resource_loader: ResourceLoader,
    block_helper: BlockConfigHelper,
}

pub struct TestRender {
    world: Arc<Mutex<World>>,
    world_state: WorldState,
    win_dimensions: Vector2<f64>,
    window: Window,
    control_state: controls::ControlState,
    cursor_reset_position: Point2<f64>,
    time_tracker: TimeTracker,
    render_state: RenderState,
    view_state: ViewParams,
    has_cursor_control: bool,
}

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

fn build_window(event_loop: &EventLoop<()>, settings: &settings::VideoSettings) -> Result<Window> {
    let builder = WindowBuilder::new();

    let builder = match settings.video_mode {
        settings::VideoMode::Windowed => builder
            .with_inner_size(winit::dpi::LogicalSize {
                width: settings.win_dimensions.x,
                height: settings.win_dimensions.y,
            })
            .with_fullscreen(None)
            .with_decorations(true),
        settings::VideoMode::Borderless => {
            let monitor = event_loop.primary_monitor();
            builder
                .with_inner_size(winit::dpi::LogicalSize {
                    width: settings.win_dimensions.x,
                    height: settings.win_dimensions.y,
                })
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(monitor)))
                .with_decorations(false)
        }
        settings::VideoMode::Fullscreen => {
            let monitor = event_loop.primary_monitor();
            let mut video_modes = monitor
                .video_modes()
                .filter(|video_mode| {
                    let mode_dimensions =
                        video_mode.size().to_logical::<f64>(monitor.scale_factor());
                    mode_dimensions.width == settings.win_dimensions.x
                        && mode_dimensions.height == settings.win_dimensions.y
                })
                .collect::<Vec<_>>();
            video_modes.sort_by_key(|mode| mode.refresh_rate());

            let video_mode = video_modes.pop().ok_or(anyhow!(
                "No fullscreen video mode matches requested dimensions"
            ))?;

            builder
                .with_inner_size(winit::dpi::LogicalSize {
                    width: settings.win_dimensions.x,
                    height: settings.win_dimensions.y,
                })
                .with_fullscreen(Some(winit::window::Fullscreen::Exclusive(video_mode)))
        }
    };

    Ok(builder.build(event_loop)?)
}

impl<'a> TestRenderBuilder<'a> {
    pub async fn build(
        self,
    ) -> Result<TestRender> {
        let window = build_window(self.event_loop, &self.test_params.video_settings)?;

        let physical_win_size = window.inner_size();
        let physical_win_size: (u32, u32) = physical_win_size.into();
        let physical_win_size = physical_win_size.into();

        let logical_win_size = window.inner_size().to_logical::<f64>(window.scale_factor());
        let win_dimensions = Vector2::new(logical_win_size.width, logical_win_size.height);

        let visible_size = controls::restrict_visible_size(
            self.test_params.max_visible_chunks,
            self.test_params.initial_visible_size,
        );

        if visible_size != self.test_params.initial_visible_size {
            let new_visible_chunks = visible_size_to_chunks(visible_size);
            log::warn!(
                "Initial visible size of {:?} would exceed max_visible_chunks setting of {}. Decreasing to {:?}. This will use {} out of {} chunks.",
                self.test_params.initial_visible_size,
                self.test_params.max_visible_chunks,
                visible_size,
                new_visible_chunks.x * new_visible_chunks.y * new_visible_chunks.z,
                self.test_params.max_visible_chunks
                );
        }

        let view_state = ViewParams {
            camera_pos: self.test_params.initial_camera_pos,
            z_level: self.test_params.initial_z_level,
            visible_size,
            look_at_dir: self.test_params.look_at_dir,
        };

        let render_state = RenderStateBuilder {
            window: &window,
            physical_win_size,
            resource_loader: self.resource_loader,
            display_size: physical_win_size,
            world: &self.world,
            block_helper: &self.block_helper,
            view_params: view_state.clone(),
            max_visible_chunks: self.test_params.max_visible_chunks,
        }
        .build(self.render_params.clone())
        .await?;

        let control_state = controls::ControlState::new(&self.test_params);

        let time_tracker = TimeTracker::new(
            TimingParams {
                window_len: 30,
                tick_size: Duration::from_millis(self.test_params.game_step_millis),
                render_interval: self
                    .test_params
                    .fixed_refresh_rate
                    .map(|rate| Duration::from_millis(1000 / rate)),
            },
            Instant::now(),
        );

        let cursor_reset_position = Point2::origin() + win_dimensions / 2.;

        let world = Arc::new(Mutex::new(self.world));

        let world_state = world_state::WorldStateBuilder {
            world: world.clone(),
            block_helper: self.block_helper.clone(),
            tree_config: self.test_params.tree.as_ref(),
        }
        .build()?;

        Ok(TestRender {
            world,
            world_state,
            win_dimensions,
            window,
            control_state,
            cursor_reset_position,
            time_tracker,
            render_state,
            view_state,
            has_cursor_control: false,
        })
    }
}

impl TestRender {
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
        use event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};

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
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state: ElementState::Pressed,
                            ..
                        } => match key {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Space => self.world_state.toggle_updates()?,
                            VirtualKeyCode::E => {
                                self.world_state.modify_filled_blocks(1)?
                            }
                            VirtualKeyCode::Q => {
                                self.world_state.modify_filled_blocks(-1)?
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    self.world_state.on_click(
                        &self.world,
                        convert_point!(self.view_state.effective_camera_pos(), f64),
                        convert_vec!(self.view_state.look_at_dir, f64),
                    )?;
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
                        self.window.set_cursor_visible(true);
                    }
                }
                WindowEvent::Resized(event_size) => {
                    let new_size = self.window.inner_size();
                    log::info!(
                        "Window resized to {:?} (event reported {:?})",
                        new_size,
                        event_size
                    );

                    self.render_state
                        .update_win_size(Vector2::new(new_size.width, new_size.height))?;
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
            self.window.set_cursor_visible(false);
        }

        Ok(())
    }

    fn redraw(&mut self) -> Result<()> {
        self.render_state.set_world_view(self.view_state.clone());

        let world_diff = self.world_state.world_diff()?;

        {
            let world = self.world.lock().unwrap();
            self.render_state
                .update(&*world, world_diff);
        }

        self.render_state.render_frame()?;
        world_diff.clear();
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
        let offset = (pos - self.cursor_reset_position) / (self.win_dimensions.x / 2.);
        self.control_state.move_cursor(offset);
        reset_cursor(&self.window, self.cursor_reset_position)?;
        Ok(())
    }

    fn tick(&mut self, _now: Instant, elapsed: Duration) -> Result<()> {
        let elapsed = elapsed.as_secs_f64();

        self.world_state.tick(elapsed)?;
        self.control_state.tick(elapsed, &mut self.view_state);
        Ok(())
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
