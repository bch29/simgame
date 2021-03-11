mod controls;
pub mod files;
pub mod settings;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Point2, Vector2};
use winit::{
    event::{self, Event},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::{Window, WindowBuilder},
};

pub use simgame_render::RenderParams;
use simgame_render::{
    resource::{ResourceLoader, TextureLoader},
    visible_size_to_chunks, RenderState, RenderStateInputs, Renderer, RendererBuilder, ViewParams,
};
use simgame_types::{Directory, Entity};
use simgame_util::{convert_point, convert_vec};
use simgame_world::World;

use settings::RenderTestParams;
use simgame_world::{WorldState, WorldStateBuilder};

pub async fn test_render(
    world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'_>,
    directory: Directory,
    resource_loader: ResourceLoader,
    texture_loader: TextureLoader,
    metrics_controller: metrics_runtime::Controller,
) -> Result<()> {
    let event_loop = EventLoop::new();

    let game = TestRenderBuilder {
        event_loop: &event_loop,
        world,
        test_params,
        render_params,
        directory,
        resource_loader,
        texture_loader,
        metrics_controller,
    }
    .build()
    .await?;
    game.run(event_loop);
    Ok(())
}

pub struct TestRenderBuilder<'a> {
    event_loop: &'a EventLoop<()>,
    world: World,
    test_params: RenderTestParams,
    render_params: RenderParams<'a>,
    resource_loader: ResourceLoader,
    texture_loader: TextureLoader,
    directory: Directory,
    metrics_controller: metrics_runtime::Controller,
}

pub struct TestRender {
    directory: Arc<Directory>,
    world: Arc<Mutex<World>>,
    world_state: WorldState,
    win_dimensions: Vector2<f64>,
    window: Window,
    control_state: controls::ControlState,
    cursor_reset_position: Point2<f64>,
    time_tracker: TimeTracker,
    renderer: Renderer,
    render_state: RenderState,
    view_params: ViewParams,
    has_cursor_control: bool,
    focused: bool,
}

struct TimingParams {
    log_interval: Option<Duration>,
    tick_size: Duration,
    render_interval: Option<Duration>,
}

type MetricsObserver = <metrics_runtime::observers::YamlBuilder as metrics_core::Builder>::Output;

struct EventTracker {
    metrics_key: Option<metrics::Key>,
    prev_event: Option<Instant>,
}

struct TimeTracker {
    params: TimingParams,

    render_event: EventTracker,
    log_event: EventTracker,

    start_time: Instant,
    total_ticks: i64,
    metrics_controller: metrics_runtime::Controller,
    metrics_observer: MetricsObserver,
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
                    let matches_width =
                        (mode_dimensions.width - settings.win_dimensions.x).abs() < f64::EPSILON;
                    let matches_height =
                        (mode_dimensions.height - settings.win_dimensions.y).abs() < f64::EPSILON;

                    matches_width && matches_height
                })
                .collect::<Vec<_>>();
            video_modes.sort_by_key(|mode| mode.refresh_rate());

            let video_mode = video_modes
                .pop()
                .ok_or_else(|| anyhow!("No fullscreen video mode matches requested dimensions"))?;

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
    pub async fn build(self) -> Result<TestRender> {
        let renderer = RendererBuilder {
            resource_loader: self.resource_loader,
            texture_loader: self.texture_loader,
            directory: &self.directory,
        }
        .build(self.render_params.clone())
        .await?;

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

        let view_params = ViewParams {
            camera_pos: self.test_params.initial_camera_pos,
            z_level: self.test_params.initial_z_level,
            visible_size,
            look_at_dir: self.test_params.look_at_dir,
        };

        let render_state = renderer.create_state(RenderStateInputs {
            window: &window,
            physical_win_size,
            world: &self.world,
            view_params,
            directory: &self.directory,
            max_visible_chunks: self.test_params.max_visible_chunks,
        })?;

        let control_state = controls::ControlState::new(&self.test_params);

        use metrics_core::Builder;
        let metrics_observer = metrics_runtime::observers::YamlBuilder::new().build();

        let time_tracker = TimeTracker::new(
            TimingParams {
                log_interval: Some(Duration::from_secs(5)),
                tick_size: Duration::from_millis(self.test_params.game_step_millis),
                render_interval: self
                    .test_params
                    .fixed_refresh_rate
                    .map(|rate| Duration::from_millis(1000 / rate)),
            },
            metrics_observer,
            self.metrics_controller,
            Instant::now(),
        );

        let cursor_reset_position = Point2::origin() + win_dimensions / 2.;

        let directory = Arc::new(self.directory);

        let mut entities = Vec::new();
        let mut entity_locations = Vec::new();

        for entity in self.test_params.entities {
            entity_locations.push(entity.location);

            let behaviors = entity
                .behaviors
                .iter()
                .map(|name| directory.entity.behavior_key(name.as_str()))
                .collect::<Result<_>>()?;

            entities.push(Entity {
                model: directory.entity.model_key(entity.model.as_str())?,
                behaviors,
            });
        }

        let mut world = self.world;
        world
            .entities
            .populate(entities, entity_locations.as_slice());

        let world = Arc::new(Mutex::new(world));

        let world_state = {
            WorldStateBuilder {
                world: world.clone(),
                directory: directory.clone(),
                tree_config: self.test_params.tree.as_ref(),
            }
            .build()?
        };

        Ok(TestRender {
            directory,
            world,
            world_state,
            win_dimensions,
            window,
            control_state,
            cursor_reset_position,
            time_tracker,
            renderer,
            render_state,
            view_params,
            has_cursor_control: false,
            focused: false,
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
                    if let KeyboardInput {
                        virtual_keycode: Some(key),
                        state: ElementState::Pressed,
                        ..
                    } = input
                    {
                        match key {
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            VirtualKeyCode::Space => self.world_state.toggle_updates()?,
                            VirtualKeyCode::E => self.world_state.modify_filled_voxels(1)?,
                            VirtualKeyCode::Q => self.world_state.modify_filled_voxels(-1)?,
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    self.world_state.on_click(
                        &*self.directory,
                        &self.world,
                        convert_point!(self.view_params.effective_camera_pos(), f64),
                        convert_vec!(self.view_params.look_at_dir, f64),
                    )?;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.has_cursor_control {
                        self.handle_cursor_move(position)?;
                    } else if self.focused {
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
                    self.focused = focused;
                }
                WindowEvent::Resized(event_size) => {
                    let new_size = self.window.inner_size();
                    log::info!(
                        "Window resized to {:?} (event reported {:?})",
                        new_size,
                        event_size
                    );

                    self.renderer.update_win_size(
                        &mut self.render_state,
                        Vector2::new(new_size.width, new_size.height),
                    )?;
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

        if *control_flow == ControlFlow::Exit {
            log::info!("Final metrics: {}", self.time_tracker.drain_metrics());
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
        self.renderer
            .set_world_view(&mut self.render_state, self.view_params);

        {
            let world = self.world.lock().unwrap();
            let entities = self.world_state.active_entities(&*self.directory, &*world, None)?;
            let world_diff = self.world_state.world_diff()?;

            self.renderer.update(
                &mut self.render_state,
                &*world,
                world_diff,
                entities.as_slice(),
            );
            world_diff.clear();
        }

        self.renderer.render_frame(&mut self.render_state)?;

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
        self.control_state.tick(elapsed, &mut self.view_params);
        Ok(())
    }
}

impl EventTracker {
    fn check_ready(&mut self, now: Instant, period: Duration) -> bool {
        let prev_event = match &mut self.prev_event {
            None => {
                self.prev_event = Some(now);
                return true;
            }
            Some(prev_event) => prev_event,
        };

        let elapsed = now.duration_since(*prev_event);
        let ready = elapsed >= period;
        if ready {
            *prev_event = now;
            if let Some(metrics_key) = &self.metrics_key {
                metrics::recorder()
                    .record_histogram(metrics_key.clone(), elapsed.as_nanos() as u64);
            }
        }

        ready
    }
}

impl TimeTracker {
    pub fn new(
        params: TimingParams,
        metrics_observer: MetricsObserver,
        metrics_controller: metrics_runtime::Controller,
        now: Instant,
    ) -> Self {
        Self {
            start_time: now,
            render_event: EventTracker {
                metrics_key: Some(metrics::Key::from_name("simgame.frame_interval")),
                prev_event: None,
            },
            log_event: EventTracker {
                metrics_key: None,
                prev_event: None,
            },
            total_ticks: 0,
            params,
            metrics_controller,
            metrics_observer,
        }
    }

    /// Advance the TimeTracker forward to a new instant. Returns an iterator over each tick since
    /// the previous call to tick().
    pub fn tick(&mut self, now: Instant) -> impl Iterator<Item = (Instant, Duration)> {
        if let Some(log_interval) = self.params.log_interval {
            if self.log_event.check_ready(now, log_interval) {
                log::info!("{}", self.drain_metrics());
            }
        }

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

    pub fn check_render(&mut self, now: Instant) -> bool {
        let render_interval = self
            .params
            .render_interval
            .unwrap_or_else(|| Duration::from_secs(0));
        self.render_event.check_ready(now, render_interval)
    }

    pub fn drain_metrics(&mut self) -> String {
        use metrics_core::{Drain, Observe};
        self.metrics_controller.observe(&mut self.metrics_observer);
        self.metrics_observer.drain()
    }
}

fn reset_cursor(window: &Window, pos: Point2<f64>) -> Result<(), winit::error::ExternalError> {
    window.set_cursor_position(winit::dpi::Position::Logical(winit::dpi::LogicalPosition {
        x: pos.x,
        y: pos.y,
    }))
}
