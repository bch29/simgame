mod controls;
pub mod files;
pub mod settings;

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, One, Point2, Vector2, Zero};
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
use simgame_types::{Directory, VoxelData};
use simgame_util::{convert_point, convert_vec};

use settings::RenderTestParams;
use simgame_world::{component, WorldStateBuilder, WorldStateHandle};

pub fn run_game(args: GameArgs<'_>) -> Result<()> {
    let event_loop = EventLoop::new();

    let game = smol::run(
        GameBuilder {
            event_loop: &event_loop,
            args,
        }
        .build(),
    )?;
    game.run(event_loop);
    Ok(())
}

pub struct GameArgs<'a> {
    pub voxels: VoxelData,
    pub entities: hecs::World,
    pub test_params: RenderTestParams,
    pub render_params: RenderParams<'a>,
    pub resource_loader: ResourceLoader,
    pub texture_loader: TextureLoader,
    pub directory: Directory,
    pub metrics_controller: metrics_runtime::Controller,
}

pub struct GameBuilder<'a> {
    event_loop: &'a EventLoop<()>,
    args: GameArgs<'a>,
}

pub struct Game {
    directory: Arc<Directory>,
    voxels: Arc<Mutex<VoxelData>>,
    world_state: WorldStateHandle,
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
    args: TimingParams,

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

impl<'a> GameBuilder<'a> {
    pub async fn build(self) -> Result<Game> {
        let directory = Arc::new(self.args.directory);

        let renderer = RendererBuilder {
            resource_loader: self.args.resource_loader,
            texture_loader: self.args.texture_loader,
            directory: directory.clone(),
        }
        .build(self.args.render_params.clone())
        .await?;

        let window = build_window(self.event_loop, &self.args.test_params.video_settings)?;

        let physical_win_size = window.inner_size();
        let physical_win_size: (u32, u32) = physical_win_size.into();
        let physical_win_size = physical_win_size.into();

        let logical_win_size = window.inner_size().to_logical::<f64>(window.scale_factor());
        let win_dimensions = Vector2::new(logical_win_size.width, logical_win_size.height);

        let visible_size = controls::restrict_visible_size(
            self.args.test_params.max_visible_chunks,
            self.args.test_params.initial_visible_size,
        );

        if visible_size != self.args.test_params.initial_visible_size {
            let new_visible_chunks = visible_size_to_chunks(visible_size);
            log::warn!(
                "Initial visible size of {:?} would exceed max_visible_chunks setting of {}. Decreasing to {:?}. This will use {} out of {} chunks.",
                self.args.test_params.initial_visible_size,
                self.args.test_params.max_visible_chunks,
                visible_size,
                new_visible_chunks.x * new_visible_chunks.y * new_visible_chunks.z,
                self.args.test_params.max_visible_chunks
                );
        }

        let view_params = ViewParams {
            camera_pos: self.args.test_params.initial_camera_pos,
            z_level: self.args.test_params.initial_z_level,
            visible_size,
            look_at_dir: self.args.test_params.look_at_dir,
        };

        let render_state = renderer.create_state(RenderStateInputs {
            window: &window,
            physical_win_size,
            voxels: &self.args.voxels,
            view_params,
            max_visible_chunks: self.args.test_params.max_visible_chunks,
        })?;

        let control_state = controls::ControlState::new(&self.args.test_params);

        use metrics_core::Builder;
        let metrics_observer = metrics_runtime::observers::YamlBuilder::new().build();

        let time_tracker = TimeTracker::new(
            TimingParams {
                log_interval: Some(Duration::from_secs(5)),
                tick_size: Duration::from_millis(self.args.test_params.game_step_millis),
                render_interval: self
                    .args
                    .test_params
                    .fixed_refresh_rate
                    .map(|rate| Duration::from_millis(1000 / rate)),
            },
            metrics_observer,
            self.args.metrics_controller,
            Instant::now(),
        );

        let cursor_reset_position = Point2::origin() + win_dimensions / 2.;

        let voxels = Arc::new(Mutex::new(self.args.voxels));

        let entities = {
            let mut entities = self.args.entities;

            let mut builder = hecs::EntityBuilder::new();

            let resolved = archetype::resolve(
                self.args.test_params.entity_archetypes.as_slice(),
                self.args.test_params.entities.as_slice(),
            )?;

            for entity in resolved {
                let model_key = directory.model.model_key(entity.model)?;
                let model_data = directory.model.model_data(model_key)?;

                builder.add(component::Bounds(model_data.bounds));
                builder.add(component::Position(entity.location));
                builder.add(component::Orientation(One::one()));
                builder.add(component::Velocity(Zero::zero()));
                builder.add(component::Ground::NotOnGround);
                builder.add(component::Model {
                    key: model_key,
                    transform: model_data.transform,
                });

                for behavior in &entity.behaviors {
                    behavior.insert(&mut builder);
                }

                entities.spawn(builder.build());
            }

            entities
        };

        let world_state = WorldStateBuilder {
            voxels: voxels.clone(),
            entities,
            directory: directory.clone(),
            tree_config: self.args.test_params.tree.as_ref(),
        }
        .build()?;

        Ok(Game {
            directory,
            voxels,
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

mod archetype {
    use std::collections::HashMap;

    use anyhow::{anyhow, bail, Result};
    use cgmath::Point3;

    use simgame_types::Behavior;

    use crate::settings;

    pub struct ResolvedEntity<'a> {
        pub model: &'a str,
        pub location: Point3<f64>,
        pub behaviors: Vec<&'a dyn Behavior>,
    }

    #[derive(Clone, Debug)]
    struct ResolvedArchetype<'a> {
        name: &'a str,
        model: &'a str,
        behaviors: Vec<&'a dyn Behavior>,
    }

    impl<'a> ResolvedArchetype<'a> {
        fn from_config_pure(archetype: &'a settings::EntityArchetype) -> Result<Self> {
            let name = archetype.name.as_str();
            let model = archetype
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("missing model in archetype {:?}", name))?
                .as_str();
            let behaviors = archetype
                .behaviors
                .iter()
                .flat_map(|behaviors| behaviors.iter().map(|b| &**b))
                .collect();

            Ok(Self {
                name,
                model,
                behaviors,
            })
        }

        fn from_config_cloning(
            archetype: &'a settings::EntityArchetype,
            base: ResolvedArchetype<'a>,
        ) -> Self {
            let mut result = base;
            result.name = archetype.name.as_str();

            if let Some(model) = &archetype.model {
                result.model = model.as_str();
            }

            if let Some(behaviors) = &archetype.behaviors {
                result.behaviors.extend(behaviors.iter().map(|b| &**b));
            }

            result
        }

        fn try_from_config(
            archetype: &'a settings::EntityArchetype,
            existing: &HashMap<String, ResolvedArchetype<'a>>,
        ) -> Result<Option<Self>> {
            let result = match &archetype.clone_from {
                Some(clone_from) => match existing.get(clone_from).cloned() {
                    Some(base) => Self::from_config_cloning(archetype, base),
                    None => return Ok(None),
                },
                None => Self::from_config_pure(archetype)?,
            };

            if existing.contains_key(result.name) {
                bail!("duplicate archetype name {:?}", result.name);
            }

            Ok(Some(result))
        }
    }

    pub fn resolve<'a>(
        archetypes: &'a [settings::EntityArchetype],
        entities: &'a [settings::RenderTestEntity],
    ) -> Result<Vec<ResolvedEntity<'a>>> {
        let mut resolved_archetypes: HashMap<String, ResolvedArchetype> = HashMap::new();
        let mut remaining_archetypes: Vec<&'a settings::EntityArchetype> =
            archetypes.iter().collect();

        while !remaining_archetypes.is_empty() {
            let previous_len = remaining_archetypes.len();
            for archetype in remaining_archetypes.drain(..).collect::<Vec<_>>() {
                match ResolvedArchetype::try_from_config(archetype, &resolved_archetypes)? {
                    Some(result) => {
                        resolved_archetypes.insert(archetype.name.to_owned(), result);
                    }
                    None => {
                        remaining_archetypes.push(archetype);
                    }
                };
            }

            if remaining_archetypes.len() == previous_len {
                bail!("cycle or missing archetype in clone_from");
            }
        }

        entities
            .iter()
            .map(|entity| {
                let archetype = resolved_archetypes
                    .get(&entity.archetype)
                    .ok_or_else(|| anyhow!("archetype {:?} not configured", entity.archetype))?;

                let mut behaviors: Vec<&'a dyn Behavior> = archetype.behaviors.clone();
                if let Some(extra_behaviors) = &entity.behaviors {
                    behaviors.extend(extra_behaviors.iter().map(|b| &**b));
                }

                Ok(ResolvedEntity {
                    model: archetype.model,
                    location: entity.location,
                    behaviors,
                })
            })
            .collect()
    }
}

impl Game {
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
                        &self.voxels,
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
            let mut voxel_delta = Default::default();
            let mut entities = Vec::new();
            self.world_state.voxel_delta(&mut voxel_delta)?;
            self.world_state.model_render_data(&mut entities)?;

            let voxels = self.voxels.lock().unwrap();
            self.renderer.update(
                &mut self.render_state,
                &*voxels,
                &voxel_delta,
                entities.as_slice(),
            )?;
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
        args: TimingParams,
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
            args,
            metrics_controller,
            metrics_observer,
        }
    }

    /// Advance the TimeTracker forward to a new instant. Returns an iterator over each tick since
    /// the previous call to tick().
    pub fn tick(&mut self, now: Instant) -> impl Iterator<Item = (Instant, Duration)> {
        if let Some(log_interval) = self.args.log_interval {
            if self.log_event.check_ready(now, log_interval) {
                log::info!("{}", self.drain_metrics());
            }
        }

        let old_total_ticks = self.total_ticks;
        self.total_ticks = (now.duration_since(self.start_time).as_secs_f64()
            / self.args.tick_size.as_secs_f64())
        .floor() as i64;

        let start_time = self.start_time;
        let tick_size = self.args.tick_size;
        (old_total_ticks..self.total_ticks).map(move |tick_ix| {
            let tick_time = start_time + tick_size * (tick_ix as u32);
            let elapsed = tick_size;
            (tick_time, elapsed)
        })
    }

    pub fn check_render(&mut self, now: Instant) -> bool {
        let render_interval = self
            .args
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
