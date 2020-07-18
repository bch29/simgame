use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};
use crossbeam::channel;
use rand::SeedableRng;

use simgame_worldgen::tree::{TreeConfig, TreeSystem};

use simgame_core::block::{self, index_utils, Block, BlockConfigHelper, BlockUpdater};
use simgame_core::convert_point;
use simgame_core::ray::Ray;
use simgame_core::util::Bounds;
use simgame_core::world::{UpdatedWorldState, World};

pub struct WorldState {
    join_handle: Option<std::thread::JoinHandle<Result<()>>>,
    action_sender: channel::Sender<BackgroundAction>,
    diff_receiver: channel::Receiver<UpdateAction>,

    block_helper: BlockConfigHelper,
    world_diff: UpdatedWorldState,
}

pub struct WorldStateBuilder<'a> {
    pub world: Arc<Mutex<World>>,
    pub block_helper: BlockConfigHelper,
    pub tree_config: Option<&'a TreeConfig>,
}

struct BackgroundThread {
    action_receiver: channel::Receiver<BackgroundAction>,
    diff_sender: channel::Sender<UpdateAction>,

    world_diff: UpdatedWorldState,
    _block_helper: BlockConfigHelper,
    rng: rand::rngs::StdRng,
    updating: bool,
    filled_blocks: i32,

    tree_system: Option<TreeSystem>,
}

enum UpdateAction {
    Updated { diff: UpdatedWorldState },
    Died,
}

enum BackgroundAction {
    SpawnTree { raycast_hit: block::RaycastHit },
    Tick { elapsed: f64 },
    ModifyFilledBlocks { delta: i32 },
    ToggleUpdates,
    Kill,
}

impl<'a> WorldStateBuilder<'a> {
    pub fn build(self) -> Result<WorldState> {
        let tree_system = match self.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.block_helper)?),
            None => None,
        };

        let (action_sender, action_receiver) = channel::bounded(16);
        let (diff_sender, diff_receiver) = channel::bounded(16);

        let background = BackgroundThread {
            action_receiver,
            diff_sender,

            _block_helper: self.block_helper.clone(),
            world_diff: UpdatedWorldState::empty(),
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_blocks: (16 * 16 * 4) / 8,
            tree_system,
        };

        let world = self.world;
        let join_handle = std::thread::spawn(move || background.run(&*world));

        Ok(WorldState {
            join_handle: Some(join_handle),
            action_sender,
            diff_receiver,

            block_helper: self.block_helper,
            world_diff: UpdatedWorldState::empty(),
        })
    }
}

impl WorldState {
    /// Moves the world forward by one tick.
    pub fn tick(&mut self, elapsed: f64) -> Result<()> {
        self.action_sender
            .send(BackgroundAction::Tick { elapsed })?;
        self.try_collect_diffs()?;
        Ok(())
    }

    pub fn modify_filled_blocks(&mut self, delta: i32) -> Result<()> {
        self.action_sender
            .send(BackgroundAction::ModifyFilledBlocks { delta })?;
        self.try_collect_diffs()?;
        Ok(())
    }

    pub fn on_click(
        &mut self,
        world: &Mutex<World>,
        camera_pos: Point3<f64>,
        camera_facing: Vector3<f64>,
    ) -> Result<()> {
        let ray = Ray {
            origin: camera_pos,
            dir: camera_facing,
        };

        let raycast_hit = {
            let world = world.lock().unwrap();
            match world.blocks.cast_ray(&ray, &self.block_helper) {
                None => return Ok(()),
                Some(raycast_hit) => raycast_hit,
            }
        };

        self.action_sender
            .send(BackgroundAction::SpawnTree { raycast_hit })?;
        self.try_collect_diffs()?;

        Ok(())
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.action_sender.send(BackgroundAction::ToggleUpdates)?;
        self.try_collect_diffs()?;

        Ok(())
    }

    pub fn kill(&mut self) -> Result<()> {
        let join_handle = match self.join_handle.take() {
            Some(handle) => handle,
            None => {
                log::warn!("Attempted to kill background thread twice");
                return Ok(());
            }
        };

        self.action_sender.send(BackgroundAction::Kill)?;
        match join_handle.join() {
            Ok(res) => res,
            Err(err) => Err(anyhow!("Error joining background thread: {:?}", err)),
        }
    }

    fn try_collect_diffs(&mut self) -> Result<()> {
        loop {
            match self.diff_receiver.try_recv() {
                Ok(UpdateAction::Updated { diff }) => self.world_diff.update_from(diff),
                Ok(UpdateAction::Died) => break self.kill(),
                Err(channel::TryRecvError::Empty) => break Ok(()),
                Err(err) => break Err(err.into()),
            }
        }
    }

    pub fn world_diff(&mut self) -> Result<&mut UpdatedWorldState> {
        self.try_collect_diffs()?;
        Ok(&mut self.world_diff)
    }
}

impl Drop for WorldState {
    fn drop(&mut self) {
        match self.kill() {
            Ok(()) => {}
            Err(end_error) => {
                for error in end_error.chain() {
                    log::error!("{}", error);
                    log::error!("========");
                }
            }
        }
    }
}

impl BackgroundThread {
    pub fn run(mut self, world: &Mutex<World>) -> Result<()> {
        match self.run_impl(world) {
            Ok(res) => Ok(res),
            Err(err) => {
                self.diff_sender.send(UpdateAction::Died)?;
                Err(err)
            }
        }
    }

    fn run_impl(&mut self, world: &Mutex<World>) -> Result<()> {
        loop {
            let action = self.action_receiver.recv()?;
            match action {
                BackgroundAction::Kill => break,
                BackgroundAction::SpawnTree { raycast_hit } => {
                    self.spawn_tree(world, raycast_hit)?
                }
                BackgroundAction::Tick { elapsed } => self.tick(world, elapsed)?,
                BackgroundAction::ToggleUpdates => self.toggle_updates()?,
                BackgroundAction::ModifyFilledBlocks { delta } => {
                    self.modify_filled_blocks(world, delta)?
                }
            }
        }

        Ok(())
    }

    /// Moves the world forward by one tick.
    pub fn tick(&mut self, _world: &Mutex<World>, _elapsed: f64) -> Result<()> {
        if !self.updating {
            return Ok(());
        }

        // let mut updater = BlockUpdater::new(&mut self.world.blocks, &mut self.world_diff.blocks);
        // shape.draw(&mut updater);

        self.updating = false;

        self.send_diff()?;
        Ok(())
    }

    pub fn modify_filled_blocks(&mut self, world: &Mutex<World>, delta: i32) -> Result<()> {
        self.filled_blocks += delta * 8;
        if self.filled_blocks < 1 {
            self.filled_blocks = 1
        } else if self.filled_blocks >= index_utils::chunk_size_total() as i32 {
            self.filled_blocks = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<i64> = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_blocks as i64;
        let mut count_filled = 0;

        {
            let mut world = world.lock().expect("mutex poisoned");
            for p in bounds.iter_points() {
                if (p.x + p.y + p.z) % step == 0 {
                    world.blocks.set_block(p, Block::from_u16(1));
                    count_filled += 1;
                } else {
                    world.blocks.set_block(p, Block::from_u16(0));
                }
                let (chunk_pos, _) = index_utils::to_chunk_pos(p);
                self.world_diff.blocks.record_chunk_update(chunk_pos);
            }
        }

        log::debug!(
            "Setting number of filled blocks to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );

        self.send_diff()?;
        Ok(())
    }

    pub fn spawn_tree(
        &mut self,
        world: &Mutex<World>,
        raycast_hit: block::RaycastHit,
    ) -> Result<()> {
        log::debug!(
            "Clicked on block {:?} at pos {:?} with intersection {:?}",
            raycast_hit.block,
            raycast_hit.block_pos,
            raycast_hit.intersection
        );

        let mut tree_system = match &self.tree_system {
            None => return Ok(()),
            Some(system) => system.clone(),
        };

        let tree = tree_system.generate(&mut self.rng)?;

        let root_pos =
            convert_point!(raycast_hit.block_pos, f64) + raycast_hit.intersection.normal;

        let draw_transform = Matrix4::from_translation(root_pos - Point3::origin());

        {
            let mut world = world.lock().unwrap();
            let mut updater = BlockUpdater::new(&mut world.blocks, &mut self.world_diff.blocks);
            tree.draw_transformed(&mut updater, draw_transform);
        }

        self.send_diff()?;
        Ok(())
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.updating = !self.updating;
        self.send_diff()?;
        Ok(())
    }

    fn send_diff(&mut self) -> Result<()> {
        let mut diff = UpdatedWorldState::empty();
        std::mem::swap(&mut self.world_diff, &mut diff);
        self.diff_sender.send(UpdateAction::Updated { diff })?;
        Ok(())
    }
}
