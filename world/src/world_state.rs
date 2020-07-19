use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};
use rand::SeedableRng;

use simgame_blocks::primitive::{self, Primitive};
use simgame_blocks::{index_utils, Block, BlockConfigHelper, BlockRaycastHit, BlockUpdater};
use simgame_util::ray::Ray;
use simgame_util::{convert_point, Bounds};

use crate::{
    background_object::{self, BackgroundObject},
    tree::{TreeConfig, TreeSystem},
    UpdatedWorldState, World,
};

pub struct WorldStateBuilder<'a> {
    pub world: Arc<Mutex<World>>,
    pub block_helper: BlockConfigHelper,
    pub tree_config: Option<&'a TreeConfig>,
}

/// Keeps track of world state. Responds immediately to user input and hands off updates to the
/// background state.
pub struct WorldState {
    connection: Option<background_object::Connection<BackgroundState>>,
    block_helper: BlockConfigHelper,
}

/// Handles background updates. While locked on the world mutex, the rendering thread will be
/// blocked, so care should be taken not to lock it for long. Other blocking in this object will
/// not block the rendering thread.
struct BackgroundState {
    world: Arc<Mutex<World>>,
    world_diff: UpdatedWorldState,
    _block_helper: BlockConfigHelper,
    rng: rand::rngs::StdRng,
    updating: bool,
    filled_blocks: i32,

    tree_system: Option<TreeSystem>,
}

#[derive(Debug, Clone)]
struct Tick {
    pub elapsed: f64,
}

#[derive(Debug, Clone)]
enum WorldUpdateAction {
    SpawnTree { raycast_hit: BlockRaycastHit },
    ModifyFilledBlocks { delta: i32 },
    ToggleUpdates,
}

impl<'a> WorldStateBuilder<'a> {
    pub fn build(self) -> Result<WorldState> {
        let tree_system = match self.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.block_helper)?),
            None => None,
        };

        let world_state = BackgroundState {
            _block_helper: self.block_helper.clone(),
            world_diff: UpdatedWorldState::empty(),
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_blocks: (16 * 16 * 4) / 8,
            tree_system,
            world: self.world,
        };

        let connection = background_object::Connection::new(world_state, Default::default())?;

        Ok(WorldState {
            connection: Some(connection),
            block_helper: self.block_helper,
        })
    }
}

impl WorldState {
    /// Moves the world forward by one tick.
    pub fn tick(&mut self, elapsed: f64) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("sending tick on a closed connection"))?;

        connection.send_tick(Tick { elapsed })
    }

    pub fn modify_filled_blocks(&mut self, delta: i32) -> Result<()> {
        self.send_action(WorldUpdateAction::ModifyFilledBlocks { delta })
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
            let world = world.lock().map_err(|_| anyhow!("world mutex poisoned"))?;
            match world.blocks.cast_ray(&ray, &self.block_helper) {
                None => return Ok(()),
                Some(raycast_hit) => raycast_hit,
            }
        };

        self.send_action(WorldUpdateAction::SpawnTree { raycast_hit })
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.send_action(WorldUpdateAction::ToggleUpdates)
    }

    pub fn world_diff(&mut self) -> Result<&mut UpdatedWorldState> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("collecting diffs on a closed connection"))?;
        connection.current_response()
    }

    fn send_action(&mut self, action: WorldUpdateAction) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("sending action on a closed connection: {:?}", action))?;
        connection.send_user(action)
    }
}

impl BackgroundState {
    /// Moves the world forward by one tick.
    fn tick(&mut self, _elapsed: f64) -> Result<()> {
        if !self.updating {
            return Ok(());
        }

        // let mut updater = BlockUpdater::new(&mut self.world.blocks, &mut self.world_diff.blocks);
        // shape.draw(&mut updater);

        self.updating = false;
        Ok(())
    }

    fn modify_filled_blocks(&mut self, delta: i32) -> Result<()> {
        self.filled_blocks += delta * 6;
        if self.filled_blocks < 1 {
            self.filled_blocks = 1
        } else if self.filled_blocks >= index_utils::chunk_size_total() as i32 {
            self.filled_blocks = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<i64> = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_blocks as i64;
        let mut count_filled = 0;

        {
            let mut world = self
                .world
                .lock()
                .map_err(|_| anyhow!("world mutex poisoned"))?;
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
        Ok(())
    }

    fn spawn_tree(&mut self, raycast_hit: BlockRaycastHit) -> Result<()> {
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
        self.draw_shape(tree, draw_transform)
    }

    /// Draws a complex shape while trying to avoid blocking the rendering thread due to world
    /// updates.
    fn draw_shape(&mut self, shape: primitive::Shape, draw_transform: Matrix4<f64>) -> Result<()> {
        for (fill_block, primitive) in shape.iter_transformed_primitives(draw_transform) {
            // lock the mutex inside the loop so that the rendering thread isn't blocked while we
            // update
            let mut world = self
                .world
                .lock()
                .map_err(|_| anyhow!("world mutex poisoned"))?;
            let mut updater = BlockUpdater::new(&mut world.blocks, &mut self.world_diff.blocks);
            primitive.draw(&mut updater, fill_block);
        }
        Ok(())
    }

    fn toggle_updates(&mut self) -> Result<()> {
        self.updating = !self.updating;
        Ok(())
    }
}

impl BackgroundObject for BackgroundState {
    type UserAction = WorldUpdateAction;
    type TickAction = Tick;
    type Response = UpdatedWorldState;

    fn receive_user(&mut self, action: WorldUpdateAction) -> Result<()> {
        match action {
            WorldUpdateAction::SpawnTree { raycast_hit } => self.spawn_tree(raycast_hit),
            WorldUpdateAction::ToggleUpdates => self.toggle_updates(),
            WorldUpdateAction::ModifyFilledBlocks { delta } => self.modify_filled_blocks(delta),
        }
    }

    fn receive_tick(&mut self, tick: Tick) -> Result<()> {
        let Tick { elapsed } = tick;
        self.tick(elapsed)
    }

    fn produce_response(&mut self) -> Result<Option<UpdatedWorldState>> {
        if self.world_diff.is_empty() {
            return Ok(None);
        }
        let mut diff = UpdatedWorldState::empty();
        std::mem::swap(&mut self.world_diff, &mut diff);
        Ok(Some(diff))
    }
}

impl background_object::Cumulative for UpdatedWorldState {
    fn empty() -> Self {
        UpdatedWorldState::empty()
    }

    fn append(&mut self, other: Self) -> Result<()> {
        self.update_from(other);
        Ok(())
    }
}
