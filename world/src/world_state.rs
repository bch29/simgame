use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};
use rand::SeedableRng;

use simgame_util::ray::Ray;
use simgame_util::{convert_point, Bounds};
use simgame_voxels::primitive::{self, Primitive};
use simgame_voxels::{index_utils, Voxel, VoxelConfigHelper, VoxelRaycastHit, VoxelUpdater};

use crate::{
    background_object::{self, BackgroundObject},
    tree::{TreeConfig, TreeSystem},
    WorldDelta, World,
};

pub struct WorldStateBuilder<'a> {
    pub world: Arc<Mutex<World>>,
    pub voxel_helper: VoxelConfigHelper,
    pub tree_config: Option<&'a TreeConfig>,
}

/// Keeps track of world state. Responds immediately to user input and hands off updates to the
/// background state.
pub struct WorldState {
    connection: Option<background_object::Connection<BackgroundState>>,
    voxel_helper: VoxelConfigHelper,
}

/// Handles background updates. While locked on the world mutex, the rendering thread will be
/// voxeled, so care should be taken not to lock it for long. Other voxeling in this object will
/// not voxel the rendering thread.
struct BackgroundState {
    world: Arc<Mutex<World>>,
    world_diff: WorldDelta,
    _voxel_helper: VoxelConfigHelper,
    rng: rand::rngs::StdRng,
    updating: bool,
    filled_voxels: i32,

    tree_system: Option<TreeSystem>,
}

#[derive(Debug, Clone)]
struct Tick {
    pub elapsed: f64,
}

#[derive(Debug, Clone)]
enum WorldUpdateAction {
    SpawnTree { raycast_hit: VoxelRaycastHit },
    ModifyFilledVoxels { delta: i32 },
    ToggleUpdates,
}

impl<'a> WorldStateBuilder<'a> {
    pub fn build(self) -> Result<WorldState> {
        let tree_system = match self.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.voxel_helper)?),
            None => None,
        };

        let world_state = BackgroundState {
            _voxel_helper: self.voxel_helper.clone(),
            world_diff: WorldDelta::new(),
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_voxels: (16 * 16 * 4) / 8,
            tree_system,
            world: self.world,
        };

        let connection = background_object::Connection::new(world_state, Default::default())?;

        Ok(WorldState {
            connection: Some(connection),
            voxel_helper: self.voxel_helper,
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

    pub fn modify_filled_voxels(&mut self, delta: i32) -> Result<()> {
        self.send_action(WorldUpdateAction::ModifyFilledVoxels { delta })
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
            match world.voxels.cast_ray(&ray, &self.voxel_helper) {
                None => return Ok(()),
                Some(raycast_hit) => raycast_hit,
            }
        };

        self.send_action(WorldUpdateAction::SpawnTree { raycast_hit })
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.send_action(WorldUpdateAction::ToggleUpdates)
    }

    pub fn world_diff(&mut self) -> Result<&mut WorldDelta> {
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
    #[allow(clippy::unnecessary_wraps)]
    fn tick(&mut self, _elapsed: f64) -> Result<()> {
        if !self.updating {
            return Ok(());
        }

        // let mut updater = VoxelUpdater::new(&mut self.world.voxels, &mut self.world_diff.voxels);
        // shape.draw(&mut updater);

        self.updating = false;
        Ok(())
    }

    fn modify_filled_voxels(&mut self, delta: i32) -> Result<()> {
        self.filled_voxels += delta * 6;
        if self.filled_voxels < 1 {
            self.filled_voxels = 1
        } else if self.filled_voxels >= index_utils::chunk_size_total() as i32 {
            self.filled_voxels = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<i64> = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_voxels as i64;
        let mut count_filled = 0;

        {
            let mut world = self
                .world
                .lock()
                .map_err(|_| anyhow!("world mutex poisoned"))?;
            for p in bounds.iter_points() {
                if (p.x + p.y + p.z) % step == 0 {
                    world.voxels.set_voxel(p, Voxel::from_u16(1));
                    count_filled += 1;
                } else {
                    world.voxels.set_voxel(p, Voxel::from_u16(0));
                }
                let (chunk_pos, _) = index_utils::to_chunk_pos(p);
                self.world_diff.voxels.record_chunk_update(chunk_pos);
            }
        }

        log::debug!(
            "Setting number of filled voxels to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );
        Ok(())
    }

    fn spawn_tree(&mut self, raycast_hit: VoxelRaycastHit) -> Result<()> {
        log::debug!(
            "Clicked on voxel {:?} at pos {:?} with intersection {:?}",
            raycast_hit.voxel,
            raycast_hit.voxel_pos,
            raycast_hit.intersection
        );

        let mut tree_system = match &self.tree_system {
            None => return Ok(()),
            Some(system) => system.clone(),
        };

        let tree = tree_system.generate(&mut self.rng)?;

        let root_pos =
            convert_point!(raycast_hit.voxel_pos, f64) + raycast_hit.intersection.normal;

        let draw_transform = Matrix4::from_translation(root_pos - Point3::origin());
        self.draw_shape(tree, draw_transform)
    }

    /// Draws a complex shape while trying to avoid voxeling the rendering thread due to world
    /// updates.
    fn draw_shape(&mut self, shape: primitive::Shape, draw_transform: Matrix4<f64>) -> Result<()> {
        for (fill_voxel, primitive) in shape.iter_transformed_primitives(draw_transform) {
            // lock the mutex inside the loop so that the rendering thread isn't voxeled while we
            // update
            let mut world = self
                .world
                .lock()
                .map_err(|_| anyhow!("world mutex poisoned"))?;
            let mut updater = VoxelUpdater::new(&mut world.voxels, &mut self.world_diff.voxels);
            primitive.draw(&mut updater, fill_voxel);
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn toggle_updates(&mut self) -> Result<()> {
        self.updating = !self.updating;
        Ok(())
    }
}

impl BackgroundObject for BackgroundState {
    type UserAction = WorldUpdateAction;
    type TickAction = Tick;
    type Response = WorldDelta;

    fn receive_user(&mut self, action: WorldUpdateAction) -> Result<()> {
        match action {
            WorldUpdateAction::SpawnTree { raycast_hit } => self.spawn_tree(raycast_hit),
            WorldUpdateAction::ToggleUpdates => self.toggle_updates(),
            WorldUpdateAction::ModifyFilledVoxels { delta } => self.modify_filled_voxels(delta),
        }
    }

    fn receive_tick(&mut self, tick: Tick) -> Result<()> {
        let Tick { elapsed } = tick;
        self.tick(elapsed)
    }

    fn produce_response(&mut self) -> Result<Option<WorldDelta>> {
        if self.world_diff.is_empty() {
            return Ok(None);
        }
        let mut diff = WorldDelta::new();
        std::mem::swap(&mut self.world_diff, &mut diff);
        Ok(Some(diff))
    }
}

impl background_object::Cumulative for WorldDelta {
    fn empty() -> Self {
        WorldDelta::new()
    }

    fn append(&mut self, other: Self) -> Result<()> {
        self.update_from(other);
        Ok(())
    }
}
