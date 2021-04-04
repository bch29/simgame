use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cgmath::{Point3, Vector3};
use rand::SeedableRng;

use simgame_types::{Directory, ModelRenderData};
use simgame_util::ray::Ray;
use simgame_voxels::{VoxelData, VoxelDelta, VoxelRaycastHit};

use crate::{
    background_object,
    tree::{TreeConfig, TreeSystem},
};

mod background;

pub struct WorldStateBuilder<'a> {
    pub voxels: Arc<Mutex<VoxelData>>,
    pub entities: hecs::World,
    pub directory: Arc<Directory>,
    pub tree_config: Option<&'a TreeConfig>,
}

/// Keeps track of world state. Responds immediately to user input and hands off updates to the
/// background state.
pub struct WorldState {
    connection: Option<background_object::Connection<background::BackgroundState>>,
}

#[derive(Debug, Clone, Default)]
struct WorldResponse {
    voxel_delta: VoxelDelta,
    models: Vec<ModelRenderData>,
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
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.directory.voxel)?),
            None => None,
        };

        let world_state = background::BackgroundState {
            voxels: self.voxels,
            response: Default::default(),
            entities: self.entities,
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_voxels: (16 * 16 * 4) / 8,
            tree_system,
        };

        let connection = background_object::Connection::new(world_state, Default::default())?;

        Ok(WorldState {
            connection: Some(connection),
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
        directory: &Directory,
        voxels: &Mutex<VoxelData>,
        camera_pos: Point3<f64>,
        camera_facing: Vector3<f64>,
    ) -> Result<()> {
        let ray = Ray {
            origin: camera_pos,
            dir: camera_facing,
        };

        let raycast_hit = {
            let voxels = voxels.lock().map_err(|_| anyhow!("voxel mutex poisoned"))?;
            match voxels.cast_ray(&ray, &directory.voxel) {
                None => return Ok(()),
                Some(raycast_hit) => raycast_hit,
            }
        };

        self.send_action(WorldUpdateAction::SpawnTree { raycast_hit })
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.send_action(WorldUpdateAction::ToggleUpdates)
    }

    fn send_action(&mut self, action: WorldUpdateAction) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("sending action on a closed connection: {:?}", action))?;
        connection.send_user(action)
    }

    pub fn voxel_delta(&mut self, result: &mut VoxelDelta) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("collecting diffs on a closed connection"))?;

        let delta = &mut connection.current_response()?.voxel_delta;
        result.update_from(delta);
        delta.clear();
        Ok(())
    }

    /// Returns entity models within the given bounds, for rendering.
    pub fn model_render_data(&mut self, result: &mut Vec<ModelRenderData>) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("collecting diffs on a closed connection"))?;

        let models = &mut connection.current_response()?.models;
        result.extend(models.drain(..));
        Ok(())
    }
}

impl background_object::Cumulative for WorldResponse {
    fn empty() -> Self {
        Self::default()
    }

    fn append(&mut self, mut other: Self) -> Result<()> {
        // voxel data gets accumulated
        self.voxel_delta.update_from(&mut other.voxel_delta);

        // models to render get overwritten
        if !other.models.is_empty() {
            self.models.clear();
            self.models.extend(other.models.into_iter());
        }
        Ok(())
    }
}

impl WorldResponse {
    fn is_empty(&self) -> bool {
        self.voxel_delta.is_empty() && self.models.is_empty()
    }
}
