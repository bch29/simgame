use std::sync::Mutex;

use anyhow::{anyhow, Result};
use cgmath::{Point3, Vector3};

use simgame_types::{Directory, ModelRenderData};
use simgame_util::{background_object, ray::Ray};
use simgame_voxels::{VoxelData, VoxelDelta};

use crate::state::{Tick, WorldState, WorldUpdateAction};

/// Keeps track of world state. Responds immediately to user input and hands off updates to the
/// background state.
pub struct WorldStateHandle {
    connection: Option<background_object::Connection<WorldState>>,
}

impl WorldStateHandle {
    pub(super) fn new(connection: background_object::Connection<WorldState>) -> Self {
        Self {
            connection: Some(connection),
        }
    }

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
