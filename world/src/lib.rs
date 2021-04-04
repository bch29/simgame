mod behavior;
pub mod component;
mod handle;
pub mod lsystem;
mod state;
pub mod tree;
pub mod turtle;
pub mod worldgen;

use std::sync::{Arc, Mutex};

use anyhow::Result;

use simgame_types::Directory;
use simgame_util::background_object;
use simgame_voxels::VoxelData;

use crate::tree::{TreeConfig, TreeSystem};

pub use handle::WorldStateHandle;

pub struct WorldStateBuilder<'a> {
    pub voxels: Arc<Mutex<VoxelData>>,
    pub entities: hecs::World,
    pub directory: Arc<Directory>,
    pub tree_config: Option<&'a TreeConfig>,
}

impl<'a> WorldStateBuilder<'a> {
    pub fn build(self) -> Result<WorldStateHandle> {
        let tree_system = match self.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.directory.voxel)?),
            None => None,
        };

        let world_state =
            state::WorldState::new(self.directory, self.voxels, self.entities, tree_system);
        let connection = background_object::Connection::new(world_state, Default::default())?;

        Ok(WorldStateHandle::new(connection))
    }
}
