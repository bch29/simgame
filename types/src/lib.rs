pub use simgame_voxels::{VoxelData, UpdatedVoxelsState};
pub use simgame_util::bsp::Bsp;

pub mod entity;

pub use entity::{Entity, EntityConfig, EntityModel, EntityModelInfo};

#[derive(Debug)]
pub struct World {
    pub voxels: VoxelData,
    pub entities: Bsp<Entity>,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub voxels: UpdatedVoxelsState,
}

impl World {
    pub fn new(voxels: VoxelData) -> World {
        World {
            voxels,
            entities: Bsp::new(),
        }
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        Self {
            voxels: UpdatedVoxelsState::empty(),
        }
    }

    pub fn clear(&mut self) {
        self.voxels.clear()
    }

    pub fn update_from(&mut self, other: UpdatedWorldState) {
        self.voxels.update_from(other.voxels)
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }
}
