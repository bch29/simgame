use simgame_util::float_octree::Octree as FloatOctree;
pub use simgame_voxels::{UpdatedVoxelsState, VoxelData};

pub mod entity;

pub use entity::{Entity, EntityConfig, EntityModel, EntityModelInfo};

#[derive(Debug)]
pub struct World {
    pub voxels: VoxelData,
    pub entities: FloatOctree<Entity>,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub voxels: UpdatedVoxelsState,
}

impl World {
    pub fn new(voxels: VoxelData) -> World {
        World {
            voxels,
            entities: FloatOctree::new(),
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
