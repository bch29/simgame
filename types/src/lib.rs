use simgame_util::float_octree::Octree as FloatOctree;
pub use simgame_voxels::{VoxelDelta, VoxelData};

pub mod entity;
pub use entity::Entity;

#[derive(Debug)]
pub struct World {
    pub voxels: VoxelData,
    pub entities: EntityState,
}

#[derive(Debug)]
pub struct EntityState {
    pub entities: Vec<Entity>,
    pub entity_locations: FloatOctree<usize>,
}

#[derive(Debug)]
pub struct WorldDelta {
    pub voxels: VoxelDelta,
}

impl World {
    pub fn new(voxels: VoxelData) -> World {
        World {
            voxels,
            entities: EntityState::new(),
        }
    }
}

impl EntityState {
    pub fn new() -> EntityState {
        EntityState {
            entities: Vec::new(),
            entity_locations: FloatOctree::new(),
        }
    }
}

impl WorldDelta {
    pub fn new() -> Self {
        Self {
            voxels: VoxelDelta::empty(),
        }
    }

    pub fn clear(&mut self) {
        self.voxels.clear();
    }

    pub fn update_from(&mut self, other: WorldDelta) {
        self.voxels.update_from(other.voxels);
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }
}

impl Default for EntityState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WorldDelta {
    fn default() -> Self {
        Self::new()
    }
}
