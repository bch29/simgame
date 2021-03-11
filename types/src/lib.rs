use cgmath::{Matrix4, Point3, Vector3};

use simgame_util::float_octree::{Octree as FloatOctree, self as octree};
use simgame_util::Bounds;
pub use simgame_voxels::{VoxelDelta, VoxelData};

pub mod entity;
pub use entity::{Entity, EntityConfig};

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

pub struct ActiveEntityModel {
    pub model_kind: entity::config::ModelKind,
    pub transform: Matrix4<f32>,
    pub face_tex_ids: Vec<u32>,
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

    pub fn populate(&mut self, entities: Vec<Entity>, locations: &[Point3<f64>]) {
        for (index, (_, location)) in entities.iter().zip(locations).enumerate() {
            self.entity_locations.insert(octree::Object {
                value: index,
                bounds: Bounds::from_center(*location, Vector3::new(2., 2., 2.))
            });
        }

        self.entities = entities;
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
