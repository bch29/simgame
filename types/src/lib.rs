use std::collections::HashMap;

use anyhow::{anyhow, Result};
use cgmath::{Point3, Vector3};

use simgame_util::float_octree::{self as octree, Octree as FloatOctree};
use simgame_util::Bounds;
pub use simgame_voxels::{VoxelData, VoxelDelta, VoxelDirectory};

pub mod entity;
pub use entity::{ActiveEntityModel, Entity, EntityConfig, EntityDirectory};

#[derive(Debug)]
pub struct World {
    pub voxels: VoxelData,
    pub entities: EntityState,
}

#[derive(Debug)]
pub struct EntityState {
    pub entities: Vec<Entity>,
    pub bounds_tree: FloatOctree<usize>,
}

#[derive(Debug)]
pub struct WorldDelta {
    pub voxels: VoxelDelta,
}

pub struct Directory {
    pub voxel: VoxelDirectory,
    pub entity: EntityDirectory,
    pub texture: TextureDirectory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextureKey {
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct TextureDirectory {
    texture_keys: HashMap<String, TextureKey>,
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
            bounds_tree: FloatOctree::new(),
        }
    }

    pub fn populate(&mut self, entities: Vec<Entity>, locations: &[Point3<f64>]) {
        for (index, (_, location)) in entities.iter().zip(locations).enumerate() {
            self.bounds_tree.insert(octree::Object {
                value: index,
                bounds: Bounds::from_center(*location, Vector3::new(2., 2., 2.)),
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

impl TextureDirectory {
    pub fn new(texture_keys: HashMap<String, TextureKey>) -> Self {
        Self { texture_keys }
    }

    pub fn texture_key(&self, name: &str) -> Result<TextureKey> {
        self.texture_keys
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("texture does not exist: {:?}", name))
    }
}
