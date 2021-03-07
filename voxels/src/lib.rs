//! Types used to represent the voxel-based world.
//!
mod voxel_data;
pub mod config;
mod core;
pub mod index_utils;
pub mod octree;
pub mod primitive;
mod update;

pub use crate::voxel_data::VoxelData;
pub use crate::config::{VoxelConfig, VoxelConfigHelper};
pub use crate::core::{voxels_to_u16, Voxel, VoxelRaycastHit, Chunk};
pub use crate::update::{VoxelUpdater, UpdatedVoxelsState};
