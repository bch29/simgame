//! Types used to represent the voxel-based world.
//!
pub mod config;
mod core;
pub mod index_utils;
pub mod primitive;
mod update;
mod voxel_data;

pub use crate::config::{VoxelConfig, VoxelConfigHelper};
pub use crate::core::{voxels_to_u16, Chunk, Voxel, VoxelRaycastHit};
pub use crate::update::{UpdatedVoxelsState, VoxelUpdater};
pub use crate::voxel_data::VoxelData;
