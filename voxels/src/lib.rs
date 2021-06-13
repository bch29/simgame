//! Types used to represent the voxel-based world.
//!
pub mod config;
mod core;
pub mod index_utils;
pub mod primitive;
mod update;
mod voxel_data;

pub use crate::{
    config::{VoxelConfig, VoxelDirectory},
    core::{voxels_to_u16, Chunk, Voxel, VoxelRaycastHit},
    update::{VoxelDelta, VoxelUpdater},
    voxel_data::VoxelData,
};

use std::sync::Arc;
use parking_lot::Mutex;

#[derive(Clone)]
pub struct SharedVoxelData {
    pub data: Arc<Mutex<VoxelData>>
}
