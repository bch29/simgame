//! Types used to represent the voxel-based world.
//!
mod block_data;
pub mod config;
mod core;
pub mod index_utils;
mod octree;
pub mod primitive;
mod update;

pub use crate::block_data::BlockData;
pub use crate::config::{BlockConfig, BlockConfigHelper};
pub use crate::core::{blocks_to_u16, Block, BlockRaycastHit, Chunk};
pub use crate::update::{BlockUpdater, UpdatedBlocksState};
