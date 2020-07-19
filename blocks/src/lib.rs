//! Types used to represent the voxel-based world.
//!
pub mod index_utils;
pub mod primitive;
pub mod ray;
pub mod util;
pub mod config;
mod block_data;
mod update;
mod core;
mod octree;

pub use crate::config::{BlockConfig, BlockConfigHelper};
pub use crate::block_data::BlockData;
pub use crate::update::{UpdatedBlocksState, BlockUpdater};
pub use crate::core::{Block, Chunk, blocks_to_u16};
