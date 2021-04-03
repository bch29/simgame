pub mod config;
pub mod mesh;
mod model;

pub use config::ResourceName;
pub use model::{
    MeshKey, ModelDirectory, ModelKey, ModelRenderData, TextureDirectory, TextureKey,
};
pub use simgame_voxels::{VoxelData, VoxelDelta, VoxelDirectory};

/// Holds any persistent data that is used outside of the ECS (and potentially also used in the
/// ECS).
pub struct Directory {
    pub voxel: VoxelDirectory,
    pub texture: TextureDirectory,
    pub model: ModelDirectory,
}
