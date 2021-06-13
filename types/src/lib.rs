pub mod config;
pub mod mesh;
mod model;

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

/// Implemented by behavior components.
#[typetag::serde]
pub trait Behavior: std::fmt::Debug {
    fn insert(&self, entity: &mut hecs::EntityBuilder);
}

#[macro_export]
macro_rules! impl_behavior {
    { $(impl Behavior for $ty:ty;)* } =>
    {
        $(
        #[typetag::serde]
        impl $crate::Behavior for $ty {
            fn insert(&self, entity: &mut hecs::EntityBuilder) {
                entity.add(self.clone());
            }
        })*
    }
}
