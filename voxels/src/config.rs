use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::Voxel;

/// A voxel category.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Category(String);

/// Specification of a voxel type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelInfo {
    pub name: String,
    pub category: Category,
    #[serde(default = "VoxelInfo::default_passable_through")]
    pub passable_through: bool,
    #[serde(default = "VoxelInfo::default_passable_above")]
    pub passable_above: bool,
    #[serde(default = "VoxelInfo::default_speed_modifier")]
    pub speed_modifier: f64,

    pub texture: VoxelTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FaceTexture {
    /// The resource name of the texture image
    pub resource: String,
    /// The texture repeats horizontally after this many voxels (1 if None)
    pub x_periodicity: Option<u32>,
    /// The texture repeats vertically after this many voxels (1 if None)
    pub y_periodicity: Option<u32>,
}

/// Specification of how a voxel is textured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoxelTexture {
    /// The voxel has the given face texture on all faces.
    Uniform(FaceTexture),
    /// The voxel has the given face textures on corresponding faces.
    Nonuniform {
        top: FaceTexture,
        bottom: FaceTexture,
        side: FaceTexture,
    },
}

/// Specification of the selection of available voxels and categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelConfig {
    voxels: Vec<VoxelInfo>,
}

#[derive(Debug, Clone)]
pub struct VoxelConfigHelper {
    voxels_by_name: HashMap<String, (Voxel, VoxelInfo)>,
    voxels_by_id: Vec<VoxelInfo>,
}

impl VoxelInfo {
    pub fn default_passable_through() -> bool {
        false
    }
    pub fn default_passable_above() -> bool {
        true
    }
    pub fn default_speed_modifier() -> f64 {
        1.0
    }
}

impl VoxelConfigHelper {
    pub fn new(config: &VoxelConfig) -> Result<Self> {
        if config.voxels.is_empty() || config.voxels[0].category != Category("air".into()) {
            bail!("First entry in voxel config must have category \"air\"");
        }

        let voxels_by_id = config.voxels.clone();

        let voxels_by_name = voxels_by_id
            .iter()
            .enumerate()
            .map(|(i, voxel_info)| {
                let name = voxel_info.name.clone();
                let voxel = Voxel::from_u16(i as u16);
                (name, (voxel, voxel_info.clone()))
            })
            .collect();

        for (voxel_id, voxel) in voxels_by_id.iter().enumerate() {
            log::info!("Voxel id {} is {}", voxel_id, voxel.name);
        }

        Ok(Self {
            voxels_by_name,
            voxels_by_id,
        })
    }

    pub fn voxel_by_name(&self, name: &str) -> Option<(Voxel, &VoxelInfo)> {
        let (voxel, voxel_info) = self.voxels_by_name.get(name)?;
        Some((*voxel, voxel_info))
    }

    pub fn voxel_info(&self, voxel: Voxel) -> Option<&VoxelInfo> {
        self.voxels_by_id.get(voxel.to_u16() as usize)
    }

    pub fn voxels(&self) -> &[VoxelInfo] {
        &self.voxels_by_id[..]
    }

    pub fn all_face_textures(&self) -> HashSet<FaceTexture> {
        let mut result = HashSet::new();

        for voxel in self.voxels() {
            match voxel.texture.clone() {
                VoxelTexture::Uniform(face_tex) => {
                    result.insert(face_tex);
                }
                VoxelTexture::Nonuniform { top, bottom, side } => {
                    result.insert(top);
                    result.insert(bottom);
                    result.insert(side);
                }
            }
        }

        result
    }
}
