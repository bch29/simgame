use std::collections::{hash_map, HashMap};

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::Block;

/// A block category.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Category(String);

/// Specification of a block type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub name: String,
    pub category: Category,
    #[serde(default = "BlockInfo::default_passable_through")]
    pub passable_through: bool,
    #[serde(default = "BlockInfo::default_passable_above")]
    pub passable_above: bool,
    #[serde(default = "BlockInfo::default_speed_modifier")]
    pub speed_modifier: f64,

    pub texture: BlockTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FaceTexture {
    /// The face has a soid color.
    SolidColor { red: u8, green: u8, blue: u8 },
    /// The face has a texture with the given resource name.
    Texture {
        resource: String,
        /// The texture repeats after this many blocks
        periodicity: u32,
    },
}

/// Specification of how a block is textured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockTexture {
    /// The block has the given face texture on all faces.
    Uniform(FaceTexture),
    /// The block has the given face textures on corresponding faces.
    Nonuniform {
        top: FaceTexture,
        bottom: FaceTexture,
        side: FaceTexture,
    },
}

/// Specification of the selection of available blocks and categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    blocks: Vec<BlockInfo>,
}

#[derive(Debug, Clone)]
pub struct BlockConfigHelper {
    blocks_by_name: HashMap<String, (Block, BlockInfo)>,
    blocks_by_id: Vec<BlockInfo>,
}

impl BlockInfo {
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

impl BlockConfigHelper {
    pub fn new(config: &BlockConfig) -> Result<Self> {
        if config.blocks.is_empty() || config.blocks[0].category != Category("air".into()) {
            bail!("First entry in block config must have category \"air\"");
        }

        let blocks_by_id = config.blocks.clone();

        let blocks_by_name = blocks_by_id
            .iter()
            .enumerate()
            .map(|(i, block_info)| {
                let name = block_info.name.clone();
                let block = Block::from_u16(i as u16);
                (name, (block, block_info.clone()))
            })
            .collect();

        for (block_id, block) in blocks_by_id.iter().enumerate() {
            log::info!("Block id {} is {}", block_id, block.name);
        }

        Ok(Self {
            blocks_by_name,
            blocks_by_id,
        })
    }

    pub fn block_by_name(&self, name: &str) -> Option<(Block, &BlockInfo)> {
        let (block, block_info) = self.blocks_by_name.get(name)?;
        Some((*block, block_info))
    }

    pub fn block_info(&self, block: Block) -> Option<&BlockInfo> {
        self.blocks_by_id.get(block.to_u16() as usize)
    }

    pub fn blocks(&self) -> &[BlockInfo] {
        &self.blocks_by_id[..]
    }

    pub fn texture_index_map(&self) -> HashMap<FaceTexture, usize> {
        let mut index_map: HashMap<FaceTexture, usize> = HashMap::new();
        let mut index = 0;

        let mut insert = |face_tex: FaceTexture| -> usize {
            let entry = index_map.entry(face_tex);
            match entry {
                hash_map::Entry::Occupied(entry) => *entry.get(),
                hash_map::Entry::Vacant(entry) => {
                    let res = *entry.insert(index);
                    index += 1;
                    res
                }
            }
        };

        for block in self.blocks() {
            match block.texture.clone() {
                BlockTexture::Uniform(face_tex) => {
                    let index = insert(face_tex);
                    log::info!(
                        "Block {} gets a uniform texture with index {}",
                        block.name,
                        index
                    );
                }
                BlockTexture::Nonuniform { top, bottom, side } => {
                    let index_top = insert(top);
                    let index_bottom = insert(bottom);
                    let index_side = insert(side);

                    log::info!(
                        "Block {} gets a non-uniform texture with indices {}/{}/{}",
                        block.name,
                        index_top,
                        index_bottom,
                        index_side
                    );
                }
            }
        }

        index_map
    }
}
