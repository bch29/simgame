pub use simgame_blocks::{BlockData, UpdatedBlocksState};
pub use simgame_util::bsp::Bsp;

pub mod entity;

pub use entity::{Entity, EntityConfig, EntityModel, EntityModelInfo};

#[derive(Debug)]
pub struct World {
    pub blocks: BlockData,
    pub entities: Bsp<Entity>,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub blocks: UpdatedBlocksState,
}

impl World {
    pub fn new(blocks: BlockData) -> World {
        World {
            blocks,
            entities: Bsp::new(),
        }
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        Self {
            blocks: UpdatedBlocksState::empty(),
        }
    }

    pub fn clear(&mut self) {
        self.blocks.clear()
    }

    pub fn update_from(&mut self, other: UpdatedWorldState) {
        self.blocks.update_from(other.blocks)
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}
