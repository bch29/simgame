pub use simgame_blocks::{BlockData, UpdatedBlocksState};

pub mod entity;

#[derive(Debug)]
pub struct World {
    pub blocks: BlockData,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub blocks: UpdatedBlocksState,
}

impl World {
    pub fn new(blocks: BlockData) -> World {
        World { blocks }
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
