use crate::block::{WorldBlockData, UpdatedBlocksState, BlockConfigHelper};

#[derive(Debug)]
pub struct World {
    pub blocks: WorldBlockData,
    pub block_helper: BlockConfigHelper,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub blocks: UpdatedBlocksState
}

impl World {
    pub fn new(blocks: WorldBlockData, block_helper: BlockConfigHelper) -> World {
        World { blocks, block_helper }
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        Self {
            blocks: UpdatedBlocksState::empty()
        }
    }

    pub fn clear(&mut self) {
        self.blocks.clear()
    }
}
