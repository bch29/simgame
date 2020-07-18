use crate::block::{WorldBlockData, UpdatedBlocksState};

#[derive(Debug)]
pub struct World {
    pub blocks: WorldBlockData,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub blocks: UpdatedBlocksState
}

impl World {
    pub fn new(blocks: WorldBlockData) -> World {
        World { blocks }
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

    pub fn update_from(&mut self, other: UpdatedWorldState) {
        self.blocks.update_from(other.blocks)
    }
}
