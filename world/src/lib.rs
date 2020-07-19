mod background_object;
pub mod lsystem;
pub mod tree;
pub mod turtle;
mod world_state;

use simgame_blocks::{UpdatedBlocksState, WorldBlockData};

pub use world_state::{WorldState, WorldStateBuilder};

#[derive(Debug)]
pub struct World {
    pub blocks: WorldBlockData,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub blocks: UpdatedBlocksState,
}

impl World {
    pub fn new(blocks: WorldBlockData) -> World {
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
