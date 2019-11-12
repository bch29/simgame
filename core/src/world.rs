#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashSet;

use cgmath::Point3;

use crate::block::WorldBlockData;

pub struct World {
    pub blocks: WorldBlockData,
}

pub struct UpdatedWorldState {
    pub modified_chunks: HashSet<Point3<usize>>
}

impl World {
    /// Moves the world forward by one tick. Records anything that changed in the 'updated_state'.
    pub fn tick(&mut self, updated_state: &mut UpdatedWorldState) {
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        UpdatedWorldState {
            modified_chunks: HashSet::new()
        }
    }

    fn clear(&mut self) {
        self.modified_chunks.clear();
    }

    fn record_chunk_update(&mut self, chunk_pos: Point3<usize>) {
        self.modified_chunks.insert(chunk_pos);
    }
}
