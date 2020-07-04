#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashSet;

use cgmath::{Point3, Vector3};
use rand::Rng;

use crate::block::{WorldBlockData, Block};
use crate::util::Bounds;

pub struct World {
    pub blocks: WorldBlockData,
    rng: rand::rngs::ThreadRng,
    updating: bool
}

pub struct UpdatedWorldState {
    pub modified_chunks: HashSet<Point3<usize>>
}

impl World {
    pub fn from_blocks(blocks: WorldBlockData) -> World {
        World {
            blocks,
            rng: rand::thread_rng(),
            updating: false
        }
    }

    /// Moves the world forward by one tick. Records anything that changed in the 'updated_state'.
    pub fn tick(&mut self, updated_state: &mut UpdatedWorldState) {
        if !self.updating {
            return;
        }

        let bounds = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));
        for _ in 0..64 {
            let point = bounds.origin()
                + Vector3 {
                    x: self.rng.gen::<usize>() % bounds.size().x,
                    y: self.rng.gen::<usize>() % bounds.size().y,
                    z: self.rng.gen::<usize>() % bounds.size().z,
                };
            self.blocks.set_block(point, Block::from_u16(1));
            let (chunk_pos, _) = crate::block::index_utils::to_chunk_pos(point);
            updated_state.record_chunk_update(chunk_pos);
        }
    }

    pub fn toggle_updates(&mut self) {
        self.updating = !self.updating;
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        UpdatedWorldState {
            modified_chunks: HashSet::new()
        }
    }

    pub fn clear(&mut self) {
        self.modified_chunks.clear();
    }

    fn record_chunk_update(&mut self, chunk_pos: Point3<usize>) {
        self.modified_chunks.insert(chunk_pos);
    }

    // pub fn 
}
