#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashSet;

use cgmath::{Point3, Vector3};
use rand::Rng;

use crate::util::Bounds;
use crate::block::{index_utils, Block, WorldBlockData};

pub struct World {
    pub blocks: WorldBlockData,
    rng: rand::rngs::ThreadRng,
    updating: bool,
    filled_blocks: i32,
}

pub struct UpdatedWorldState {
    pub modified_chunks: HashSet<Point3<usize>>,
}

impl World {
    pub fn from_blocks(blocks: WorldBlockData) -> World {
        World {
            blocks,
            rng: rand::thread_rng(),
            updating: false,
            filled_blocks: (16 * 16 * 4) / 8,
        }
    }

    /// Moves the world forward by one tick. Records anything that changed in the 'updated_state'.
    pub fn tick(&mut self, updated_state: &mut UpdatedWorldState) {
        if !self.updating {
            return;
        }

        let bounds: Bounds<usize> =
            Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        for _ in 0..64 {
            let point = bounds.origin()
                + Vector3 {
                    x: self.rng.gen::<usize>() % bounds.size().x,
                    y: self.rng.gen::<usize>() % bounds.size().y,
                    z: self.rng.gen::<usize>() % bounds.size().z,
                };
            self.blocks.set_block(point, Block::from_u16(1));
            let (chunk_pos, _) = index_utils::to_chunk_pos(point);
            updated_state.record_chunk_update(chunk_pos);
        }
    }

    pub fn modify_filled_blocks(&mut self, delta: i32, updated_state: &mut UpdatedWorldState) {
        self.filled_blocks += delta * 8;
        if self.filled_blocks <= 0 {
            self.filled_blocks = 0
        } else if self.filled_blocks >= index_utils::chunk_size_total() as i32 {
            self.filled_blocks = index_utils::chunk_size_total() as i32
        }

        self.tick(updated_state);

        let bounds: Bounds<usize> =
            Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_blocks as usize;
        let mut count_filled = 0;

        for p in bounds.iter_points() {
            if (p.x + p.y + p.z) % step as usize == 0 {
                self.blocks.set_block(p, Block::from_u16(1));
                count_filled += 1;
            } else {
                self.blocks.set_block(p, Block::from_u16(0));
            }
            let (chunk_pos, _) = index_utils::to_chunk_pos(p);
            updated_state.record_chunk_update(chunk_pos);
        }

        log::info!(
            "Setting number of filled blocks to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );
    }

    pub fn toggle_updates(&mut self) {
        self.updating = !self.updating;
    }
}

impl UpdatedWorldState {
    pub fn empty() -> Self {
        UpdatedWorldState {
            modified_chunks: HashSet::new(),
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
