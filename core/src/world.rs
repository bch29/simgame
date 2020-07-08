#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashSet;

use cgmath::{Point3, Vector3};
use rand::Rng;

use crate::block::{index_utils, Block, Chunk, WorldBlockData};
use crate::util::Bounds;
use crate::worldgen::primitives::{self, Primitive};

#[derive(Debug)]
pub struct World {
    pub blocks: WorldBlockData,
    rng: rand::rngs::ThreadRng,
    updating: bool,
    filled_blocks: i32,
}

#[derive(Debug)]
pub struct UpdatedWorldState {
    pub modified_chunks: HashSet<Point3<usize>>,
}

#[derive(Debug)]
pub struct BlockUpdater<'a> {
    blocks: &'a mut WorldBlockData,
    updated_state: &'a mut UpdatedWorldState,
}

impl<'a> BlockUpdater<'a> {
    pub fn new(blocks: &'a mut WorldBlockData, updated_state: &'a mut UpdatedWorldState) -> Self {
        Self {
            blocks,
            updated_state,
        }
    }

    #[inline]
    pub fn set_block(&mut self, point: Point3<usize>, block: Block) {
        self.blocks.set_block(point, block);
        let (chunk_pos, _) = index_utils::to_chunk_pos(point);
        self.updated_state.record_chunk_update(chunk_pos);
    }

    /// Set multiple blocks in one pass. Can be more efficient than many calls of 'set_block' if
    /// the input iterator has good chunk locality (multiple blocks in the same chunk follow each
    /// other directly).
    #[inline]
    pub fn set_blocks<I>(&mut self, blocks: I)
    where
        I: IntoIterator<Item = (Point3<usize>, Block)>,
    {
        let chunks = self.blocks.chunks_mut();
        let mut iter = blocks.into_iter();

        let ((mut chunk_pos, mut first_local_point), mut first_block) =
            if let Some((point, block)) = iter.next() {
                (index_utils::to_chunk_pos(point), block)
            } else {
                return;
            };

        let mut done = false;

        while !done {
            // Only do one hash map insert and one octree chunk lookup when we start a run of
            // blocks in a new chunk. While we are in a chunk, everything gets to be plain array
            // operations.

            while !chunks.bounds().contains_point(chunk_pos) {
                chunks.grow(0);
            }

            let chunk = chunks.get_or_insert(chunk_pos, Chunk::empty);
            self.updated_state.record_chunk_update(chunk_pos);
            chunk.set_block(first_local_point, first_block);

            done = true;

            while let Some((point, block)) = iter.next() {
                let (new_chunk_pos, local_point) = index_utils::to_chunk_pos(point);
                if new_chunk_pos != chunk_pos {
                    chunk_pos = new_chunk_pos;
                    first_local_point = local_point;
                    first_block = block;
                    done = false;
                    break;
                }

                chunk.set_block(local_point, block);
            }
        }
    }
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

        // let boxes = vec![
        //     Bounds::new(Point3::new(32., 32., 0.), Vector3::new(16., 16., 1024.)),
        //     Bounds::new(Point3::new(100., 100., 0.), Vector3::new(17., 33., 400.)),
        //     Bounds::new(Point3::new(200., 200., 0.), Vector3::new(8., 5., 1024.)),
        //     Bounds::new(Point3::new(60., 100., 0.), Vector3::new(16., 16., 1024.)),
        //     Bounds::new(Point3::new(0., 400., 0.), Vector3::new(3000., 4., 8000.)),
        // ];

        // let shapes: Vec<Box<dyn Primitive>> = boxes
        //     .into_iter()
        //     .map(|bounds| Box::new(primitives::Box { bounds }) as Box<dyn Primitive>)
        //     .collect();
        let shapes: Vec<Box<dyn Primitive>> = vec![
            Box::new(primitives::FilledLine {
                start: Point3::new(32., 32., 128.),
                end: Point3::new(256., 256., 384.),
                radius: 30.,
                round_start: false,
                round_end: true
            })
        ];

        let mut updater = BlockUpdater::new(&mut self.blocks, updated_state);

        // for _ in 0..256 {
        let shape = &*shapes[self.rng.gen::<usize>() % shapes.len()];
        shape.draw(&mut updater, Block::from_u16(4));
        self.updating = false;

        // let point = bounds.origin()
        //     + Vector3 {
        //         x: self.rng.gen::<usize>() % bounds.size().x,
        //         y: self.rng.gen::<usize>() % bounds.size().y,
        //         z: self.rng.gen::<usize>() % bounds.size().z,
        //     };
        // self.blocks.set_block(point, Block::from_u16(1));
        // let (chunk_pos, _) = index_utils::to_chunk_pos(point);
        // updated_state.record_chunk_update(chunk_pos);
        // }
    }

    pub fn modify_filled_blocks(&mut self, delta: i32, updated_state: &mut UpdatedWorldState) {
        self.filled_blocks += delta * 8;
        if self.filled_blocks < 1 {
            self.filled_blocks = 1
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
