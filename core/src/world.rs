use std::collections::HashSet;

use cgmath::Point3;

use crate::block::{index_utils, Block, Chunk, WorldBlockData};

#[derive(Debug)]
pub struct World {
    pub blocks: WorldBlockData,
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
        World { blocks }
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

    pub fn record_chunk_update(&mut self, chunk_pos: Point3<usize>) {
        self.modified_chunks.insert(chunk_pos);
    }

    // pub fn
}
