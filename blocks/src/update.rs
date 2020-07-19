use std::collections::HashSet;

use cgmath::Point3;

use crate::block_data::BlockData;
use crate::index_utils;
use crate::{Block, Chunk};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdatedBlocksState {
    pub modified_chunks: HashSet<Point3<i64>>,
}

#[derive(Debug)]
pub struct BlockUpdater<'a> {
    blocks: &'a mut BlockData,
    updated_state: &'a mut UpdatedBlocksState,
}

impl UpdatedBlocksState {
    pub fn empty() -> Self {
        Self {
            modified_chunks: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        self.modified_chunks.clear();
    }

    pub fn record_chunk_update(&mut self, chunk_pos: Point3<i64>) {
        self.modified_chunks.insert(chunk_pos);
    }

    pub fn update_from(&mut self, other: UpdatedBlocksState) {
        for chunk in other.modified_chunks.into_iter() {
            self.modified_chunks.insert(chunk);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.modified_chunks.is_empty()
    }
}

impl<'a> BlockUpdater<'a> {
    pub fn new(blocks: &'a mut BlockData, updated_state: &'a mut UpdatedBlocksState) -> Self {
        Self {
            blocks,
            updated_state,
        }
    }

    #[inline]
    pub fn set_block(&mut self, point: Point3<i64>, block: Block) {
        self.blocks.set_block(point, block);
        let (chunk_pos, _) = index_utils::to_chunk_pos(point);
        self.updated_state.record_chunk_update(chunk_pos);
    }

    /// Set multiple blocks in one pass. Can be more efficient than many calls of 'set_block' if
    /// the input iterator has good chunk locality (multiple blocks in the same chunk follow each
    /// other directly).
    #[inline]
    pub fn set_blocks<I>(&mut self, set_blocks: I)
    where
        I: IntoIterator<Item = (Point3<i64>, Block)>,
    {
        let chunks = self.blocks.chunks_mut();
        let mut iter = set_blocks.into_iter();

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
                chunks.grow();
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
