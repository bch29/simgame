use std::collections::HashSet;

use cgmath::Point3;

use crate::index_utils;
use crate::voxel_data::VoxelData;
use crate::{Chunk, Voxel};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelDelta {
    pub modified_chunks: HashSet<Point3<i64>>,
}

#[derive(Debug)]
pub struct VoxelUpdater<'a> {
    voxels: &'a mut VoxelData,
    updated_state: &'a mut VoxelDelta,
}

impl VoxelDelta {
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

    pub fn update_from(&mut self, other: VoxelDelta) {
        for chunk in other.modified_chunks.into_iter() {
            self.modified_chunks.insert(chunk);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.modified_chunks.is_empty()
    }
}

impl<'a> VoxelUpdater<'a> {
    pub fn new(voxels: &'a mut VoxelData, updated_state: &'a mut VoxelDelta) -> Self {
        Self {
            voxels,
            updated_state,
        }
    }

    #[inline]
    pub fn set_voxel(&mut self, point: Point3<i64>, voxel: Voxel) {
        self.voxels.set_voxel(point, voxel);
        let (chunk_pos, _) = index_utils::to_chunk_pos(point);
        self.updated_state.record_chunk_update(chunk_pos);
    }

    /// Set multiple voxels in one pass. Can be more efficient than many calls of 'set_voxel' if
    /// the input iterator has good chunk locality (multiple voxels in the same chunk follow each
    /// other directly).
    #[inline]
    pub fn set_voxels<I>(&mut self, set_voxels: I)
    where
        I: IntoIterator<Item = (Point3<i64>, Voxel)>,
    {
        let chunks = self.voxels.chunks_mut();
        let mut iter = set_voxels.into_iter();

        let ((mut chunk_pos, mut first_local_point), mut first_voxel) =
            if let Some((point, voxel)) = iter.next() {
                (index_utils::to_chunk_pos(point), voxel)
            } else {
                return;
            };

        let mut done = false;

        while !done {
            // Only do one hash map insert and one octree chunk lookup when we start a run of
            // voxels in a new chunk. While we are in a chunk, everything gets to be plain array
            // operations.

            while !chunks.bounds().contains_point(chunk_pos) {
                chunks.grow();
            }

            let chunk = chunks.get_or_insert(chunk_pos, Chunk::empty);
            self.updated_state.record_chunk_update(chunk_pos);
            chunk.set_voxel(first_local_point, first_voxel);

            done = true;

            for (point, voxel) in &mut iter {
                let (new_chunk_pos, local_point) = index_utils::to_chunk_pos(point);
                if new_chunk_pos != chunk_pos {
                    chunk_pos = new_chunk_pos;
                    first_local_point = local_point;
                    first_voxel = voxel;
                    done = false;
                    break;
                }

                chunk.set_voxel(local_point, voxel);
            }
        }
    }
}
