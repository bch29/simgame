use std::slice;

use cgmath::{EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::config::BlockConfigHelper;
use crate::convert_bounds;
use crate::index_utils;
use crate::ray::{BlockRaycastHit, Ray};
use crate::util::Bounds;

/// Represents the value of a single block in the world. The wrapped value is an index into the
/// BlockConfig's list of BlockInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Block(u16);

/// Encapsulates a single chunk of blocks. The world is split into chunks to help with cache
/// locality: if there were no chunks, then any movement in at least one of the dimensions
/// (probably z) would always result in moving to a different cache line. Under the chunking
/// system, blocks that are close to each other in space, and are in the same chunk, should be in
/// the same cache line.
#[derive(Clone)]
#[repr(transparent)]
pub struct Chunk {
    /* Memory layout (if chunk size were 4x3x2)
    z-level 0
       x - >
     y 0  1  2  3
     | 4  5  6  7
     v 8  9  10 11

     z-level 1
       x - >
     y 12 13 14 15
     | 16 17 18 19
     v 20 21 22 23
     */
    pub blocks: [Block; index_utils::chunk_size_total() as usize],
}

impl Block {
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn from_u16(val: u16) -> Self {
        Block(val)
    }

    pub fn air() -> Self {
        Block(0)
    }

    pub(crate) fn to_u16(self) -> u16 {
        self.0
    }
}

impl Chunk {
    #[inline]
    pub fn get_block(&self, p: Point3<i64>) -> Block {
        self.blocks[index_utils::pack_within_chunk(p) as usize]
    }

    #[inline]
    pub fn set_block(&mut self, p: Point3<i64>, val: Block) {
        self.blocks[index_utils::pack_within_chunk(p) as usize] = val;
    }

    /// Creates a chunk filled with zeroes (i.e. empty blocks).
    pub fn empty() -> Chunk {
        let blocks = [Block(0); index_utils::chunk_size_total() as usize];
        Chunk { blocks }
    }

    pub fn cast_ray(
        &self,
        ray: &Ray<f64>,
        origin: Point3<i64>,
        _block_helper: &BlockConfigHelper,
    ) -> Option<BlockRaycastHit> {
        // TODO: use space subdivision to avoid looping through every single block
        // TODO: use collision info in block config to handle blocks that are not full cubes

        let mut nearest_hit: Option<BlockRaycastHit> = None;

        for z in 0..index_utils::chunk_size().z {
            for y in 0..index_utils::chunk_size().y {
                for x in 0..index_utils::chunk_size().x {
                    let block_offset = Vector3::new(x, y, z);
                    let block = self.get_block(Point3::origin() + block_offset);

                    if block.is_empty() {
                        continue;
                    }

                    let block_pos = origin + block_offset;
                    let bounds =
                        convert_bounds!(Bounds::new(block_pos, Vector3::new(1, 1, 1)), f64);
                    let hit = match bounds.cast_ray(ray).entry() {
                        Some(intersection) => BlockRaycastHit {
                            block,
                            block_pos,
                            intersection,
                        },
                        None => continue,
                    };

                    nearest_hit = match nearest_hit.take() {
                        Some(old_hit) if old_hit.intersection.t < hit.intersection.t => {
                            Some(old_hit)
                        }
                        _ => Some(hit),
                    }
                }
            }
        }

        nearest_hit
    }
}

#[inline]
pub fn blocks_to_u16(buf: &[Block]) -> &[u16] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

#[inline]
pub fn blocks_to_u16_mut(buf: &mut [Block]) -> &mut [u16] {
    unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len()) }
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Chunk {{")?;
        for z in 0..index_utils::chunk_size().z {
            writeln!(f, "z={}", z)?;
            for y in 0..index_utils::chunk_size().y {
                for x in 0..index_utils::chunk_size().x {
                    let ix = index_utils::pack_xyz(index_utils::chunk_size(), Point3 { x, y, z });
                    write!(f, "{}", self.blocks[ix as usize].0)?;
                }
                writeln!(f)?;
            }
        }
        write!(f, "}}")
    }
}

impl std::cmp::PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..index_utils::chunk_size_total() {
            if self.blocks[i as usize] != other.blocks[i as usize] {
                return false;
            }
        }

        true
    }
}

impl Eq for Chunk {}
