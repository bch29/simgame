use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{ElementWise, EuclideanSpace, Point3};
use serde::{Deserialize, Serialize};
/// Types used to represent the voxel-based world.
use std::io::{self, Read, Write};
use std::slice;

use crate::octree::Octree;
use crate::util::Bounds;

pub mod index_utils;

/// Represents the value of a single block in the world. The wrapped value is an index into the
/// BlockConfig's list of BlockInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Block(u16);

/// A block category.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Category(String);

/// Specification of a block type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub name: String,
    pub category: Category,
    pub char_repr: char,
    #[serde(default = "BlockInfo::default_passable_through")]
    pub passable_through: bool,
    #[serde(default = "BlockInfo::default_passable_above")]
    pub passable_above: bool,
    #[serde(default = "BlockInfo::default_speed_modifier")]
    pub speed_modifier: f64,
}

impl BlockInfo {
    pub fn default_passable_through() -> bool {
        false
    }
    pub fn default_passable_above() -> bool {
        true
    }
    pub fn default_speed_modifier() -> f64 {
        1.0
    }
}

/// Specification of the selection of available blocks and categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    block_info: Vec<BlockInfo>,
}

/// Stores the world's blocks, but not other things like entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldBlockData {
    /// Storage for chunks.
    chunks: Octree<Chunk>,
}

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
    pub blocks: [Block; index_utils::chunk_size_total()],
}

impl std::fmt::Debug for Chunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Chunk {{")?;
        for z in 0..index_utils::chunk_size().z {
            writeln!(f, "z={}", z)?;
            for y in 0..index_utils::chunk_size().y {
                for x in 0..index_utils::chunk_size().x {
                    let ix = index_utils::pack_xyz(index_utils::chunk_size(), Point3 { x, y, z });
                    write!(f, "{}", self.blocks[ix].0)?;
                }
                writeln!(f)?;
            }
        }
        write!(f, "}}")
    }
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
}

impl Chunk {
    #[inline]
    pub fn get_block(&self, p: Point3<usize>) -> Block {
        self.blocks[index_utils::pack_within_chunk(p)]
    }

    #[inline]
    pub fn set_block(&mut self, p: Point3<usize>, val: Block) {
        self.blocks[index_utils::pack_within_chunk(p)] = val;
    }

    /// Creates a chunk filled with zeroes (i.e. empty blocks).
    pub fn empty() -> Chunk {
        let blocks = [Block(0); index_utils::chunk_size_total()];
        Chunk { blocks }
    }
}

impl std::cmp::PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..index_utils::chunk_size_total() {
            if self.blocks[i] != other.blocks[i] {
                return false;
            }
        }

        true
    }
}

impl Eq for Chunk {}

impl WorldBlockData {
    /// Returns the chunk in which the given block position resides.
    #[inline]
    pub fn get_chunk(&self, p: Point3<usize>) -> &Chunk {
        let (chunk_pos, _) = index_utils::to_chunk_pos(p);
        self.chunks
            .get(chunk_pos)
            .expect("requested chunk which is not present")
    }

    #[inline]
    pub fn get_block(&self, p: Point3<usize>) -> Block {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);
        let chunk = self
            .chunks
            .get(chunk_pos)
            .expect("requested chunk which is not present");
        chunk.get_block(inner_pos)
    }

    /// Visits every chunk in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_chunks(&self) -> impl Iterator<Item = (Point3<usize>, &Chunk)> + '_ {
        self.chunks.iter()
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks(&self) -> impl Iterator<Item = (Point3<usize>, Block)> + '_ {
        self.chunks.iter().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .blocks
                .iter()
                .enumerate()
                .map(move |(inner_index, block)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index));
                    (loc, *block)
                })
        })
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = (Point3<usize>, &mut Block)> + '_ {
        self.chunks.iter_mut().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .blocks
                .iter_mut()
                .enumerate()
                .map(move |(inner_index, block)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index));
                    (loc, block)
                })
        })
    }

    /// At every possible block position within the given bounds, replace the existing block
    /// using the given function.
    pub fn replace_blocks<E, F>(
        &mut self,
        bounds: Bounds<usize>,
        mut replace_block: F,
    ) -> Result<(), E>
    where
        F: FnMut(Point3<usize>, Block) -> Result<Block, E>,
    {
        while !self.bounds().contains_bounds(bounds) {
            self.chunks.grow(0);
        }

        let chunk_bounds = bounds.quantize_down(index_utils::chunk_size());

        // TODO: the possible positions in the Octree could be iterated over more efficiently
        // with a (not yet present) `replace_all_points` method.
        for chunk_pos in chunk_bounds.iter_points() {
            let chunk_start =
                chunk_pos.mul_element_wise(Point3::origin() + index_utils::chunk_size());

            let current_chunk = self.chunks.get_or_insert(chunk_pos, Chunk::empty);
            let mut count_nonempty = 0;

            for inner_pos in Bounds::from_size(index_utils::chunk_size()).iter_points() {
                let block_pos = chunk_start + (inner_pos - Point3::origin());
                if !bounds.contains_point(block_pos) {
                    continue;
                }

                let current_block = current_chunk.get_block(inner_pos);
                let new_block = replace_block(block_pos, current_block)?;
                current_chunk.set_block(inner_pos, new_block);

                if new_block != Block(0) {
                    count_nonempty += 1;
                }
            }

            if count_nonempty == 0 {
                self.chunks.remove(chunk_pos);
            }
        }
        Ok(())
    }

    /// Sets the value of a block at the given point.
    #[inline]
    pub fn set_block(&mut self, p: Point3<usize>, val: Block) {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);
        let chunk = self
            .chunks
            .get_mut(chunk_pos)
            .expect("requested chunk which is not present");
        chunk.set_block(inner_pos, val);
    }

    /// Returns a bounding box that is guaranteed to contain every block in the world. No
    /// guarantees are made about whether it is the smallest such bounding box.
    pub fn bounds(&self) -> Bounds<usize> {
        self.chunks.bounds().scale_up(index_utils::chunk_size())
    }

    /// Serialize the blocks in the world. Returns the number of bytes written.
    pub fn serialize_blocks<W>(&self, target: &mut W) -> io::Result<usize>
    where
        W: Write,
    {
        self.chunks.serialize(target, &mut |chunk, target| {
            let mut bytes_written = 0;
            for block in chunk.blocks.iter() {
                target.write_u16::<BigEndian>(block.0)?;
                bytes_written += 2;
            }
            Ok(bytes_written)
        })
    }

    /// Deserialize the world blocks from the given reader.
    pub fn deserialize_blocks<R: Read>(&mut self, src: &mut R) -> io::Result<()> {
        self.chunks = Octree::<Chunk>::deserialize(src, &mut |src| {
            let mut blocks = [Block(0u16); index_utils::chunk_size_total()];
            src.read_u16_into::<BigEndian>(blocks_to_u16_mut(&mut blocks))?;
            Ok(Chunk { blocks })
        })?;
        Ok(())
    }

    /// Constructs an empty WorldBlocks capable of holding blocks within the given bounds.
    pub fn empty(min_bounds: Bounds<usize>) -> WorldBlockData {
        WorldBlockData {
            chunks: Octree::new(bounds_to_octree_height(min_bounds)),
        }
    }

    pub fn debug_summary(&self) -> WorldBlockDataSummary {
        let size = self.bounds().size();
        let count_total = size.x * size.y * size.z;
        let mut count_nonempty = 0;
        for (_, block) in self.iter_blocks() {
            if !block.is_empty() {
                count_nonempty += 1;
            }
        }

        let pct_nonempty = (count_nonempty as f64 / count_total as f64) * 100.0;
        let byte_size = std::mem::size_of::<Block>() * count_total;
        let mb_size = byte_size / (1024 * 1024);

        WorldBlockDataSummary {
            count_total,
            count_nonempty,
            count_chunks: self.iter_chunks().count(),
            pct_nonempty,
            byte_size,
            mb_size,
            bounds: self.bounds(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct WorldBlockDataSummary {
    count_total: usize,
    count_nonempty: usize,
    count_chunks: usize,
    pct_nonempty: f64,
    byte_size: usize,
    mb_size: usize,
    bounds: Bounds<usize>,
}

#[inline]
pub fn blocks_to_u16(buf: &[Block]) -> &[u16] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

#[inline]
pub fn blocks_to_u16_mut(buf: &mut [Block]) -> &mut [u16] {
    unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len()) }
}

fn bounds_to_octree_height(min_bounds: Bounds<usize>) -> usize {
    let chunk_size = index_utils::chunk_size();
    let max_dim = [
        min_bounds.limit().x as f64 / chunk_size.x as f64,
        min_bounds.limit().y as f64 / chunk_size.y as f64,
        min_bounds.limit().z as f64 / chunk_size.z as f64,
    ]
    .iter()
    .copied()
    .fold(0.0, f64::max);

    1 + f64::max(max_dim.log2(), 0.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};

    use crate::util::Bounds;

    use super::*;

    #[test]
    fn test_bounds_to_octree_height() {
        let bounds = Bounds::from_size(
            index_utils::chunk_size() * 2
                - Vector3 {
                    x: 5,
                    y: 1,
                    z: index_utils::chunk_size().z - 2,
                },
        );
        assert_eq!(bounds_to_octree_height(bounds), 2);

        let bounds = Bounds::from_size(index_utils::chunk_size() * 2);
        assert_eq!(bounds_to_octree_height(bounds), 2);

        let bounds =
            Bounds::from_size(index_utils::chunk_size() * 2 + Vector3 { x: 1, y: 0, z: 0 });
        assert_eq!(bounds_to_octree_height(bounds), 3);
    }

    #[test]
    fn test_reserialize() {
        let bounds = Bounds::from_size(Vector3 {
            x: 32,
            y: 48,
            z: 64,
        });
        let mut buf = Vec::new();

        let mut original = WorldBlockData::empty(bounds);

        let points1 = vec![(0, 0, 0), (3, 4, 5), (24, 25, 26), (24, 35, 36)];

        let points2 = vec![(1, 2, 3), (27, 16, 5), (2, 2, 2), (31, 47, 63)];

        original
            .replace_blocks::<(), _>(bounds, |p, _| {
                Ok(if points1.contains(&p.into()) {
                    Block::from_u16(1)
                } else if points2.contains(&p.into()) {
                    Block::from_u16(257)
                } else {
                    Block::air()
                })
            })
            .unwrap();
        assert!(original.chunks.check_height_invariant());

        original.serialize_blocks(&mut buf).unwrap();

        let mut reserialized = WorldBlockData::empty(bounds);
        reserialized
            .deserialize_blocks(&mut buf.as_slice())
            .unwrap();

        assert_eq!(
            original.debug_summary().count_nonempty,
            reserialized.debug_summary().count_nonempty
        );

        let chunk_limit = Point3::origin() + index_utils::chunk_size();

        assert_eq!(
            Block::from_u16(1),
            dbg!(original
                .get_chunk(Point3::new(24, 35, 36)))
                .get_block(dbg!(Point3::new(24, 35, 36).rem_element_wise(chunk_limit)))
        );
        assert_eq!(
            Block::from_u16(257),
            reserialized
                .get_chunk(Point3::new(31, 47, 63))
                .get_block(Point3::new(31, 47, 63).rem_element_wise(chunk_limit))
        );

        assert_eq!(Block::from_u16(1), reserialized.get_block(Point3::origin()));
        assert_eq!(
            Block::from_u16(1),
            reserialized.get_block(Point3::new(24, 35, 36))
        );
        assert_eq!(
            Block::from_u16(257),
            reserialized.get_block(Point3::new(31, 47, 63))
        );

        let collect_blocks = |world: &WorldBlockData| -> Vec<(Point3<usize>, Block)> {
            world.iter_blocks().filter(|(_, b)| !b.is_empty()).collect()
        };

        assert!(reserialized.chunks.check_height_invariant());
        assert_eq!(collect_blocks(&original), collect_blocks(&reserialized));
        assert_eq!(original.chunks.height(), reserialized.chunks.height());

        assert_eq!(original, reserialized);
    }
}
