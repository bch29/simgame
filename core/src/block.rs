use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{Point3, Vector3};
use serde::{Deserialize, Serialize};
/// Types used to represent the voxel-based world.
use std::io::{self, Read, Write};
use std::slice;

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
#[derive(Clone, PartialEq, Eq)]
pub struct WorldBlockData {
    /// Number of chunks in each dimension
    count_chunks: Vector3<usize>,
    /// Storage for chunks.
    chunks: Vec<Chunk>,
}

impl std::fmt::Debug for WorldBlockData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "WorldBlocks {{ count_chunks: {:?}, chunks: <Vec<Chunk> of length {}> }}",
            self.count_chunks,
            self.chunks.len()
        )
    }
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
    fn empty() -> Chunk {
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
    pub fn get_chunk(&self, p: Point3<usize>) -> &Chunk {
        let (chunk_idx, _) = index_utils::to_chunk_index(self.count_chunks, p);
        &self.chunks[chunk_idx]
    }

    #[inline]
    pub fn get_block(&self, p: Point3<usize>) -> Block {
        let (chunk_idx, inner_pos) = index_utils::to_chunk_index(self.count_chunks, p);
        let chunk = &self.chunks[chunk_idx];
        chunk.get_block(inner_pos)
    }

    /// Visits every block in the world exaclty once, in an unspecified order.
    #[inline]
    pub fn iter_blocks(&self) -> impl Iterator<Item = Block> + '_ {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.blocks.iter().copied())
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> + '_ {
        self.chunks
            .iter_mut()
            .flat_map(|chunk| chunk.blocks.iter_mut().map(|block| block))
    }

    /// Visits every chunk in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_chunks_with_loc(&self) -> impl Iterator<Item = (Point3<usize>, &Chunk)> + '_ {
        let count_chunks = self.count_chunks;
        self.chunks.iter().enumerate().map(move |(chunk_index, chunk)| {
            let p = index_utils::unpack_xyz(count_chunks, chunk_index);
            (p, chunk)
        })
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks_with_loc(&self) -> impl Iterator<Item = (Point3<usize>, Block)> + '_ {
        let count_chunks = self.count_chunks;
        self.chunks
            .iter()
            .enumerate()
            .flat_map(move |(chunk_index, chunk)| {
                chunk
                    .blocks
                    .iter()
                    .enumerate()
                    .map(move |(inner_index, block)| {
                        let loc =
                            index_utils::unpack_index(count_chunks, (chunk_index, inner_index));
                        (loc, *block)
                    })
            })
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks_with_loc_mut(
        &mut self,
    ) -> impl Iterator<Item = (Point3<usize>, &mut Block)> + '_ {
        let count_chunks = self.count_chunks;
        self.chunks
            .iter_mut()
            .enumerate()
            .flat_map(move |(chunk_index, chunk)| {
                chunk
                    .blocks
                    .iter_mut()
                    .enumerate()
                    .map(move |(inner_index, block)| {
                        let loc =
                            index_utils::unpack_index(count_chunks, (chunk_index, inner_index));
                        (loc, block)
                    })
            })
    }

    /// Sets the value of a block at the given point.
    #[inline]
    pub fn set_block(&mut self, p: Point3<usize>, val: Block) {
        let (chunk_idx, inner_pos) = index_utils::to_chunk_index(self.count_chunks, p);
        let chunk = &mut self.chunks[chunk_idx];
        chunk.set_block(inner_pos, val);
    }
    /// Returns the full size of the world, in number of blocks in each dimension.
    #[inline]
    pub fn size(&self) -> Vector3<usize> {
        index_utils::make_size(self.count_chunks)
    }

    pub fn get_count_chunks(&self) -> Vector3<usize> {
        self.count_chunks
    }

    /// Serialize the blocks in the world in a chunk-size-independent manner.
    pub fn serialize_blocks<W>(&self, target: &mut W) -> io::Result<()>
    where
        W: Write,
    {
        for z in 0..self.size().z {
            for y in 0..self.size().y {
                for x in 0..self.size().x {
                    let p = Point3 { x, y, z };
                    target.write_u16::<LittleEndian>(self.get_block(p).0)?;
                }
            }
        }
        Ok(())
    }

    /// Deserialize a chunk-size-independent buffer of block values into a list of chunks.
    /// The source buffer must contain at least count_chunks.x * count_chunks.y * CHUNK_SIZE_TOTAL
    /// elements. The target buffer must contain at least count_chunks.x * count_chunks.y elements.
    fn deserialize_chunks<R>(
        src: &mut R,
        count_chunks: Vector3<usize>,
        target: &mut [Chunk],
    ) -> io::Result<()>
    where
        R: Read,
    {
        let size = index_utils::make_size(count_chunks);
        assert!(target.len() >= count_chunks.x * count_chunks.y * count_chunks.z);

        let mut buf = [0u16; index_utils::chunk_size().x];

        for z in 0..size.z {
            for y in 0..size.y {
                for chunk_x in 0..count_chunks.x {
                    let start_pos = Point3 {
                        x: chunk_x * index_utils::chunk_size().x,
                        y,
                        z,
                    };

                    let (chunk_idx, inner_start_idx) =
                        index_utils::pack_index(count_chunks, start_pos);
                    let inner_end_idx = inner_start_idx + index_utils::chunk_size().x;

                    src.read_u16_into::<LittleEndian>(&mut buf)?;
                    let blocks = blocks_from_u16(&buf);
                    target[chunk_idx].blocks[inner_start_idx..inner_end_idx]
                        .copy_from_slice(blocks);
                }
            }
        }
        Ok(())
    }

    /// Deserialize a chunk-size-independent buffer of block values into this world's chunk list.
    /// The source buffer must contain at least count_chunks.x * count_chunks.y * CHUNK_SIZE_TOTAL
    /// elements. The target buffer must contain at least count_chunks.x * count_chunks.y elements.
    pub fn deserialize_blocks<R: Read>(&mut self, src: &mut R) -> io::Result<()> {
        Self::deserialize_chunks(src, self.count_chunks, &mut *self.chunks)
    }

    /// Constructs a WorldBlocks with the given size and block config, which is initially empty.
    pub fn empty(count_chunks: Vector3<usize>) -> WorldBlockData {
        let empty_chunk = Chunk::empty();
        let empty_chunks: Vec<_> = (0..count_chunks.x * count_chunks.y * count_chunks.z)
            .map(|_| empty_chunk.clone())
            .collect();

        WorldBlockData {
            count_chunks,
            chunks: empty_chunks,
        }
    }

    pub fn debug_summary(&self) -> WorldBlockDataSummary {
        let count_total = self.size().x * self.size().y * self.size().z;
        let count_empty = self.iter_blocks().filter(|block| block.is_empty()).count();
        let pct_empty = (count_empty as f64 / count_total as f64) * 100.0;
        let byte_size = std::mem::size_of::<Chunk>() * self.chunks.len();
        let mb_size = byte_size / (1024 * 1024);

        WorldBlockDataSummary {
            count_total,
            count_empty,
            pct_empty,
            byte_size,
            mb_size,
            size: self.size()
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct WorldBlockDataSummary {
    count_total: usize,
    count_empty: usize,
    pct_empty: f64,
    byte_size: usize,
    mb_size: usize,
    size: Vector3<usize>
}

#[inline]
fn blocks_from_u16(buf: &[u16]) -> &[Block] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const Block, buf.len()) }
}

#[inline]
pub fn blocks_to_u16(buf: &[Block]) -> &[u16] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use super::*;

    #[test]
    fn test_reserialize() {
        let size = Vector3 { x: 2, y: 3, z: 4 };
        let mut buf = Vec::new();

        let mut original = WorldBlockData::empty(size);

        let points1 = vec![(0, 0, 0), (3, 4, 5), (24, 25, 26), (24, 35, 36)];

        let points2 = vec![(1, 2, 3), (27, 16, 5), (2, 2, 2), (31, 47, 63)];

        for (p, block) in original.iter_blocks_with_loc_mut() {
            if points1.contains(&p.into()) {
                *block = Block::from_u16(1)
            } else if points2.contains(&p.into()) {
                *block = Block::from_u16(257)
            }
        }

        original.serialize_blocks(&mut buf).unwrap();

        let mut reserialized = WorldBlockData::empty(size);
        reserialized
            .deserialize_blocks(&mut buf.as_slice())
            .unwrap();

        assert_eq!(
            original.debug_summary().count_empty,
            reserialized.debug_summary().count_empty
        );

        assert_eq!(
            Block::from_u16(1),
            original
                .get_chunk(Point3::new(24, 35, 36))
                .get_block(Point3::new(8, 3, 4))
        );
        assert_eq!(
            Block::from_u16(1),
            reserialized
                .get_chunk(Point3::new(24, 35, 36))
                .get_block(Point3::new(8, 3, 4))
        );

        assert_eq!(
            Block::from_u16(1),
            reserialized.get_block(Point3::new(0, 0, 0))
        );
        assert_eq!(
            Block::from_u16(1),
            reserialized.get_block(Point3::new(24, 35, 36))
        );
        assert_eq!(
            Block::from_u16(257),
            reserialized.get_block(Point3::new(31, 47, 63))
        );

        assert_eq!(original, reserialized);
    }
}
