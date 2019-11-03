use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{Point3, Vector3};
use serde::{Deserialize, Serialize};
/// Types used to represent the voxel-based world.
use std::io::{self, Read, Write};
use std::slice;

pub const CHUNK_SIZE_X: usize = 16;
pub const CHUNK_SIZE_Y: usize = 16;
pub const CHUNK_SIZE_Z: usize = 16;
pub const CHUNK_SIZE_XY: usize = CHUNK_SIZE_X * CHUNK_SIZE_Y;
pub const CHUNK_SIZE_TOTAL: usize = CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z;

/// Represents the value of a single block in the world. The wrapped value is an index into the
/// BlockConfig's list of BlockInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub struct WorldBlocks {
    /// Number of chunks in each dimension
    count_chunks: Vector3<usize>,
    /// Storage for chunks.
    chunks: Vec<Chunk>,
}

impl std::fmt::Debug for WorldBlocks {
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
    blocks: [Block; CHUNK_SIZE_TOTAL],
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
        let blocks = [Block(0); CHUNK_SIZE_TOTAL];
        Chunk { blocks }
    }
}

impl std::cmp::PartialEq for Chunk {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..CHUNK_SIZE_TOTAL {
            if self.blocks[i] != other.blocks[i] {
                return false;
            }
        }

        true
    }
}

impl Eq for Chunk {}

impl WorldBlocks {
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

        let mut buf: [u16; CHUNK_SIZE_X] = [0; CHUNK_SIZE_X];

        for z in 0..size.z {
            for y in 0..size.y {
                for chunk_x in 0..count_chunks.x {
                    let start_pos = Point3 {
                        x: chunk_x * CHUNK_SIZE_X,
                        y,
                        z,
                    };

                    let (chunk_idx, inner_start_idx) =
                        index_utils::pack_index(count_chunks, start_pos);
                    let inner_end_idx = inner_start_idx + CHUNK_SIZE_X;

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
    pub fn empty(count_chunks: Vector3<usize>) -> WorldBlocks {
        let empty_chunk = Chunk::empty();
        let empty_chunks: Vec<_> = (0..count_chunks.x * count_chunks.y * count_chunks.z)
            .map(|_| empty_chunk.clone())
            .collect();

        WorldBlocks {
            count_chunks,
            chunks: empty_chunks,
        }
    }

    pub fn debug_summary(&self) -> WorldBlocksSummary {
        let count_total = self.size().x * self.size().y * self.size().z;
        let count_empty = self.iter_blocks().filter(|block| block.is_empty()).count();
        let pct_empty = (count_empty as f64 / count_total as f64) * 100.0;
        let byte_size = std::mem::size_of::<Chunk>() * self.chunks.len();
        let mb_size = byte_size / (1024 * 1024);

        WorldBlocksSummary {
            count_total,
            count_empty,
            pct_empty,
            byte_size,
            mb_size,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct WorldBlocksSummary {
    count_total: usize,
    count_empty: usize,
    pct_empty: f64,
    byte_size: usize,
    mb_size: usize,
}

#[inline]
fn blocks_from_u16(buf: &[u16]) -> &[Block] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const Block, buf.len()) }
}

pub mod index_utils {
    use crate::block::{CHUNK_SIZE_X, CHUNK_SIZE_XY, CHUNK_SIZE_Y, CHUNK_SIZE_Z};
    use cgmath::{ElementWise, Point3, Vector3};

    #[inline]
    pub fn chunk_size() -> Vector3<usize> {
        Vector3 {
            x: CHUNK_SIZE_X,
            y: CHUNK_SIZE_Y,
            z: CHUNK_SIZE_Z,
        }
    }

    #[inline]
    pub fn pack_xyz(bounds: Vector3<usize>, p: Point3<usize>) -> usize {
        p.x + p.y * bounds.x + p.z * bounds.x * bounds.y
    }

    #[inline]
    pub fn unpack_xyz(bounds: Vector3<usize>, index: usize) -> Point3<usize> {
        let xy = index % (bounds.x * bounds.y);

        Point3 {
            x: xy % bounds.x,
            y: xy / bounds.x,
            z: index / (bounds.x * bounds.y),
        }
    }

    /// indices is (chunk_index, inner_index)
    #[inline]
    pub fn unpack_index(count_chunks: Vector3<usize>, indices: (usize, usize)) -> Point3<usize> {
        let (chunk_index, inner_index) = indices;

        let origin = Point3::from((0, 0, 0));
        let chunk_pos = unpack_xyz(count_chunks, chunk_index);
        let inner_pos = unpack_xyz(chunk_size(), inner_index);
        let inner_offset = inner_pos - origin;
        chunk_pos.mul_element_wise(origin + chunk_size()) + inner_offset
    }

    /// From a point in block coordinates, return the chunk index and the position of the block
    /// within that chunk.
    #[inline]
    pub fn to_chunk_index(
        count_chunks: Vector3<usize>,
        p: Point3<usize>,
    ) -> (usize, Point3<usize>) {
        let size = make_size(count_chunks);
        assert!(p.x < size.x);
        assert!(p.y < size.y);
        assert!(p.z < size.z);

        let origin = Point3::from((0, 0, 0));
        let inner_pos = p.rem_element_wise(origin + chunk_size());
        let chunk_pos = p.div_element_wise(origin + chunk_size());
        let chunk_idx = pack_xyz(count_chunks, chunk_pos);
        (chunk_idx, inner_pos)
    }

    #[inline]
    pub fn pack_within_chunk(p: Point3<usize>) -> usize {
        assert!(p.x < CHUNK_SIZE_X);
        assert!(p.y < CHUNK_SIZE_Y);
        assert!(p.z < CHUNK_SIZE_Z);
        p.x + p.y * CHUNK_SIZE_X + p.z * CHUNK_SIZE_XY
    }

    /// Returns (chunk_index, inner_index),
    #[inline]
    pub fn pack_index(count_chunks: Vector3<usize>, p: Point3<usize>) -> (usize, usize) {
        let (chunk_index, inner_point) = to_chunk_index(count_chunks, p);
        let inner_index = pack_within_chunk(inner_point);
        (chunk_index, inner_index)
    }

    #[inline]
    pub fn make_size(count_chunks: Vector3<usize>) -> Vector3<usize> {
        count_chunks.mul_element_wise(chunk_size())
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use super::*;

    #[test]
    fn test_pack_index() {
        let count_chunks = Vector3 { x: 17, y: 11, z: 7 };

        assert_eq!(
            (0, 0),
            index_utils::pack_index(count_chunks, Point3 { x: 0, y: 0, z: 0 })
        );

        assert_eq!(
            (0, 7 + 3 * CHUNK_SIZE_X),
            index_utils::pack_index(count_chunks, Point3 { x: 7, y: 3, z: 0 })
        );

        assert_eq!(
            (
                2 + 8 * count_chunks.x + 3 * count_chunks.x * count_chunks.y,
                5 + 4 * CHUNK_SIZE_X + 12 * CHUNK_SIZE_XY
            ),
            index_utils::pack_index(
                count_chunks,
                Point3 {
                    x: 37,
                    y: 132,
                    z: 60
                }
            )
        );
    }

    #[test]
    fn test_unpack_index() {
        let count_chunks = Vector3 { x: 16, y: 16, z: 8 };

        assert_eq!(
            Point3 { x: 7, y: 3, z: 0 },
            index_utils::unpack_xyz(index_utils::chunk_size(), 7 + 3 * CHUNK_SIZE_X)
        );

        let check_point = |p| {
            assert_eq!(
                p,
                index_utils::unpack_index(count_chunks, index_utils::pack_index(count_chunks, p))
            );
        };

        check_point(Point3 { x: 0, y: 0, z: 0 });
        check_point(Point3 { x: 7, y: 3, z: 0 });
        check_point(Point3 { x: 21, y: 17, z: 4 });
        check_point(Point3 {
            x: 37,
            y: 132,
            z: 60,
        });
    }

    #[test]
    fn test_reserialize() {
        let size = Vector3 { x: 2, y: 3, z: 4 };
        let mut buf = Vec::new();

        let mut original = WorldBlocks::empty(size);

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

        let mut reserialized = WorldBlocks::empty(size);
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
