//! Types used to represent the voxel-based world.

use std::collections::{hash_map, HashMap, HashSet};
use std::io::{self, Read, Write};
use std::slice;

use anyhow::{bail, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::octree::Octree;
use crate::ray::{Intersection, Ray};
use crate::util::Bounds;
use crate::{convert_bounds, convert_vec};

pub mod index_utils;

/// Represents the value of a single block in the world. The wrapped value is an index into the
/// BlockConfig's list of BlockInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Block(u16);

/// A block category.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Category(String);

/// Specification of a block type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    pub name: String,
    pub category: Category,
    #[serde(default = "BlockInfo::default_passable_through")]
    pub passable_through: bool,
    #[serde(default = "BlockInfo::default_passable_above")]
    pub passable_above: bool,
    #[serde(default = "BlockInfo::default_speed_modifier")]
    pub speed_modifier: f64,

    pub texture: BlockTexture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FaceTexture {
    /// The face has a soid color.
    SolidColor { red: u8, green: u8, blue: u8 },
    /// The face has a texture with the given resource name.
    Texture {
        resource: String,
        /// The texture repeats after this many blocks
        periodicity: u32
    },
}

/// Specification of how a block is textured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockTexture {
    /// The block has the given face texture on all faces.
    Uniform(FaceTexture),
    /// The block has the given face textures on corresponding faces.
    Nonuniform {
        top: FaceTexture,
        bottom: FaceTexture,
        side: FaceTexture,
    },
}

/// Specification of the selection of available blocks and categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockConfig {
    blocks: Vec<BlockInfo>,
}

#[derive(Debug, Clone)]
pub struct BlockConfigHelper {
    blocks_by_name: HashMap<String, (Block, BlockInfo)>,
    blocks_by_id: Vec<BlockInfo>,
}

/// Stores the world's blocks, but not other things like entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldBlockData {
    /// Storage for chunks.
    chunks: Octree<Chunk>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdatedBlocksState {
    pub modified_chunks: HashSet<Point3<i64>>,
}

#[derive(Debug)]
pub struct BlockUpdater<'a> {
    blocks: &'a mut WorldBlockData,
    updated_state: &'a mut UpdatedBlocksState,
}

#[derive(Debug, Clone)]
pub struct RaycastHit {
    pub block: Block,
    pub block_pos: Point3<i64>,
    pub intersection: Intersection<f64>,
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
    ) -> Option<RaycastHit> {
        // TODO: use space subdivision to avoid looping through every single block
        // TODO: use collision info in block config to handle blocks that are not full cubes

        let mut nearest_hit: Option<RaycastHit> = None;

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
                        Some(intersection) => RaycastHit {
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

impl WorldBlockData {
    /// Constructs an empty WorldBlocks capable of holding blocks within the given bounds.
    pub fn empty(min_bounds: Bounds<i64>) -> WorldBlockData {
        WorldBlockData {
            chunks: Octree::new(bounds_to_octree_height(min_bounds)),
        }
    }

    /// Returns the chunk in which the given block position resides.
    #[inline]
    pub fn get_chunk(&self, p: Point3<i64>) -> &Chunk {
        let (chunk_pos, _) = index_utils::to_chunk_pos(p);
        self.chunks
            .get(chunk_pos)
            .expect("requested chunk which is not present")
    }

    pub fn chunks(&self) -> &Octree<Chunk> {
        &self.chunks
    }

    pub fn chunks_mut(&mut self) -> &mut Octree<Chunk> {
        &mut self.chunks
    }

    #[inline]
    pub fn get_block(&self, p: Point3<i64>) -> Block {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);
        let chunk = self
            .chunks
            .get(chunk_pos)
            .expect("requested chunk which is not present");
        chunk.get_block(inner_pos)
    }

    /// Visits every chunk in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_chunks(&self) -> impl Iterator<Item = (Point3<i64>, &Chunk)> + '_ {
        self.chunks.iter()
    }

    /// Visits every chunk containing blocks inside the given bounding box exactly once, in an
    /// unspecified order.
    #[inline]
    pub fn iter_chunks_in_bounds(
        &self,
        block_bounds: Bounds<i64>,
    ) -> impl Iterator<Item = (Point3<i64>, &Chunk)> + '_ {
        let chunk_bounds = block_bounds.quantize_down(index_utils::chunk_size());
        self.chunks.iter_in_bounds(chunk_bounds)
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks(&self) -> impl Iterator<Item = (Point3<i64>, Block)> + '_ {
        self.chunks.iter().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .blocks
                .iter()
                .enumerate()
                .map(move |(inner_index, block)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index as i64));
                    (loc, *block)
                })
        })
    }

    /// Visits every block in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = (Point3<i64>, &mut Block)> + '_ {
        self.chunks.iter_mut().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .blocks
                .iter_mut()
                .enumerate()
                .map(move |(inner_index, block)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index as i64));
                    (loc, block)
                })
        })
    }

    /// At every possible block position within the given bounds, replace the existing block
    /// using the given function.
    pub fn replace_blocks<E, F>(
        &mut self,
        bounds: Bounds<i64>,
        mut replace_block: F,
    ) -> Result<(), E>
    where
        F: FnMut(Point3<i64>, Block) -> Result<Block, E>,
    {
        while !self.bounds().contains_bounds(bounds) {
            self.chunks.grow();
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
    pub fn set_block(&mut self, p: Point3<i64>, val: Block) {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);

        while !self.bounds().contains_point(p) {
            self.chunks.grow();
        }

        let chunk = self.chunks.get_or_insert(chunk_pos, Chunk::empty);
        chunk.set_block(inner_pos, val);
    }

    /// Returns a bounding box that is guaranteed to contain every block in the world. No
    /// guarantees are made about whether it is the smallest such bounding box.
    pub fn bounds(&self) -> Bounds<i64> {
        self.chunks.bounds().scale_up(index_utils::chunk_size())
    }

    pub fn cast_ray(
        &self,
        ray: &Ray<f64>,
        block_helper: &BlockConfigHelper,
    ) -> Option<RaycastHit> {
        let chunk_size = convert_vec!(index_utils::chunk_size(), f64);

        let chunk_ray = Ray {
            origin: ray.origin.div_element_wise(Point3::origin() + chunk_size),
            dir: ray.dir.div_element_wise(chunk_size),
        };

        let chunk_hit: RaycastHit = self
            .chunks
            .cast_ray(&chunk_ray, |chunk_pos, chunk| {
                let chunk_origin =
                    chunk_pos.mul_element_wise(Point3::origin() + index_utils::chunk_size());
                chunk.cast_ray(ray, chunk_origin, block_helper)
            })?
            .data;

        Some(chunk_hit)
    }

    /// Serialize the blocks in the world. Returns the number of bytes written.
    pub fn serialize_blocks<W>(&self, target: &mut W) -> io::Result<i64>
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
            let mut blocks = [Block(0u16); index_utils::chunk_size_total() as usize];
            src.read_u16_into::<BigEndian>(blocks_to_u16_mut(&mut blocks))?;
            Ok(Chunk { blocks })
        })?;
        Ok(())
    }

    pub fn debug_summary(&self) -> WorldBlockDataSummary {
        let size = self.bounds().size();
        let count_total = (size.x * size.y * size.z) as usize;
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
}

impl<'a> BlockUpdater<'a> {
    pub fn new(blocks: &'a mut WorldBlockData, updated_state: &'a mut UpdatedBlocksState) -> Self {
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
    pub fn set_blocks<I>(&mut self, blocks: I)
    where
        I: IntoIterator<Item = (Point3<i64>, Block)>,
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

#[derive(Debug, PartialEq)]
pub struct WorldBlockDataSummary {
    count_total: usize,
    count_nonempty: usize,
    count_chunks: usize,
    pct_nonempty: f64,
    byte_size: usize,
    mb_size: usize,
    bounds: Bounds<i64>,
}

#[inline]
pub fn blocks_to_u16(buf: &[Block]) -> &[u16] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

#[inline]
pub fn blocks_to_u16_mut(buf: &mut [Block]) -> &mut [u16] {
    unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u16, buf.len()) }
}

fn bounds_to_octree_height(min_bounds: Bounds<i64>) -> i64 {
    let chunk_size = index_utils::chunk_size();
    let max_dim = [
        min_bounds.limit().x.abs() as f64 / chunk_size.x as f64,
        min_bounds.limit().y.abs() as f64 / chunk_size.y as f64,
        min_bounds.limit().z.abs() as f64 / chunk_size.z as f64,
        min_bounds.origin().x.abs() as f64 / chunk_size.x as f64,
        min_bounds.origin().y.abs() as f64 / chunk_size.y as f64,
        min_bounds.origin().z.abs() as f64 / chunk_size.z as f64,
    ]
    .iter()
    .copied()
    .fold(0.0, f64::max);

    2 + f64::max(max_dim.log2(), 0.0).ceil() as i64
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

impl BlockConfigHelper {
    pub fn new(config: &BlockConfig) -> Result<Self> {
        if config.blocks.is_empty() || config.blocks[0].category != Category("air".into()) {
            bail!("First entry in block config must have category \"air\"");
        }

        let blocks_by_id = config.blocks.clone();

        let blocks_by_name = blocks_by_id
            .iter()
            .enumerate()
            .map(|(i, block_info)| {
                let name = block_info.name.clone();
                let block = Block::from_u16(i as u16);
                (name, (block, block_info.clone()))
            })
            .collect();

        for (block_id, block) in blocks_by_id.iter().enumerate() {
            log::info!("Block id {} is {}", block_id, block.name);
        }

        Ok(Self {
            blocks_by_name,
            blocks_by_id,
        })
    }

    pub fn block_by_name(&self, name: &str) -> Option<(Block, &BlockInfo)> {
        let (block, block_info) = self.blocks_by_name.get(name)?;
        Some((*block, block_info))
    }

    pub fn block_info(&self, block: Block) -> Option<&BlockInfo> {
        self.blocks_by_id.get(block.0 as usize)
    }

    pub fn blocks(&self) -> &[BlockInfo] {
        &self.blocks_by_id[..]
    }

    pub fn texture_index_map(&self) -> HashMap<FaceTexture, usize> {
        let mut index_map: HashMap<FaceTexture, usize> = HashMap::new();
        let mut index = 0;

        let mut insert = |face_tex: FaceTexture| -> usize {
            let entry = index_map.entry(face_tex);
            match entry {
                hash_map::Entry::Occupied(entry) => *entry.get(),
                hash_map::Entry::Vacant(entry) => {
                    let res = *entry.insert(index);
                    index += 1;
                    res
                }
            }
        };

        for block in self.blocks() {
            match block.texture.clone() {
                BlockTexture::Uniform(face_tex) => {
                    let index = insert(face_tex);
                    log::info!(
                        "Block {} gets a uniform texture with index {}",
                        block.name,
                        index
                    );
                }
                BlockTexture::Nonuniform { top, bottom, side } => {
                    let index_top = insert(top);
                    let index_bottom = insert(bottom);
                    let index_side = insert(side);

                    log::info!(
                        "Block {} gets a non-uniform texture with indices {}/{}/{}",
                        block.name,
                        index_top,
                        index_bottom,
                        index_side
                    );
                }
            }
        }

        index_map
    }
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
        assert_eq!(bounds_to_octree_height(bounds), 3);

        let bounds = Bounds::from_size(index_utils::chunk_size() * 2);
        assert_eq!(bounds_to_octree_height(bounds), 3);

        let bounds =
            Bounds::from_size(index_utils::chunk_size() * 2 + Vector3 { x: 1, y: 0, z: 0 });
        assert_eq!(bounds_to_octree_height(bounds), 4);
    }

    #[test]
    fn test_reserialize() {
        let bounds = Bounds::from_size(Vector3 {
            x: 32,
            y: 48,
            z: 64,
        });

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

        let mut buf = Vec::new();
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
            dbg!(original.get_chunk(Point3::new(24, 35, 36)))
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

        let collect_blocks = |world: &WorldBlockData| -> Vec<(Point3<i64>, Block)> {
            world.iter_blocks().filter(|(_, b)| !b.is_empty()).collect()
        };

        assert!(reserialized.chunks.check_height_invariant());
        assert_eq!(collect_blocks(&original), collect_blocks(&reserialized));
        assert_eq!(original.chunks.height(), reserialized.chunks.height());

        assert_eq!(original, reserialized);
    }

    #[test]
    fn test_iter_chunks_in_bounds() {
        let bounds = Bounds::from_size(8 * index_utils::chunk_size());

        let mut blocks = WorldBlockData::empty(bounds);
        blocks
            .replace_blocks::<(), _>(bounds, |_, _| Ok(Block::from_u16(1)))
            .unwrap();

        let cs = index_utils::chunk_size();

        assert_eq!(
            point_range(
                blocks
                    .iter_chunks_in_bounds(Bounds::new(Point3::origin() + 2 * cs, 3 * cs,))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(4, 4, 4)))
        );

        assert_eq!(
            point_range(
                blocks
                    .iter_chunks_in_bounds(Bounds::new(
                        Point3::origin() + 2 * cs,
                        3 * cs - Vector3::new(1, 1, 1),
                    ))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(4, 4, 4)))
        );

        assert_eq!(
            point_range(
                blocks
                    .iter_chunks_in_bounds(Bounds::from_limit(
                        Point3::new(6, 6, 6),
                        Point3::new(38, 38, 26),
                    ))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(0, 0, 1), Point3::new(2, 2, 6)))
        );

        assert_eq!(
            point_range(
                blocks
                    .iter_chunks_in_bounds(Bounds::new(
                        Point3::origin() + 3 * cs - cs / 2,
                        3 * cs - cs / 2 + Vector3::new(1, 0, 0),
                    ))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(5, 4, 4)))
        );

        assert_eq!(
            point_range(
                blocks
                    .iter_chunks_in_bounds(Bounds::new(
                        Point3::origin() + 3 * cs - cs / 2,
                        2 * cs + Vector3::new(1, 1, 1),
                    ))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(4, 4, 4)))
        );
    }

    fn point_range<Points>(points: Points) -> Option<(Point3<i64>, Point3<i64>)>
    where
        Points: IntoIterator<Item = Point3<i64>>,
    {
        let points: Vec<_> = points.into_iter().collect();
        let x = || points.iter().map(|p| p.x);
        let y = || points.iter().map(|p| p.y);
        let z = || points.iter().map(|p| p.z);

        let l = x().min().and_then(|lx| {
            y().min()
                .and_then(|ly| z().min().map(|lz| Point3::new(lx, ly, lz)))
        });

        let h = x().max().and_then(|hx| {
            y().max()
                .and_then(|hy| z().max().map(|hz| Point3::new(hx, hy, hz)))
        });

        l.and_then(|l| h.map(|h| (l, h)))
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
