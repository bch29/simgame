use std::slice;

use cgmath::{EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

use simgame_util::convert_bounds;
use simgame_util::ray::{Intersection, Ray};
use simgame_util::Bounds;

use crate::config::VoxelConfigHelper;
use crate::index_utils;

/// Represents the value of a single voxel in the world. The wrapped value is an index into the
/// VoxelConfig's list of VoxelInfo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Voxel(u16);

/// Encapsulates a single chunk of voxels. The world is split into chunks to help with cache
/// locality: if there were no chunks, then any movement in at least one of the dimensions
/// (probably z) would always result in moving to a different cache line. Under the chunking
/// system, voxels that are close to each other in space, and are in the same chunk, should be in
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
    pub voxels: [Voxel; index_utils::chunk_size_total() as usize],
}

#[derive(Debug, Clone)]
pub struct VoxelRaycastHit {
    pub voxel: Voxel,
    pub voxel_pos: Point3<i64>,
    pub intersection: Intersection<f64>,
}

impl Voxel {
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn from_u16(val: u16) -> Self {
        Voxel(val)
    }

    pub fn air() -> Self {
        Voxel(0)
    }

    pub(crate) fn to_u16(self) -> u16 {
        self.0
    }
}

impl Chunk {
    #[inline]
    pub fn get_voxel(&self, p: Point3<i64>) -> Voxel {
        self.voxels[index_utils::pack_within_chunk(p) as usize]
    }

    #[inline]
    pub fn set_voxel(&mut self, p: Point3<i64>, val: Voxel) {
        self.voxels[index_utils::pack_within_chunk(p) as usize] = val;
    }

    /// Creates a chunk filled with zeroes (i.e. empty voxels).
    pub fn empty() -> Chunk {
        let voxels = [Voxel(0); index_utils::chunk_size_total() as usize];
        Chunk { voxels }
    }

    pub fn cast_ray(
        &self,
        ray: &Ray<f64>,
        origin: Point3<i64>,
        _voxel_helper: &VoxelConfigHelper,
    ) -> Option<VoxelRaycastHit> {
        // TODO: use space subdivision to avoid looping through every single voxel
        // TODO: use collision info in voxel config to handle voxels that are not full cubes

        let mut nearest_hit: Option<VoxelRaycastHit> = None;

        for z in 0..index_utils::chunk_size().z {
            for y in 0..index_utils::chunk_size().y {
                for x in 0..index_utils::chunk_size().x {
                    let voxel_offset = Vector3::new(x, y, z);
                    let voxel = self.get_voxel(Point3::origin() + voxel_offset);

                    if voxel.is_empty() {
                        continue;
                    }

                    let voxel_pos = origin + voxel_offset;
                    let bounds =
                        convert_bounds!(Bounds::new(voxel_pos, Vector3::new(1, 1, 1)), f64);
                    let hit = match bounds.cast_ray(ray).entry() {
                        Some(intersection) => VoxelRaycastHit {
                            voxel,
                            voxel_pos,
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
pub fn voxels_to_u16(buf: &[Voxel]) -> &[u16] {
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const u16, buf.len()) }
}

#[inline]
pub fn voxels_to_u16_mut(buf: &mut [Voxel]) -> &mut [u16] {
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
                    write!(f, "{}", self.voxels[ix as usize].0)?;
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
            if self.voxels[i as usize] != other.voxels[i as usize] {
                return false;
            }
        }

        true
    }
}

impl Eq for Chunk {}
