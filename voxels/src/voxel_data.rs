use std::io::{self, Read, Write};

use anyhow::Result;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{ElementWise, EuclideanSpace, Point3};

use simgame_util::{convert_vec, octree::Octree, ray::Ray, Bounds};

use crate::{
    config::VoxelDirectory,
    core::{voxels_to_u16_mut, Chunk, Voxel, VoxelRaycastHit},
    index_utils,
};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VoxelData {
    /// Storage for chunks.
    chunks: Octree<Chunk>,
}

#[derive(Debug, PartialEq)]
pub struct VoxelDataSummary {
    count_total: usize,
    count_nonempty: usize,
    count_chunks: usize,
    pct_nonempty: f64,
    byte_size: usize,
    mb_size: usize,
    bounds: Bounds<i64>,
}

impl VoxelData {
    /// Constructs an empty WorldVoxels capable of holding voxels within the given bounds.
    pub fn empty(min_bounds: Bounds<i64>) -> VoxelData {
        VoxelData {
            chunks: Octree::new(bounds_to_octree_height(min_bounds)),
        }
    }

    /// Returns the chunk in which the given voxel position resides.
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
    pub fn get_voxel(&self, p: Point3<i64>) -> Voxel {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);
        let chunk = self
            .chunks
            .get(chunk_pos)
            .expect("requested chunk which is not present");
        chunk.get_voxel(inner_pos)
    }

    /// Visits every chunk in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_chunks(&self) -> impl Iterator<Item = (Point3<i64>, &Chunk)> + '_ {
        self.chunks.iter()
    }

    /// Visits every chunk containing voxels inside the given bounding box exactly once, in an
    /// unspecified order.
    #[inline]
    pub fn iter_chunks_in_bounds(
        &self,
        voxel_bounds: Bounds<i64>,
    ) -> impl Iterator<Item = (Point3<i64>, &Chunk)> + '_ {
        let chunk_bounds = voxel_bounds.quantize_down(index_utils::chunk_size());
        self.chunks.iter_in_bounds(chunk_bounds)
    }

    /// Visits every voxel in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_voxels(&self) -> impl Iterator<Item = (Point3<i64>, Voxel)> + '_ {
        self.chunks.iter().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .voxels
                .iter()
                .enumerate()
                .map(move |(inner_index, voxel)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index as i64));
                    (loc, *voxel)
                })
        })
    }

    /// Visits every voxel in the world exactly once, in an unspecified order.
    #[inline]
    pub fn iter_voxels_mut(&mut self) -> impl Iterator<Item = (Point3<i64>, &mut Voxel)> + '_ {
        self.chunks.iter_mut().flat_map(move |(chunk_pos, chunk)| {
            chunk
                .voxels
                .iter_mut()
                .enumerate()
                .map(move |(inner_index, voxel)| {
                    let loc = index_utils::unpack_index((chunk_pos, inner_index as i64));
                    (loc, voxel)
                })
        })
    }

    /// At every possible voxel position within the given bounds, replace the existing voxel
    /// using the given function.
    pub fn replace_voxels<E, F>(
        &mut self,
        bounds: Bounds<i64>,
        mut replace_voxel: F,
    ) -> Result<(), E>
    where
        F: FnMut(Point3<i64>, Voxel) -> Result<Voxel, E>,
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
                let voxel_pos = chunk_start + (inner_pos - Point3::origin());
                if !bounds.contains_point(voxel_pos) {
                    continue;
                }

                let current_voxel = current_chunk.get_voxel(inner_pos);
                let new_voxel = replace_voxel(voxel_pos, current_voxel)?;
                current_chunk.set_voxel(inner_pos, new_voxel);

                if new_voxel != Voxel::from_u16(0) {
                    count_nonempty += 1;
                }
            }

            if count_nonempty == 0 {
                self.chunks.remove(chunk_pos);
            }
        }
        Ok(())
    }

    /// Sets the value of a voxel at the given point.
    #[inline]
    pub fn set_voxel(&mut self, p: Point3<i64>, val: Voxel) {
        let (chunk_pos, inner_pos) = index_utils::to_chunk_pos(p);

        while !self.bounds().contains_point(p) {
            self.chunks.grow();
        }

        let chunk = self.chunks.get_or_insert(chunk_pos, Chunk::empty);
        chunk.set_voxel(inner_pos, val);
    }

    /// Returns a bounding box that is guaranteed to contain every voxel in the world. No
    /// guarantees are made about whether it is the smallest such bounding box.
    pub fn bounds(&self) -> Bounds<i64> {
        self.chunks.bounds().scale_up(index_utils::chunk_size())
    }

    pub fn cast_ray(
        &self,
        ray: &Ray<f64>,
        voxel_directory: &VoxelDirectory,
    ) -> Option<VoxelRaycastHit> {
        let chunk_size = convert_vec!(index_utils::chunk_size(), f64);

        let chunk_ray = Ray {
            origin: ray.origin.div_element_wise(Point3::origin() + chunk_size),
            dir: ray.dir.div_element_wise(chunk_size),
        };

        let chunk_hit: VoxelRaycastHit = self
            .chunks
            .cast_ray(&chunk_ray, |chunk_pos, chunk| {
                let chunk_origin =
                    chunk_pos.mul_element_wise(Point3::origin() + index_utils::chunk_size());
                chunk.cast_ray(ray, chunk_origin, voxel_directory)
            })?
            .data;

        Some(chunk_hit)
    }

    /// Serialize the voxels in the world. Returns the number of bytes written.
    pub fn serialize_voxels<W>(&self, target: &mut W) -> io::Result<i64>
    where
        W: Write,
    {
        self.chunks.serialize(target, &mut |chunk, target| {
            let mut bytes_written = 0;
            for voxel in chunk.voxels.iter() {
                target.write_u16::<BigEndian>(voxel.to_u16())?;
                bytes_written += 2;
            }
            Ok(bytes_written)
        })
    }

    /// Deserialize the world voxels from the given reader.
    pub fn deserialize_voxels<R: Read>(&mut self, src: &mut R) -> io::Result<()> {
        self.chunks = Octree::<Chunk>::deserialize(src, &mut |src| {
            let mut voxels = [Voxel::from_u16(0); index_utils::chunk_size_total() as usize];
            src.read_u16_into::<BigEndian>(voxels_to_u16_mut(&mut voxels))?;
            Ok(Chunk { voxels })
        })?;
        Ok(())
    }

    pub fn debug_summary(&self) -> VoxelDataSummary {
        let size = self.bounds().size();
        let count_total = (size.x * size.y * size.z) as usize;
        let mut count_nonempty = 0;
        for (_, voxel) in self.iter_voxels() {
            if !voxel.is_empty() {
                count_nonempty += 1;
            }
        }

        let pct_nonempty = (count_nonempty as f64 / count_total as f64) * 100.0;
        let byte_size = std::mem::size_of::<Voxel>() * count_total;
        let mb_size = byte_size / (1024 * 1024);

        VoxelDataSummary {
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

#[cfg(test)]
mod tests {
    use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};

    use simgame_util::Bounds;

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

        let mut original = VoxelData::empty(bounds);

        let points1 = vec![(0, 0, 0), (3, 4, 5), (24, 25, 26), (24, 35, 36)];

        let points2 = vec![(1, 2, 3), (27, 16, 5), (2, 2, 2), (31, 47, 63)];

        original
            .replace_voxels::<(), _>(bounds, |p, _| {
                Ok(if points1.contains(&p.into()) {
                    Voxel::from_u16(1)
                } else if points2.contains(&p.into()) {
                    Voxel::from_u16(257)
                } else {
                    Voxel::air()
                })
            })
            .unwrap();
        assert!(original.chunks.check_height_invariant());

        let mut buf = Vec::new();
        original.serialize_voxels(&mut buf).unwrap();

        let mut reserialized = VoxelData::empty(bounds);
        reserialized
            .deserialize_voxels(&mut buf.as_slice())
            .unwrap();

        assert_eq!(
            original.debug_summary().count_nonempty,
            reserialized.debug_summary().count_nonempty
        );

        let chunk_limit = Point3::origin() + index_utils::chunk_size();

        assert_eq!(
            Voxel::from_u16(1),
            dbg!(original.get_chunk(Point3::new(24, 35, 36)))
                .get_voxel(dbg!(Point3::new(24, 35, 36).rem_element_wise(chunk_limit)))
        );
        assert_eq!(
            Voxel::from_u16(257),
            reserialized
                .get_chunk(Point3::new(31, 47, 63))
                .get_voxel(Point3::new(31, 47, 63).rem_element_wise(chunk_limit))
        );

        assert_eq!(Voxel::from_u16(1), reserialized.get_voxel(Point3::origin()));
        assert_eq!(
            Voxel::from_u16(1),
            reserialized.get_voxel(Point3::new(24, 35, 36))
        );
        assert_eq!(
            Voxel::from_u16(257),
            reserialized.get_voxel(Point3::new(31, 47, 63))
        );

        let collect_voxels = |world: &VoxelData| -> Vec<(Point3<i64>, Voxel)> {
            world.iter_voxels().filter(|(_, b)| !b.is_empty()).collect()
        };

        assert!(reserialized.chunks.check_height_invariant());
        assert_eq!(collect_voxels(&original), collect_voxels(&reserialized));
        assert_eq!(original.chunks.height(), reserialized.chunks.height());

        assert_eq!(original, reserialized);
    }

    #[test]
    fn test_iter_chunks_in_bounds() {
        let bounds = Bounds::from_size(8 * index_utils::chunk_size());

        let mut voxels = VoxelData::empty(bounds);
        voxels
            .replace_voxels::<(), _>(bounds, |_, _| Ok(Voxel::from_u16(1)))
            .unwrap();

        let cs = index_utils::chunk_size();

        assert_eq!(
            point_range(
                voxels
                    .iter_chunks_in_bounds(Bounds::new(Point3::origin() + 2 * cs, 3 * cs,))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(4, 4, 4)))
        );

        assert_eq!(
            point_range(
                voxels
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
                voxels
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
                voxels
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
                voxels
                    .iter_chunks_in_bounds(Bounds::new(
                        Point3::origin() + 3 * cs - cs / 2,
                        2 * cs + Vector3::new(1, 1, 1),
                    ))
                    .map(|(pos, _)| pos)
            ),
            Some((Point3::new(2, 2, 2), Point3::new(4, 4, 4)))
        );
    }

    #[allow(clippy::many_single_char_names)]
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
