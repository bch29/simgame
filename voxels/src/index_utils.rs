use std::ops::{Index, IndexMut};

use cgmath::{ElementWise, Point3, Vector3};

#[inline]
pub const fn chunk_size() -> Vector3<i64> {
    Vector3 { x: 16, y: 16, z: 4 }
}

#[inline]
pub const fn chunk_size_total() -> i64 {
    chunk_size().x * chunk_size().y * chunk_size().z
}

#[inline]
pub fn pack_xyz(bounds: Vector3<i64>, p: Point3<i64>) -> i64 {
    assert!(point_within_size(p, bounds));
    p.x + p.y * bounds.x + p.z * bounds.x * bounds.y
}

#[inline]
pub fn unpack_xyz(bounds: Vector3<i64>, index: i64) -> Point3<i64> {
    assert!(index < bounds.x * bounds.y * bounds.z);
    let xy = index % (bounds.x * bounds.y);

    Point3 {
        x: xy % bounds.x,
        y: xy / bounds.x,
        z: index / (bounds.x * bounds.y),
    }
}

#[inline]
pub fn point_within_size(point: Point3<i64>, bounds: Vector3<i64>) -> bool {
    point.x < bounds.x && point.y < bounds.y && point.z < bounds.z
}

/// indices is (chunk_pos, inner_index)
#[inline]
pub fn unpack_index(indices: (Point3<i64>, i64)) -> Point3<i64> {
    let (chunk_pos, inner_index) = indices;
    let origin = Point3::from((0, 0, 0));
    let inner_pos = unpack_xyz(chunk_size(), inner_index);
    let inner_offset = inner_pos - origin;
    chunk_pos.mul_element_wise(origin + chunk_size()) + inner_offset
}

/// From a point in voxel coordinates, return the chunk index and the position of the voxel
/// within that chunk.
#[inline]
pub fn to_chunk_pos(p: Point3<i64>) -> (Point3<i64>, Point3<i64>) {
    let origin = Point3::from((0, 0, 0));
    let mut inner_pos = p.rem_element_wise(origin + chunk_size());
    let mut chunk_pos = p.div_element_wise(origin + chunk_size());

    if inner_pos.x < 0 {
        inner_pos.x += chunk_size().x;
        chunk_pos.x -= 1;
    }

    if inner_pos.y < 0 {
        inner_pos.y += chunk_size().y;
        chunk_pos.y -= 1;
    }

    if inner_pos.z < 0 {
        inner_pos.z += chunk_size().z;
        chunk_pos.z -= 1;
    }

    (chunk_pos, inner_pos)
}

#[inline]
pub fn pack_within_chunk(p: Point3<i64>) -> i64 {
    assert!(point_within_size(p, chunk_size()));
    p.x + p.y * chunk_size().x + p.z * chunk_size().x * chunk_size().y
}

/// Returns (chunk_pos, inner_index),
#[inline]
pub fn pack_index(p: Point3<i64>) -> (Point3<i64>, i64) {
    let (chunk_pos, inner_point) = to_chunk_pos(p);
    let inner_index = pack_within_chunk(inner_point);
    (chunk_pos, inner_index)
}

#[inline]
pub fn make_size(count_chunks: Vector3<i64>) -> Vector3<i64> {
    count_chunks.mul_element_wise(chunk_size())
}

pub struct Indexed3D<'a, T> {
    slice: &'a [T],
    size: Vector3<i64>,
}

impl<'a, T> Indexed3D<'a, T> {
    pub fn new(slice: &'a [T], size: Vector3<i64>) -> Self {
        assert!(slice.len() >= (size.x * size.y * size.z) as usize);
        Indexed3D { slice, size }
    }
}

impl<'a, T> Index<Point3<i64>> for Indexed3D<'a, T> {
    type Output = T;
    fn index(&self, loc: Point3<i64>) -> &Self::Output {
        assert!(point_within_size(loc, self.size));
        let index = pack_xyz(self.size, loc);
        &self.slice[index as usize]
    }
}

pub struct Indexed3DMut<'a, T> {
    slice: &'a mut [T],
    size: Vector3<i64>,
}

impl<'a, T> Indexed3DMut<'a, T> {
    pub fn new(slice: &'a mut [T], size: Vector3<i64>) -> Self {
        assert!(slice.len() >= (size.x * size.y * size.z) as usize);
        Indexed3DMut { slice, size }
    }
}

impl<'a, T> Index<Point3<i64>> for Indexed3DMut<'a, T> {
    type Output = T;
    fn index(&self, loc: Point3<i64>) -> &Self::Output {
        assert!(point_within_size(loc, self.size));
        let index = pack_xyz(self.size, loc);
        unsafe { self.slice.get_unchecked(index as usize) }
    }
}

impl<'a, T> IndexMut<Point3<i64>> for Indexed3DMut<'a, T> {
    fn index_mut(&mut self, loc: Point3<i64>) -> &mut Self::Output {
        assert!(point_within_size(loc, self.size));
        let index = pack_xyz(self.size, loc);
        unsafe { self.slice.get_unchecked_mut(index as usize) }
    }
}

pub struct Vec3D<T> {
    data: Vec<T>,
    size: Vector3<i64>,
}

impl<T> Vec3D<T>
where
    T: Clone,
{
    pub fn new(val: &T, size: Vector3<i64>) -> Self {
        Vec3D {
            data: (0..size.x * size.y * size.z).map(|_| val.clone()).collect(),
            size,
        }
    }
}

impl<T> Index<Point3<i64>> for Vec3D<T> {
    type Output = T;
    fn index(&self, loc: Point3<i64>) -> &Self::Output {
        assert!(point_within_size(loc, self.size));
        let index = pack_xyz(self.size, loc);
        unsafe { self.data.get_unchecked(index as usize) }
    }
}

impl<T> IndexMut<Point3<i64>> for Vec3D<T> {
    fn index_mut(&mut self, loc: Point3<i64>) -> &mut Self::Output {
        assert!(point_within_size(loc, self.size));
        let index = pack_xyz(self.size, loc);
        unsafe { self.data.get_unchecked_mut(index as usize) }
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use super::*;

    #[test]
    fn test_indexed_3d() {
        let size = Vector3 { x: 4, y: 3, z: 2 };
        let mut numbers: Vec<i32> = (0..(size.x * size.y * size.z) as i32).collect();

        let mut as_3d = Indexed3DMut::new(numbers.as_mut_slice(), size);

        assert_eq!(as_3d[Point3::new(2, 1, 1)], 18);

        as_3d[Point3::new(3, 2, 1)] = 1000;

        assert_eq!(as_3d[Point3::new(2, 1, 1)], 18);
        assert_eq!(as_3d[Point3::new(3, 2, 1)], 1000);
    }

    #[test]
    fn test_pack_index() {
        assert_eq!(
            (Point3 { x: 0, y: 0, z: 0 }, 0),
            pack_index(Point3 { x: 0, y: 0, z: 0 })
        );

        assert_eq!(
            (Point3 { x: 0, y: 0, z: 0 }, 2 + 3 * chunk_size().x),
            pack_index(Point3 { x: 2, y: 3, z: 0 })
        );
    }

    #[test]
    fn test_unpack_index() {
        assert_eq!(
            Point3 { x: 2, y: 3, z: 0 },
            unpack_xyz(chunk_size(), 2 + 3 * chunk_size().x)
        );

        let check_point = |p| {
            assert_eq!(p, unpack_index(pack_index(p)));
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
}
