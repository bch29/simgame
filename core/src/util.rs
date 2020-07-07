use cgmath::{BaseNum, ElementWise, EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Represents a half-open cuboid of points: the origin is inclusive and the limit is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[repr(C)]
pub struct Bounds<T> {
    origin: Point3<T>,
    size: Vector3<T>,
}

/// Represents the difference between two Bounds objects, i.e. all the points in the LHS which are
/// not also in the RHS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[repr(C)]
pub struct BoundsDiff<T> {
    lhs: Bounds<T>,
    intersection: Option<Bounds<T>>,
}

/// Iterator through all the points in the difference between two bounds objects.
#[derive(Debug, Clone)]
pub struct IterBoundsDiffPoints<T> {
    diff: BoundsDiff<T>,
    pos: Option<Point3<T>>,
}

pub trait DivDown {
    fn div_down(&self, divisor: &Self) -> Self;
}

pub trait DivUp {
    fn div_up(&self, divisor: &Self) -> Self;
}

impl<T> Bounds<T> {
    pub fn origin(self) -> Point3<T> {
        self.origin
    }

    pub fn size(self) -> Vector3<T> {
        self.size
    }

    /// Moves the origin while keeping the size the same. Note this also changes the limit.
    pub fn translate(mut self, new_origin: Point3<T>) -> Self {
        self.origin = new_origin;
        self
    }
}

impl<T: BaseNum> Bounds<T> {
    #[inline]
    pub fn new(origin: Point3<T>, size: Vector3<T>) -> Self {
        assert!(size.x >= T::zero());
        assert!(size.y >= T::zero());
        assert!(size.z >= T::zero());
        Bounds { origin, size }
    }

    #[inline]
    pub fn from_size(size: Vector3<T>) -> Self {
        Bounds::new(Point3::<T>::origin(), size)
    }

    #[inline]
    pub fn from_limit(origin: Point3<T>, limit: Point3<T>) -> Self {
        assert!(origin.x <= limit.x);
        assert!(origin.y <= limit.y);
        assert!(origin.z <= limit.z);
        Bounds::new(origin, limit - origin)
    }

    /// Returns the smallest bounds object containing all of the given points. If the input is
    /// empty, returns None.
    #[inline]
    pub fn from_points<I: IntoIterator<Item = Point3<T>>>(points: I) -> Option<Self> {
        let mut iter = points.into_iter();

        let first_point = iter.next()?;

        let mut origin = first_point;
        let mut limit = first_point;

        for point in iter {
            if point.x < origin.x {
                origin.x = point.x
            }
            if point.y < origin.y {
                origin.y = point.y
            }
            if point.z < origin.z {
                origin.z = point.z
            }
            if point.x > limit.x {
                limit.x = point.x
            }
            if point.y > limit.y {
                limit.y = point.y
            }
            if point.z > limit.z {
                limit.z = point.z
            }
        }

        Some(Self::from_limit(origin, limit))
    }

    #[inline]
    pub fn limit(self) -> Point3<T> {
        self.origin + self.size
    }

    #[inline]
    pub fn contains_point(self, point: Point3<T>) -> bool {
        let limit = self.limit();
        point.x >= self.origin.x
            && point.x < limit.x
            && point.y >= self.origin.y
            && point.y < limit.y
            && point.z >= self.origin.z
            && point.z < limit.z
    }

    /// True if the given bounds is fully contained within self (including if the limits
    /// intersect).
    #[inline]
    pub fn contains_bounds(self, bounds: Bounds<T>) -> bool {
        let limit = self.limit();
        let a = bounds.origin();
        let b = bounds.limit();
        a.x >= self.origin.x
            && b.x <= limit.x
            && a.y >= self.origin.y
            && b.y <= limit.y
            && a.z >= self.origin.z
            && b.z <= limit.z
    }

    #[inline]
    /// If the size is 0 in any dimension, the bounds object is empty.
    pub fn is_empty(self) -> bool {
        self.size.x == T::zero() || self.size.y == T::zero() || self.size.z == T::zero()
    }

    /// Computes a bounds object that is the intersection of self with the given bounds. If the
    /// intersection would be empty, returns `None`.
    #[inline]
    pub fn intersection(self, bounds: Bounds<T>) -> Option<Self> {
        let max = |a, b| {
            if a >= b {
                a
            } else {
                b
            }
        };
        let min = |a, b| {
            if a <= b {
                a
            } else {
                b
            }
        };

        let ao = self.origin();
        let al = self.limit();
        let bo = bounds.origin();
        let bl = bounds.limit();

        let origin = Point3 {
            x: max(ao.x, bo.x),
            y: max(ao.y, bo.y),
            z: max(ao.z, bo.z),
        };

        let limit = Point3 {
            x: min(al.x, bl.x),
            y: min(al.y, bl.y),
            z: min(al.z, bl.z),
        };

        if origin.x >= limit.x || origin.y >= limit.y || origin.z >= limit.z {
            None
        } else {
            Some(Self::from_limit(origin, limit))
        }
    }

    #[inline]
    pub fn x_range(self) -> std::ops::Range<T> {
        self.origin().x..self.limit().x
    }

    #[inline]
    pub fn y_range(self) -> std::ops::Range<T> {
        self.origin().y..self.limit().y
    }

    #[inline]
    pub fn z_range(self) -> std::ops::Range<T> {
        self.origin().z..self.limit().z
    }

    #[inline]
    pub fn clamp(self, point: Point3<T>) -> Point3<T> {
        let limit = self.limit();

        let clamp = |x, a, b| {
            if x >= a {
                if x < b {
                    x
                } else {
                    b
                }
            } else {
                a
            }
        };

        Point3 {
            x: clamp(point.x, self.origin.x, limit.x),
            y: clamp(point.y, self.origin.y, limit.y),
            z: clamp(point.z, self.origin.z, limit.z),
        }
    }

    /// Moves the origin while keeping the limit the same. Note this also changes the size.
    #[inline]
    pub fn with_origin(self, origin: Point3<T>) -> Self {
        let limit = self.limit();
        assert!(origin.x <= limit.x);
        assert!(origin.y <= limit.y);
        assert!(origin.z <= limit.z);
        let size = limit - origin;

        Bounds { origin, size }
    }

    /// Changes the size while keeping the origin the same. Note this also changes the limit.
    #[inline]
    pub fn with_size(mut self, size: Vector3<T>) -> Self {
        assert!(size.x >= T::zero());
        assert!(size.y >= T::zero());
        assert!(size.z >= T::zero());
        self.size = size;
        self
    }

    /// Moves the limit while keeping the origin the same. Note this also changes the size.
    #[inline]
    pub fn with_limit(mut self, limit: Point3<T>) -> Self {
        assert!(limit.x >= self.origin.x);
        assert!(limit.y >= self.origin.y);
        assert!(limit.z >= self.origin.z);
        self.size = limit - self.origin;
        self
    }

    /// Returns the smallest bounding box that holds all the points in `self`, when `self` is
    /// divided into quanta of size `quantum_size`.
    ///
    /// The result is given in units of quanta.
    #[inline]
    pub fn quantize_down(self, quantum_size: Vector3<T>) -> Self
    where
        T: DivUp + DivDown,
    {
        Bounds::from_limit(
            self.origin.div_down(&(Point3::origin() + quantum_size)),
            self.limit().div_up(&(Point3::origin() + quantum_size)),
        )
    }

    #[inline]
    pub fn scale_up(self, scale: Vector3<T>) -> Self {
        assert!(scale.x >= T::zero());
        assert!(scale.y >= T::zero());
        assert!(scale.z >= T::zero());

        Bounds {
            origin: self.origin.mul_element_wise(Point3::origin() + scale),
            size: self.size.mul_element_wise(scale),
        }
    }

    #[inline]
    pub fn iter_points(self) -> impl Iterator<Item = Point3<T>>
    where
        std::ops::Range<T>: Iterator<Item = T>,
        T: Copy + 'static,
    {
        let origin = self.origin();
        let limit = self.limit();

        (origin.z..limit.z).flat_map(move |z| {
            (origin.y..limit.y)
                .flat_map(move |y| (origin.x..limit.x).map(move |x| Point3 { x, y, z }))
        })
    }

    // TODO: this is not implemented properly yet. It should iterate in an order that facilitates
    // chunk locality better.
    #[inline]
    pub fn iter_points_aligned(self, _align_to: Vector3<T>) -> impl Iterator<Item = Point3<T>>
    where
        std::ops::Range<T>: Iterator<Item = T>,
        T: Copy + 'static,
    {
        self.iter_points()
    }

    #[inline]
    pub fn volume(self) -> T {
        self.size.x * self.size.y * self.size.z
    }

    pub fn diff(self, other: Bounds<T>) -> BoundsDiff<T> {
        BoundsDiff {
            lhs: self,
            intersection: self.intersection(other),
        }
    }

    /// Iterates through all the points in 'self' which are not in 'other'.
    pub fn iter_diff(self, other: Bounds<T>) -> IterBoundsDiffPoints<T> {
        self.diff(other).iter_points()
    }
}

impl<T> BoundsDiff<T>
where
    T: BaseNum,
{
    pub fn iter_points(self) -> IterBoundsDiffPoints<T> {
        // Algorithm:
        // - compute intersection I
        // - for each z level in self whose plane is not contained in the intersection
        // - for each y level in self such that the y, z line is not contained in the intersection
        // - for each x level in self such that the x, y, z point is not contained in the
        //   intersection
        // - yield (x, y, z)

        IterBoundsDiffPoints {
            diff: self,
            pos: self.origin(),
        }
    }

    fn first_x(&self, y: T, z: T) -> Option<T> {
        self.next_x(y, z, self.lhs.origin().x)
    }

    fn first_y(&self, z: T) -> Option<T> {
        self.next_y(z, self.lhs.origin().y)
    }

    fn first_z(&self) -> Option<T> {
        self.next_z(self.lhs.origin().z)
    }

    fn origin(&self) -> Option<Point3<T>> {
        self.first_z().and_then(|z| {
            self.first_y(z)
                .and_then(|y| self.first_x(y, z).map(|x| Point3::new(x, y, z)))
        })
    }

    /// Given an x coordinate on a particular y-z line, find the next smallest x coordinate
    /// (including start_x) which lies in the diff.
    fn next_x(&self, y: T, z: T, start_x: T) -> Option<T> {
        let mut result = start_x;
        // if the x-y-z point would lie fully inside the intersection, advance to the x limit of
        // the intersection
        if let Some(intersection) = self.intersection {
            if intersection.contains_point(Point3::new(start_x, y, z)) {
                result = intersection.limit().x;
            }
        }

        if result < self.lhs.limit().x {
            Some(result)
        } else {
            None
        }
    }

    /// Given a y coordinate on a particular z plane, find the next smallest y coordinate
    /// (including start_y) which lies in the diff.
    fn next_y(&self, z: T, start_y: T) -> Option<T> {
        let mut result = start_y;

        // if the y-z line would lie fully inside the intersection, advance to the y limit of the
        // intersection
        if let Some(intersection) = self.intersection {
            let yz_line = {
                let origin = Point3::new(self.lhs.origin().x, start_y, z);
                let size = Vector3::new(self.lhs.size().x, T::one(), T::one());
                Bounds::new(origin, size)
            };

            if intersection.contains_bounds(yz_line) {
                result = intersection.limit().y;
            }
        }

        if result < self.lhs.limit().y {
            Some(result)
        } else {
            None
        }
    }

    /// Given a z coordinate, find the next smallest z coordinate (including start_z) which lies in
    /// the diff.
    fn next_z(&self, start_z: T) -> Option<T> {
        let mut result = start_z;

        // if the z plane would lie fully inside the intersection, advance to the z limit of the
        // intersection
        if let Some(intersection) = self.intersection {
            let z_plane = {
                let mut origin = self.lhs.origin();
                origin.z = start_z;
                let mut size = self.lhs.size();
                size.z = T::one();
                Bounds::new(origin, size)
            };

            if intersection.contains_bounds(z_plane) {
                result = intersection.limit().z;
            }
        }

        if result < self.lhs.limit().z {
            Some(result)
        } else {
            None
        }
    }

    fn advance_x(&mut self, mut pos: Point3<T>, new_x: T) -> Option<Point3<T>> {
        pos.x = new_x;
        Some(pos)
    }

    fn advance_y(&mut self, mut pos: Point3<T>, new_y: T) -> Option<Point3<T>> {
        pos.y = new_y;
        pos.x = self.first_x(new_y, pos.z)?;
        Some(pos)
    }

    fn advance_z(&mut self, mut pos: Point3<T>, new_z: T) -> Option<Point3<T>> {
        pos.z = new_z;
        pos.y = self.first_y(new_z)?;
        pos.x = self.first_x(pos.y, new_z)?;
        Some(pos)
    }
}

impl<T> IterBoundsDiffPoints<T> where T: BaseNum {}

impl<T> Iterator for IterBoundsDiffPoints<T>
where
    T: BaseNum,
{
    type Item = Point3<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.pos.as_mut()?;
        let result = *pos;

        if let Some(x) = self.diff.next_x(result.y, result.z, T::one() + result.x) {
            *pos = self.diff.advance_x(*pos, x).expect("failed to advance x");
            return Some(result);
        }

        if let Some(y) = self.diff.next_y(result.z, T::one() + result.y) {
            *pos = self.diff.advance_y(*pos, y).expect("failed to advance y");
            return Some(result);
        }

        if let Some(z) = self.diff.next_z(T::one() + result.z) {
            *pos = self.diff.advance_z(*pos, z).expect("failed to advance z");
        } else {
            self.pos = None;
        }

        Some(result)
    }
}

#[macro_export]
macro_rules! convert_point {
    ($val:expr, $type:ty) => {
        Point3 {
            x: $val.x as $type,
            y: $val.y as $type,
            z: $val.z as $type,
        }
    };
}

#[macro_export]
macro_rules! convert_vec {
    ($val:expr, $type:ty) => {
        Vector3 {
            x: $val.x as $type,
            y: $val.y as $type,
            z: $val.z as $type,
        }
    };
}

macro_rules! impl_div_traits_int {
    ($type:ty) => {
        impl DivDown for $type {
            #[inline]
            fn div_down(&self, divisor: &$type) -> $type {
                self / divisor
            }
        }

        impl DivUp for $type {
            #[inline]
            fn div_up(&self, divisor: &$type) -> $type {
                let d = self / divisor;
                let r = self % divisor;
                if r > 0 {
                    1 + d
                } else {
                    d
                }
            }
        }
    };
}

macro_rules! impl_div_trait_pv {
    ($pv:tt, $trait:path, $method:tt) => {
        impl<T> $trait for $pv<T>
        where
            T: $trait,
        {
            #[inline]
            fn $method(&self, divisor: &$pv<T>) -> Self {
                $pv {
                    x: self.x.$method(&divisor.x),
                    y: self.y.$method(&divisor.y),
                    z: self.z.$method(&divisor.z),
                }
            }
        }
    };
}

impl_div_traits_int!(u8);
impl_div_traits_int!(u16);
impl_div_traits_int!(u32);
impl_div_traits_int!(u64);
impl_div_traits_int!(i8);
impl_div_traits_int!(i16);
impl_div_traits_int!(i32);
impl_div_traits_int!(i64);
impl_div_traits_int!(usize);
impl_div_traits_int!(isize);

impl_div_trait_pv!(Vector3, DivDown, div_down);
impl_div_trait_pv!(Vector3, DivUp, div_up);
impl_div_trait_pv!(Point3, DivDown, div_down);
impl_div_trait_pv!(Point3, DivUp, div_up);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_up() {
        assert_eq!(38.div_up(&16), 3);
        assert_eq!(38.div_up(&4), 10);
    }

    fn check_diff(lhs: Bounds<i32>, rhs: Bounds<i32>) {
        use std::collections::HashSet;

        let diff: HashSet<Point3<i32>> = lhs.iter_diff(rhs).collect();
        let naive_diff: HashSet<Point3<i32>> = lhs
            .z_range()
            .flat_map(|z| {
                lhs.y_range().flat_map(move |y| {
                    lhs.x_range().flat_map(move |x| {
                        let p = Point3::new(x, y, z);
                        let p = if rhs.contains_point(p) { None } else { Some(p) };
                        p.into_iter()
                    })
                })
            })
            .collect();

        assert_eq!(diff, naive_diff);
    }

    #[test]
    fn test_iter_diff() {
        let small = Bounds::from_limit(Point3::new(0, 0, 0), Point3::new(3, 3, 3));
        let large = Bounds::from_limit(Point3::new(0, 0, 0), Point3::new(4, 4, 3));

        check_diff(small, large);
        check_diff(large, small);

        let size = Vector3::new(2, 2, 2);
        check_diff(
            Bounds::new(Point3::new(0, 1, 0), size),
            Bounds::new(Point3::new(0, 0, 0), size),
        );

        let size = Vector3::new(2, 2, 2);
        check_diff(
            Bounds::new(Point3::new(1, 1, 0), size),
            Bounds::new(Point3::new(0, 0, 0), size),
        );

        let size = Vector3::new(128, 128, 3);
        check_diff(
            Bounds::new(Point3::new(124, 365, 7), size),
            Bounds::new(Point3::new(70, 340, 6), size),
        );

        let size = Vector3::new(128, 128, 10);
        check_diff(
            Bounds::new(Point3::new(124, 340, 11), size),
            Bounds::new(Point3::new(70, 365, 6), size),
        );

        check_diff(
            Bounds::new(Point3::new(0, 0, 0), Vector3::new(8, 8, 5)),
            Bounds::new(Point3::new(0, 0, 0), Vector3::new(8, 8, 4)),
        );
    }
}
