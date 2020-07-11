use cgmath::{BaseFloat, BaseNum, ElementWise, EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::ray::{ConvexRaycastResult, Intersection, Ray, Rect};

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
    /// Divides and rounds the result towards negative infinity.
    fn div_down(self, divisor: Self) -> Self;
}

pub trait DivUp {
    /// Divides and rounds the result towards positive infinity.
    fn div_up(self, divisor: Self) -> Self;
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

    /// Returns rects describing each face of the bounds.
    #[inline]
    pub fn face_rects(&self) -> [Rect<T>; 6]
    where
        Vector3<T>: std::ops::Neg<Output = Vector3<T>>,
    {
        let Point3 {
            x: left,
            y: bottom,
            z: front,
        } = self.origin;
        let Point3 {
            x: right,
            y: top,
            z: back,
        } = self.limit();

        let sx = Vector3::new(self.size.x, T::zero(), T::zero());
        let sy = Vector3::new(T::zero(), self.size.y, T::zero());
        let sz = Vector3::new(T::zero(), T::zero(), self.size.z);

        let p = Point3::new;

        [
            // front
            Rect {
                origin: p(left, bottom, front),
                horizontal: sx,
                vertical: sy,
            }
            .flipped(),
            // back
            Rect {
                origin: p(right, bottom, back),
                horizontal: -sx,
                vertical: sy,
            }
            .flipped(),
            // left
            Rect {
                origin: p(left, bottom, back),
                horizontal: -sz,
                vertical: sy,
            }
            .flipped(),
            // right
            Rect {
                origin: p(right, bottom, front),
                horizontal: sz,
                vertical: sy,
            }
            .flipped(),
            // bottom
            Rect {
                origin: p(left, bottom, back),
                horizontal: sx,
                vertical: -sz,
            }
            .flipped(),
            // top
            Rect {
                origin: p(left, top, front),
                horizontal: sx,
                vertical: sz,
            }
            .flipped(),
        ]
    }

    #[inline]
    /// Checks if a ray intersects the bounds and if so returns the entry and exit points. These
    /// may be the same if the ray just touches the bounds on an edge or corner.
    pub fn cast_ray(&self, ray: &Ray<T>) -> ConvexRaycastResult<T>
    where
        T: BaseFloat,
    {
        // algorithm: test for intersection with each plane of the bounds and insert into the
        // `results` array when we get a hit. Sort results by t-value to find entry and exit
        // points.

        let faces = self.face_rects();
        let mut results: [Option<Intersection<T>>; 6] = [None; 6];

        for i in 0..6 {
            results[i] = match ray.test_rect(&faces[i]) {
                Some(int) if int.t >= T::zero() => Some(int),
                _ => None,
            };
        }

        results.sort_by_key(|x| x.map(|p| OrdFloat(p.t)));

        let mut results_iter = results.iter().flatten().cloned();

        let first = results_iter.next();
        let second = results_iter.next();

        match (first, second) {
            (Some(exit), None) if self.contains_point(ray.origin) => {
                ConvexRaycastResult::ExitOnly { exit }
            }
            (Some(clip), None) => ConvexRaycastResult::Clip { clip },
            (Some(entry), Some(exit)) => ConvexRaycastResult::PassThrough { entry, exit },
            _ => ConvexRaycastResult::Miss,
        }
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
            self.origin.div_down(Point3::origin() + quantum_size),
            self.limit().div_up(Point3::origin() + quantum_size),
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
    pub fn corners(self) -> [Point3<T>; 8] {
        let a = self.origin();
        let b = self.limit();

        [
            Point3::new(a.x, a.y, a.z),
            Point3::new(a.x, a.y, b.z),
            Point3::new(a.x, b.y, a.z),
            Point3::new(a.x, b.y, b.z),
            Point3::new(b.x, a.y, a.z),
            Point3::new(b.x, a.y, b.z),
            Point3::new(b.x, b.y, a.z),
            Point3::new(b.x, b.y, b.z),
        ]
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrdFloat<T>(T);

impl<T> Eq for OrdFloat<T> where T: PartialEq {}
impl<T> Ord for OrdFloat<T>
where
    T: PartialOrd,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Less)
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
    ($type:ty, $div_up_positive:ident) => {
        // `0 - x` used for negation so that unsigned types can share the implementation. We will
        // never actually compute `0 - x` for an unsigned x because there is always an `x <= 0`
        // check before a negation. These checks and the corresponding negation code should be
        // optimized away for unsigned types.

        #[allow(unused_comparisons)]
        fn $div_up_positive(dividend: $type, divisor: $type) -> $type {
            assert!(dividend >= 0);
            assert!(divisor > 0);

            let d = dividend / divisor;
            let r = dividend % divisor;
            if r > 0 {
                1 + d
            } else {
                d
            }
        }

        impl DivDown for $type {
            #[inline]
            #[allow(unused_comparisons)]
            fn div_down(mut self, mut divisor: $type) -> $type {
                assert!(divisor != 0);
                if divisor < 0 {
                    self = 0 - self;
                    divisor = 0 - divisor;
                }

                if self >= 0 {
                    self / divisor
                } else {
                    0 - $div_up_positive(0 - self, divisor)
                }
            }
        }

        impl DivUp for $type {
            #[inline]
            #[allow(unused_comparisons)]
            fn div_up(mut self, mut divisor: $type) -> $type {
                assert!(divisor != 0);
                if divisor < 0 {
                    self = 0 - self;
                    divisor = 0 - divisor;
                }

                if self >= 0 {
                    $div_up_positive(self, divisor)
                } else {
                    0 - ((0 - self) / divisor)
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
            fn $method(self, divisor: $pv<T>) -> Self {
                $pv {
                    x: self.x.$method(divisor.x),
                    y: self.y.$method(divisor.y),
                    z: self.z.$method(divisor.z),
                }
            }
        }
    };
}

impl_div_traits_int!(u8, div_up_positive_u8);
impl_div_traits_int!(u16, div_up_positive_u16);
impl_div_traits_int!(u32, div_up_positive_u32);
impl_div_traits_int!(u64, div_up_positive_u64);
impl_div_traits_int!(i8, div_up_positive_i8);
impl_div_traits_int!(i16, div_up_positive_i16);
impl_div_traits_int!(i32, div_up_positive_i32);
impl_div_traits_int!(i64, div_up_positive_i64);
impl_div_traits_int!(usize, div_up_positive_usize);
impl_div_traits_int!(isize, div_up_positive_isize);

impl_div_trait_pv!(Vector3, DivDown, div_down);
impl_div_trait_pv!(Vector3, DivUp, div_up);
impl_div_trait_pv!(Point3, DivDown, div_down);
impl_div_trait_pv!(Point3, DivUp, div_up);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_traits() {
        assert_eq!(38u32.div_up(16), 3);
        assert_eq!(38i32.div_up(4), 10);
        assert_eq!(38u32.div_down(4), 9);

        assert_eq!((-38).div_up(-4), 10);

        assert_eq!((-38).div_up(4), -9);
        assert_eq!((-38).div_down(4), -10);
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
                        if rhs.contains_point(p) {
                            None
                        } else {
                            Some(p)
                        }
                    })
                })
            })
            .collect();

        assert_eq!(diff, naive_diff);
    }

    #[test]
    fn test_iter_diff() {
        let small = Bounds::from_limit(Point3::new(0, 0, 0), Point3::new(3, 3, 3));
        let large = Bounds::from_limit(Point3::new(-1, 0, 0), Point3::new(4, 4, 3));

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

    #[test]
    fn test_quantize_down() {
        let large = Bounds::from_limit(Point3::new(-15, -3, -31), Point3::new(-1, 30, -16));

        let small = large.quantize_down(Vector3::new(16, 16, 16));

        assert_eq!(small.origin(), Point3::new(-1, -1, -2));
        assert_eq!(small.limit(), Point3::new(0, 2, -1));
    }

    #[test]
    fn test_rects() {
        let bounds = Bounds::from_size(Vector3::new(4.0, 5.0, 6.0));

        let faces = bounds.face_rects();

        assert_eq!(faces[0].normal(), -faces[0].flipped().normal());

        assert_eq!(faces[0].normal(), -Vector3::unit_z());
        assert_eq!(faces[1].normal(), Vector3::unit_z());
        assert_eq!(faces[2].normal(), -Vector3::unit_x());
        assert_eq!(faces[3].normal(), Vector3::unit_x());
        assert_eq!(faces[4].normal(), -Vector3::unit_y());
        assert_eq!(faces[5].normal(), Vector3::unit_y());
    }

    #[test]
    fn test_raycast() {
        let bounds = Bounds::from_size(Vector3::new(4.0, 5.0, 6.0));

        {
            let ray = Ray {
                origin: Point3::new(0.0, 0.0, -1.0),
                dir: Vector3::new(1.0, 1.0, 1.0),
            };
            let res = bounds.cast_ray(&ray);

            assert_eq!(
                res,
                ConvexRaycastResult::PassThrough {
                    entry: Intersection {
                        t: 1.0,
                        normal: Vector3::new(0.0, 0.0, -1.0)
                    },
                    exit: Intersection {
                        t: 4.0,
                        normal: Vector3::new(1.0, 0.0, 0.0)
                    }
                }
            );
        }

        {
            let ray = Ray {
                origin: Point3::new(4.0, 0.0, 7.0),
                dir: Vector3::new(-1.0, 1.0, -1.0),
            };
            let res = bounds.cast_ray(&ray);

            assert_eq!(
                res,
                ConvexRaycastResult::PassThrough {
                    entry: Intersection {
                        t: 1.0,
                        normal: Vector3::new(0.0, 0.0, 1.0)
                    },
                    exit: Intersection {
                        t: 4.0,
                        normal: Vector3::new(-1.0, 0.0, 0.0)
                    }
                }
            );
        }
    }
}
