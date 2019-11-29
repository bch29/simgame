use cgmath::{BaseNum, ElementWise, EuclideanSpace, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Represents a half-open cuboid of points: the origin is inclusive and the limit is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Bounds<T> {
    origin: Point3<T>,
    size: Vector3<T>,
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
    pub fn new(origin: Point3<T>, size: Vector3<T>) -> Self {
        assert!(size.x >= T::zero());
        assert!(size.y >= T::zero());
        assert!(size.z >= T::zero());
        Bounds { origin, size }
    }

    pub fn from_size(size: Vector3<T>) -> Self {
        Bounds::new(Point3::<T>::origin(), size)
    }

    pub fn limit(self) -> Point3<T> {
        self.origin + self.size
    }

    pub fn contains(self, point: Point3<T>) -> bool {
        let limit = self.limit();
        point.x >= self.origin.x
            && point.x < limit.x
            && point.y >= self.origin.y
            && point.y < limit.y
            && point.z >= self.origin.z
            && point.z < limit.z
    }

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
    pub fn with_origin(self, origin: Point3<T>) -> Self {
        let limit = self.limit();
        assert!(origin.x <= limit.x);
        assert!(origin.y <= limit.y);
        assert!(origin.z <= limit.z);
        let size = limit - origin;

        Bounds { origin, size }
    }

    /// Changes the size while keeping the origin the same. Note this also changes the limit.
    pub fn with_size(mut self, size: Vector3<T>) -> Self {
        assert!(size.x >= T::zero());
        assert!(size.y >= T::zero());
        assert!(size.z >= T::zero());
        self.size = size;
        self
    }

    /// Moves the limit while keeping the origin the same. Note this also changes the size.
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
    pub fn quantize_down(self, quantum_size: Vector3<T>) -> Self
    where
        T: DivUp + DivDown,
    {
        Bounds {
            origin: Point3 {
                x: self.origin.x.div_down(&quantum_size.x),
                y: self.origin.y.div_down(&quantum_size.y),
                z: self.origin.z.div_down(&quantum_size.z),
            },
            size: Vector3 {
                x: self.size.x.div_up(&quantum_size.x),
                y: self.size.y.div_up(&quantum_size.y),
                z: self.size.z.div_up(&quantum_size.z),
            },
        }
    }

    pub fn scale_up(self, scale: Vector3<T>) -> Self {
        assert!(scale.x >= T::zero());
        assert!(scale.y >= T::zero());
        assert!(scale.z >= T::zero());

        Bounds {
            origin: self.origin.mul_element_wise(Point3::origin() + scale),
            size: self.size.mul_element_wise(scale),
        }
    }

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
}

impl DivDown for usize {
    fn div_down(&self, divisor: &usize) -> usize {
        (*self as f64 / *divisor as f64).floor() as usize
    }
}

impl DivUp for usize {
    fn div_up(&self, divisor: &usize) -> usize {
        (*self as f64 / *divisor as f64).ceil() as usize
    }
}
