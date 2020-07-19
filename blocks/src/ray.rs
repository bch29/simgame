use cgmath::{BaseFloat, BaseNum, InnerSpace, Point3, Vector3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray<T> {
    pub origin: Point3<T>,
    pub dir: Vector3<T>,
}

/// Invariant: `vertical` is orthogonal to `horizontal`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect<T> {
    pub origin: Point3<T>,
    pub horizontal: Vector3<T>,
    pub vertical: Vector3<T>,
}

/// A point where a ray intersected with a surface. Contains the t-value identifying the point
/// along the ray where it hit, and the normal to the surface at the point of intersection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Intersection<T> {
    pub t: T,
    pub normal: Vector3<T>,
}

/// The result of a ray intersection test with a convex 3-dimensional object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvexRaycastResult<T> {
    /// The ray missed the object.
    Miss,
    /// The ray started within the object and hit it on the way out.
    ExitOnly { exit: Intersection<T> },
    /// The ray passed through the object, hitting it on entry and exit.
    PassThrough {
        entry: Intersection<T>,
        exit: Intersection<T>,
    },
    /// The ray clip a corner or edge of the object, intersecting in one place.
    Clip { clip: Intersection<T> },
}

impl<T> Ray<T>
where
    T: BaseNum,
{
    /// Evaluate the ray at a t-value to get a concrete position.
    pub fn get(&self, t: T) -> Point3<T> {
        self.origin + self.dir * t
    }

    /// Move the ray's origin by some offset.
    pub fn offset(&self, offset: Vector3<T>) -> Self {
        Self {
            origin: self.origin + offset,
            dir: self.dir,
        }
    }
}

impl<T> Ray<T>
where
    T: BaseFloat,
{
    /// Tests if a plane intersects the ray. If this function returns `None` it means the ray is
    /// parallel to the plane and therefore does not intersect it.
    pub fn test_plane(&self, origin: Point3<T>, normal: Vector3<T>) -> Option<Intersection<T>> {
        // ray: x = a + t*d
        // plane: n . (x - p) = 0

        // intersection calculation:
        // sub x into plane equation: n . (a + t * d - p) = 0
        // expand dot product:        n . (a - p) + n . (t * d) = 0
        // rearrange:                 t * n . d = n . (p - a)
        // solve for t:               t = (n . (p - a)) / (n . d)
        // N.B. if the divisor is zero it means there is no intersection

        let divisor = normal.dot(self.dir);
        if divisor.abs_diff_eq(&T::zero(), T::default_epsilon()) {
            return None;
        }

        let t = normal.dot(origin - self.origin) / divisor;
        Some(Intersection { t, normal })
    }

    /// Tests if a rect intersects the ray. If so, returns the ray parameter at the point of
    /// intersection.
    ///
    /// Returns bogus results if the rect's `horizontal` and `vertical` are not orthogonal.
    pub fn test_rect(&self, rect: &Rect<T>) -> Option<Intersection<T>> {
        let int = self.test_plane(rect.origin, rect.normal())?;
        let point = self.get(int.t);

        // `point` lies on the plane that contains the rect. Now find the x and y coordinates
        // within the rect (treating `horizontal` as the x axis and `vertical` as the y axis).
        // The point at x=1, y=1 is the top-right corner and the point at x=0, y=0 is the origin.
        // So if x and y lie between those bounds then the ray intersects the rect at t.

        let x = (point - rect.origin).dot(rect.horizontal) / rect.horizontal.magnitude2();
        let y = (point - rect.origin).dot(rect.vertical) / rect.vertical.magnitude2();

        if x >= T::zero() && x <= T::one() && y >= T::zero() && y <= T::one() {
            Some(int)
        } else {
            None
        }
    }
}

impl<T> ConvexRaycastResult<T> {
    pub fn entry(&self) -> Option<Intersection<T>>
    where
        T: Clone,
    {
        use ConvexRaycastResult::*;

        match self {
            PassThrough { entry, .. } => Some(entry.clone()),
            Clip { clip } => Some(clip.clone()),
            _ => None,
        }
    }

    pub fn exit(&self) -> Option<Intersection<T>>
    where
        T: Clone,
    {
        use ConvexRaycastResult::*;

        match self {
            ExitOnly { exit } => Some(exit.clone()),
            PassThrough { exit, .. } => Some(exit.clone()),
            Clip { clip } => Some(clip.clone()),
            _ => None,
        }
    }
}

impl<T> Rect<T> {
    pub fn normal(&self) -> Vector3<T>
    where
        T: BaseFloat,
    {
        self.horizontal.cross(self.vertical).normalize()
    }

    /// Returns a rect representing the back face of `self`.
    pub fn flipped(&self) -> Rect<T>
    where
        T: BaseNum,
        Vector3<T>: std::ops::Neg<Output = Vector3<T>>,
    {
        Rect {
            origin: self.origin + self.horizontal,
            horizontal: -self.horizontal,
            vertical: self.vertical,
        }
    }
}
