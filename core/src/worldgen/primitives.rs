use crate::block::{index_utils, Block};
use crate::util::Bounds;
use crate::world::BlockUpdater;
use crate::{convert_point, convert_vec};
use cgmath::{InnerSpace, Point3, Vector3};

pub struct Box {
    pub bounds: Bounds<f64>,
}

pub struct FilledLine {
    pub start: Point3<f64>,
    pub end: Point3<f64>,
    pub radius: f64,
    pub round_start: bool,
    pub round_end: bool
}

pub trait Primitive {
    /// Returns true if the given point is within the shape.
    fn test(&self, point: Point3<f64>) -> bool;

    /// Returns the smallest cuboid containing the shape.
    fn bounds(&self) -> Bounds<f64>;

    fn draw(&self, blocks: &mut BlockUpdater, fill_block: Block) {
        blocks.set_blocks(
            round_bounds(self.bounds())
                .iter_points_aligned(convert_vec!(index_utils::chunk_size(), i64))
                .filter(|p| self.test(convert_point!(p, f64)))
                .map(|p| convert_point!(p, usize))
                .zip(std::iter::repeat(fill_block)),
        );
    }
}

impl Primitive for FilledLine {
    fn test(&self, point: Point3<f64>) -> bool {
        // x = a + t * (b - a)
        //
        // 1. find t and x
        // 2. if t < 0, distance to line is distance to a
        // 3. if t > 1, distance to line is distance to b
        // 4. otherwise, distance to line is distance to x

        let (t, proj) = line_projection(self.start, self.end, point);

        let distance = if t < 0. {
            if !self.round_start {
                return false;
            }

            (point - self.start).magnitude()
        } else if t > 1. {
            if !self.round_end {
                return false;
            }

            (self.end - point).magnitude()
        } else {
            (proj - point).magnitude()
        };

        return distance <= self.radius
    }

    fn bounds(&self) -> Bounds<f64> {
        line_bounds(self.start, self.end, self.radius)
    }
}

impl Primitive for Box {
    fn test(&self, point: Point3<f64>) -> bool {
        self.bounds.contains_point(point)
    }

    fn bounds(&self) -> Bounds<f64> {
        self.bounds
    }
}

fn round_bounds(bounds: Bounds<f64>) -> Bounds<i64> {
    let limit = bounds.limit();
    Bounds::from_limit(
        convert_point!(bounds.origin(), i64),
        Point3::new(
            limit.x.ceil() as i64,
            limit.y.ceil() as i64,
            limit.z.ceil() as i64,
        ),
    )
}

fn line_bounds(start: Point3<f64>, end: Point3<f64>, radius: f64) -> Bounds<f64> {
    let ux = Vector3::new(1., 0., 0.);
    let uy = Vector3::new(0., 1., 0.);
    let uz = Vector3::new(0., 0., 1.);

    let points = &[
        start + radius * ux,
        start + radius * uy,
        start + radius * uz,
        start - radius * ux,
        start - radius * uy,
        start - radius * uz,
        end + radius * ux,
        end + radius * uy,
        end + radius * uz,
        end - radius * ux,
        end - radius * uy,
        end - radius * uz,
    ];

    Bounds::from_points(points.iter().copied()).unwrap()
}

/// Given a line in the form `x = a + t * (b - a)` and a point `p`, find the closest point on the
/// line to `p` and the corresponding value of `t`.
fn line_projection(start: Point3<f64>, end: Point3<f64>, point: Point3<f64>) -> (f64, Point3<f64>) {
    let direction = end - start;

    let t = (point - start).dot(direction) / direction.magnitude2();
    let proj = start + t * direction;

    (t, proj)
}
