use cgmath::{InnerSpace, Matrix4, Point3, Transform, Vector3};

use simgame_core::block::{index_utils, Block, BlockUpdater};
use simgame_core::convert_point;
use simgame_core::util::Bounds;

pub trait Primitive {
    /// Returns true if the given point is within the shape.
    fn test(&self, point: Point3<f64>) -> bool;

    /// Returns the smallest cuboid containing the shape.
    fn bounds(&self) -> Bounds<f64>;

    fn draw(&self, blocks: &mut BlockUpdater, fill_block: Block) {
        blocks.set_blocks(
            round_bounds(self.bounds())
                .iter_points_aligned(index_utils::chunk_size())
                .filter(|p| self.test(convert_point!(p, f64)))
                .zip(std::iter::repeat(fill_block)),
        );
    }
}

pub struct ShapeComponent {
    pub fill_block: Block,
    pub primitive: Box<dyn Primitive>,
}

pub struct Shape {
    components: Vec<ShapeComponent>,
}

pub struct Cuboid {
    pub bounds: Bounds<f64>,
}

pub struct FilledLine {
    pub start: Point3<f64>,
    pub end: Point3<f64>,
    pub radius: f64,
    pub round_start: bool,
    pub round_end: bool,
}

pub struct Sphere {
    pub center: Point3<f64>,
    pub radius: f64,
}

pub struct AffineTransform<T> {
    primitive: T,
    transform: Matrix4<f64>,
    inv_transform: Matrix4<f64>,
}

impl Shape {
    pub fn new(components: Vec<ShapeComponent>) -> Self {
        Self { components }
    }

    pub fn empty() -> Self {
        Self::new(vec![])
    }

    pub fn segmented_line<Points>(
        points: Points,
        radius: f64,
        round_start: bool,
        round_end: bool,
        fill_block: Block,
    ) -> Self
    where
        Points: IntoIterator<Item = Point3<f64>>,
    {
        let iter = util::IterTagged::new(util::IterPairs::new(points.into_iter()));
        let mut segments: Vec<ShapeComponent> = Vec::new();

        for (is_start, is_end, (point0, point1)) in iter {
            segments.push(ShapeComponent {
                primitive: Box::new(FilledLine {
                    start: point0,
                    end: point1,
                    radius,
                    round_start: !is_start || round_start,
                    round_end: !is_end || round_end,
                }),
                fill_block,
            });
        }

        Self::new(segments)
    }

    pub fn components(&self) -> &[ShapeComponent] {
        self.components.as_slice()
    }

    pub fn push_primitive<T>(&mut self, fill_block: Block, primitive: T)
    where
        T: Primitive + 'static,
    {
        self.components.push(ShapeComponent {
            fill_block,
            primitive: Box::new(primitive),
        })
    }

    pub fn from_primitive<T>(fill_block: Block, primitive: T) -> Self
    where
        T: Primitive + 'static,
    {
        Self::new(vec![ShapeComponent {
            fill_block,
            primitive: Box::new(primitive),
        }])
    }

    pub fn draw_transformed(&self, blocks: &mut BlockUpdater, transform: Matrix4<f64>) {
        let inv_transform = transform
            .inverse_transform()
            .expect("draw_transformed expects an invertible transform matrix");
        for component in &self.components {
            let primitive = AffineTransform {
                primitive: &*component.primitive,
                transform,
                inv_transform,
            };
            primitive.draw(blocks, component.fill_block);
        }
    }

    pub fn draw(&self, blocks: &mut BlockUpdater) {
        for component in &self.components {
            component.primitive.draw(blocks, component.fill_block);
        }
    }
}

fn check_matrix_valid(matrix: Matrix4<f64>) {
    let cols: [[f64; 4]; 4] = matrix.into();
    for col in &cols {
        for cell in col {
            assert!(!cell.is_nan());
        }
    }
}

impl<T> AffineTransform<T> {
    /// Returns `None` if `transform` is not invertible.
    pub fn new(primitive: T, transform: Matrix4<f64>) -> Option<Self> {
        let inv_transform = transform.inverse_transform()?;
        if cfg!(debug) {
            check_matrix_valid(transform);
            check_matrix_valid(inv_transform);
        }
        Some(Self {
            primitive,
            transform,
            inv_transform,
        })
    }
}

impl<T> Primitive for AffineTransform<T>
where
    T: Primitive,
{
    fn bounds(&self) -> Bounds<f64> {
        Bounds::from_points(
            self.primitive
                .bounds()
                .corners()
                .iter()
                .map(|&p| self.transform.transform_point(p)),
        )
        .unwrap()
    }

    fn test(&self, point: Point3<f64>) -> bool {
        self.primitive
            .test(self.inv_transform.transform_point(point))
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

        return distance <= self.radius;
    }

    fn bounds(&self) -> Bounds<f64> {
        line_bounds(self.start, self.end, self.radius)
    }
}

impl Primitive for Cuboid {
    fn test(&self, point: Point3<f64>) -> bool {
        self.bounds.contains_point(point)
    }

    fn bounds(&self) -> Bounds<f64> {
        self.bounds
    }
}

impl Primitive for Sphere {
    fn bounds(&self) -> Bounds<f64> {
        let radius_vec = Vector3::new(self.radius, self.radius, self.radius);
        Bounds::from_limit(self.center - radius_vec, self.center + radius_vec)
    }

    fn test(&self, point: Point3<f64>) -> bool {
        return (point - self.center).magnitude2() < self.radius * self.radius;
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
fn line_projection(
    start: Point3<f64>,
    end: Point3<f64>,
    point: Point3<f64>,
) -> (f64, Point3<f64>) {
    let direction = end - start;

    let t = (point - start).dot(direction) / direction.magnitude2();
    let proj = start + t * direction;

    (t, proj)
}

mod util {
    pub struct IterPairs<Iter>
    where
        Iter: Iterator,
    {
        prev: Option<Iter::Item>,
        iter: Iter,
    }

    impl<Iter> IterPairs<Iter>
    where
        Iter: Iterator,
    {
        pub fn new(iter: Iter) -> Self {
            Self { prev: None, iter }
        }
    }

    impl<Iter> Iterator for IterPairs<Iter>
    where
        Iter: Iterator,
        Iter::Item: Clone,
    {
        type Item = (Iter::Item, Iter::Item);

        fn next(&mut self) -> Option<Self::Item> {
            let prev = match self.prev.take() {
                None => self.iter.next()?,
                Some(prev) => prev,
            };

            let item = self.iter.next()?;
            self.prev = Some(item.clone());
            Some((prev, item))
        }
    }

    pub struct IterTagged<Iter>
    where
        Iter: Iterator,
    {
        is_start: bool,
        iter: std::iter::Peekable<Iter>,
    }

    impl<Iter> IterTagged<Iter>
    where
        Iter: Iterator,
    {
        pub fn new(iter: Iter) -> Self {
            Self {
                is_start: true,
                iter: iter.peekable(),
            }
        }
    }

    impl<Iter> Iterator for IterTagged<Iter>
    where
        Iter: Iterator,
    {
        type Item = (bool, bool, Iter::Item);

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.iter.next()?;
            let is_end = self.iter.peek().is_none();
            let is_start = self.is_start;
            self.is_start = false;
            Some((is_start, is_end, item))
        }
    }
}

impl<P> Primitive for &P
where
    P: Primitive,
{
    fn test(&self, point: Point3<f64>) -> bool {
        use std::ops::Deref;
        self.deref().test(point)
    }

    fn bounds(&self) -> Bounds<f64> {
        use std::ops::Deref;
        self.deref().bounds()
    }
}

impl Primitive for &dyn Primitive {
    fn test(&self, point: Point3<f64>) -> bool {
        use std::ops::Deref;
        self.deref().test(point)
    }

    fn bounds(&self) -> Bounds<f64> {
        use std::ops::Deref;
        self.deref().bounds()
    }
}
