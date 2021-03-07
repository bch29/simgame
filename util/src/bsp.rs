//! Binary space partitioning tree.

use cgmath::Point3;

use crate::bounds::Bounds;
use crate::ray::Ray;

#[derive(Clone, Debug)]
pub struct Bsp<T> {
    root_node: OptNode,
    nodes: NodePool<T>,
}

impl<T> Bsp<T> {
    pub fn new() -> Self {
        Self {
            root_node: OptNode::new(),
            nodes: Pool::new(),
        }
    }

    pub fn clear(&mut self) {
        self.root_node = OptNode::new();
        self.nodes.clear();
    }

    #[inline]
    pub fn insert(&mut self, point: Point3<f64>, value: T) -> usize {
        self.root_node
            .insert(&mut self.nodes, Axis::X, point, value)
    }

    #[inline]
    pub fn iter<F>(&self, bounds: Option<Bounds<f64>>, mut func: F)
    where
        F: FnMut(usize, Point3<f64>, &T),
    {
        self.root_node.iter(&self.nodes, Axis::X, bounds, &mut func)
    }

    #[inline]
    pub fn iter_mut<F>(&mut self, bounds: Option<Bounds<f64>>, mut func: F)
    where
        F: FnMut(usize, Point3<f64>, &mut T),
    {
        self.root_node
            .iter_mut(&mut self.nodes, Axis::X, bounds, &mut func)
    }

    /// Iterates over nodes whose contained space intersects with `bounds`, even if the stored
    /// point does not intersect with `bounds`. The `Bounds` passed to the callback is the
    /// intersection of the node's space with `bounds`.
    #[inline]
    pub fn iter_with_bounds<F>(&self, bounds: Bounds<f64>, mut func: F)
    where
        F: FnMut(usize, Bounds<f64>, Point3<f64>, &T),
    {
        self.root_node
            .iter_with_bounds(&self.nodes, Axis::X, bounds, &mut func);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.root_node.depth(&self.nodes)
    }

    pub fn build_boundmap<F>(&self, mut object_bounds: F) -> BspBoundmap
    where
        F: FnMut(&T) -> Bounds<f64>,
    {
        let mut nodes: Vec<BoundmapNodeInfo> = self
            .nodes
            .iter()
            .map(|branch| BoundmapNodeInfo {
                bounds: object_bounds(&branch.value),
                contained_indices: Vec::new(),
            })
            .collect();

        for contained_ix in 0..nodes.len() {
            let bounds = nodes[contained_ix].bounds;

            self.iter_with_bounds(bounds, |containing_ix, _, _, _| {
                nodes[containing_ix].contained_indices.push(contained_ix);
            });
        }

        BspBoundmap { nodes }
    }

    /// Finds objects in the Bsp which intersect the given ray. The `test` function is called on
    /// each object whose bounding box intersects with the ray. Stops if the `test` function
    /// returns `Some`. Guaranteed to visit items in sorted order of t-value.
    pub fn cast_ray<'a, R, F>(
        &'a self,
        boundmap: &BspBoundmap,
        ray: &Ray<f64>,
        mut test: F,
    ) -> Option<RaycastHit<R>>
    where
        F: FnMut(Point3<f64>, &'a T) -> Option<R>,
        R: 'a,
    {
        self.root_node
            .cast_ray(&self.nodes, boundmap, Axis::X, ray, &mut test)
    }
}

/// Data structure for accelerating raycasts into a Bsp whose objects have 3D bounding boxes.
#[derive(Clone, Debug)]
pub struct BspBoundmap {
    // contained_indices[i] contains every index j such that node j's bounding box intersects with
    // the space contained within node i
    nodes: Vec<BoundmapNodeInfo>,
}

#[derive(Clone, Debug)]
struct BoundmapNodeInfo {
    bounds: Bounds<f64>,
    // contains every index j such that node j's bounding box intersects with the space contained
    // within the entry's node
    contained_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaycastHit<T> {
    pub data: T,
    pub pos: Point3<i64>,
}

impl<T> Default for Bsp<T> {
    fn default() -> Self {
        Self::new()
    }
}

type NodePool<T> = Pool<Node<T>>;

#[derive(Clone, Copy, Debug)]
struct OptNode {
    branch_ix: Option<usize>,
}

#[derive(Clone, Debug)]
struct Node<T> {
    point: Point3<f64>,
    value: T,
    lower: OptNode,
    upper: OptNode,
}

#[derive(Clone, Debug)]
struct Pool<T> {
    nodes: Vec<T>,
}

impl<T> std::ops::Index<usize> for Pool<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<T> std::ops::IndexMut<usize> for Pool<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug)]
enum DividePointResult {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
enum DivideBoundsResult {
    Lower,
    Upper,
    Both,
}

impl<T> Pool<T> {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    #[inline]
    fn push(&mut self, item: T) -> usize {
        let ix = self.nodes.len();
        self.nodes.push(item);
        ix
    }

    #[inline]
    fn clear(&mut self) {
        self.nodes.clear()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.nodes.iter()
    }
}

impl OptNode {
    fn new() -> Self {
        Self { branch_ix: None }
    }

    fn insert<T>(
        &mut self,
        nodes: &mut NodePool<T>,
        axis: Axis,
        point: Point3<f64>,
        value: T,
    ) -> usize {
        match self.branch_ix {
            None => {
                let res = nodes.push(Node {
                    point,
                    value,
                    lower: OptNode { branch_ix: None },
                    upper: OptNode { branch_ix: None },
                });
                self.branch_ix = Some(res);
                res
            }
            Some(branch_ix) => {
                // look up branch and take temporary copies of nodes to avoid holding on to nodes
                // reference, which we later need as mutable
                let Node {
                    point: branch_point,
                    lower,
                    upper,
                    ..
                } = &nodes[branch_ix];
                let mut tmp_lower = *lower;
                let mut tmp_upper = *upper;

                // update temporary copies
                let res = match axis.divide_point(*branch_point, point) {
                    DividePointResult::Lower => tmp_lower.insert(nodes, axis.next(), point, value),
                    DividePointResult::Upper => tmp_upper.insert(nodes, axis.next(), point, value),
                };

                // insert updated nodes back into tree
                let Node { lower, upper, .. } = &mut nodes[branch_ix];
                *lower = tmp_lower;
                *upper = tmp_upper;

                res
            }
        }
    }

    fn iter<T, F>(
        &self,
        nodes: &NodePool<T>,
        axis: Axis,
        bounds: Option<Bounds<f64>>,
        func: &mut F,
    ) where
        F: FnMut(usize, Point3<f64>, &T),
    {
        let branch_ix = match self.branch_ix {
            None => return,
            Some(branch_ix) => branch_ix,
        };
        let Node {
            point,
            lower,
            upper,
            value,
        } = &nodes[branch_ix];

        let div_result = match bounds {
            None => {
                func(branch_ix, *point, value);
                DivideBoundsResult::Both
            }
            Some(bounds) => {
                if bounds.contains_point(*point) {
                    func(branch_ix, *point, value);
                }

                axis.divide_bounds(*point, bounds)
            }
        };

        if div_result.contains_upper() {
            upper.iter(nodes, axis.next(), bounds, func);
        }

        if div_result.contains_lower() {
            lower.iter(nodes, axis.next(), bounds, func);
        }
    }

    fn iter_mut<T, F>(
        &self,
        nodes: &mut NodePool<T>,
        axis: Axis,
        bounds: Option<Bounds<f64>>,
        func: &mut F,
    ) where
        F: FnMut(usize, Point3<f64>, &mut T),
    {
        let branch_ix = match self.branch_ix {
            None => return,
            Some(branch_ix) => branch_ix,
        };
        let Node {
            point,
            lower,
            upper,
            value,
        } = &mut nodes[branch_ix];

        // take copies of nodes in order to drop mutable reference to nodes
        let lower = *lower;
        let upper = *upper;

        let div_result = match bounds {
            None => {
                func(branch_ix, *point, value);
                DivideBoundsResult::Both
            }
            Some(bounds) => {
                if bounds.contains_point(*point) {
                    func(branch_ix, *point, value);
                }

                axis.divide_bounds(*point, bounds)
            }
        };

        if div_result.contains_upper() {
            upper.iter_mut(nodes, axis.next(), bounds, func);
        }

        if div_result.contains_lower() {
            lower.iter_mut(nodes, axis.next(), bounds, func);
        }
    }

    fn depth<T>(&self, nodes: &NodePool<T>) -> usize {
        match self.branch_ix {
            None => 0,
            Some(branch_ix) => {
                let Node { lower, upper, .. } = &nodes[branch_ix];
                1 + lower.depth(nodes).max(upper.depth(nodes))
            }
        }
    }

    fn iter_with_bounds<T, F>(
        &self,
        nodes: &NodePool<T>,
        axis: Axis,
        bounds: Bounds<f64>,
        func: &mut F,
    ) where
        F: FnMut(usize, Bounds<f64>, Point3<f64>, &T),
    {
        let branch_ix = match self.branch_ix {
            None => return,
            Some(ix) => ix,
        };
        let Node {
            upper,
            lower,
            value,
            point,
        } = &nodes[branch_ix];

        func(branch_ix, bounds, *point, value);

        let (upper_bounds, lower_bounds) = axis.cut_bounds(*point, bounds);

        if let Some(upper_bounds) = upper_bounds {
            upper.iter_with_bounds(nodes, axis.next(), upper_bounds, func);
        }

        if let Some(lower_bounds) = lower_bounds {
            lower.iter_with_bounds(nodes, axis.next(), lower_bounds, func);
        }
    }

    fn cast_ray<'a, T, R, F>(
        &self,
        nodes: &'a NodePool<T>,
        boundmap: &BspBoundmap,
        axis: Axis,
        ray: &Ray<f64>,
        test: &mut F,
    ) -> Option<RaycastHit<R>>
    where
        F: FnMut(Point3<f64>, &'a T) -> Option<R>,
        R: 'a,
    {
        unimplemented!()
    }
}

impl Axis {
    fn next(self) -> Self {
        match self {
            Axis::X => Axis::Y,
            Axis::Y => Axis::Z,
            Axis::Z => Axis::X,
        }
    }

    fn divide_point(self, center: Point3<f64>, dividend: Point3<f64>) -> DividePointResult {
        let offset = dividend - center;
        match self {
            Axis::X => {
                if offset.x < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
            Axis::Y => {
                if offset.y < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
            Axis::Z => {
                if offset.z < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
        }
    }

    fn divide_bounds(self, center: Point3<f64>, bounds: Bounds<f64>) -> DivideBoundsResult {
        match (
            self.divide_point(center, bounds.origin()),
            self.divide_point(center, bounds.limit()),
        ) {
            (DividePointResult::Upper, _) => DivideBoundsResult::Upper,
            (DividePointResult::Lower, DividePointResult::Upper) => DivideBoundsResult::Both,
            (_, DividePointResult::Lower) => DivideBoundsResult::Lower,
        }
    }

    /// Cuts the bounds along the plane aligned with the axis containing the given center point.
    /// Returns (bounds with lower axis coordinate, bounds with higher axis coordinate).
    fn cut_bounds(
        self,
        center: Point3<f64>,
        bounds: Bounds<f64>,
    ) -> (Option<Bounds<f64>>, Option<Bounds<f64>>) {
        match self {
            Axis::X => {
                if center.x < bounds.origin().x {
                    (None, Some(bounds))
                } else if center.x > bounds.limit().x {
                    (Some(bounds), None)
                } else {
                    let a = bounds.origin();
                    let mut b1 = bounds.limit();
                    b1.x = center.x;
                    let mut b2 = bounds.origin();
                    b2.x = center.x;
                    let c = bounds.limit();
                    (
                        Some(Bounds::from_limit(a, b1)),
                        Some(Bounds::from_limit(b2, c)),
                    )
                }
            }
            Axis::Y => {
                if center.y < bounds.origin().y {
                    (None, Some(bounds))
                } else if center.y > bounds.limit().y {
                    (Some(bounds), None)
                } else {
                    let a = bounds.origin();
                    let mut b1 = bounds.limit();
                    b1.y = center.y;
                    let mut b2 = bounds.origin();
                    b2.y = center.y;
                    let c = bounds.limit();
                    (
                        Some(Bounds::from_limit(a, b1)),
                        Some(Bounds::from_limit(b2, c)),
                    )
                }
            }
            Axis::Z => {
                if center.z < bounds.origin().z {
                    (None, Some(bounds))
                } else if center.z > bounds.limit().z {
                    (Some(bounds), None)
                } else {
                    let a = bounds.origin();
                    let mut b1 = bounds.limit();
                    b1.z = center.z;
                    let mut b2 = bounds.origin();
                    b2.z = center.z;
                    let c = bounds.limit();
                    (
                        Some(Bounds::from_limit(a, b1)),
                        Some(Bounds::from_limit(b2, c)),
                    )
                }
            }
        }
    }
}

impl DivideBoundsResult {
    fn contains_upper(self) -> bool {
        !matches!(self, DivideBoundsResult::Lower)
    }

    fn contains_lower(self) -> bool {
        !matches!(self, DivideBoundsResult::Upper)
    }
}

#[cfg(test)]
mod tests {
    use super::Bsp;

    use cgmath::{Point3, Vector3};

    use crate::bounds::Bounds;

    #[test]
    fn test_bsp() {
        let mut tree: Bsp<i32> = Bsp::new();

        tree.insert(Point3::new(0., 0., 0.), 0);
        tree.insert(Point3::new(-10., 5., 4.), 1);
        tree.insert(Point3::new(10., 5., 4.), 2);
        tree.insert(Point3::new(5., 2., -8.), 3);
        tree.insert(Point3::new(0.1, 0.1, 0.1), 4);
        tree.insert(Point3::new(0.2, 0.2, 0.2), 5);

        let mut found = Vec::new();
        tree.iter(
            Some(Bounds::new(
                Point3::new(-0.1, -0.1, -0.1),
                Vector3::new(2., 2., 2.),
            )),
            |ix, p, x| found.push((ix, p, *x)),
        );

        assert_eq!(
            found.into_iter().map(|(_, _, x)| x).collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
    }
}
