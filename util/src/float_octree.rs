//! Binary space partitioning tree.

#![allow(unused)]

use cgmath::{EuclideanSpace, Point3, Vector3};

use crate::bounds::Bounds;
use crate::ray::{ConvexIntersection, Ray};

type ObjectPool<T> = ListNodePool<Object<T>>;

#[derive(Clone, Debug)]
pub struct Octree<T> {
    root: Option<OctreeLevel>,
    nodes: OctreeNodePool,
    objects: ObjectPool<T>,
}

#[derive(Clone, Copy, Debug)]
pub struct NodeHandle {
    index: usize,
}

impl<T> Octree<T> {
    pub fn new() -> Self {
        Self {
            root: None,
            nodes: NodePool::new(),
            objects: NodePool::new(),
        }
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
        match self.root {
            None => 0,
            Some(root) => 1 + self.nodes.depth(root.node),
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.objects.clear();
        self.root = None;
    }

    #[inline]
    pub fn insert(&mut self, object: Object<T>) -> NodeHandle {
        let root = self.ensure_fits_bounds(object.bounds);
        self.nodes.insert(&mut self.objects, root, object)
    }

    /// Iterates through all objects in the tree whose bounds intersect with the given bounds.
    #[inline]
    pub fn iter<'a, E, F>(&'a self, bounds: Option<Bounds<f64>>, mut func: F) -> Result<(), E>
    where
        F: FnMut(NodeHandle, Bounds<f64>, &'a T) -> Result<(), E>,
        E: 'a,
    {
        if let Some(root) = self.root {
            self.nodes.iter(
                &self.objects,
                root,
                bounds.unwrap_or(root.bounds),
                &mut func,
            )
        } else {
            Ok(())
        }
    }

    /// Iterates through all objects in the tree whose bounds intersect with the given bounds.
    #[inline]
    pub fn iter_mut<'a, E, F>(&mut self, bounds: Option<Bounds<f64>>, mut func: F) -> Result<(), E>
    where
        F: FnMut(NodeHandle, Bounds<f64>, &mut T) -> Result<(), E>,
        E: 'a,
    {
        if let Some(root) = self.root {
            self.nodes.iter_mut(
                &mut self.objects,
                root,
                bounds.unwrap_or(root.bounds),
                &mut func,
            )?
        }
        Ok(())
    }

    /// Finds objects in the Octree which intersect the given ray. The `test` function is called on
    /// each object whose bounding box intersects with the ray.  returns `Some`. No guarantees are
    /// made about visit order.
    #[inline]
    pub fn cast_ray<'a, E, F>(&'a self, ray: &Ray<f64>, mut test: F) -> Result<(), E>
    where
        F: FnMut(ConvexIntersection<f64>, &'a Object<T>) -> Result<(), E>,
        E: 'a,
    {
        if let Some(root) = self.root {
            self.nodes.cast_ray(&self.objects, root, ray, &mut test)?;
        }
        Ok(())
    }

    /// Ensures the tree is large enough to contain the given bounds. If not, expands until it is
    /// large enough. Returns the root node index (after expanding, if necessary).
    fn ensure_fits_bounds(&mut self, object_bounds: Bounds<f64>) -> OctreeLevel {
        let mut root = match self.root {
            Some(root) => root,
            None => {
                let node = self.nodes.push(OctreeNode::new());
                let root = OctreeLevel {
                    node,
                    bounds: object_bounds,
                    axis: Axis::X,
                };
                self.root = Some(root);
                return root;
            }
        };

        while !root.bounds.contains_bounds(object_bounds) {
            let axis = root.axis.prev();
            let (bounds, old_location) = if !root.bounds.contains_point(object_bounds.origin()) {
                axis.expand_bounds(root.bounds, object_bounds.origin())
            } else {
                axis.expand_bounds(root.bounds, object_bounds.limit())
            };

            let mut new_root = OctreeNode::new();
            match old_location {
                NodeLocation::Lower => new_root.lower = Some(root.node),
                NodeLocation::Upper => new_root.upper = Some(root.node),
            }

            root.node = self.nodes.push(new_root);
            root.axis = axis;
            root.bounds = bounds;
        }

        self.root = Some(root);
        root
    }
}

impl<T> Default for Octree<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
struct OctreeNode {
    lower: Option<NodeHandle>,
    upper: Option<NodeHandle>,
    // stores objects whose bounding boxes fit fully inside this node, but not inside either child
    // node
    objects: Option<NodeHandle>,
}

#[derive(Clone, Debug)]
struct NodePool<T> {
    items: Vec<T>,
}

type OctreeNodePool = NodePool<OctreeNode>;

#[derive(Clone, Debug)]
pub struct Object<T> {
    pub bounds: Bounds<f64>,
    pub value: T,
}

impl<T> std::ops::Index<NodeHandle> for NodePool<T> {
    type Output = T;

    fn index(&self, index: NodeHandle) -> &Self::Output {
        &self.items[index.index]
    }
}

impl<T> std::ops::IndexMut<NodeHandle> for NodePool<T> {
    fn index_mut(&mut self, index: NodeHandle) -> &mut Self::Output {
        &mut self.items[index.index]
    }
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug)]
enum NodeLocation {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
enum DivideBoundsResult {
    Lower,
    Upper,
    Both,
}

impl<T> NodePool<T> {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    #[inline]
    fn push(&mut self, item: T) -> NodeHandle {
        let index = self.items.len();
        self.items.push(item);
        NodeHandle { index }
    }

    #[inline]
    fn clear(&mut self) {
        self.items.clear()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn iter_nodes(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

impl OctreeNode {
    fn new() -> Self {
        Self {
            lower: None,
            upper: None,
            objects: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OctreeLevel {
    node: NodeHandle,
    bounds: Bounds<f64>,
    axis: Axis,
}

impl OctreeLevel {
    fn next(self, node: NodeHandle, bounds: Bounds<f64>) -> Self {
        OctreeLevel {
            node,
            bounds,
            axis: self.axis.next(),
        }
    }
}

impl OctreeNodePool {
    fn insert<T>(
        &mut self,
        objects: &mut ObjectPool<T>,
        level: OctreeLevel,
        object: Object<T>,
    ) -> NodeHandle {
        let (lower_bounds, upper_bounds) = level.axis.cut_bounds(level.bounds);

        if lower_bounds.contains_bounds(object.bounds) {
            let lower = match self[level.node].lower {
                None => {
                    let lower = self.push(OctreeNode::new());
                    self[level.node].lower = Some(lower);
                    lower
                }
                Some(lower) => lower,
            };

            self.insert(objects, level.next(lower, lower_bounds), object)
        } else if upper_bounds.contains_bounds(object.bounds) {
            let upper = match self[level.node].upper {
                None => {
                    let upper = self.push(OctreeNode::new());
                    self[level.node].upper = Some(upper);
                    upper
                }
                Some(upper) => upper,
            };

            self.insert(objects, level.next(upper, upper_bounds), object)
        } else {
            self[level.node].objects = Some(objects.add(self[level.node].objects, object));
            level.node
        }
    }

    fn iter<'a, T, E, F>(
        &'a self,
        objects: &'a ObjectPool<T>,
        level: OctreeLevel,
        iter_bounds: Bounds<f64>,
        func: &mut F,
    ) -> Result<(), E>
    where
        F: FnMut(NodeHandle, Bounds<f64>, &'a T) -> Result<(), E>,
        E: 'a,
    {
        let (lower_bounds, upper_bounds) = level.axis.cut_bounds(level.bounds);

        if let Some(lower) = self[level.node].lower {
            if iter_bounds.intersection(lower_bounds).is_some() {
                self.iter(objects, level.next(lower, lower_bounds), iter_bounds, func)?
            }
        }

        objects.iter(self[level.node].objects, &mut |object| {
            if iter_bounds.intersection(object.bounds).is_some() {
                func(level.node, object.bounds, &object.value)?;
            }
            Ok(())
        })?;

        if let Some(upper) = self[level.node].upper {
            if iter_bounds.intersection(upper_bounds).is_some() {
                self.iter(objects, level.next(upper, upper_bounds), iter_bounds, func)?
            }
        }

        Ok(())
    }

    fn iter_mut<'a, T, E, F>(
        &'a mut self,
        objects: &'a mut ObjectPool<T>,
        level: OctreeLevel,
        iter_bounds: Bounds<f64>,
        func: &mut F,
    ) -> Result<(), E>
    where
        F: FnMut(NodeHandle, Bounds<f64>, &mut T) -> Result<(), E>,
        E: 'a,
    {
        let (lower_bounds, upper_bounds) = level.axis.cut_bounds(level.bounds);

        if let Some(lower) = self[level.node].lower {
            if iter_bounds.intersection(lower_bounds).is_some() {
                self.iter_mut(objects, level.next(lower, lower_bounds), iter_bounds, func)?
            }
        }

        objects.iter_mut(self[level.node].objects, &mut |object| {
            if iter_bounds.intersection(object.bounds).is_some() {
                func(level.node, object.bounds, &mut object.value)?;
            }
            Ok(())
        })?;

        if let Some(upper) = self[level.node].upper {
            if iter_bounds.intersection(upper_bounds).is_some() {
                self.iter_mut(objects, level.next(upper, upper_bounds), iter_bounds, func)?
            }
        }

        Ok(())
    }

    fn depth(&self, node: NodeHandle) -> usize {
        let lower_depth = self[node]
            .lower
            .map(|node| 1 + self.depth(node))
            .unwrap_or(0);
        let upper_depth = self[node]
            .upper
            .map(|node| 1 + self.depth(node))
            .unwrap_or(0);
        usize::max(lower_depth, upper_depth)
    }

    fn cast_ray<'a, T, E, F>(
        &self,
        objects: &'a ObjectPool<T>,
        level: OctreeLevel,
        ray: &Ray<f64>,
        test: &mut F,
    ) -> Result<(), E>
    where
        F: FnMut(ConvexIntersection<f64>, &'a Object<T>) -> Result<(), E>,
        E: 'a,
    {
        let (lower_bounds, upper_bounds) = level.axis.cut_bounds(level.bounds);

        if let Some(lower) = self[level.node].lower {
            if lower_bounds.cast_ray(ray).is_some() {
                self.cast_ray(objects, level.next(lower, lower_bounds), ray, test)?;
            }
        }

        objects.iter(self[level.node].objects, &mut |object| {
            if let Some(intersection) = object.bounds.cast_ray(ray) {
                test(intersection, object)?
            }
            Ok(())
        })?;

        if let Some(upper) = self[level.node].upper {
            if upper_bounds.cast_ray(ray).is_some() {
                self.cast_ray(objects, level.next(upper, upper_bounds), ray, test)?
            }
        }

        Ok(())
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

    fn prev(self) -> Self {
        match self {
            Axis::X => Axis::Z,
            Axis::Y => Axis::X,
            Axis::Z => Axis::Y,
        }
    }

    /// Returns expanded bounds, and the relative location of the old bounds within the new bounds.
    fn expand_bounds(
        self,
        bounds: Bounds<f64>,
        toward_point: Point3<f64>,
    ) -> (Bounds<f64>, NodeLocation) {
        let mut origin = bounds.origin();
        let mut limit = bounds.limit();
        let size = bounds.size();
        let original_location = match self {
            Axis::X => {
                if toward_point.x < origin.x {
                    origin.x -= size.x;
                    NodeLocation::Upper
                } else {
                    limit.x += size.x;
                    NodeLocation::Lower
                }
            }
            Axis::Y => {
                if toward_point.y < origin.y {
                    origin.y -= size.y;
                    NodeLocation::Upper
                } else {
                    limit.y += size.y;
                    NodeLocation::Lower
                }
            }
            Axis::Z => {
                if toward_point.z < origin.z {
                    origin.z -= size.z;
                    NodeLocation::Upper
                } else {
                    limit.z += size.z;
                    NodeLocation::Lower
                }
            }
        };

        (Bounds::from_limit(origin, limit), original_location)
    }

    /// Cuts the bounds along the plane normal to the given axis and containing the center of the
    /// bounds.  Returns (bounds with lower axis coordinate, bounds with higher axis coordinate).
    fn cut_bounds(self, bounds: Bounds<f64>) -> (Bounds<f64>, Bounds<f64>) {
        let center = bounds.origin() + bounds.size() / 2.;

        match self {
            Axis::X => {
                let a = bounds.origin();
                let mut b1 = bounds.limit();
                b1.x = center.x;
                let mut b2 = bounds.origin();
                b2.x = center.x;
                let c = bounds.limit();
                (Bounds::from_limit(a, b1), Bounds::from_limit(b2, c))
            }
            Axis::Y => {
                let a = bounds.origin();
                let mut b1 = bounds.limit();
                b1.y = center.y;
                let mut b2 = bounds.origin();
                b2.y = center.y;
                let c = bounds.limit();
                (Bounds::from_limit(a, b1), Bounds::from_limit(b2, c))
            }
            Axis::Z => {
                let a = bounds.origin();
                let mut b1 = bounds.limit();
                b1.z = center.z;
                let mut b2 = bounds.origin();
                b2.z = center.z;
                let c = bounds.limit();
                (Bounds::from_limit(a, b1), Bounds::from_limit(b2, c))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ListNode<T> {
    value: T,
    next: Option<NodeHandle>,
}

type ListNodePool<T> = NodePool<ListNode<T>>;

impl<T> ListNodePool<T> {
    fn add(&mut self, next: Option<NodeHandle>, value: T) -> NodeHandle {
        self.push(ListNode { value, next })
    }

    fn iter<'a, E, F>(&'a self, mut node: Option<NodeHandle>, func: &mut F) -> Result<(), E>
    where
        F: FnMut(&'a T) -> Result<(), E>,
        E: 'a,
    {
        while let Some(node_ix) = node {
            let item = &self[node_ix];
            func(&item.value)?;
            node = item.next;
        }

        Ok(())
    }

    fn iter_mut<'a, E, F>(&mut self, mut node: Option<NodeHandle>, func: &mut F) -> Result<(), E>
    where
        F: FnMut(&mut T) -> Result<(), E>,
        E: 'a,
    {
        while let Some(node_ix) = node {
            let item = &mut self[node_ix];
            func(&mut item.value)?;
            node = item.next;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Octree;

    use cgmath::{Point3, Vector3};

    use crate::bounds::Bounds;

    fn object(pos: Point3<f64>, value: i32) -> super::Object<i32> {
        let size = Vector3::new(2., 2., 2.);
        super::Object {
            bounds: Bounds::new(pos - size / 2., size),
            value,
        }
    }

    #[test]
    fn test_basic() {
        let mut tree: Octree<i32> = Octree::new();

        tree.insert(object(Point3::new(0., 0., 0.), 0));
        tree.insert(object(Point3::new(-10., 5., 4.), 1));
        tree.insert(object(Point3::new(10., 5., 4.), 2));
        tree.insert(object(Point3::new(5., 2., -8.), 3));
        tree.insert(object(Point3::new(0.1, 0.1, 0.1), 4));
        tree.insert(object(Point3::new(0.2, 0.2, 0.2), 5));
        tree.insert(super::Object {
            value: 6,
            bounds: Bounds::from_limit(Point3::new(-5., -2., 1.), Point3::new(10., 5., 10.)),
        });

        assert_eq!(tree.depth(), 15);

        {
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
                vec![6, 0, 5, 4]
            );
        }

        {
            let mut found = Vec::new();
            tree.iter(
                Some(Bounds::from_limit(
                    Point3::new(-20., 3.5, 3.),
                    Point3::new(20., 4.5, 4.),
                )),
                |ix, p, x| found.push((ix, p, *x)),
            );

            assert_eq!(
                found.into_iter().map(|(_, _, x)| x).collect::<Vec<_>>(),
                vec![6, 1, 2]
            );
        }
    }

    #[test]
    fn test_cast_ray() {
        let mut tree: Octree<i32> = Octree::new();

        tree.insert(object(Point3::new(0., 0., 0.), 0));
        tree.insert(object(Point3::new(-10., 5., 4.), 1));
        tree.insert(object(Point3::new(10., 5., 4.), 2));
        tree.insert(object(Point3::new(5., 2., -8.), 3));
        tree.insert(object(Point3::new(0.1, 0.1, 0.1), 4));
        tree.insert(object(Point3::new(0.2, 0.2, 0.2), 5));
        tree.insert(super::Object {
            value: 6,
            bounds: Bounds::from_limit(Point3::new(-5., -2., 1.), Point3::new(10., 5., 10.)),
        });

        // TODO
    }
}
