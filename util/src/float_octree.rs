use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};

use crate::ray::{ConvexRaycastResult, Ray};
use crate::{convert_bounds, convert_vec, Bounds, OrdFloat};

/// A tree structure providing a sparse representation of values in a 3D grid.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Octree<T> {
    scale: i64,
    node: Option<Box<Node<T>>>,
}

/// Refers to one octant of an area that has been split into 8, along 3 orthogonal planes.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
#[repr(packed)]
pub struct Octant {
    x: i8,
    y: i8,
    z: i8,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OctantMap<T> {
    children: (T, T, T, T, T, T, T, T),
}

pub struct Iter<'a, T> {
    /// Traces the path currently taken through the tree. Each point along the path records the
    /// node and the index of the next branch to take.
    stack: Vec<IterNode<'a, T>>,
    bounds: Bounds<f64>,
}

pub struct IterMut<'a, T> {
    // This could be implemented safely without using raw pointers but that would require using a
    // stack of size 8 * tree scale instead of the 1 + tree scale that we can get away with in
    // unsafe code.
    stack: Vec<IterMutNode<'a, T>>,

    // this is used to test that the stack never re-allocates
    #[cfg(test)]
    capacity: usize,
}

pub struct IterOctantMap<'a, T> {
    map: &'a OctantMap<T>,
    next_octant: Option<Octant>,
}

// #[derive(Clone, Debug, PartialEq)]
// pub struct RaycastHit<T> {
//     pub data: T,
//     pub pos: Point3<f64>,
// }

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Node<T> {
    Branch(OctantMap<Octree<T>>),
    Leaf(T),
}

impl<T> Octree<T> {
    /// Creates a new octree with the given scale. It will have space for `2^scale` distinct
    /// points.
    pub fn with_scale(scale: i64) -> Octree<T> {
        // scales of 64 or more would cause points within the tree to overflow their 64 bits
        assert!(scale < 64);
        Octree { scale, node: None }
    }

    pub fn new() -> Octree<T> {
        Self::with_scale(0)
    }

    /// Increase the scale of the Octree by one step.
    pub fn grow(&mut self) {
        // octant (-1, -1, -1) becomes octant (1, 1, 1) of the new octant (-1, -1, -1)
        // octant (1, 1, 1) becomes octant (-1, -1, -1) of the new octant (1, 1, 1)
        // etc

        let new_self = Octree::with_scale(self.scale + 1);
        let old = std::mem::replace(self, new_self);

        let old_scale = old.scale;
        let old_node = match old.node {
            None => {
                return;
            }
            Some(node) => *node,
        };

        let octants = match old_node {
            Node::Branch(octants) => octants.transform(|octant, octree| {
                let mut branch = Self::new_branch(old_scale - 1);
                assert_eq!(octree.scale, old_scale - 1);
                octree.assert_scale_invariant();
                branch[octant.opposite()] = octree;
                Octree {
                    scale: old_scale,
                    node: Some(Box::new(Node::Branch(branch))),
                }
            }),
            Node::Leaf(data) => {
                let mut octants = Self::new_branch(old_scale);
                octants[Octant::from_index(0)] = Octree {
                    scale: old_scale,
                    node: Some(Box::new(Node::Leaf(data))),
                };
                octants
            }
        };

        self.node = Some(Box::new(Node::Branch(octants)));
    }

    /// Shrinks the scale of the octree by one step, by discarding all but the given octant.
    /// If the octree has no children, returns None. Otherwise returns the discarded octants.
    /// In the returned array, the octant that was shrunk into is made empty.
    pub fn shrink(&mut self, octant: Octant) -> Option<OctantMap<Self>> {
        let old_scale = self.scale;
        self.scale = old_scale - 1;

        let node = std::mem::replace(&mut self.node, None);

        match node {
            None => None,
            Some(boxed_node) => match *boxed_node {
                Node::Leaf(_) => unreachable!("Leaf node at the top level in a tree of scale > 0"),
                Node::Branch(mut children) => {
                    self.node = std::mem::replace(&mut children[octant].node, None);
                    Some(children)
                }
            },
        }
    }

    /// Inserts data at the given point in the octree. Returns a mutable reference to the data at
    /// its new location inside the tree.
    pub fn insert(&mut self, pos: Point3<f64>, data: T) -> &mut T {
        assert!(
            self.bounds().contains_point(pos),
            "insert of point {:?} is outside bounds {:?}",
            pos,
            self.bounds()
        );

        let node = match self.node {
            None => {
                if self.scale == 0 {
                    self.node = Some(Box::new(Node::Leaf(data)));
                    let node = &mut **self.node.as_mut().expect("unreachable");
                    match node {
                        Node::Leaf(data) => {
                            return data;
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.node = Some(Box::new(Node::Branch(Self::new_branch(self.scale - 1))));
                    &mut **self.node.as_mut().expect("unreachable")
                }
            }
            Some(ref mut node) => &mut **node,
        };

        match node {
            Node::Leaf(old_data) => {
                *old_data = data;
                old_data
            }
            Node::Branch(children) => {
                let (octant, next_pos) = Self::subdivide(self.scale, pos);
                children[octant].insert(next_pos, data)
            }
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.iter_in_bounds(self.bounds())
    }

    #[inline]
    pub fn iter_in_bounds(&self, bounds: Bounds<f64>) -> Iter<T> {
        Iter::new(self, bounds)
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
    }

    const fn new_branch(scale: i64) -> OctantMap<Self> {
        OctantMap {
            children: (
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
                Octree { scale, node: None },
            ),
        }
    }

    /// Subdivide a point into its octant and the point within that octant.
    #[inline(always)]
    pub fn subdivide(scale: i64, pos: Point3<f64>) -> (Octant, Point3<f64>) {
        let half_size: f64 = 2f64.powi((scale - 1) as i32);

        let get_step = |value| {
            if value < 0 && value >= -half_size {
                (-1, value + (half_size + 1) / 2)
            } else if value >= 0 && value < half_size {
                (1, value - half_size / 2)
            } else {
                panic!(
                    "Subdividing point {:?} into octants but it falls outside the current octant
                     at scale {} with allowed range [{}, {})",
                    pos, scale, -half_size, half_size
                )
            }
        };

        let (x_octant, x) = get_step(pos.x);
        let (y_octant, y) = get_step(pos.y);
        let (z_octant, z) = get_step(pos.z);

        (
            Octant {
                x: x_octant,
                y: y_octant,
                z: z_octant,
            },
            Point3::new(x, y, z),
        )
    }

    /// Calculates the offset from the parent node's origin where a child node's origin will be,
    /// within a tree of the given scale, for the child in the given octant.
    ///
    /// # Example
    /// ```
    /// use cgmath::{Point3, Vector3};
    /// use simgame_util::octree::{self, Octree};
    /// let scale = 7;
    /// let octant = octree::Octant::from_index(3);
    /// let offset = Octree::<()>::octant_offset(scale, octant);
    /// assert_eq!(offset, Vector3::new(32, 32, -32));
    /// let sub = Octree::<()>::subdivide(scale, Point3::new(0, 0, 0) + offset);
    /// assert_eq!(sub.0, octant);
    /// assert_eq!(sub.1, Point3::new(0, 0, 0));
    /// ```
    pub fn octant_offset(scale: i64, octant: Octant) -> Vector3<i64> {
        assert!(scale > 0);
        if scale >= 2 {
            let distance = 1 << (scale - 2);
            distance * octant.as_direction()
        } else {
            (octant.as_direction() - Vector3::new(1, 1, 1)) / 2
        }
    }

    #[inline]
    pub fn scale(&self) -> i64 {
        self.scale
    }

    /// Returns a bounding box that is guaranteed to contain every point currently within the
    /// Octree. No guarantees are made about whether it is the smallest such bounding box.
    #[inline]
    pub fn bounds(&self) -> Bounds<f64> {
        let width = 1 << self.scale;
        let origin = Point3::new(-width / 2, -width / 2, -width / 2);
        let size = Vector3::new(width, width, width);

        Bounds::new(origin, size)
    }

    pub fn check_scale_invariant(&self) -> bool {
        match &self.node {
            None => true,
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(_) => true,
                Node::Branch(children) => children
                    .iter()
                    .all(|child| 1 + child.scale == self.scale && child.check_scale_invariant()),
            },
        }
    }

    pub fn assert_scale_invariant(&self) {
        match &self.node {
            None => {}
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(_) => {},
                Node::Branch(children) => {
                    for child in children {
                        assert_eq!(child.scale, self.scale - 1);
                        child.assert_scale_invariant();
                    }
                }
            },
        }
    }
}

struct IterNode<'a, T> {
    node: &'a Node<T>,
    next_octant: Option<Octant>,
    octant_origin: Point3<f64>,
}

struct IterMutNode<'a, T> {
    node: *mut Node<T>,
    next_octant: Option<Octant>,
    octant_origin: Point3<f64>,
    _marker: std::marker::PhantomData<&'a mut Octree<T>>,
}

impl<'a, T> Iter<'a, T> {
    #[inline]
    fn new(tree: &'a Octree<T>, bounds: Bounds<f64>) -> Self {
        let mut stack = Vec::new();
        match tree.node {
            None => {}
            Some(ref boxed_node) => {
                stack.reserve_exact(1 + tree.scale as usize);
                stack.push(IterNode {
                    node: &**boxed_node,
                    next_octant: Some(Octant::from_index(0)),
                    octant_origin: tree.bounds().origin(),
                });
            }
        }

        Iter { stack, bounds }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Point3<f64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut iter_node = self.stack.pop()?;

            let children = match &iter_node.node {
                Node::Leaf(data) => break Some((iter_node.octant_origin, data)),
                Node::Branch(children) => children,
            };

            let next_child = loop {
                let next_octant = if let Some(next_octant) = iter_node.next_octant {
                    next_octant
                } else {
                    break None;
                };

                let child = &children[next_octant];
                match &child.node {
                    None => iter_node.next_octant = next_octant.next(),
                    Some(boxed_node) => {
                        let scale = child.scale;
                        let octant_size = 1 << scale;
                        let dir = next_octant.as_direction();
                        let offset = (dir + Vector3::new(1, 1, 1)) / 2;

                        iter_node.next_octant = next_octant.next();
                        let octant_origin = iter_node.octant_origin + (octant_size * offset);
                        let octant_bounds =
                            Bounds::new(octant_origin, octant_size * Vector3::new(1, 1, 1));

                        // Exclude children that contain no points within the bounds we're
                        // iterating through.  This will both prevent them from being yielded by
                        // the iterator, and prevent subsequent iteration from searching through
                        // this child's children.
                        if self.bounds.intersection(octant_bounds).is_none() {
                            continue;
                        }

                        break Some((octant_origin, &**boxed_node));
                    }
                }
            };

            match next_child {
                None => {}
                Some((octant_origin, next_node)) => {
                    self.stack.push(iter_node);
                    self.stack.push(IterNode {
                        node: next_node,
                        next_octant: Some(Octant::from_index(0)),
                        octant_origin,
                    });
                }
            }
        }
    }
}

impl<'a, T> IterMut<'a, T> {
    #[inline]
    fn new(tree: &'a mut Octree<T>) -> Self {
        let mut stack = Vec::new();
        #[allow(unused_assignments)]
        let mut capacity = 0;

        match &mut tree.node {
            None => {}
            Some(node) => {
                capacity = 1 + tree.scale as usize;
                stack.reserve_exact(capacity);
                stack.push(IterMutNode {
                    node: &mut **node as *mut Node<T>,
                    next_octant: Some(Octant::from_index(0)),
                    octant_origin: tree.bounds().origin(),
                    _marker: std::marker::PhantomData,
                });
            }
        }

        IterMut {
            stack,
            #[cfg(test)]
            capacity,
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (Point3<f64>, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut iter_node = self.stack.pop()?;
            let next_child = {
                let this_node = unsafe { &mut *iter_node.node };

                let children = match this_node {
                    Node::Leaf(data) => break Some((iter_node.octant_origin, data)),
                    Node::Branch(children) => children,
                };

                loop {
                    let next_octant = if let Some(next_octant) = iter_node.next_octant {
                        next_octant
                    } else {
                        break None;
                    };

                    let child = &mut children[next_octant];
                    match &mut child.node {
                        None => iter_node.next_octant = next_octant.next(),
                        Some(ref mut boxed_node) => {
                            let scale = child.scale;
                            let distance = 1 << scale;
                            let dir = (next_octant.as_direction() + Vector3::new(1, 1, 1)) / 2;
                            iter_node.next_octant = next_octant.next();
                            let octant_origin = iter_node.octant_origin + (distance * dir);
                            break Some((octant_origin, &mut **boxed_node));
                        }
                    }
                }
            };

            match next_child {
                None => {}
                Some((octant_origin, next_node)) => {
                    self.stack.push(iter_node);
                    self.stack.push(IterMutNode {
                        node: next_node as *mut Node<T>,
                        next_octant: Some(Octant::from_index(0)),
                        octant_origin,
                        _marker: std::marker::PhantomData,
                    });
                    #[cfg(test)]
                    assert_eq!(self.stack.capacity(), self.capacity);
                }
            }
        }
    }
}

impl<T> Node<T> {
    const fn dummy() -> Node<T> {
        Node::Branch(Octree::new_branch(0))
    }
}

impl<T> OctantMap<T> {
    pub fn iter(&self) -> IterOctantMap<T> {
        self.into_iter()
    }
}

impl<T> std::ops::Index<Octant> for OctantMap<T> {
    type Output = T;

    fn index(&self, octant: Octant) -> &Self::Output {
        let index = octant.as_index();
        if index == 0 {
            &self.children.0
        } else if index == 1 {
            &self.children.1
        } else if index == 2 {
            &self.children.2
        } else if index == 3 {
            &self.children.3
        } else if index == 4 {
            &self.children.4
        } else if index == 5 {
            &self.children.5
        } else if index == 6 {
            &self.children.6
        } else if index == 7 {
            &self.children.7
        } else {
            panic!("Octant has invalid index {}", index);
        }
    }
}

impl<T> std::ops::IndexMut<Octant> for OctantMap<T> {
    fn index_mut(&mut self, octant: Octant) -> &mut Self::Output {
        let index = octant.as_index();
        if index == 0 {
            &mut self.children.0
        } else if index == 1 {
            &mut self.children.1
        } else if index == 2 {
            &mut self.children.2
        } else if index == 3 {
            &mut self.children.3
        } else if index == 4 {
            &mut self.children.4
        } else if index == 5 {
            &mut self.children.5
        } else if index == 6 {
            &mut self.children.6
        } else if index == 7 {
            &mut self.children.7
        } else {
            panic!("Octant has invalid index {}", index);
        }
    }
}

impl<'a, T> Iterator for IterOctantMap<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let octant = self.next_octant?;
        self.next_octant = octant.next();
        Some(&self.map[octant])
    }
}

impl<'a, T> IntoIterator for &'a OctantMap<T> {
    type Item = &'a T;
    type IntoIter = IterOctantMap<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterOctantMap {
            map: self,
            next_octant: Some(Octant::from_index(0)),
        }
    }
}

impl<T> OctantMap<T> {
    pub fn transform<R, F>(self, mut f: F) -> OctantMap<R>
    where
        F: FnMut(Octant, T) -> R,
    {
        OctantMap {
            children: (
                f(Octant::from_index(0), self.children.0),
                f(Octant::from_index(1), self.children.1),
                f(Octant::from_index(2), self.children.2),
                f(Octant::from_index(3), self.children.3),
                f(Octant::from_index(4), self.children.4),
                f(Octant::from_index(5), self.children.5),
                f(Octant::from_index(6), self.children.6),
                f(Octant::from_index(7), self.children.7),
            ),
        }
    }
}

impl Octant {
    #[inline]
    pub fn as_index(self) -> usize {
        ((self.x + 1) / 2 + (self.y + 1) + (self.z + 1) * 2) as usize
    }

    #[inline]
    pub fn from_index(index: usize) -> Octant {
        assert!(index < 8);

        Octant {
            x: ((index % 2) as i8) * 2 - 1,
            y: (((index % 4) / 2) as i8) * 2 - 1,
            z: ((index / 4) as i8) * 2 - 1,
        }
    }

    #[inline]
    pub fn as_direction(self) -> Vector3<i64> {
        Vector3 {
            x: self.x as i64,
            y: self.y as i64,
            z: self.z as i64,
        }
    }

    #[inline]
    pub fn next(self) -> Option<Octant> {
        let index = self.as_index();
        if 1 + index < 8 {
            Some(Octant::from_index(1 + index))
        } else {
            None
        }
    }

    #[inline]
    pub fn opposite(self) -> Octant {
        Octant {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[inline]
    pub fn enumerate() -> [Self; 8] {
        let mut octants = [Self::from_index(0); 8];
        for i in 0..8 {
            octants[i] = Self::from_index(i);
        }
        octants
    }
}

#[cfg(test)]
mod tests;
