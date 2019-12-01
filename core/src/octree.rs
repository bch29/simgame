use anyhow::anyhow;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{Point3, Vector3};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

use crate::util::Bounds;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum Node<T> {
    Branch([Octree<T>; 8]),
    Leaf(T),
}

/// A tree structure providing a sparse representation of values in a 3D grid.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Octree<T> {
    height: usize,
    node: Option<Box<Node<T>>>,
}

impl<T> Octree<T> {
    /// Creates a new octree with the given height. It will have space for `2^height` distinct
    /// points.
    pub fn new(height: usize) -> Octree<T> {
        // heights of 64 or more would cause points within the tree to overflow their 64 bits
        assert!(height < 64);
        Octree { height, node: None }
    }

    /// Increase the height of the Octree by one step, by growing 7 new leaves and moving the
    /// existing tree into the eighth. The given octant is the one where the existing tree ends up.
    /// To keep the set of points in the tree unchanged, grow into the zeroth octant.
    pub fn grow(&mut self, octant: usize) {
        assert!(octant < 8);
        assert!(self.height < 63);

        let mut branch = Self::new_branch(self.height);
        branch[octant].node = std::mem::replace(&mut self.node, None);

        self.height += 1;
        self.node = Some(Box::new(Node::Branch(branch)))
    }

    /// Shrinks the height of the octree by one step, by discarding all but the given octant.
    /// If the octree has no children, returns None. Otherwise returns the discarded octants.
    /// In the returned array, the octant that was shrunk into is made empty.
    pub fn shrink(&mut self, octant: usize) -> Option<[Self; 8]> {
        assert!(self.height > 0);
        assert!(octant < 8);

        let old_height = self.height;
        self.height = old_height - 1;

        let node = std::mem::replace(&mut self.node, None);

        match node {
            None => None,
            Some(boxed_node) => match *boxed_node {
                Node::Leaf(_) => {
                    unreachable!("Leaf node at the top level in a tree of height > 0")
                }
                Node::Branch(mut children) => {
                    self.node = std::mem::replace(&mut children[octant].node, None);
                    Some(children)
                }
            },
        }
    }

    /// Inserts data at the given point in the octree. Returns a mutable reference to the data at
    /// its new location inside the tree.
    pub fn insert(&mut self, pos: Point3<usize>, data: T) -> &mut T {
        let node = match self.node {
            None => {
                if self.height == 0 {
                    self.node = Some(Box::new(Node::Leaf(data)));
                    let node = &mut **self.node.as_mut().expect("unreachable");
                    match node {
                        Node::Leaf(data) => {
                            return data;
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.node = Some(Box::new(Node::Branch(Self::new_branch(self.height - 1))));
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
                let (octant, next_pos) = Self::subdivide(self.height, pos);
                children[octant].insert(next_pos, data)
            }
        }
    }

    /// Removes and returns object inside the octree at the given point. If there is nothing there,
    /// returns None.
    pub fn remove(&mut self, pos: Point3<usize>) -> Option<T> {
        let node = std::mem::replace(&mut self.node, None);
        match node {
            None => None,
            Some(mut boxed_node) => {
                // We want ownership of the node inside the box in case we need to return the
                // data inside. So we swap out the node with a dummy value, then replace it later
                // if need be. This could also be done by moving out of the box unconditionally and
                // then making a new box to replace it, but this would require an allocation per
                // level of height in the tree.
                let branch = std::mem::replace(&mut *boxed_node, Node::dummy());

                match branch {
                    Node::Leaf(data) => Some(data),
                    Node::Branch(mut children) => {
                        let (octant, next_pos) = Self::subdivide(self.height, pos);
                        let res = children[octant].remove(next_pos);
                        std::mem::replace(&mut *boxed_node, Node::Branch(children));
                        self.node = Some(boxed_node);
                        res
                    }
                }
            }
        }
    }

    pub fn get(&self, pos: Point3<usize>) -> Option<&T> {
        match &self.node {
            None => None,
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(data) => Some(data),
                Node::Branch(children) => {
                    let (octant, next_pos) = Self::subdivide(self.height, pos);
                    children[octant].get(next_pos)
                }
            },
        }
    }

    pub fn get_mut(&mut self, pos: Point3<usize>) -> Option<&mut T> {
        match &mut self.node {
            None => None,
            Some(boxed_node) => match &mut **boxed_node {
                Node::Leaf(data) => Some(data),
                Node::Branch(children) => {
                    let (octant, next_pos) = Self::subdivide(self.height, pos);
                    children[octant].get_mut(next_pos)
                }
            },
        }
    }

    pub fn get_or_insert<F>(&mut self, pos: Point3<usize>, make_data: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self.node {
            None => self.insert(pos, make_data()),
            Some(ref mut boxed_node) => match &mut **boxed_node {
                Node::Leaf(data) => data,
                Node::Branch(children) => {
                    let (octant, next_pos) = Self::subdivide(self.height, pos);
                    children[octant].get_or_insert(next_pos, make_data)
                }
            },
        }
    }

    /// Serializes the octree in a binary format. Returns the number of bytes written.
    /// Accepts a function to serialize individual data leaves. This function must precisely report
    /// the number of bytes it wrote if it succeeds.
    pub fn serialize<W, F>(&self, writer: &mut W, serialize_data: &mut F) -> io::Result<usize>
    where
        W: Write,
        F: FnMut(&T, &mut W) -> io::Result<usize>,
    {
        let mut bytes_written = 0;

        writer.write_u64::<BigEndian>(self.height as u64)?;
        bytes_written += 8;

        match &self.node {
            None => {
                writer.write_u8(0)?;
                bytes_written += 1;
            }
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(data) => {
                    writer.write_u8(1)?;
                    bytes_written += 1;
                    bytes_written += serialize_data(data, writer)?;
                }
                Node::Branch(children) => {
                    writer.write_u8(2)?;
                    bytes_written += 1;
                    for child in children {
                        bytes_written += child.serialize(writer, serialize_data)?;
                    }
                }
            },
        };

        Ok(bytes_written)
    }

    pub fn deserialize<R, F>(reader: &mut R, deserialize_data: &mut F) -> io::Result<Self>
    where
        R: Read,
        F: FnMut(&mut R) -> io::Result<T>,
    {
        let height = reader.read_u64::<BigEndian>()? as usize;
        let tag = reader.read_u8()?;

        assert!(height < 64);

        match tag {
            0 => Ok(Octree { height, node: None }),
            1 => {
                if height != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        anyhow!("Leaf with nonzero height value {}", height),
                    ));
                }

                Ok(Octree {
                    height: 0,
                    node: Some(Box::new(Node::Leaf(deserialize_data(reader)?))),
                })
            }
            2 => {
                let mut de = || Self::deserialize(reader, deserialize_data);
                let node = Node::Branch([de()?, de()?, de()?, de()?, de()?, de()?, de()?, de()?]);

                Ok(Octree {
                    height,
                    node: Some(Box::new(node)),
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                anyhow!("Invalid tag byte value {}", tag),
            )),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<T> {
        self.iter_in_bounds(self.bounds())
    }

    #[inline]
    pub fn iter_in_bounds(&self, bounds: Bounds<usize>) -> Iter<T> {
        Iter::new(self, bounds)
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
    }

    const fn new_branch(height: usize) -> [Self; 8] {
        [
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
            Octree { height, node: None },
        ]
    }

    /// Subdivide a point into its octant and the point within that octant.
    #[inline(always)]
    pub fn subdivide(height: usize, pos: Point3<usize>) -> (usize, Point3<usize>) {
        assert!(height > 0);
        let half_size: usize = 1 << (height - 1);
        let size: usize = half_size * 2;

        let get_step = |value| {
            if value < half_size {
                (0, value)
            } else if value < size {
                (1, value - half_size)
            } else {
                panic!(
                    "Subdividing point {:?} into octants but it falls outside the current octant
                     at height {} with max size {}",
                    pos, height, size
                )
            }
        };

        let (x_octant, x) = get_step(pos.x);
        let (y_octant, y) = get_step(pos.y);
        let (z_octant, z) = get_step(pos.z);

        (x_octant + y_octant * 2 + z_octant * 4, Point3::new(x, y, z))
    }

    /// Calculates the offset from the parent node's origin where a child node's origin will be,
    /// within a tree of the given height, for the child in the given octant.
    ///
    /// # Example
    /// ```
    /// use cgmath::{Point3, Vector3};
    /// use simgame_core::octree::Octree;
    /// let height = 7;
    /// let octant = 3;
    /// let offset = Octree::<()>::quadrant_offset(height, octant);
    /// assert_eq!(offset, Vector3::new(64, 64, 0));
    /// let sub = Octree::<()>::subdivide(height, Point3::new(0, 0, 0) + offset);
    /// assert_eq!(sub.0, octant);
    /// assert_eq!(sub.1, Point3::new(0, 0, 0));
    /// ```
    pub fn quadrant_offset(height: usize, octant: usize) -> Vector3<usize> {
        assert!(height > 0);
        assert!(octant < 8);
        let distance = 1 << (height - 1);
        let direction = Vector3 {
            x: octant % 2,
            y: (octant % 4) / 2,
            z: octant / 4,
        };

        distance * direction
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns a bounding box that is guaranteed to contain every point currently within the
    /// Octree. No guarantees are made about whether it is the smallest such bounding box.
    #[inline]
    pub fn bounds(&self) -> Bounds<usize> {
        let width = 1 << self.height;
        Bounds::from_size(Vector3::new(width, width, width))
    }

    pub fn check_height_invariant(&self) -> bool {
        match &self.node {
            None => true,
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(_) => self.height == 0,
                Node::Branch(children) => children.iter().all(|child| {
                    1 + child.height == self.height && child.check_height_invariant()
                }),
            },
        }
    }

    pub fn assert_height_invariant(&self) {
        match &self.node {
            None => {}
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(_) => assert_eq!(self.height, 0),
                Node::Branch(children) => {
                    assert!(self.height > 0);
                    for child in children {
                        assert_eq!(child.height, self.height - 1);
                        child.assert_height_invariant();
                    }
                }
            },
        }
    }
}

pub struct Iter<'a, T> {
    /// Traces the path currently taken through the tree. Each point along the path records the
    /// node and the index of the next branch to take.
    stack: Vec<IterNode<'a, T>>,
    bounds: Bounds<usize>,
}

struct IterNode<'a, T> {
    node: &'a Node<T>,
    next_index: usize,
    octant_origin: Point3<usize>,
}

pub struct IterMut<'a, T> {
    // This could be implemented safely without using raw pointers but that would require using a
    // stack of size 8 * tree height instead of the 1 + tree height that we can get away with in
    // unsafe code.
    stack: Vec<IterMutNode<'a, T>>,

    // this is used to test that the stack never re-allocates
    #[cfg(test)]
    capacity: usize,
}

struct IterMutNode<'a, T> {
    node: *mut Node<T>,
    next_index: usize,
    octant_origin: Point3<usize>,
    _marker: std::marker::PhantomData<&'a mut Octree<T>>,
}

impl<'a, T> Iter<'a, T> {
    #[inline]
    fn new(tree: &'a Octree<T>, bounds: Bounds<usize>) -> Self {
        let mut stack = Vec::new();
        match tree.node {
            None => {}
            Some(ref boxed_node) => {
                stack.reserve_exact(1 + tree.height as usize);
                stack.push(IterNode {
                    node: &**boxed_node,
                    next_index: 0,
                    octant_origin: Point3::new(0, 0, 0),
                });
            }
        }

        Iter { stack, bounds }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (Point3<usize>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut iter_node = self.stack.pop()?;

            let children = match &iter_node.node {
                Node::Leaf(data) => break Some((iter_node.octant_origin, data)),
                Node::Branch(children) => children,
            };

            let next_child = loop {
                if iter_node.next_index == 8 {
                    break None;
                }

                let child = &children[iter_node.next_index];
                match &child.node {
                    None => iter_node.next_index += 1,
                    Some(boxed_node) => {
                        let height = child.height;
                        let octant_size = 1 << height;
                        let dir = Vector3 {
                            x: iter_node.next_index % 2,
                            y: (iter_node.next_index % 4) / 2,
                            z: iter_node.next_index / 4,
                        };

                        iter_node.next_index += 1;
                        let octant_origin = iter_node.octant_origin + (octant_size * dir);
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
                        next_index: 0,
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
                capacity = 1 + tree.height as usize;
                stack.reserve_exact(capacity);
                stack.push(IterMutNode {
                    node: &mut **node as *mut Node<T>,
                    next_index: 0,
                    octant_origin: Point3::new(0, 0, 0),
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
    type Item = (Point3<usize>, &'a mut T);

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
                    if iter_node.next_index == 8 {
                        break None;
                    }

                    let child = &mut children[iter_node.next_index];
                    match &mut child.node {
                        None => iter_node.next_index += 1,
                        Some(ref mut boxed_node) => {
                            let height = child.height;
                            let distance = 1 << height;
                            let dir = Vector3 {
                                x: iter_node.next_index % 2,
                                y: (iter_node.next_index % 4) / 2,
                                z: iter_node.next_index / 4,
                            };

                            iter_node.next_index += 1;
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
                        next_index: 0,
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

#[cfg(test)]
mod tests;
