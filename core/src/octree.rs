use cgmath::Point3;
use std::io::{self, Read, Write};

enum Node<T> {
    Branch([Octree<T>; 8]),
    Leaf(T),
}

pub struct Octree<T> {
    height: u8,
    node: Option<Box<Node<T>>>,
}

impl<T> Octree<T> {
    pub const fn new(height: u8) -> Octree<T> {
        Octree { height, node: None }
    }

    /// Increase the height of the Octree by one step, by growing 7 new leaves and moving the
    /// existing tree into the eighth. The given octant is the one where the existing tree ends up.
    /// To keep the set of points in the tree unchanged, grow into the zeroth octant.
    pub fn grow(&mut self, octant: usize) {
        assert!(self.height < 255);
        assert!(octant < 8);

        let new_height = 1 + self.height;

        let active_child = Octree {
            height: self.height,
            node: std::mem::replace(&mut self.node, None),
        };

        let mut branch = Self::new_branch(new_height);
        branch[octant] = active_child;

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
            Some(boxed_node) => {
                let node: Node<_> = *boxed_node;
                match node {
                    Node::Leaf(_) => {
                        unreachable!("Leaf node at the top level in a tree of height > 0")
                    }
                    Node::Branch(mut children) => {
                        self.node = std::mem::replace(&mut children[octant].node, None);
                        Some(children)
                    }
                }
            }
        }
    }

    /// Inserts data at the given point in the octree. Returns a mutable reference to the data at
    /// its new location inside the tree.
    pub fn insert(&mut self, pos: Point3<usize>, data: T) -> &mut T {
        let node: &mut Node<T> = match self.node {
            None => {
                if self.height == 0 {
                    self.node = Some(Box::new(Node::Leaf(data)));
                    let node: &mut Node<T> = &mut *self.node.as_mut().expect("unreachable");
                    match node {
                        Node::Leaf(data) => {
                            return data;
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.node = Some(Box::new(Node::Branch(Self::new_branch(self.height - 1))));
                    &mut *self.node.as_mut().expect("unreachable")
                }
            }
            Some(ref mut node) => &mut *node,
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
                let branch =
                    std::mem::replace(&mut *boxed_node, DetachedNode::dummy_node());

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
            Some(boxed_node) => {
                let node: &Node<T> = &*boxed_node;
                match node {
                    Node::Leaf(data) => Some(data),
                    Node::Branch(children) => {
                        let (octant, next_pos) = Self::subdivide(self.height, pos);
                        children[octant].get(next_pos)
                    }
                }
            }
        }
    }

    pub fn get_mut(&mut self, pos: Point3<usize>) -> Option<&mut T> {
        match &mut self.node {
            None => None,
            Some(boxed_node) => {
                let node: &mut Node<T> = &mut *boxed_node;
                match node {
                    Node::Leaf(data) => Some(data),
                    Node::Branch(children) => {
                        let (octant, next_pos) = Self::subdivide(self.height, pos);
                        children[octant].get_mut(next_pos)
                    }
                }
            }
        }
    }

    pub fn get_or_insert<F>(&mut self, pos: Point3<usize>, make_data: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self.node {
            None => self.insert(pos, make_data()),
            Some(ref mut boxed_node) => {
                let node: &mut Node<T> = &mut *boxed_node;
                match node {
                    Node::Leaf(data) => data,
                    Node::Branch(children) => {
                        let (octant, next_pos) = Self::subdivide(self.height, pos);
                        children[octant].get_or_insert(next_pos, make_data)
                    }
                }
            }
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
        // First byte
        // - 0 for empty
        // - 1 for leaf, followed by serialize_data encoding
        // - height + 1 for tree, followed by tree encoding
        //
        // Cannot serialize a tree of height 255 because first byte is height + 1
        assert!(self.height < 255);

        let mut bytes_written = 0;

        match &self.node {
            None => bytes_written += writer.write(&[0])?,
            Some(boxed_node) => {
                let node: &Node<T> = &*boxed_node;
                match node {
                    Node::Leaf(data) => {
                        bytes_written += writer.write(&[1])?;
                        bytes_written += serialize_data(data, writer)?;
                    }
                    Node::Branch(children) => {
                        for child in children {
                            bytes_written += child.serialize(writer, serialize_data)?;
                        }
                    }
                }
            }
        };

        Ok(bytes_written)
    }

    pub fn deserialize<R, F>(reader: &mut R, deserialize_data: &mut F) -> io::Result<Self>
    where
        R: Read,
        F: FnMut(&mut R) -> io::Result<T>,
    {
        let mut buf0: [u8; 1] = [0];
        reader.read_exact(&mut buf0)?;
        let byte0 = buf0[0];

        match byte0 {
            0 => Ok(Octree {
                height: 0,
                node: None,
            }),
            1 => Ok(Octree {
                height: 0,
                node: Some(Box::new(Node::Leaf(deserialize_data(reader)?))),
            }),
            _ => {
                let mut de = || Self::deserialize(reader, deserialize_data);
                let node = Node::Branch([de()?, de()?, de()?, de()?, de()?, de()?, de()?, de()?]);

                Ok(Octree {
                    height: byte0 - 1,
                    node: Some(Box::new(node)),
                })
            }
        }
    }

    pub fn iter(&self) -> RefIterator<T> {
        RefIterator::new(self)
    }

    const fn new_branch(height: u8) -> [Self; 8] {
        [
            Self::new(height),
            Self::new(height),
            Self::new(height),
            Self::new(height),
            Self::new(height),
            Self::new(height),
            Self::new(height),
            Self::new(height),
        ]
    }

    /// Subdivide a point into its octant and the point within that octant.
    #[inline(always)]
    fn subdivide(height: u8, pos: Point3<usize>) -> (usize, Point3<usize>) {
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
                     at height {}",
                    pos, height
                )
            }
        };

        let (x_octant, x) = get_step(pos.x);
        let (y_octant, y) = get_step(pos.y);
        let (z_octant, z) = get_step(pos.z);

        (x_octant + y_octant * 2 + z_octant * 4, Point3::new(x, y, z))
    }
}

pub struct RefIterator<'a, T> {
    /// Traces the path currently taken through the tree. Each point along the path records the
    /// node and the index of the next branch to take.
    stack: Vec<(&'a Node<T>, usize)>,
}

impl<'a, T> RefIterator<'a, T> {
    fn new(tree: &'a Octree<T>) -> Self {
        match tree.node {
            None => RefIterator { stack: Vec::new() },
            Some(ref boxed_node) => RefIterator {
                stack: vec![(&*boxed_node, 0)],
            },
        }
    }
}

impl<'a, T> Iterator for RefIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            let (this_node, this_index) = match self.stack.pop() {
                None => break None,
                Some(x) => x,
            };

            match this_node {
                Node::Leaf(x) => break Some(x),
                Node::Branch(children) => {
                    if this_index == 8 {
                        continue;
                    };
                    let next_tree = &children[this_index];
                    self.stack.push((this_node, 1 + this_index));
                    match &next_tree.node {
                        None => continue,
                        Some(next_node) => self.stack.push((next_node, 0)),
                    }
                }
            }
        }
    }
}

// pub struct MutIterator<'a, T> {
//     stack: Vec<(DetachedNode<'a, T>, usize)>,
// }

// impl<'a, T> MutIterator<'a, T> {
//     fn new(tree: &'a mut Octree<T>) -> Self {
//         MutIterator {
//             stack: vec![(&mut tree.node, 0)]
//         }
//     }
// }

// impl<'a, T> Iterator for MutIterator<'a, T> {
//     type Item = &'a mut T;

//     fn next(&mut self) -> Option<&'a mut T> {
//         loop {
//             let (this_node_ref, this_index) = match self.stack.pop() {
//                 None => break None,
//                 Some(x) => x,
//             };

//             let mut this_node = std::mem::replace(this_node_ref, None);

//             match this_node {
//                 None => continue,
//                 Some(mut boxed_node) => {
//                     match *boxed_node {
//                         Node::Leaf(x) => break Some(&mut x),
//                         Node::Branch(children) => {
//                         }
//                     }
//                 }
//             }

//             // match this_node {
//             //     Node::Leaf(x) => break Some(x),
//             //     Node::Branch(children) => {
//             //         if this_index == 8 {
//             //             continue;
//             //         };
//             //         let next_tree = &mut children[this_index];
//             //         self.stack.push((this_node, 1 + this_index));
//             //         match &mut next_tree.node {
//             //             None => continue,
//             //             Some(next_node) => self.stack.push((next_node, 0)),
//             //         }
//             //     }
//             // }
//         }
//     }
// }

struct DetachedNode<'a, T> {
    original: &'a mut Node<T>,
    stolen: Node<T>,
}

impl<'a, T> DetachedNode<'a, T> {
    fn new(node: &'a mut Node<T>) -> Self {
        let stolen = std::mem::replace(node, Self::dummy_node());
        DetachedNode {
            original: node,
            stolen
        }
    }

    const fn dummy_node() -> Node<T> {
        Node::Branch(Octree::new_branch(0))
    }
}

impl<'a, T> std::ops::Deref for DetachedNode<'a, T> {
    type Target = Node<T>;

    fn deref(&self) -> &Self::Target {
        &self.stolen
    }
}

impl<'a, T> std::ops::DerefMut for DetachedNode<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stolen
    }
}

impl<'a, T> Drop for DetachedNode<'a, T> {
    fn drop(&mut self) {
        std::mem::swap(self.original, &mut self.stolen);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use cgmath::Point3;

    #[test]
    fn test_octree() {
        let mut tree: Octree<i64> = Octree::new(8);
        tree.insert(Point3::new(34, 2, 17), 6);
        assert_eq!(tree.get(Point3::new(34, 2, 17)), Some(&6));
        assert_eq!(tree.get(Point3::new(0, 2, 17)), None);

        {
            let inner_ref = tree.get_or_insert(Point3::new(2, 3, 4), || 7);
            assert_eq!(*inner_ref, 7);
            *inner_ref = 5;
        }

        assert_eq!(tree.get_or_insert(Point3::new(2, 3, 4), || 4), &mut 5);

        tree.insert(Point3::new(34, 3, 16), 2);

        // Iteration
        let all_vals: Vec<i64> = tree.iter().copied().collect();
        assert_eq!(all_vals, vec![5, 2, 6]);

        // Remove
        assert_eq!(tree.remove(Point3::new(34, 3, 16)), Some(2));
        assert_eq!(tree.remove(Point3::new(34, 3, 16)), None);

        // Iteration after removal still yields the values that were not removed
        let all_vals: Vec<i64> = tree.iter().copied().collect();
        assert_eq!(all_vals, vec![5, 6]);
    }
}
