use cgmath::{Point3, Vector3};
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
            Some(boxed_node) => match &**boxed_node {
                Node::Leaf(data) => {
                    bytes_written += writer.write(&[1])?;
                    bytes_written += serialize_data(data, writer)?;
                }
                Node::Branch(children) => {
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

    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
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

    pub fn bounds(&self) -> Vector3<usize> {
        let width = 1 << self.height;
        Vector3::new(width, width, width)
    }
}

pub struct Iter<'a, T> {
    /// Traces the path currently taken through the tree. Each point along the path records the
    /// node and the index of the next branch to take.
    stack: Vec<(&'a Node<T>, usize)>,
}

impl<'a, T> Iter<'a, T> {
    fn new(tree: &'a Octree<T>) -> Self {
        let mut stack = Vec::new();
        match tree.node {
            None => {}
            Some(ref boxed_node) => {
                stack.reserve_exact(1 + tree.height as usize);
                stack.push((&**boxed_node, 0));
            }
        }

        Iter { stack }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
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

pub struct IterMut<'a, T> {
    // This could be implemented safely without using raw pointers but that would require using a
    // stack of size 8 * tree height instead of the 1 + tree height that we can get away with in
    // unsafe code.
    stack: Vec<(*mut Node<T>, usize)>,
    _marker: std::marker::PhantomData<&'a mut Octree<T>>,

    // this is used to test that the stack never re-allocates
    #[cfg(test)]
    capacity: usize,
}

impl<'a, T> IterMut<'a, T> {
    fn new(tree: &'a mut Octree<T>) -> Self {
        let mut stack = Vec::new();
        #[allow(unused_assignments)]
        let mut capacity = 0;

        match &mut tree.node {
            None => {}
            Some(node) => {
                capacity = 1 + tree.height as usize;
                stack.reserve_exact(capacity);
                stack.push((&mut **node as *mut Node<T>, 0));
            }
        }

        IterMut {
            stack,
            _marker: std::marker::PhantomData,
            #[cfg(test)]
            capacity,
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        loop {
            let (this_node_ptr, mut next_index) = self.stack.pop()?;
            let next_child = {
                let this_node = unsafe { &mut *this_node_ptr };

                let children = match this_node {
                    Node::Leaf(data) => break Some(data),
                    Node::Branch(children) => children,
                };

                loop {
                    if next_index == 8 {
                        break None;
                    }

                    match children[next_index].node {
                        None => next_index += 1,
                        Some(ref mut boxed_node) => {
                            break Some(&mut **boxed_node);
                        }
                    }
                }
            };

            match next_child {
                None => {}
                Some(next_child) => {
                    self.stack.push((this_node_ptr, 1 + next_index));
                    self.stack.push((next_child as *mut Node<T>, 0));
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

        tree.insert(Point3::new(1, 1, 1), 7);
        tree.insert(Point3::new(1, 2, 1), 8);
        tree.insert(Point3::new(1, 1, 2), 9);

        let mut iter_count = 0;
        for x in tree.iter_mut() {
            *x *= 2;
            iter_count += 1;
        }

        let all_vals: Vec<i64> = tree.iter().copied().collect();
        assert_eq!(all_vals.len(), iter_count);
        assert_eq!(all_vals, vec![14, 16, 18, 10, 12]);
    }

    #[test]
    fn test_dense_iter_mut() {
        let mut tree: Octree<i64> = Octree::new(3);

        let bounds = tree.bounds();

        use std::collections::HashSet;

        let mut expected_vals = HashSet::new();

        for x in 0..bounds.x {
            for y in 0..bounds.y {
                for z in 0..bounds.z {
                    let k = Point3::new(x, y, z);
                    let v = (x + y * bounds.x + z * bounds.x * bounds.y) as i64;
                    expected_vals.insert(v);
                    tree.insert(k, v);
                }
            }
        }

        let actual_vals: HashSet<i64> = tree.iter().copied().collect();
        assert_eq!(actual_vals, expected_vals);

        for v in tree.iter_mut() {
            *v *= 3;
        }

        let expected_vals: HashSet<i64> = expected_vals.iter().map(|x| *x * 3).collect();
        let actual_vals: HashSet<i64> = tree.iter().copied().collect();
        assert_eq!(actual_vals, expected_vals);
    }
}
