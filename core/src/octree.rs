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
    pub fn new(height: u8) -> Octree<T> {
        Octree { height, node: None }
    }

    fn new_branch(height: u8) -> [Self; 8] {
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
                    "Subdividing point into octants but it falls outside the current octant
                     at height {}",
                    height
                )
            }
        };

        let (x_octant, x) = get_step(pos.x);
        let (y_octant, y) = get_step(pos.y);
        let (z_octant, z) = get_step(pos.z);

        (x_octant + y_octant * 2 + z_octant * 4, Point3::new(x, y, z))
    }

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
}
