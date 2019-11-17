use anyhow::anyhow;
use cgmath::Point3;
use std::io::{self, Read, Write};

/// Indexed by (x, y, z)
type Branch<T> = [[[T; 2]; 2]; 2];

enum Node<T> {
    Tree(Octree<T>),
    Leaf(T),
    Empty,
}

pub struct Octree<T> {
    height: u8,
    children: Branch<Box<Node<T>>>,
}

impl<T> Octree<T> {
    fn empty_branch() -> Branch<Box<Node<T>>> {
        [
            [
                [Box::new(Node::Empty), Box::new(Node::Empty)],
                [Box::new(Node::Empty), Box::new(Node::Empty)],
            ],
            [
                [Box::new(Node::Empty), Box::new(Node::Empty)],
                [Box::new(Node::Empty), Box::new(Node::Empty)],
            ],
        ]
    }

    pub fn new(height: u8) -> Octree<T> {
        assert!(height >= 1);
        Octree {
            height,
            children: Self::empty_branch(),
        }
    }

    /// Given a point, subdivide it into its octant and the point within that octant.
    fn subdivide(height: u8, pos: Point3<usize>) -> (Point3<u8>, Point3<usize>) {
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
        (
            Point3::new(x_octant, y_octant, z_octant),
            Point3::new(x, y, z),
        )
    }

    pub fn insert(&mut self, pos: Point3<usize>, data: T) -> &mut T {
        let (quadrant, next_pos) = Self::subdivide(self.height, pos);
        let next_node =
            &mut *self.children[quadrant.x as usize][quadrant.y as usize][quadrant.z as usize];

        match next_node {
            Node::Tree(child_tree) => child_tree.insert(next_pos, data),
            Node::Leaf(old_data) => {
                *old_data = data;
                old_data
            }
            Node::Empty => {
                if self.height == 1 {
                    *next_node = Node::Leaf(data);
                    match next_node {
                        Node::Leaf(data) => data,
                        _ => unreachable!(),
                    }
                } else {
                    *next_node = Node::Tree(Self::new(self.height - 1));
                    match next_node {
                        Node::Tree(child_tree) => child_tree.insert(next_pos, data),
                        _ => unreachable!(),
                    }
                }
            }
        }
    }

    pub fn get(&self, pos: Point3<usize>) -> Option<&T> {
        let (quadrant, next_pos) = Self::subdivide(self.height, pos);
        let next_node =
            &*self.children[quadrant.x as usize][quadrant.y as usize][quadrant.z as usize];

        match next_node {
            Node::Tree(child_tree) => child_tree.get(next_pos),
            Node::Leaf(data) => Some(data),
            Node::Empty => None,
        }
    }

    pub fn get_mut(&mut self, pos: Point3<usize>) -> Option<&mut T> {
        let (quadrant, next_pos) = Self::subdivide(self.height, pos);
        let next_node =
            &mut *self.children[quadrant.x as usize][quadrant.y as usize][quadrant.z as usize];

        match next_node {
            Node::Tree(child_tree) => child_tree.get_mut(next_pos),
            Node::Leaf(data) => Some(data),
            Node::Empty => None,
        }
    }

    pub fn get_or_insert<F>(&mut self, pos: Point3<usize>, make_data: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        let (quadrant, next_pos) = Self::subdivide(self.height, pos);
        let next_node =
            &mut *self.children[quadrant.x as usize][quadrant.y as usize][quadrant.z as usize];

        match next_node {
            Node::Tree(child_tree) => child_tree.get_or_insert(next_pos, make_data),
            Node::Leaf(data) => data,
            Node::Empty => {
                *next_node = Node::Tree(Self::new(self.height - 1));
                match next_node {
                    Node::Tree(child_tree) => child_tree.insert(pos, make_data()),
                    _ => unreachable!(),
                }
            }
        }
    }

    // pub fn iter(&self) -> impl Iterator<Item=&T> {
    //     self.children.iter().flat_map(|cx| {
    //         cx.iter().flat_map(|cy| {
    //             cy.iter().flat_map(|child| {
    //                 match child {
    //                     Node::Empty => unimplemented!(),
    //                     Node::Leaf(data) => Some(data).iter(),
    //                     _ => unimplemented!()
    //                 }
    //             })
    //         })
    //     })

    //     // for cx in &self.children {
    //     //     for cy in cx {
    //     //         for child in cy {
    //     //             match &*child {
    //     //                 Node::Empty => Box::new((&[]).iter()),
    //     //                 _ => unimplemented!()
    //     //             }
    //     //         }
    //     //     }
    //     // }
    // }

    /// Serializes the octree in a binary format. Returns the number of bytes written.
    /// Accepts a function to serialize individual data leaves. This function must precisely report
    /// the number of bytes it wrote if it succeeds.
    pub fn serialize<W, F>(&self, writer: &mut W, serialize_data: &mut F) -> io::Result<usize>
    where
        W: Write,
        F: FnMut(&T, &mut W) -> io::Result<usize>,
    {
        // Cannot serialize a tree of height 255 because first byte is height + 1
        assert!(self.height < 255);

        let mut bytes_written = 0;
        bytes_written += writer.write(&[self.height + 1])?;

        // Loop through all 3 layers in the children array
        for cx in &self.children {
            for cy in cx {
                for child in cy {
                    bytes_written += Self::serialize_node(&*child, writer, serialize_data)?;
                }
            }
        }

        Ok(bytes_written)
    }

    fn serialize_node<W, F>(
        node: &Node<T>,
        writer: &mut W,
        serialize_data: &mut F,
    ) -> io::Result<usize>
    where
        W: Write,
        F: FnMut(&T, &mut W) -> io::Result<usize>,
    {
        // First byte
        // - 0 for empty
        // - 1 for leaf, followed by serialize_data encoding
        // - height + 1 for tree, followed by tree encoding

        match node {
            Node::Empty => writer.write(&[0]),
            Node::Leaf(data) => {
                let initial_size = writer.write(&[1])?;
                Ok(initial_size + serialize_data(data, writer)?)
            }
            Node::Tree(tree) => {
                // Tree encodes its own height
                tree.serialize(writer, serialize_data)
            }
        }
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
            0 | 1 => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                anyhow!("Initial byte value of {} not valid: must be > 1", byte0),
            )),
            _ => Self::deserialize_known_height(byte0 - 1, reader, deserialize_data),
        }
    }

    fn deserialize_known_height<R, F>(
        height: u8,
        reader: &mut R,
        deserialize_data: &mut F,
    ) -> io::Result<Self>
    where
        R: Read,
        F: FnMut(&mut R) -> io::Result<T>,
    {
        let mut node =
            || Ok::<_, io::Error>(Box::new(Self::deserialize_node(reader, deserialize_data)?));

        let children = [
            [[node()?, node()?], [node()?, node()?]],
            [[node()?, node()?], [node()?, node()?]],
        ];

        Ok(Octree { height, children })
    }

    fn deserialize_node<R, F>(reader: &mut R, deserialize_data: &mut F) -> io::Result<Node<T>>
    where
        R: Read,
        F: FnMut(&mut R) -> io::Result<T>,
    {
        let mut buf0: [u8; 1] = [0];
        reader.read_exact(&mut buf0)?;
        let byte0 = buf0[0];

        match byte0 {
            0 => Ok(Node::Empty),
            1 => {
                let data = deserialize_data(reader)?;
                Ok(Node::Leaf(data))
            }
            _ => Ok(Node::Tree(Self::deserialize_known_height(
                byte0 - 1,
                reader,
                deserialize_data,
            )?)),
        }
    }
}

// struct RefIterator<'a, T> {
//     tree: Octree
// }
