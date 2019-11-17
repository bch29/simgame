use std::collections::HashMap;

use cgmath::Point2;

use simgame_core::block;
use crate::mesh;

pub struct Triangulator {
    /// For each block, the lower-left coordinates in the texture map for that block's texture
    /// Each texture is assumed to be of dimensions 1.0 x 1.0
    block_to_uv: HashMap<block::Block, Point2<f32>>,
}

// impl Triangulator {
//     pub fn triangulate_chunk(chunk: &block::Chunk) -> 
// }
