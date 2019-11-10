use anyhow::{bail, Result};
use cgmath::{Vector3, ElementWise};
use rand::Rng;

use crate::block::{index_utils, Block, WorldBlockData};

#[derive(Debug)]
pub struct GenerateWorldConfig {
    pub size: Vector3<usize>,
}

pub fn generate_world(config: &GenerateWorldConfig) -> Result<WorldBlockData> {
    let chunk_size = index_utils::chunk_size();

    let size_rem = config.size.rem_element_wise(chunk_size);
    if size_rem.x != 0 || size_rem.y != 0 || size_rem.z != 0 {
        bail!("size must be a multiple of [{}, {}, {}] (got [{}, {}, {}])", 
              chunk_size.x, chunk_size.y, chunk_size.y,
              config.size.x, config.size.y, config.size.z);
    }

    let count_chunks = config.size.div_element_wise(chunk_size);

    println!("Creating empty world");
    let mut blocks = WorldBlockData::empty(count_chunks);

    let world_size = Vector3 {
        x: blocks.size().x as f64,
        y: blocks.size().y as f64,
        z: blocks.size().z as f64,
    };

    println!("Generating terrain");
    let base_z = 0.0;
    let terrain_height = world_size.z;

    let cos_factor = 2.0;
    let sin_factor = 1.0;

    let mut blocks_done = 0;
    let total_blocks = blocks.size().x * blocks.size().y * blocks.size().z;

    let progress_count = 10;
    let mut next_progress_step = 1;

    let mut rng = rand::thread_rng();
    for (loc, block) in blocks.iter_blocks_with_loc_mut() {
        // Normalized point with coords in range (0, 1)
        let p = Vector3 {
            x: loc.x as f64 / world_size.x,
            y: loc.y as f64 / world_size.y,
            z: loc.z as f64 / world_size.z,
        };

        // Empty above curve, filled below curve
        use std::f64::consts::PI;
        // val in range [0, 1]
        let val = 0.5 * (1.0 + (p.x * PI * sin_factor).sin() * (p.y * PI * cos_factor).cos());
        assert!(0. <= val && val <= 1.);

        // this is the height of the terrain at current x, y coordinate
        let height_here = base_z + val * terrain_height;

        // fill with air if current z is above height_here, else rock
        if p.z * world_size.z > height_here {
            *block = Block::air();
        } else {
            // let block_val = rng.gen::<u16>() % 7;
            *block = Block::from_u16(1);
        }

        blocks_done += 1;
        if blocks_done == (next_progress_step * total_blocks) / progress_count {
            println!("{}%", next_progress_step * 10);
            next_progress_step += 1;
        }
    }

    println!("Generated world: {:?}", blocks.debug_summary());

    Ok(blocks)
}
