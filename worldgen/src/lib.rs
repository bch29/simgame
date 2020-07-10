use anyhow::Result;
use cgmath::Vector3;
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};

use simgame_core::block::{Block, WorldBlockData, BlockConfigHelper};
use simgame_core::world::{BlockUpdater, UpdatedWorldState};
use simgame_core::util::Bounds;

pub mod primitives;
pub mod lsystem;
pub mod turtle;
pub mod tree;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateWorldConfig {
    pub size: Vector3<usize>,
    pub tree: Option<tree::TreeConfig>,
}

pub fn generate_world(config: &GenerateWorldConfig, block_helper: &BlockConfigHelper) -> Result<WorldBlockData> {
    info!("Creating empty world: {:?}", config);
    let world_bounds = Bounds::from_size(config.size);
    let mut blocks = WorldBlockData::empty(world_bounds);

    let world_size = Vector3 {
        x: config.size.x as f64,
        y: config.size.y as f64,
        z: config.size.z as f64,
    };

    info!("Generating terrain");
    let base_z = 0.0;
    let terrain_height = world_size.z;

    let cos_factor = 2.0;
    let sin_factor = 1.0;

    let mut blocks_done = 0;
    let total_blocks = config.size.x * config.size.y * config.size.z;

    let progress_count = 10;
    let mut next_progress_step = 1;

    info!("Bounds are {:?}", world_bounds);
    let mut rng = rand::thread_rng();
    blocks
        .replace_blocks(world_bounds, |loc, _| -> Result<Block> {
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
            let block = if p.z * world_size.z > height_here {
                Block::air()
            } else {
                let block_val = 1 + rng.gen::<u16>() % 8;
                Block::from_u16(block_val)
            };

            blocks_done += 1;
            if blocks_done == (next_progress_step * total_blocks) / progress_count {
                info!("{}%", next_progress_step * 10);
                next_progress_step += 1;
            }

            Ok(block)
        })
        .unwrap();

    if let Some(tree_config) = &config.tree {
        let mut updated_state = UpdatedWorldState::empty();
        let mut block_updater = BlockUpdater::new(&mut blocks, &mut updated_state);

        let tree = tree::generate(tree_config, block_helper, &mut rng)?;
        tree.draw(&mut block_updater);
    }

    info!("Generated world: {:?}", blocks.debug_summary());

    Ok(blocks)
}
