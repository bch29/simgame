use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Point3, Vector3, ElementWise};
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};

use simgame_core::block::{Block, BlockConfigHelper, BlockUpdater, WorldBlockData};
use simgame_core::util::Bounds;
use simgame_core::world::UpdatedWorldState;

pub mod lsystem;
pub mod primitive;
pub mod tree;
pub mod turtle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateWorldConfig {
    pub size: Vector3<i64>,
    pub tree: Option<tree::TreeConfig>,
}

pub struct WorldGenerator<'a, R> {
    config: &'a GenerateWorldConfig,
    block_helper: &'a BlockConfigHelper,
    bounds: Bounds<i64>,
    blocks: WorldBlockData,
    rng: &'a mut R,
}

pub fn generate_world(
    config: &GenerateWorldConfig,
    block_helper: &BlockConfigHelper,
) -> Result<WorldBlockData> {
    info!("Creating empty world: {:?}", config);

    let mut rng = rand::thread_rng();

    let mut generator = WorldGenerator::new(config, block_helper, &mut rng);
    generator.generate()?;
    Ok(generator.finish())
}

impl<'a, R> WorldGenerator<'a, R> {
    pub fn new(
        config: &'a GenerateWorldConfig,
        block_helper: &'a BlockConfigHelper,
        rng: &'a mut R,
    ) -> Self {
        let bounds = {
            let mut origin = Point3::origin() - config.size / 2;
            origin.z = 0;
            Bounds::new(origin, config.size)
        };

        let blocks = WorldBlockData::empty(bounds);

        Self {
            config,
            block_helper,
            bounds,
            blocks,
            rng,
        }
    }

    pub fn generate(&mut self) -> Result<()>
    where
        R: Rng,
    {
        self.generate_terrain()?;
        self.generate_trees()?;
        Ok(())
    }

    pub fn finish(self) -> WorldBlockData {
        self.blocks
    }

    fn generate_terrain(&mut self) -> Result<()>
    where
        R: Rng,
    {
        info!("Generating terrain");
        let world_size = Vector3 {
            x: self.config.size.x as f64,
            y: self.config.size.y as f64,
            z: self.config.size.z as f64,
        } / 2.;

        let base_z = 0.0;
        let terrain_height = world_size.z;

        let cos_factor = 2.0;
        let sin_factor = 1.0;

        let mut blocks_done = 0;
        let total_blocks = self.config.size.x * self.config.size.y * self.config.size.z;

        let progress_count = 10;
        let mut next_progress_step = 1;

        let rock_block = self
            .block_helper
            .block_by_name("Rock")
            .ok_or_else(|| anyhow!("Missing block config for Rock"))?.0;
        let dirt_block = self
            .block_helper
            .block_by_name("Dirt")
            .ok_or_else(|| anyhow!("Missing block config for Dirt"))?.0;

        info!("Bounds are {:?}", self.bounds);
        self.blocks
            .replace_blocks(self.bounds, |loc, _| -> Result<Block> {
                // Normalized point with coords in range (0, 1)
                let p = Vector3 {
                    x: loc.x as f64 / world_size.x,
                    y: loc.y as f64 / world_size.y,
                    z: loc.z as f64 / world_size.z,
                };

                // Empty above curve, filled below curve
                use std::f64::consts::PI;
                // val in range [0, 1]
                let val =
                    0.5 * (1.0 + (p.x * PI * sin_factor).sin() * (p.y * PI * cos_factor).cos());
                assert!(0. <= val && val <= 1.);

                // this is the height of the terrain at current x, y coordinate
                let height_here = base_z + val * terrain_height;
                let p = p.mul_element_wise(world_size);

                let block = if p.z > height_here {
                    Block::air()
                } else if p.z > height_here - 2. {
                    dirt_block
                } else {
                    rock_block
                };

                blocks_done += 1;
                if blocks_done == (next_progress_step * total_blocks) / progress_count {
                    info!("{}%", next_progress_step * 10);
                    next_progress_step += 1;
                }

                Ok(block)
            })
    }

    fn generate_trees(&mut self) -> Result<()>
    where
        R: Rng,
    {
        let tree_config = match &self.config.tree {
            Some(x) => x,
            None => return Ok(()),
        };

        let mut updated_state = UpdatedWorldState::empty();
        let mut block_updater = BlockUpdater::new(&mut self.blocks, &mut updated_state.blocks);

        let tree = tree::generate(tree_config, self.block_helper, &mut self.rng)?;
        tree.draw(&mut block_updater);

        Ok(())
    }
}
