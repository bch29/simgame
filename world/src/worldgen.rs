use std::time::Instant;

use anyhow::{anyhow, Result};
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
use log::info;
use rand::Rng;
use serde::{Deserialize, Serialize};

use simgame_voxels::{Voxel, VoxelConfigHelper, VoxelData, VoxelUpdater};
use simgame_util::Bounds;

use crate::{tree, UpdatedWorldState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateWorldConfig {
    pub size: Vector3<i64>,
    pub tree: Option<tree::TreeConfig>,
}

pub struct WorldGenerator<'a, R> {
    config: &'a GenerateWorldConfig,
    voxel_helper: &'a VoxelConfigHelper,
    bounds: Bounds<i64>,
    voxels: VoxelData,
    rng: &'a mut R,
}

pub fn generate_world(
    config: &GenerateWorldConfig,
    voxel_helper: &VoxelConfigHelper,
) -> Result<VoxelData> {
    info!("Creating empty world: {:?}", config);

    let mut rng = rand::thread_rng();

    let mut generator = WorldGenerator::new(config, voxel_helper, &mut rng);
    generator.generate()?;
    Ok(generator.finish())
}

impl<'a, R> WorldGenerator<'a, R> {
    pub fn new(
        config: &'a GenerateWorldConfig,
        voxel_helper: &'a VoxelConfigHelper,
        rng: &'a mut R,
    ) -> Self {
        let bounds = {
            let mut origin = Point3::origin() - config.size / 2;
            origin.z = 0;
            Bounds::new(origin, config.size)
        };

        let voxels = VoxelData::empty(bounds);

        Self {
            config,
            voxel_helper,
            bounds,
            voxels,
            rng,
        }
    }

    pub fn generate(&mut self) -> Result<()>
    where
        R: Rng,
    {
        let ts_start = Instant::now();
        self.generate_terrain()?;
        let ts_terrain = Instant::now();
        self.generate_trees()?;
        let ts_trees = Instant::now();

        metrics::timing!(
            "world.worldgen.terrain",
            ts_terrain.duration_since(ts_start)
        );
        metrics::timing!("world.worldgen.trees", ts_trees.duration_since(ts_terrain));

        Ok(())
    }

    pub fn finish(self) -> VoxelData {
        self.voxels
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

        let mut voxels_done = 0;
        let total_voxels = self.config.size.x * self.config.size.y * self.config.size.z;

        let progress_count = 10;
        let mut next_progress_step = 1;

        let rock_voxel = self
            .voxel_helper
            .voxel_by_name("Rock")
            .ok_or_else(|| anyhow!("Missing voxel config for Rock"))?
            .0;
        let dirt_voxel = self
            .voxel_helper
            .voxel_by_name("Dirt")
            .ok_or_else(|| anyhow!("Missing voxel config for Dirt"))?
            .0;
        let grass_voxel = self
            .voxel_helper
            .voxel_by_name("Grass")
            .ok_or_else(|| anyhow!("Missing voxel config for Grass"))?
            .0;

        info!("Bounds are {:?}", self.bounds);
        self.voxels
            .replace_voxels(self.bounds, |loc, _| -> Result<Voxel> {
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
                assert!((0. ..=1.).contains(&val));

                // this is the height of the terrain at current x, y coordinate
                let height_here = base_z + val * terrain_height;
                let p = p.mul_element_wise(world_size);

                let voxel = if p.z > height_here {
                    Voxel::air()
                } else if p.z > height_here - 1. {
                    grass_voxel
                } else if p.z > height_here - 4. {
                    dirt_voxel
                } else {
                    rock_voxel
                };

                voxels_done += 1;
                if voxels_done == (next_progress_step * total_voxels) / progress_count {
                    info!("{}%", next_progress_step * 10);
                    next_progress_step += 1;
                }

                Ok(voxel)
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
        let mut voxel_updater = VoxelUpdater::new(&mut self.voxels, &mut updated_state.voxels);

        let tree = tree::generate(tree_config, self.voxel_helper, &mut self.rng)?;
        tree.draw(&mut voxel_updater);

        Ok(())
    }
}
