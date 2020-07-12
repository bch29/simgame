use anyhow::Result;
use cgmath::{Point3, Vector3, Matrix4, EuclideanSpace};

use simgame_worldgen::tree::{TreeConfig, TreeSystem};

use simgame_core::block::{index_utils, Block, BlockUpdater};
use simgame_core::convert_point;
use simgame_core::ray::Ray;
use simgame_core::util::Bounds;
use simgame_core::world::{UpdatedWorldState, World};

pub struct WorldState {
    world: World,
    world_diff: UpdatedWorldState,
    rng: rand::rngs::ThreadRng,
    updating: bool,
    filled_blocks: i32,

    tree_system: Option<TreeSystem>,
}

pub struct WorldStateInit<'a> {
    pub world: World,
    pub tree_config: Option<&'a TreeConfig>,
}

impl WorldState {
    pub fn new(init: WorldStateInit) -> Result<Self> {
        let tree_system = match init.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &init.world.block_helper)?),
            None => None
        };

        Ok(Self {
            world: init.world,
            world_diff: UpdatedWorldState::empty(),
            rng: rand::thread_rng(),
            updating: false,
            filled_blocks: (16 * 16 * 4) / 8,
            tree_system
        })
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_diff_mut(&mut self) -> &mut UpdatedWorldState {
        &mut self.world_diff
    }

    pub fn world_diff(&self) -> &UpdatedWorldState {
        &self.world_diff
    }

    /// Moves the world forward by one tick.
    pub fn tick(&mut self, _elapsed: f64) -> Result<()> {
        if !self.updating {
            return Ok(());
        }

        let shape = {
            use simgame_worldgen::turtle::{Turtle, TurtleBrush, TurtleState};

            let trunk_brush = TurtleBrush::FilledLine {
                fill_block: Block::from_u16(10),
                round_start: false,
                round_end: false,
            };

            let leaf_brush = TurtleBrush::Spheroid {
                fill_block: Block::from_u16(1),
                stretch: true,
            };

            let mut turtle = Turtle::new(TurtleState {
                pos: Point3::new(32., 32., 32.),
                direction: Vector3::new(0., 0., 1.),
                thickness: 5.,
                brush: trunk_brush,
            });

            // this should look a bit like a tree
            turtle.draw(30.);

            turtle.push_state();
            turtle.state_mut().direction = Vector3::new(0.5, 0.5, 0.5);
            turtle.state_mut().thickness *= 0.5;
            turtle.draw(20.);
            turtle.push_state();
            turtle.state_mut().brush = leaf_brush;
            turtle.state_mut().thickness *= 4.;
            turtle.draw(2.);
            turtle.pop_state()?;
            turtle.pop_state()?;

            turtle.push_state();
            turtle.state_mut().direction = Vector3::new(-0.5, 0.5, 0.5);
            turtle.state_mut().thickness *= 0.5;
            turtle.draw(20.);
            turtle.push_state();
            turtle.state_mut().brush = leaf_brush;
            turtle.state_mut().thickness *= 4.;
            turtle.draw(2.);
            turtle.pop_state()?;
            turtle.pop_state()?;

            turtle.into_shape()
        };

        let mut updater = BlockUpdater::new(&mut self.world.blocks, &mut self.world_diff.blocks);
        shape.draw(&mut updater);

        self.updating = false;

        Ok(())
    }

    pub fn modify_filled_blocks(&mut self, delta: i32) {
        self.filled_blocks += delta * 8;
        if self.filled_blocks < 1 {
            self.filled_blocks = 1
        } else if self.filled_blocks >= index_utils::chunk_size_total() as i32 {
            self.filled_blocks = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<i64> = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_blocks as i64;
        let mut count_filled = 0;

        for p in bounds.iter_points() {
            if (p.x + p.y + p.z) % step == 0 {
                self.world.blocks.set_block(p, Block::from_u16(1));
                count_filled += 1;
            } else {
                self.world.blocks.set_block(p, Block::from_u16(0));
            }
            let (chunk_pos, _) = index_utils::to_chunk_pos(p);
            self.world_diff.blocks.record_chunk_update(chunk_pos);
        }

        log::debug!(
            "Setting number of filled blocks to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );
    }

    pub fn on_click(
        &mut self,
        camera_pos: Point3<f64>,
        camera_facing: Vector3<f64>,
    ) -> Result<()> {
        let ray = Ray {
            origin: camera_pos,
            dir: camera_facing,
        };

        let raycast_hit = match self.world.blocks.cast_ray(&ray, &self.world.block_helper) {
            None => return Ok(()),
            Some(raycast_hit) => raycast_hit,
        };

        log::debug!(
            "Clicked on block {:?} at pos {:?} with intersection {:?}",
            raycast_hit.block,
            raycast_hit.block_pos,
            raycast_hit.intersection
        );

        let mut tree_system = match &self.tree_system {
            None => return Ok(()),
            Some(system) => system.clone()
        };

        let mut updater = BlockUpdater::new(&mut self.world.blocks, &mut self.world_diff.blocks);
        let tree = tree_system.generate(&mut self.rng)?;

        let root_pos =
            convert_point!(raycast_hit.block_pos, f64) + raycast_hit.intersection.normal;

        let draw_transform = Matrix4::from_translation(root_pos - Point3::origin());

        tree.draw_transformed(&mut updater, draw_transform);

        Ok(())
    }

    pub fn toggle_updates(&mut self) {
        self.updating = !self.updating;
    }
}
