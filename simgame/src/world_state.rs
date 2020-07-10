use cgmath::{Point3, Vector3};
use anyhow::Result;

use simgame_core::block::{index_utils, Block};
use simgame_core::util::Bounds;
use simgame_core::world::{BlockUpdater, UpdatedWorldState, World};

pub struct WorldState {
    world: World,
    rng: rand::rngs::ThreadRng,
    updating: bool,
    filled_blocks: i32,
}

impl WorldState {
    pub fn new(world: World) -> Self {
        Self {
            world,
            rng: rand::thread_rng(),
            updating: false,
            filled_blocks: (16 * 16 * 4) / 8,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    /// Moves the world forward by one tick. Records anything that changed in the 'updated_state'.
    pub fn tick(&mut self, _elapsed: f64, updated_state: &mut UpdatedWorldState) -> Result<()> {
        if !self.updating {
            return Ok(());
        }

        use simgame_worldgen::turtle::{Turtle, TurtleState, TurtleBrush};

        let trunk_brush = TurtleBrush::FilledLine {
            fill_block: Block::from_u16(10),
            round_start: false,
            round_end: false
        };

        let leaf_brush = TurtleBrush::Spheroid {
            fill_block: Block::from_u16(1),
            stretch: true
        };

        let mut turtle = Turtle::new(TurtleState {
            pos: Point3::new(32., 32., 32.),
            direction: Vector3::new(0., 0., 1.),
            thickness: 5.,
            brush: trunk_brush
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

        let shape = turtle.into_shape();

        let mut updater = BlockUpdater::new(&mut self.world.blocks, updated_state);

        shape.draw(&mut updater);
        self.updating = false;

        Ok(())
    }

    pub fn modify_filled_blocks(&mut self, delta: i32, updated_state: &mut UpdatedWorldState) {
        self.filled_blocks += delta * 8;
        if self.filled_blocks < 1 {
            self.filled_blocks = 1
        } else if self.filled_blocks >= index_utils::chunk_size_total() as i32 {
            self.filled_blocks = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<usize> =
            Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_blocks as usize;
        let mut count_filled = 0;

        for p in bounds.iter_points() {
            if (p.x + p.y + p.z) % step as usize == 0 {
                self.world.blocks.set_block(p, Block::from_u16(1));
                count_filled += 1;
            } else {
                self.world.blocks.set_block(p, Block::from_u16(0));
            }
            let (chunk_pos, _) = index_utils::to_chunk_pos(p);
            updated_state.record_chunk_update(chunk_pos);
        }

        log::info!(
            "Setting number of filled blocks to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );
    }

    pub fn toggle_updates(&mut self) {
        self.updating = !self.updating;
    }
}
