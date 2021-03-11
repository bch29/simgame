use anyhow::{anyhow, Context, Result};
use cgmath::{Deg, Matrix3, Point3, Transform, Vector3};
use rand::Rng;
use serde::{Deserialize, Serialize};

use simgame_voxels::{primitive, Voxel, VoxelDirectory};

use crate::lsystem;
use crate::turtle::{Turtle, TurtleBrush, TurtleInterpreter, TurtleState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSystemConfig {
    trunk_voxel: String,
    foliage_voxel: String,
    trunk_radius: f64,
    branch_radius_step: f64,
    trunk_length: f64,
    foliage_length: f64,
    foliage_radius: f64,
    rotate_degrees: f64,
    steps: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    tree: TreeSystemConfig,
    l_system: lsystem::LSystemConfig<Symbol>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Symbol {
    Expand,
    Contract,
    Base { id: u8 },
    Push,
    Pop,
    Wood { id: u8 },
    Leaf { id: u8 },
    YawCw,
    YawCcw,
    PitchCw,
    PitchCcw,
    RollCw,
    RollCcw,
    Flip,
}

pub fn generate<R: Rng>(
    config: &TreeConfig,
    voxels: &VoxelDirectory,
    rng: &mut R,
) -> Result<primitive::Shape> {
    let mut system = TreeSystem::new(config, voxels)?;
    system.generate(rng)
}

#[derive(Debug, Clone)]
pub struct TreeSystem {
    root_pos: Point3<f64>,
    trunk_voxel: Voxel,
    foliage_voxel: Voxel,
    config: TreeSystemConfig,
    l_system: lsystem::LSystem<Symbol>,
}

impl TreeSystem {
    pub fn new(config: &TreeConfig, voxels: &VoxelDirectory) -> Result<Self> {
        let l_system = config.l_system.as_l_system()?;

        let trunk_voxel = voxels
            .voxel_by_name(&config.tree.trunk_voxel[..])
            .ok_or_else(|| {
                anyhow!(
                    "Unknown voxel configured for trunk: {}",
                    config.tree.trunk_voxel
                )
            })?
            .0;

        let foliage_voxel = voxels
            .voxel_by_name(&config.tree.foliage_voxel[..])
            .ok_or_else(|| {
                anyhow!(
                    "Unknown voxel configured for foliage: {}",
                    config.tree.foliage_voxel
                )
            })?
            .0;

        Ok(Self {
            root_pos: Point3::new(0., 0., 0.),
            trunk_voxel,
            foliage_voxel,
            l_system,
            config: config.tree.clone(),
        })
    }

    pub fn generate<R: Rng>(&mut self, rng: &mut R) -> Result<primitive::Shape> {
        let symbols = self
            .l_system
            .run(self.config.steps, rng)
            .with_context(|| "tree L-system failed")?;
        log::info!("Generated tree L-system with {} symbols", symbols.len());
        let turtle = self.run(symbols)?;
        Ok(turtle.into_shape())
    }
}

impl TurtleInterpreter<Symbol> for TreeSystem {
    fn initial_state(&self) -> TurtleState {
        TurtleState {
            pos: self.root_pos,
            direction: Vector3::unit_z(),
            thickness: self.config.trunk_radius,
            brush: TurtleBrush::FilledLine {
                fill_voxel: self.trunk_voxel,
                round_start: false,
                round_end: true,
            },
        }
    }

    fn apply(&mut self, symbol: Symbol, turtle: &mut Turtle) -> Result<()> {
        let apply_rotate = |axis, rot_amount, direction: &mut Vector3<f64>| {
            let rotation: Matrix3<f64> = Matrix3::from_axis_angle(axis, rot_amount);
            let transformed =
                <Matrix3<f64> as Transform<Point3<f64>>>::transform_vector(&rotation, *direction);
            *direction = transformed;
        };

        let rot_amount = Deg(self.config.rotate_degrees);

        use Symbol::*;
        match symbol {
            Push => turtle.push_state(),
            Pop => turtle.pop_state()?,
            Expand => turtle.state_mut().thickness += self.config.branch_radius_step,
            Contract => turtle.state_mut().thickness -= self.config.branch_radius_step,
            Wood { .. } => {
                turtle.state_mut().brush = TurtleBrush::FilledLine {
                    fill_voxel: self.trunk_voxel,
                    round_end: true,
                    round_start: true,
                };
                turtle.draw(self.config.trunk_length);
            }
            Leaf { .. } => {
                turtle.push_state();
                turtle.state_mut().thickness = self.config.foliage_radius;

                turtle.state_mut().brush = TurtleBrush::Spheroid {
                    stretch: true,
                    fill_voxel: self.foliage_voxel,
                };
                turtle.draw(self.config.foliage_length);
                turtle.pop_state()?;
            }
            YawCw => {
                apply_rotate(
                    Vector3::unit_z(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            YawCcw => {
                apply_rotate(
                    -Vector3::unit_z(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            PitchCw => {
                apply_rotate(
                    Vector3::unit_x(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            PitchCcw => {
                apply_rotate(
                    -Vector3::unit_x(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            RollCw => {
                apply_rotate(
                    Vector3::unit_y(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            RollCcw => {
                apply_rotate(
                    -Vector3::unit_y(),
                    rot_amount,
                    &mut turtle.state_mut().direction,
                );
            }
            Flip => {
                apply_rotate(
                    Vector3::unit_y(),
                    Deg(180.),
                    &mut turtle.state_mut().direction,
                );
            }
            _ => {}
        };

        Ok(())
    }
}
