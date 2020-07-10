use anyhow::{anyhow, Result};
use cgmath::{Angle, EuclideanSpace, InnerSpace, Matrix4, Point3, Rad, Vector3};

use simgame_core::block::Block;

use crate::primitives;

pub struct Turtle {
    state: TurtleState,
    state_stack: Vec<TurtleState>,
    components: Vec<primitives::ShapeComponent>,
}

#[derive(Debug, Clone, Copy)]
pub enum TurtleBrush {
    FilledLine {
        fill_block: Block,
        round_start: bool,
        round_end: bool,
    },
    Spheroid {
        /// If true, the spheroid is stretched between the start and end points of the drawing.
        /// Otherwise, it is a proper sphere placed directly between the two points.
        stretch: bool,
        fill_block: Block,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct TurtleState {
    pub pos: Point3<f64>,
    pub direction: Vector3<f64>,
    pub thickness: f64,
    pub brush: TurtleBrush,
}

pub trait TurtleInterpreter<Symbol> {
    fn initial_state(&self) -> TurtleState;

    fn apply(&mut self, symbol: Symbol, turtle: &mut Turtle) -> Result<()>;

    fn run<Iter>(&mut self, symbols: Iter) -> Result<Turtle>
    where
        Iter: IntoIterator<Item = Symbol>,
    {
        let mut turtle = Turtle::new(self.initial_state());

        for symbol in symbols.into_iter() {
            self.apply(symbol, &mut turtle)?;
        }

        Ok(turtle)
    }
}

impl Turtle {
    pub fn new(state: TurtleState) -> Self {
        Self {
            state,
            state_stack: vec![],
            components: vec![],
        }
    }

    pub fn into_shape(self) -> primitives::Shape {
        primitives::Shape::new(self.components)
    }

    pub fn push_state(&mut self) {
        self.state_stack.push(self.state);
    }

    pub fn pop_state(&mut self) -> Result<()> {
        self.state = self
            .state_stack
            .pop()
            .ok_or_else(|| anyhow!("popped turtle state with empty stack"))?;
        Ok(())
    }

    pub fn state(&self) -> &TurtleState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut TurtleState {
        &mut self.state
    }

    pub fn skip(&mut self, distance: f64) {
        self.state.pos += distance * self.state.direction.normalize();
    }

    pub fn draw(&mut self, distance: f64) {
        let end_pos = self.state.pos + distance * self.state.direction.normalize();

        match self.state.brush {
            TurtleBrush::FilledLine {
                fill_block,
                round_start,
                round_end,
            } => self.components.push(primitives::ShapeComponent {
                fill_block,
                primitive: Box::new(primitives::FilledLine {
                    start: self.state.pos,
                    end: end_pos,
                    radius: self.state.thickness,
                    round_start,
                    round_end,
                }),
            }),
            TurtleBrush::Spheroid {
                fill_block,
                stretch,
            } => {
                let sphere = primitives::Sphere {
                    center: Point3::origin(),
                    radius: 1.,
                };

                let rotation: Matrix4<f64> = {
                    let dir = self.state.direction.normalize();
                    let up = Vector3::unit_z();
                    let rot_axis = up.cross(dir).normalize();
                    let rot_angle = Rad::acos(up.dot(dir));
                    Matrix4::from_axis_angle(rot_axis, rot_angle)
                };

                let scale = {
                    let z_scale = if stretch {
                        (end_pos - self.state.pos).magnitude()
                    } else {
                        self.state.thickness
                    };
                    Matrix4::from_nonuniform_scale(
                        self.state.thickness,
                        self.state.thickness,
                        z_scale,
                    )
                };

                let translation = {
                    let center = self.state.pos + (end_pos - self.state.pos) / 2.;
                    Matrix4::from_translation(center - Point3::origin())
                };

                let transform = translation * rotation * scale;

                let primitive = primitives::AffineTransform::new(sphere, transform)
                    .expect("spheroid brush resulted in uninvertible transform");

                self.components.push(primitives::ShapeComponent {
                    fill_block,
                    primitive: Box::new(primitive),
                });
            }
        }

        self.state.pos = end_pos;
    }
}
