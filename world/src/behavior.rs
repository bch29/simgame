use cgmath::Vector3;
use serde::{Deserialize, Serialize};

use simgame_types::impl_behavior;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Fall;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Accelerate(pub Vector3<f64>);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Impulse(pub Vector3<f64>);

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Friction {
    pub air: f64,
    pub ground: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bounce {
    pub coeff: f64,
}

impl_behavior! {
    impl Behavior for Fall;
    impl Behavior for Accelerate;
    impl Behavior for Impulse;
    impl Behavior for Friction;
    impl Behavior for Bounce;
}
