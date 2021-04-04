use cgmath::{Matrix4, Point3, Quaternion, Vector3};
use serde::{Deserialize, Serialize};

use simgame_types::ModelKey;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Name {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub point: Point3<f64>,
}

pub type Bounds = simgame_util::Bounds<f64>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orientation {
    pub quat: Quaternion<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub key: ModelKey,
    pub transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Velocity(Vector3<f64>);
