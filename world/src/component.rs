use cgmath::{Matrix4, Point3, Quaternion, Vector3};
use serde::{Deserialize, Serialize};

use simgame_types::ModelKey;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Name(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position(pub Point3<f64>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounds(pub simgame_util::Bounds<f64>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Orientation(pub Quaternion<f64>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub key: ModelKey,
    pub transform: Matrix4<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Velocity(pub Vector3<f64>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ground {
    OnGround,
    NotOnGround,
}
