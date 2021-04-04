use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};
use serde::{Deserialize, Serialize};

use simgame_util::Bounds;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityConfig {
    pub models: Vec<Model>,
}

pub type ResourceName = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub kind: ModelKind,
    pub face_texture_resources: Vec<ResourceName>,
    pub transforms: Vec<Transform>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelKind {
    Sphere,
    Cube,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transform {
    Offset(Vector3<f32>),
    Rotation { axis: Vector3<f32>, degrees: f32 },
    UniformScale(f32),
    NonuniformScale(Vector3<f32>),
}

impl Transform {
    pub fn to_matrix(&self) -> Matrix4<f32> {
        match *self {
            Transform::Offset(offset) => Matrix4::from_translation(offset),
            Transform::Rotation { axis, degrees } => {
                Matrix4::from_axis_angle(axis, cgmath::Deg(degrees))
            }
            Transform::UniformScale(scale) => Matrix4::from_scale(scale),
            Transform::NonuniformScale(scale) => {
                Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z)
            }
        }
    }
}

impl ModelKind {
    pub fn bounds(&self) -> Bounds<f64> {
        Bounds::from_center(Point3::origin(), Vector3::new(0.5, 0.5, 0.5))
    }
}
