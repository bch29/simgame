use cgmath::{Matrix4, Point3, Vector3, SquareMatrix};
use serde::{Deserialize, Serialize};

use crate::{WorldDelta, World};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub model: ModelKey,
    pub behaviours: Vec<BehaviorKey>,
    pub location: Point3<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ModelKey(u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BehaviorKey(u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub kind: ModelKind,
    pub face_texture_resources: Vec<String>,
    pub transforms: Vec<Transform>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

pub trait Behavior {
    fn name(&self) -> &str;
    fn update(&self, entity: &Entity, world: &World, updates: &mut WorldDelta);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub models: Vec<Model>,
}

pub struct Directory {
    pub config: Config,
    pub behaviors: Vec<Box<dyn Behavior>>,
}

impl Directory {
    pub fn model(&self, key: ModelKey) -> Option<&Model> {
        self.config.models.get(key.0 as usize)
    }

    pub fn behavior(&self, key: BehaviorKey) -> Option<&dyn Behavior> {
        self.behaviors.get(key.0 as usize).map(|x| &**x)
    }
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

impl Model {
    pub fn transform_matrix(&self) -> Matrix4<f32> {
        let mut result = Matrix4::identity();

        for transform in &self.transforms {
            result = transform.to_matrix() * result;
        }

        result
    }
}
