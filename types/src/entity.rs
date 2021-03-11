use std::collections::HashMap;

use anyhow::Result;
use cgmath::Matrix4;
use serde::{Deserialize, Serialize};

use crate::{World, WorldDelta};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub model: ModelKey,
    pub behaviors: Vec<BehaviorKey>,
}

pub use config::EntityConfig;

pub trait Behavior {
    fn name(&self) -> &str;
    fn update(&self, entity: &Entity, world: &World, updates: &mut WorldDelta);
}

pub struct Directory {
    models: Vec<ConcreteModel>,
    behaviors: Vec<Box<dyn Behavior>>,

    model_keys: HashMap<String, ModelKey>,
    behavior_keys: HashMap<String, BehaviorKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ModelKey {
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BehaviorKey {
    pub index: u32,
}

#[derive(Debug, Clone)]
pub struct ConcreteModel {
    pub name: String,
    pub kind: config::ModelKind,
    pub face_texture_ids: Vec<u32>,
    pub transform: Matrix4<f32>,
}

pub mod config {
    use cgmath::{Matrix4, SquareMatrix, Vector3};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EntityConfig {
        pub models: Vec<Model>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Model {
        pub name: String,
        pub kind: ModelKind,
        pub face_texture_resources: Vec<String>,
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

    impl Model {
        pub fn transform_matrix(&self) -> Matrix4<f32> {
            let mut result = Matrix4::identity();

            for transform in &self.transforms {
                result = transform.to_matrix() * result;
            }

            result
        }
    }
}

impl Directory {
    pub fn new<F>(
        config: &EntityConfig,
        behaviors: Vec<Box<dyn Behavior>>,
        mut lookup_resource: F,
    ) -> Result<Self>
    where
        F: FnMut(&str) -> Result<u32>,
    {
        let models: Vec<ConcreteModel> = config
            .models
            .iter()
            .map(|model| {
                let face_texture_ids = model
                    .face_texture_resources
                    .iter()
                    .map(|resource| lookup_resource(resource.as_str()))
                    .collect::<Result<_>>()?;

                let transform = model.transform_matrix();

                Ok(ConcreteModel {
                    name: model.name.clone(),
                    kind: model.kind.clone(),
                    face_texture_ids,
                    transform,
                })
            })
            .collect::<Result<_>>()?;

        let model_keys: HashMap<_, _> = models
            .iter()
            .enumerate()
            .map(|(index, model)| {
                (
                    model.name.clone(),
                    ModelKey {
                        index: index as u32,
                    },
                )
            })
            .collect();

        let behavior_keys: HashMap<_, _> = behaviors
            .iter()
            .enumerate()
            .map(|(index, behavior)| {
                (
                    behavior.name().into(),
                    BehaviorKey {
                        index: index as u32,
                    },
                )
            })
            .collect();

        Ok(Self {
            models,
            behaviors,
            model_keys,
            behavior_keys,
        })
    }

    pub fn model_key(&self, name: &str) -> Option<ModelKey> {
        self.model_keys.get(name).copied()
    }

    pub fn behavior_key(&self, name: &str) -> Option<BehaviorKey> {
        self.behavior_keys.get(name).copied()
    }

    pub fn model(&self, key: ModelKey) -> Option<&ConcreteModel> {
        self.models.get(key.index as usize)
    }

    pub fn behavior(&self, key: BehaviorKey) -> Option<&dyn Behavior> {
        self.behaviors
            .get(key.index as usize)
            .map(std::ops::Deref::deref)
    }
}
