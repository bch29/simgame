use std::collections::HashMap;

use anyhow::{anyhow, Result};
use cgmath::{Matrix4, SquareMatrix};
use serde::{Deserialize, Serialize};

use simgame_util::{convert_matrix4, Bounds};

use crate::{TextureDirectory, World, WorldDelta};

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

pub struct EntityDirectory {
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
    pub bounds: Bounds<f64>,
}

pub struct ActiveEntityModel {
    pub model_kind: config::ModelKind,
    pub transform: Matrix4<f32>,
    pub face_tex_ids: Vec<u32>,
}

pub mod config {
    use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};
    use serde::{Deserialize, Serialize};

    use simgame_util::Bounds;

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

    impl ModelKind {
        pub fn bounds(&self) -> Bounds<f64> {
            Bounds::from_center(Point3::origin(), Vector3::new(2., 2., 2.))
        }
    }
}

impl ConcreteModel {
    fn from_config_model(
        model: &config::Model,
        texture_directory: &TextureDirectory,
    ) -> Result<Self> {
        let transform = {
            let mut result = Matrix4::identity();

            for transform in &model.transforms {
                result = transform.to_matrix() * result;
            }

            result
        };

        let bounds = model
            .kind
            .bounds()
            .transform(convert_matrix4!(transform, f64))
            .ok_or_else(|| anyhow!("unable to transform bounds for model {:?}", model.name))?;

        let face_texture_ids = model
            .face_texture_resources
            .iter()
            .map(|resource| {
                let key = texture_directory.texture_key(resource.as_str())?;

                Ok(key.index as u32)
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            name: model.name.clone(),
            kind: model.kind.clone(),
            face_texture_ids,
            transform,
            bounds,
        })
    }
}

impl EntityDirectory {
    pub fn new(
        config: &EntityConfig,
        behaviors: Vec<Box<dyn Behavior>>,
        texture_directory: &TextureDirectory,
    ) -> Result<Self> {
        let models: Vec<ConcreteModel> = config
            .models
            .iter()
            .map(|model| ConcreteModel::from_config_model(model, texture_directory))
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

    pub fn model_key(&self, name: &str) -> Result<ModelKey> {
        self.model_keys
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("model does not exist: {:?}", name))
    }

    pub fn behavior_key(&self, name: &str) -> Result<BehaviorKey> {
        self.behavior_keys
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("behavior does not exist: {:?}", name))
    }

    pub fn model(&self, key: ModelKey) -> Result<&ConcreteModel> {
        self.models
            .get(key.index as usize)
            .ok_or_else(|| anyhow!("model does not exist: {:?}", key))
    }

    pub fn behavior(&self, key: BehaviorKey) -> Result<&dyn Behavior> {
        self.behaviors
            .get(key.index as usize)
            .map(std::ops::Deref::deref)
            .ok_or_else(|| anyhow!("behavior does not exist: {:?}", key))
    }
}
