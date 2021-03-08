use serde::{Deserialize, Serialize};

use crate::{World, UpdatedWorldState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub model: ModelKey,
    pub behaviours: Vec<BehaviorKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ModelKey(u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BehaviorKey(u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Model {
    Sphere { radius: f32 },
}

pub trait Behavior {
    fn name(&self) -> &str;
    fn update(&self, entity: &Entity, world: &World, updates: &mut UpdatedWorldState);
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
