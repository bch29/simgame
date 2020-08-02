use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub model: EntityModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityModelInfo {
    Sphere { radius: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityConfig {
    models: Vec<EntityModelInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct EntityModel(u16);

impl EntityConfig {
    pub fn model_info(&self, model: EntityModel) -> Option<&EntityModelInfo> {
        self.models.get(model.0 as usize)
    }
}
