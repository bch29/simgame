use std::collections::HashMap;

use anyhow::{anyhow, Result};
use cgmath::{Matrix4, SquareMatrix};
use serde::{Deserialize, Serialize};

use simgame_util::{convert_matrix4, Bounds};

use crate::{
    config::{self, ModelKind, ResourceName},
    mesh::Mesh,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ModelKey {
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct MeshKey {
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct TextureKey {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDirectory {
    meshes: Vec<Mesh>,
    models: Vec<ModelData>,
    model_keys: HashMap<String, ModelKey>,
}

#[derive(Debug, Clone)]
pub struct TextureDirectory {
    texture_keys: HashMap<ResourceName, TextureKey>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelData {
    pub name: String,
    pub mesh: MeshKey,
    pub face_texture_ids: Vec<u32>,
    pub transform: Matrix4<f32>,
    pub bounds: Bounds<f64>,
}

#[derive(Debug, Clone)]
pub struct ModelRenderData {
    pub model: ModelKey,
    pub transform: Matrix4<f32>,
}

// pub struct Model

impl ModelDirectory {
    pub fn new(
        cfg_models: &[config::Model],
        texture_directory: &TextureDirectory,
    ) -> Result<Self> {
        let mut models: Vec<ModelData> = Vec::new();
        let mut model_keys: HashMap<String, ModelKey> = HashMap::new();

        let mesh_kinds = vec![ModelKind::Cube, ModelKind::Sphere];

        let mesh_keys: HashMap<ModelKind, MeshKey> = mesh_kinds
            .iter()
            .enumerate()
            .map(|(ix, kind)| (kind.clone(), MeshKey { index: ix as _ }))
            .collect();

        let meshes: Vec<Mesh> = mesh_kinds
            .iter()
            .map(|kind| Mesh::from_model_kind(kind))
            .collect();

        for (index, model) in cfg_models.iter().enumerate() {
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

            let mesh = *mesh_keys
                .get(&model.kind)
                .ok_or_else(|| anyhow!("unknown model kind {:?}", model.kind))?;

            let face_texture_ids = model
                .face_texture_resources
                .iter()
                .map(|resource| {
                    let key = texture_directory.texture_key(&resource)?;

                    Ok(key.index as u32)
                })
                .collect::<Result<_>>()?;

            models.push(ModelData {
                name: model.name.clone(),
                mesh,
                face_texture_ids,
                transform,
                bounds,
            });

            model_keys.insert(model.name.clone(), ModelKey { index: index as _ });
        }

        Ok(Self {
            meshes,
            models,
            model_keys,
        })
    }

    pub fn meshes(&self) -> impl Iterator<Item = (MeshKey, &'_ Mesh)> {
        self.meshes
            .iter()
            .enumerate()
            .map(|(ix, mesh)| (MeshKey { index: ix as _ }, mesh))
    }

    pub fn model_data(&self, key: ModelKey) -> Result<&ModelData> {
        self.models
            .get(key.index as usize)
            .ok_or_else(|| anyhow!("model key does not exist: {:?}", key))
    }

    pub fn model_key(&self, name: &str) -> Result<ModelKey> {
        self.model_keys
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("model name does not exist: {:?}", name))
    }
}

impl TextureDirectory {
    pub fn new(texture_keys: HashMap<ResourceName, TextureKey>) -> Self {
        Self { texture_keys }
    }

    pub fn texture_key(&self, name: &str) -> Result<TextureKey> {
        self.texture_keys
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("texture does not exist: {:?}", name))
    }
}
