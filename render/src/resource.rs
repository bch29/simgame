use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::shaders::{CompileParams, ShaderCompiler, ShaderKind};

pub struct ResourceLoader {
    root: PathBuf,
    artifact_root: PathBuf,
    resources: HashMap<String, Resource>,
    shader_compiler: ShaderCompiler,
    config: ResourceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub root: PathBuf,
    pub artifact_root: PathBuf,
    pub resources: Vec<Resource>,
    pub force_shader_compile: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    Image,
    Shader { kind: ShaderKind },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub name: String,
    pub relative_path: PathBuf,
    pub kind: ResourceKind,
}

impl ResourceLoader {
    pub fn new(data_root: &Path, config: ResourceConfig) -> Result<Self> {
        let resources = config
            .resources
            .iter()
            .map(|res| (res.name.clone(), res.clone()))
            .collect();

        std::fs::create_dir_all(&config.artifact_root)?;

        let mut root = data_root.to_path_buf();
        root.push(config.root.clone());

        let mut artifact_root = data_root.to_path_buf();
        artifact_root.push(config.artifact_root.clone());

        let shader_compiler = ShaderCompiler::new(CompileParams {
            chunk_size: simgame_core::block::index_utils::chunk_size().into(),
        })?;

        Ok(Self {
            root,
            artifact_root,
            resources,
            shader_compiler,
            config,
        })
    }

    pub fn load_image(&self, name: &str) -> Result<DynamicImage> {
        let resource = self.get_resource(name)?;

        if resource.kind != ResourceKind::Image {
            bail!(
                "Resource {:?} was loaded as an image, but configured as {:?}",
                name,
                resource.kind
            );
        }

        let file = self.open_resource(&resource)?;
        let reader = image::io::Reader::new(file).with_guessed_format()?;
        Ok(reader.decode()?)
    }

    pub fn get_resource(&self, name: &str) -> Result<&Resource> {
        Ok(self
            .resources
            .get(name)
            .ok_or_else(|| anyhow!("Unknown resource {:?}", name))?)
    }

    pub fn load_shader(&self, name: &str) -> Result<Vec<u32>> {
        let resource = self.get_resource(name)?;

        let shader_kind = match &resource.kind {
            ResourceKind::Shader { kind } => kind.clone(),
            _ => bail!(
                "Resource {:?} was loaded as a shader, but configured as {:?}",
                name,
                resource.kind
            ),
        };

        let src_path = self.full_resource_path(resource);
        let spirv_path = self.make_artifact_path(resource)?;

        let load_spirv = || -> Result<Vec<u32>> {
            let mut file = std::fs::File::open(&spirv_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;

            let mut words = vec![0; bytes.len() / 4];
            bytes
                .as_slice()
                .read_u32_into::<byteorder::LittleEndian>(&mut words)?;
            log::info!("Loaded precompiled shader: {:?}", spirv_path);
            Ok(words)
        };

        let save_spirv = |words| -> Result<()> {
            log::info!("Saving compiled shader: {:?}", spirv_path);
            let mut file = std::fs::File::create(&spirv_path)?;
            for &word in words {
                file.write_u32::<byteorder::LittleEndian>(word)?;
            }
            Ok(())
        };

        if self.config.force_shader_compile {
            log::info!("Compiling shader from source: {:?}", src_path);
            let result = self
                .shader_compiler
                .compile(src_path.as_path(), shader_kind)?;
            save_spirv(&result[..])?;
            Ok(result)
        } else {
            match load_spirv() {
                Ok(result) => Ok(result),
                Err(_) => {
                    log::info!("Compiling shader from source: {:?}", src_path);
                    let result = self
                        .shader_compiler
                        .compile(src_path.as_path(), shader_kind)?;
                    save_spirv(&result[..])?;
                    Ok(result)
                }
            }
        }
    }

    fn open_resource(&self, resource: &Resource) -> Result<BufReader<File>> {
        let path = resource.relative_path.as_path();
        self.open_relative_path(path)
    }

    fn open_relative_path(&self, relative_path: &Path) -> Result<BufReader<File>> {
        let full_path = self.path_to_absolute(relative_path);
        let file = File::open(full_path.as_path())?;
        Ok(BufReader::new(file))
    }

    fn full_resource_path(&self, resource: &Resource) -> PathBuf {
        self.path_to_absolute(resource.relative_path.as_path())
    }

    fn make_artifact_path(&self, resource: &Resource) -> Result<PathBuf> {
        match resource.kind {
            ResourceKind::Shader { .. } => {
                let dir_path = {
                    let mut path = self.artifact_root.clone();
                    path.push("shaders");
                    path
                };
                std::fs::create_dir_all(&dir_path)?;
                let mut res = dir_path;
                res.push(format!("{}.spirv", resource.name));
                Ok(res)
            }
            _ => bail!(
                "Cannot make artifact path for resource {:?} of kind {:?}",
                resource.name,
                resource.kind
            ),
        }
    }

    fn path_to_absolute(&self, relative_path: &Path) -> PathBuf {
        let mut res = self.root.clone();
        res.push(relative_path);
        res
    }
}
