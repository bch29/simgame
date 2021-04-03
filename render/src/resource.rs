use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};

use crate::shaders::{CompileParams, ShaderCompiler, ShaderKind};

use simgame_types::{TextureDirectory, TextureKey};

pub struct ResourceLoader {
    root: PathBuf,
    artifact_root: PathBuf,
    resources: HashMap<String, Resource>,
    shader_compiler: ShaderCompiler,
    _config: ResourceConfig,
    options: ResourceOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceOptions {
    pub recompile_option: RecompileOption,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecompileOption {
    None,
    All,
    Subset(HashSet<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub root: PathBuf,
    pub artifact_root: PathBuf,
    pub resources: Vec<Resource>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    Image {
        relative_path: PathBuf,
        mip: MipType,
    },
    SolidColor {
        red: u8,
        green: u8,
        blue: u8,
    },
    Shader {
        relative_path: PathBuf,
        kind: ShaderKind,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MipType {
    NoMip,
    Mip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub name: String,
    pub kind: ResourceKind,
}

pub struct TextureLoader {
    directory: TextureDirectory,
    images: Vec<ImageLoader>,
}

pub struct ImageLoader {
    name: String,
    sample_generator: SampleGeneratorBuilder,
    mip: MipType,
}

pub(crate) struct Textures {
    textures: Vec<TextureData>,
}

pub(crate) struct TextureData {
    pub texture: wgpu::Texture,
    pub mip_level_count: u32,
}

impl ResourceLoader {
    pub fn new(
        data_root: &Path,
        config: ResourceConfig,
        options: ResourceOptions,
    ) -> Result<Self> {
        let mut resources: HashMap<String, Resource> = HashMap::new();
        for res in &config.resources {
            if res.name.is_empty() {
                bail!("Resource {:?} has empty name", res);
            }
            if resources.contains_key(res.name.as_str()) {
                bail!("Duplicate resource name {:?}", res.name);
            }
            resources.insert(res.name.clone(), res.clone());
        }

        if let RecompileOption::Subset(subset) = &options.recompile_option {
            let all_shaders: HashSet<String> = config
                .resources
                .iter()
                .filter(|resource| matches!(&resource.kind, ResourceKind::Shader { .. }))
                .map(|resource| resource.name.clone())
                .collect();
            let missing: Vec<&String> = subset
                .iter()
                .filter(|name| !all_shaders.contains(name.as_str()))
                .collect();

            if !missing.is_empty() {
                bail!(
                    "Invalid --force-recompile-shaders arguments {:?}. Valid arguments are {:?}.",
                    missing,
                    all_shaders
                );
            }
        }

        let mut root = data_root.to_path_buf();
        root.push(config.root.clone());

        let mut artifact_root = data_root.to_path_buf();
        artifact_root.push(config.artifact_root.clone());

        let shader_compiler = ShaderCompiler::new(CompileParams {
            chunk_size: simgame_voxels::index_utils::chunk_size().into(),
        })?;

        Ok(Self {
            root,
            artifact_root,
            resources,
            shader_compiler,
            _config: config,
            options,
        })
    }

    pub fn load_image(&self, name: &str) -> Result<ImageLoader> {
        let resource = self.get_resource(name)?;
        if let Some(res) = self.load_image_impl(resource) {
            Ok(res)
        } else {
            bail!(
                "Resource {:?} was loaded as an image, but configured as {:?}",
                resource.name,
                resource.kind
            )
        }
    }

    fn load_image_impl(&self, resource: &Resource) -> Option<ImageLoader> {
        match resource.kind {
            ResourceKind::Image {
                ref relative_path,
                mip,
            } => {
                let absolute_path = self.path_to_absolute(relative_path.as_path());
                let sample_generator = SampleGeneratorBuilder::Image {
                    relative_path: relative_path.clone(),
                    absolute_path,
                };
                Some(ImageLoader {
                    name: resource.name.clone(),
                    sample_generator,
                    mip,
                })
            }
            ResourceKind::SolidColor { red, green, blue } => {
                let sample_generator = SampleGeneratorBuilder::SolidColor { red, green, blue };
                Some(ImageLoader {
                    name: resource.name.clone(),
                    sample_generator,
                    mip: MipType::NoMip,
                })
            }
            _ => None,
        }
    }

    pub fn get_resource(&self, name: &str) -> Result<&Resource> {
        self.resources
            .get(name)
            .ok_or_else(|| anyhow!("Unknown resource {:?}", name))
    }

    pub fn load_shader(&self, name: &str) -> Result<Vec<u32>> {
        let resource = self.get_resource(name)?;

        let (shader_kind, relative_path) = match &resource.kind {
            ResourceKind::Shader {
                kind,
                relative_path,
            } => (kind.clone(), relative_path.as_path()),
            _ => bail!(
                "Resource {:?} was loaded as a shader, but configured as {:?}",
                name,
                resource.kind
            ),
        };

        let src_path = self.path_to_absolute(relative_path);
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

        let force_recompile = match &self.options.recompile_option {
            RecompileOption::All => true,
            RecompileOption::Subset(subset) => subset.contains(name),
            RecompileOption::None => false,
        };

        if force_recompile {
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

    #[allow(unused)]
    fn open_relative_path(&self, relative_path: &Path) -> Result<BufReader<File>> {
        let full_path = self.path_to_absolute(relative_path);
        let file = File::open(full_path.as_path())?;
        Ok(BufReader::new(file))
    }

    fn make_artifact_path(&self, resource: &Resource) -> Result<PathBuf> {
        if resource.name.is_empty() {
            bail!("Cannot make artifact path from empty resource name");
        }

        let extension = match resource.kind {
            ResourceKind::Shader { .. } => "spirv",
            _ => bail!(
                "Cannot make artifact path for resource {:?} of kind {:?}",
                resource.name,
                resource.kind
            ),
        };

        let mut path = self.artifact_root.clone();
        let mut path_components = resource.name.split_terminator('/').peekable();

        let fname = loop {
            let component = path_components
                .next()
                .ok_or_else(|| anyhow!("got empty iterator after splitting on '/'"))?;
            if path_components.peek().is_none() {
                break format!("{}.{}", component, extension);
            } else {
                path.push(component);
            }
        };

        std::fs::create_dir_all(&path)?;
        path.push(fname);

        Ok(path)
    }

    fn path_to_absolute(&self, relative_path: &Path) -> PathBuf {
        let mut res = self.root.clone();
        res.push(relative_path);
        res
    }

    pub fn texture_loader(&self) -> Result<TextureLoader> {
        let mut texture_keys = HashMap::new();
        let mut images = Vec::new();
        for resource in self.resources.values() {
            if let Some(img) = self.load_image_impl(resource) {
                texture_keys.insert(
                    resource.name.clone(),
                    TextureKey {
                        index: texture_keys.len() as _,
                    },
                );
                images.push(img);
            }
        }
        Ok(TextureLoader {
            images,
            directory: TextureDirectory::new(texture_keys),
        })
    }
}

impl TextureLoader {
    pub fn directory(&self) -> TextureDirectory {
        self.directory.clone()
    }

    pub(crate) fn load(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Textures> {
        let mut textures: Vec<TextureData> = Vec::new();

        for img in &self.images {
            let texture = Self::make_texture(
                device,
                queue,
                img.name.as_str(),
                textures.len(),
                img.sample_generator.build()?.as_ref(),
                img.mip,
            )?;
            textures.push(texture);
        }

        Ok(Textures { textures })
    }

    fn make_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
        index: usize,
        sample_generator: &dyn SampleGenerator,
        mip_type: MipType,
    ) -> Result<TextureData> {
        log::info!("Creating texture {:?} with index {}", name, index);

        let (width, height) = sample_generator.source_dimensions();

        match mip_type {
            MipType::NoMip => {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(name),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
                });
                let copy_view = wgpu::TextureCopyView {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                };

                let (samples, bytes_per_row) = sample_generator.generate(width, height);

                queue.write_texture(
                    copy_view,
                    samples.as_slice(),
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: 0,
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                );

                Ok(TextureData {
                    texture,
                    mip_level_count: 1,
                })
            }
            MipType::Mip => {
                if log2_exact(width).is_none() || log2_exact(height).is_none() {
                    bail!(
                        "Texture resource {:?} has dimensions {}x{}. Expected powers of 2 for mipped textures.",
                        name,
                        width,
                        height
                    );
                }

                let mip_level_count = log2_exact(width.min(height))
                    .expect("expected dimensions to be powers of 2")
                    .max(1);

                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("voxel texture array"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth: 1,
                    },
                    mip_level_count,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
                });

                log::info!("writing mip levels for {:?}", name);
                for mip_level in 0..mip_level_count {
                    let (mip_width, mip_height) = (width >> mip_level, height >> mip_level);

                    let copy_view = wgpu::TextureCopyView {
                        texture: &texture,
                        mip_level,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    };

                    let (samples, bytes_per_row) =
                        sample_generator.generate(mip_width, mip_height);

                    queue.write_texture(
                        copy_view,
                        samples.as_slice(),
                        wgpu::TextureDataLayout {
                            offset: 0,
                            bytes_per_row,
                            rows_per_image: 0,
                        },
                        wgpu::Extent3d {
                            width: mip_width,
                            height: mip_height,
                            depth: 1,
                        },
                    );
                }

                Ok(TextureData {
                    texture,
                    mip_level_count,
                })
            }
        }
    }
}

use sample_gen::{SampleGenerator, SampleGeneratorBuilder};

mod sample_gen {
    use std::path::PathBuf;

    use anyhow::Result;

    pub trait SampleGenerator {
        /// Returns the buffer of samples and the number of bytes per row in the result.
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32);

        fn source_dimensions(&self) -> (u32, u32);
    }

    pub enum SampleGeneratorBuilder {
        Image {
            relative_path: PathBuf,
            absolute_path: PathBuf,
        },
        SolidColor {
            red: u8,
            green: u8,
            blue: u8,
        },
    }

    impl SampleGeneratorBuilder {
        pub fn build(&self) -> Result<Box<dyn SampleGenerator>> {
            match *self {
                SampleGeneratorBuilder::Image {
                    ref relative_path,
                    ref absolute_path,
                } => {
                    log::info!("Loading texture from file {:?}", relative_path);
                    let file =
                        std::io::BufReader::new(std::fs::File::open(absolute_path.as_path())?);
                    let reader = image::io::Reader::new(file).with_guessed_format()?;
                    let image = reader.decode()?;
                    Ok(Box::new(ImageSampleGenerator {
                        image: image.to_rgba8(),
                    }))
                }
                SampleGeneratorBuilder::SolidColor { red, green, blue } => {
                    log::info!(
                        "Creating solid color texture with R/G/B {}/{}/{}",
                        red,
                        green,
                        blue
                    );
                    Ok(Box::new(SolidColorSampleGenerator { red, green, blue }))
                }
            }
        }
    }

    pub struct ImageSampleGenerator {
        pub image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    }

    pub struct SolidColorSampleGenerator {
        pub red: u8,
        pub green: u8,
        pub blue: u8,
    }

    impl SampleGenerator for ImageSampleGenerator {
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32) {
            let (img_width, img_height) = self.image.dimensions();

            let resized;
            let image = if img_width != width || img_height != height {
                resized = image::imageops::resize(
                    &self.image,
                    width,
                    height,
                    image::imageops::FilterType::CatmullRom,
                );
                &resized
            } else {
                &self.image
            };

            let samples = image.as_flat_samples();
            let bytes_per_row = samples.layout.height_stride as u32;
            (samples.as_slice().to_vec(), bytes_per_row)
        }

        fn source_dimensions(&self) -> (u32, u32) {
            self.image.dimensions()
        }
    }

    impl SampleGenerator for SolidColorSampleGenerator {
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32) {
            let mut buf = Vec::with_capacity(width as usize * height as usize * 4);
            for _row_ix in 0..height {
                for _col_ix in 0..width {
                    buf.push(self.red);
                    buf.push(self.green);
                    buf.push(self.blue);
                    buf.push(255); // alpha
                }
            }
            (buf, 4 * width)
        }

        fn source_dimensions(&self) -> (u32, u32) {
            (1, 1)
        }
    }
}

impl Textures {
    pub fn textures(&self) -> &[TextureData] {
        self.textures.as_slice()
    }

    pub fn from_resource(
        &self,
        directory: &TextureDirectory,
        resource: &str,
    ) -> Result<&TextureData> {
        Ok(&self.textures[directory.texture_key(resource)?.index as usize])
    }
}

/// If the value is a power of 2, returns its log base 2. Otherwise, returns `None`.
fn log2_exact(value: u32) -> Option<u32> {
    if value == 0 {
        return None;
    }

    let mut log2 = 0;
    let mut shifted = value;
    while shifted > 1 {
        shifted >>= 1;
        log2 += 1;
    }

    if 1 << log2 == value {
        Some(log2)
    } else {
        None
    }
}
