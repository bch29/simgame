use std::{collections::HashMap, convert::TryInto, num::NonZeroU32, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use bevy::{
    app::{App, Plugin},
    asset::{AssetIo, Assets, FileAssetIo, Handle, HandleId},
    ecs::system::{Commands, IntoSystem, Res, ResMut},
    render2::texture::{
        AddressMode, Extent3d, FilterMode, SamplerDescriptor, Texture, TextureDimension,
        TextureFormat,
    },
    tasks::{ComputeTaskPool, TaskPool},
};
use crossbeam_channel::{Receiver, Sender};

use simgame_types::{config, TextureDirectory, TextureKey};

pub struct SimgameAssetsPlugin;

impl Plugin for SimgameAssetsPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(setup_system.system())
            .add_system(await_load_system.system());
    }
}

pub struct SimgameAssetsParams {
    pub config: config::AssetConfig,
}

fn setup_system(
    mut commands: Commands,
    params: Res<SimgameAssetsParams>,
    task_pool: Res<ComputeTaskPool>,
) {
    let loader = TextureLoader::new(&params.config).unwrap();
    let loading_texture_assets = loader.load(&*task_pool).unwrap();

    commands.insert_resource(loading_texture_assets);
}

fn await_load_system(
    mut core_texture_assets: ResMut<Assets<Texture>>,
    mut loading_texture_assets: ResMut<TextureAssets>,
) {
    let loading_texture_assets = &mut *loading_texture_assets;

    for LoadedTexture {
        index,
        handle,
        texture,
    } in loading_texture_assets.receiver.try_iter()
    {
        let handle = core_texture_assets.set(handle, texture);
        loading_texture_assets.textures[index] = handle;
        loading_texture_assets.count_loaded += 1;
    }

    if !loading_texture_assets.all_loaded
        && loading_texture_assets.count_loaded == loading_texture_assets.count_expected
    {
        log::info!(
            "Finished loading {} texture assets",
            loading_texture_assets.count_expected
        );
        loading_texture_assets.all_loaded = true;
    }
}

struct LoadedTexture {
    index: usize,
    handle: Handle<Texture>,
    texture: Texture,
}

#[derive(Clone)]
pub struct TextureAssets {
    pub directory: TextureDirectory,
    pub textures: Vec<Handle<Texture>>,
    pub all_loaded: bool,
    receiver: Receiver<LoadedTexture>,
    count_loaded: usize,
    count_expected: usize,
}

impl TextureAssets {
    pub fn texture_from_name(&self, name: &str) -> Result<Handle<Texture>> {
        Ok(self.textures[self.directory.texture_key(name)?.index as usize].clone())
    }
}

struct ImageLoader {
    name: String,
    sample_generator: SampleGeneratorBuilder,
    mip: config::TextureMipKind,
}

impl ImageLoader {
    fn new(name: &str, kind: &config::TextureKind) -> ImageLoader {
        match *kind {
            config::TextureKind::Image {
                ref asset_path,
                mip,
            } => {
                let sample_generator = SampleGeneratorBuilder::Image {
                    asset_path: asset_path.into(),
                };
                ImageLoader {
                    name: name.into(),
                    sample_generator,
                    mip,
                }
            }
            config::TextureKind::SolidColor { red, green, blue } => {
                let sample_generator = SampleGeneratorBuilder::SolidColor { red, green, blue };
                ImageLoader {
                    name: name.into(),
                    sample_generator,
                    mip: config::TextureMipKind::NoMip,
                }
            }
        }
    }
}

struct TextureLoader {
    images: Vec<ImageLoader>,
    directory: TextureDirectory,
    file_io: Arc<FileAssetIo>,
}

impl TextureLoader {
    fn new(config: &config::AssetConfig) -> Result<TextureLoader> {
        let mut texture_keys = HashMap::new();
        let mut images = Vec::new();
        for config::Texture { name, kind } in config.textures.iter() {
            let img = ImageLoader::new(name.as_str(), kind);
            texture_keys.insert(
                name.clone(),
                TextureKey {
                    index: texture_keys.len() as _,
                },
            );
            images.push(img);
        }

        Ok(TextureLoader {
            file_io: Arc::new(FileAssetIo::new("assets")),
            images,
            directory: TextureDirectory::new(texture_keys),
        })
    }

    fn load(&self, task_pool: &TaskPool) -> Result<TextureAssets> {
        log::info!("Beginning loaded of {} textures", self.images.len());

        let (sender, receiver) = crossbeam_channel::bounded(self.images.len());
        let sender = Arc::new(sender);

        let mut textures = Vec::new();

        for (index, img) in self.images.iter().enumerate() {
            let handle = Handle::weak(HandleId::random::<Texture>());
            textures.push(handle.clone());

            let file_io = self.file_io.clone();
            let name = img.name.clone();
            let sample_generator = img.sample_generator.clone();
            let mip = img.mip;
            let sender = sender.clone();

            task_pool
                .spawn(async move {
                    let res = Self::make_texture(
                        name.as_str(),
                        index,
                        sample_generator,
                        file_io,
                        mip,
                        handle,
                        sender,
                    )
                    .await;
                    match res {
                        Ok(()) => {}
                        Err(err) => log::error!("Failed to load texture {:?}: {:?}", name, err),
                    }
                })
                .detach();
        }

        Ok(TextureAssets {
            textures,
            receiver,
            count_loaded: 0,
            count_expected: self.images.len(),
            directory: self.directory.clone(),
            all_loaded: false,
        })
    }

    async fn make_texture(
        name: &str,
        index: usize,
        sample_generator: SampleGeneratorBuilder,
        file_io: Arc<FileAssetIo>,
        _mip_kind: config::TextureMipKind,
        handle: Handle<Texture>,
        sender: Arc<Sender<LoadedTexture>>,
    ) -> Result<()> {
        log::info!("Creating texture {:?} with index {}", name, index);

        let sample_generator = sample_generator.build(&*file_io).await?;

        let (width, height) = sample_generator.source_dimensions();

        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let format = TextureFormat::Rgba8UnormSrgb;
        let dimension = TextureDimension::D2;
        let sampler = SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            anisotropy_clamp: None,
            ..Default::default()
        };
        let data = sample_generator.generate(width, height)?.0;

        let texture = Texture {
            data,
            gpu_data: None,
            size,
            format,
            dimension,
            sampler,
        };

        sender.send(LoadedTexture {
            index,
            handle,
            texture,
        })?;

        log::info!("Loading texture {:?} done", name,);

        Ok(())

        // TODO: mips

        // match mip_kind {
        //     config::TextureMipKind::NoMip => {
        //     }
        //     config::TextureMipKind::Mip => {
        //         if log2_exact(width).is_none() || log2_exact(height).is_none() {
        //             bail!(
        //                 "Texture {:?} has dimensions {}x{}. Expected powers of 2 for mipped textures.",
        //                 name,
        //                 width,
        //                 height
        //             );
        //         }

        //         let mip_level_count = log2_exact(width.min(height))
        //             .expect("expected dimensions to be powers of 2")
        //             .max(1);

        //         let texture = device.create_texture(&wgpu::TextureDescriptor {
        //             label: Some("voxel texture array"),
        //             size: wgpu::Extent3d {
        //                 width,
        //                 height,
        //                 depth_or_array_layers: 1,
        //             },
        //             mip_level_count,
        //             sample_count: 1,
        //             dimension: wgpu::TextureDimension::D2,
        //             format: wgpu::TextureFormat::Rgba8UnormSrgb,
        //             usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
        //         });

        //         log::info!("writing mip levels for {:?}", name);
        //         for mip_level in 0..mip_level_count {
        //             let (mip_width, mip_height) = (width >> mip_level, height >> mip_level);

        //             let copy_view = wgpu::ImageCopyTexture {
        //                 texture: &texture,
        //                 mip_level,
        //                 origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        //             };

        //             let (samples, bytes_per_row) =
        //                 sample_generator.generate(mip_width, mip_height)?;

        //             queue.write_texture(
        //                 copy_view,
        //                 samples.as_slice(),
        //                 wgpu::ImageDataLayout {
        //                     offset: 0,
        //                     bytes_per_row: Some(bytes_per_row),
        //                     rows_per_image: None,
        //                 },
        //                 wgpu::Extent3d {
        //                     width: mip_width,
        //                     height: mip_height,
        //                     depth_or_array_layers: 1,
        //                 },
        //             );
        //         }

        //         Ok(TextureData {
        //             texture,
        //             mip_level_count,
        //         })
        //     }
        // }
    }
}

pub trait SampleGenerator {
    /// Returns the buffer of samples and the number of bytes per row in the result.
    fn generate(&self, width: u32, height: u32) -> Result<(Vec<u8>, NonZeroU32)>;

    fn source_dimensions(&self) -> (u32, u32);
}

#[derive(Clone)]
pub enum SampleGeneratorBuilder {
    Image { asset_path: PathBuf },
    SolidColor { red: u8, green: u8, blue: u8 },
}

impl SampleGeneratorBuilder {
    pub async fn build(
        &self,
        file_asset_io: &FileAssetIo,
    ) -> Result<Box<dyn SampleGenerator + Send>> {
        match *self {
            SampleGeneratorBuilder::Image { ref asset_path } => {
                log::info!("Loading texture from file {:?}", asset_path);
                let data = file_asset_io.load_path(asset_path.as_path()).await?;
                let reader =
                    image::io::Reader::new(std::io::Cursor::new(data)).with_guessed_format()?;
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
    fn generate(&self, width: u32, height: u32) -> Result<(Vec<u8>, NonZeroU32)> {
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
        let bytes_per_row: NonZeroU32 = (samples.layout.height_stride as u32)
            .try_into()
            .context("image height stride is 0")?;
        Ok((samples.as_slice().to_vec(), bytes_per_row))
    }

    fn source_dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }
}

impl SampleGenerator for SolidColorSampleGenerator {
    fn generate(&self, width: u32, height: u32) -> Result<(Vec<u8>, NonZeroU32)> {
        let mut buf = Vec::with_capacity(width as usize * height as usize * 4);
        for _row_ix in 0..height {
            for _col_ix in 0..width {
                buf.push(self.red);
                buf.push(self.green);
                buf.push(self.blue);
                buf.push(255); // alpha
            }
        }
        Ok((buf, (4 * width).try_into()?))
    }

    fn source_dimensions(&self) -> (u32, u32) {
        (1, 1)
    }
}

// /// If the value is a power of 2, returns its log base 2. Otherwise, returns `None`.
// fn log2_exact(value: u32) -> Option<u32> {
//     if value == 0 {
//         return None;
//     }

//     let mut log2 = 0;
//     let mut shifted = value;
//     while shifted > 1 {
//         shifted >>= 1;
//         log2 += 1;
//     }

//     if 1 << log2 == value {
//         Some(log2)
//     } else {
//         None
//     }
// }
