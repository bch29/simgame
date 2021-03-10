use std::collections::HashMap;
use std::convert::TryInto;

use anyhow::{anyhow, bail, Result};
use zerocopy::{AsBytes, FromBytes};

use simgame_voxels::{config as voxel, VoxelConfigHelper};

use crate::buffer_util::{InstancedBuffer, InstancedBufferDesc};
use crate::mesh::cube::Cube;
use crate::resource::ResourceLoader;

/// Manages all GPU buffers that are static for any given set of voxel types.
pub(crate) struct VoxelInfoManager {
    pub index_map: HashMap<voxel::FaceTexture, usize>,

    #[allow(unused)]
    texture_arr: Vec<TextureData>,
    pub texture_arr_views: Vec<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,

    pub voxel_info_buf: InstancedBuffer,
    pub texture_metadata_buf: InstancedBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct VoxelTextureMetadata {
    x_periodicity: u32,
    y_periodicity: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default)]
struct VoxelRenderInfo {
    // index into texture list, by face
    face_tex_ids: [u32; 6],
    _padding: [u32; 2],
    vertex_data: Cube,
}

struct TextureData {
    texture: wgpu::Texture,
    mip_level_count: u32,
}

impl VoxelInfoManager {
    pub fn new(
        config: &VoxelConfigHelper,
        resource_loader: &ResourceLoader,
        ctx: &crate::GraphicsContext,
    ) -> Result<Self> {
        let index_map = config.texture_index_map();
        log::info!("Got {} voxel textures total", index_map.len());

        let face_texes = {
            let mut face_texes: Vec<_> = index_map
                .iter()
                .map(|(face_tex, &index)| (index, face_tex))
                .collect();
            face_texes.sort_by_key(|(index, _)| *index);
            face_texes
        };

        let texture_arr: Vec<TextureData> = {
            let mut res: Vec<(usize, TextureData)> = face_texes
                .into_iter()
                .map(|(index, face_tex)| {
                    Ok((index, Self::make_texture(ctx, resource_loader, face_tex)?))
                })
                .collect::<Result<_>>()?;
            res.sort_by_key(|(index, _)| *index);
            res.into_iter().map(|(_, tex)| tex).collect()
        };

        let texture_arr_views = texture_arr
            .iter()
            .map(|data| {
                data.texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("voxel textures"),
                    format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    level_count: Some(
                        data.mip_level_count
                            .try_into()
                            .expect("mip level count cannot be 0"),
                    ),
                    base_array_layer: 0,
                    array_layer_count: Some(1.try_into().expect("nonzero")),
                })
            })
            .collect();

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: None,
            ..Default::default()
        });

        let voxel_info_buf = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "voxel info",
                instance_len: std::mem::size_of::<VoxelRenderInfo>(),
                n_instances: config.voxels().len(),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            },
        );

        for (index, voxel) in config.voxels().iter().enumerate() {
            let render_info = Self::get_voxel_render_info(&index_map, voxel)?;
            voxel_info_buf.write(&ctx.queue, index, render_info.as_bytes());
        }

        let texture_metadata_buf = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "texture metadata",
                instance_len: std::mem::size_of::<VoxelTextureMetadata>(),
                n_instances: index_map.len(),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            },
        );

        for (face_tex, &index) in index_map.iter() {
            let texture_metadata = Self::get_texture_metadata(face_tex);
            texture_metadata_buf.write(&ctx.queue, index, texture_metadata.as_bytes())
        }

        Ok(Self {
            index_map,

            texture_arr_views,
            texture_arr,
            sampler,

            voxel_info_buf,
            texture_metadata_buf,
        })
    }

    fn make_texture(
        ctx: &crate::GraphicsContext,
        resource_loader: &ResourceLoader,
        face_tex: &voxel::FaceTexture,
    ) -> Result<TextureData> {
        let sample_generator = sample_gen::to_sample_generator(resource_loader, face_tex)?;

        let (width, height) = sample_generator.source_dimensions();
        if log2_exact(width).is_none() || log2_exact(height).is_none() {
            bail!(
                "Voxel texture resource {:?} has dimensions {}x{}. Expected powers of 2.",
                face_tex,
                width,
                height
            );
        }

        let mip_level_count = log2_exact(width.min(height))
            .expect("expected dimensions to be powers of 2")
            .max(1);

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
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

        log::info!("writing mip levels for {:?}", face_tex);
        for mip_level in 0..mip_level_count {
            let (mip_width, mip_height) = (width >> mip_level, height >> mip_level);

            let copy_view = wgpu::TextureCopyView {
                texture: &texture,
                mip_level,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            };

            let (samples, bytes_per_row) = sample_generator.generate(mip_width, mip_height);

            ctx.queue.write_texture(
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

    fn get_voxel_render_info(
        index_map: &HashMap<voxel::FaceTexture, usize>,
        voxel: &voxel::VoxelInfo,
    ) -> Result<VoxelRenderInfo> {
        let to_index = |face_tex| {
            index_map
                .get(face_tex)
                .ok_or_else(|| anyhow!("missing index for face tex {:?}", face_tex))
                .map(|&ix| ix)
        };

        let face_tex_ids = match &voxel.texture {
            voxel::VoxelTexture::Uniform(face_tex) => {
                let index = to_index(face_tex)? as u32;
                [index; 6]
            }
            voxel::VoxelTexture::Nonuniform { top, bottom, side } => {
                let index_top = to_index(top)? as u32;
                let index_bottom = to_index(bottom)? as u32;
                let index_sides = to_index(side)? as u32;
                [
                    index_top,
                    index_bottom,
                    index_sides,
                    index_sides,
                    index_sides,
                    index_sides,
                ]
            }
        };
        Ok(VoxelRenderInfo {
            face_tex_ids,
            _padding: [0; 2],
            vertex_data: Cube::new(),
        })
    }

    fn get_texture_metadata(face_tex: &voxel::FaceTexture) -> VoxelTextureMetadata {
        match face_tex {
            voxel::FaceTexture::SolidColor { .. } => VoxelTextureMetadata {
                x_periodicity: 1,
                y_periodicity: 1,
            },
            voxel::FaceTexture::Texture {
                x_periodicity,
                y_periodicity,
                ..
            } => VoxelTextureMetadata {
                x_periodicity: *x_periodicity,
                y_periodicity: *y_periodicity,
            },
        }
    }
}

mod sample_gen {
    use super::voxel;
    use super::ResourceLoader;
    use super::Result;

    pub trait SampleGenerator {
        /// Returns the buffer of samples and the number of bytes per row in the result.
        fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32);

        fn source_dimensions(&self) -> (u32, u32);
    }

    pub fn to_sample_generator(
        resource_loader: &ResourceLoader,
        face_tex: &voxel::FaceTexture,
    ) -> Result<Box<dyn SampleGenerator>> {
        match face_tex {
            voxel::FaceTexture::SolidColor { red, green, blue } => {
                log::info!(
                    "Creating solid color texture with R/G/B {}/{}/{}",
                    red,
                    green,
                    blue
                );
                Ok(Box::new(SolidColorSampleGenerator {
                    red: *red,
                    green: *green,
                    blue: *blue,
                }))
            }
            voxel::FaceTexture::Texture { resource, .. } => {
                log::info!("Loading voxel texture {:?}", resource);
                let image = resource_loader.load_image(&resource[..])?.into_rgba8();

                Ok(Box::new(ImageSampleGenerator { image }))
            }
        }
    }

    struct ImageSampleGenerator {
        image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    }

    struct SolidColorSampleGenerator {
        red: u8,
        green: u8,
        blue: u8,
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
