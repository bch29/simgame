use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use zerocopy::AsBytes;

use simgame_core::block::{self, BlockConfigHelper};

use crate::buffer_util::{InstancedBuffer, InstancedBufferDesc};
use crate::mesh::cube::Cube;
use crate::resource::ResourceLoader;

use super::BlockRenderInfo;

#[allow(unused)]
pub(super) struct BlockInfoHandler {
    pub index_map: HashMap<block::FaceTexture, usize>,
    dimensions: (u32, u32),

    texture_arr: wgpu::Texture,
    pub texture_arr_views: Vec<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,

    pub block_info_buf: InstancedBuffer,
}

impl BlockInfoHandler {
    pub fn new(
        config: &BlockConfigHelper,
        resource_loader: &ResourceLoader,
        ctx: &crate::GraphicsContext,
    ) -> Result<Self> {
        let index_map = config.texture_index_map();
        log::info!("Got {} block textures total", index_map.len());
        let dimensions = Self::find_max_dimensions(&index_map, resource_loader)?;
        log::info!(
            "Block texture dimensions are {}x{}",
            dimensions.0,
            dimensions.1
        );

        let mip_level_count = log2_exact(dimensions.0.min(dimensions.1))
            .expect("expected dimensions to be powers of 2");

        let texture_arr = Self::make_texture(
            dimensions,
            mip_level_count,
            &index_map,
            resource_loader,
            ctx,
        )?;

        let texture_arr_views = (0..index_map.len())
            .map(|index| {
                texture_arr.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("block textures"),
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    dimension: wgpu::TextureViewDimension::D2,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    level_count: mip_level_count,
                    base_array_layer: index as u32,
                    array_layer_count: 1,
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

        let block_info_buf = InstancedBuffer::new(
            &ctx.device,
            InstancedBufferDesc {
                label: "block info",
                instance_len: std::mem::size_of::<BlockRenderInfo>(),
                n_instances: config.blocks().len(),
                usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
            },
        );

        for (index, block) in config.blocks().iter().enumerate() {
            let render_info = BlockRenderInfo {
                face_tex_ids: Self::get_face_textures(&index_map, block)
                    .ok_or_else(|| anyhow!("Unable to get face textures for block {:?}", block))?,
                _padding: [0; 2],
                vertex_data: Cube::new(),
            };

            block_info_buf.write(&ctx.queue, index, render_info.as_bytes());
        }

        Ok(Self {
            index_map,
            dimensions,

            texture_arr_views,
            texture_arr,
            sampler,

            block_info_buf,
        })
    }

    fn make_texture(
        dimensions: (u32, u32),
        mip_level_count: u32,
        index_map: &HashMap<block::FaceTexture, usize>,
        resource_loader: &ResourceLoader,
        ctx: &crate::GraphicsContext,
    ) -> Result<wgpu::Texture> {
        let texture_arr = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block texture array"),
            size: wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth: index_map.len() as u32,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::COPY_DST | wgpu::TextureUsage::SAMPLED,
        });

        let face_texes = {
            let mut face_texes: Vec<_> = index_map
                .iter()
                .map(|(face_tex, &index)| (index, face_tex))
                .collect();
            face_texes.sort_by_key(|(index, _)| *index);
            face_texes
        };

        for (index, face_tex) in face_texes {
            let sample_generator = to_sample_generator(resource_loader, face_tex)?;

            for mip_level in 0..mip_level_count {
                let mip_dimensions = (dimensions.0 >> mip_level, dimensions.1 >> mip_level);

                let copy_view = wgpu::TextureCopyView {
                    texture: &texture_arr,
                    mip_level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: index as u32,
                    },
                };

                let (samples, bytes_per_row) =
                    sample_generator.generate(mip_dimensions.0, mip_dimensions.1);

                ctx.queue.write_texture(
                    copy_view,
                    samples.as_slice(),
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: 0,
                    },
                    wgpu::Extent3d {
                        width: mip_dimensions.0,
                        height: mip_dimensions.1,
                        depth: 1,
                    },
                );
            }
        }

        Ok(texture_arr)
    }

    fn find_max_dimensions(
        index_map: &HashMap<block::FaceTexture, usize>,
        resource_loader: &ResourceLoader,
    ) -> Result<(u32, u32)> {
        // result will be 1x1 if every face texture is a solid color
        let mut res: (u32, u32) = (1, 1);

        for (face_tex, _) in index_map.iter() {
            match face_tex {
                block::FaceTexture::Texture { resource } => {
                    let reader = resource_loader.open_image(&resource[..])?;
                    let (width, height) = reader.into_dimensions()?;
                    if log2_exact(width).is_none() || log2_exact(height).is_none() {
                        bail!("Block texture resource {:?} has dimensions {}x{}. Expected powers of 2.",
                          resource, width, height);
                    }

                    let (old_width, old_height) = &mut res;
                    if width > *old_width {
                        *old_width = width;
                    }
                    if height > *old_height {
                        *old_height = height;
                    }
                }
                _ => {}
            }
        }

        Ok(res)
    }

    fn get_face_textures(
        index_map: &HashMap<block::FaceTexture, usize>,
        block: &block::BlockInfo,
    ) -> Option<[u32; 6]> {
        match &block.texture {
            block::BlockTexture::Uniform(face_tex) => {
                let index = *index_map.get(face_tex)? as u32;
                Some([index; 6])
            }
            block::BlockTexture::Nonuniform { top, bottom, side } => {
                let index_top = *index_map.get(top)? as u32;
                let index_bottom = *index_map.get(bottom)? as u32;
                let index_sides = *index_map.get(side)? as u32;
                Some([
                    index_top,
                    index_bottom,
                    index_sides,
                    index_sides,
                    index_sides,
                    index_sides,
                ])
            }
        }
    }
}

trait SampleGenerator {
    /// Returns the buffer of samples and the number of bytes per row in the result.
    fn generate(&self, width: u32, height: u32) -> (Vec<u8>, u32);
}

fn to_sample_generator(
    resource_loader: &ResourceLoader,
    face_tex: &block::FaceTexture,
) -> Result<Box<dyn SampleGenerator>> {
    match face_tex {
        block::FaceTexture::SolidColor { red, green, blue } => {
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
        block::FaceTexture::Texture { resource } => {
            log::info!("Loading block texture {:?}", resource);
            let image = resource_loader.load_image(&resource[..])?.into_rgba();

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
}

/// If the value is a power of 2, returns its log base 2. Otherwise, returns `None`.
fn log2_exact(value: u32) -> Option<u32> {
    if value == 0 {
        return None;
    }

    let mut log2 = 0;
    let mut shifted = value;
    while shifted > 1 {
        shifted = shifted >> 1;
        log2 += 1;
    }

    if 1 << log2 == value {
        Some(log2)
    } else {
        None
    }
}
