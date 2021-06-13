use std::collections::HashMap;

use anyhow::{anyhow, Result};
use bevy::{render2::render_resource::BufferUsage, wgpu2::WgpuRenderResourceContext};
use zerocopy::{AsBytes, FromBytes};

use simgame_types::{mesh::cube::Cube, VoxelDirectory};
use simgame_voxels::config as voxel;

use crate::{
    assets::TextureAssets,
    buffer_util::{InstancedBuffer, InstancedBufferDesc},
};

/// Manages all GPU buffers that are static for any given set of voxel types.
pub(crate) struct VoxelInfoManager {
    // pub texture_arr_views: Vec<wgpu::TextureView>,
    // pub sampler: wgpu::Sampler,
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

impl VoxelInfoManager {
    pub fn new(
        voxel_directory: &VoxelDirectory,
        texture_assets: &TextureAssets,
        ctx: &WgpuRenderResourceContext,
    ) -> Result<Self> {
        let index_map = HashMap::new();
        // let index_map = voxel_directory
        //     .all_face_textures()
        //     .into_iter()
        //     .map(|face_tex| {
        //         let index = texture_assets
        //             .directory
        //             .texture_key(face_tex.resource.as_str())?
        //             .index;
        //         Ok((face_tex, index))
        //     })
        //     .collect::<Result<HashMap<_, _>>>()?;

        // TODO: texture creation is handled by asset system, need to add our textures to
        // Assets<Texture>

        // ctx.create_texture(descriptor);

        // let texture_arr_views = ctx
        //     .textures
        //     .textures()
        //     .iter()
        //     .map(|data| {
        //         data.texture.create_view(&wgpu::TextureViewDescriptor {
        //             label: Some("voxel textures"),
        //             format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        //             dimension: Some(wgpu::TextureViewDimension::D2),
        //             aspect: wgpu::TextureAspect::All,
        //             base_mip_level: 0,
        //             mip_level_count: Some(
        //                 data.mip_level_count
        //                     .try_into()
        //                     .expect("mip level count cannot be 0"),
        //             ),
        //             base_array_layer: 0,
        //             array_layer_count: Some(1.try_into().expect("nonzero")),
        //         })
        //     })
        //     .collect();

        let voxel_info_buf = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "voxel info",
                instance_len: std::mem::size_of::<VoxelRenderInfo>(),
                n_instances: voxel_directory.voxels().len(),
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            },
        );

        for (index, voxel) in voxel_directory.voxels().iter().enumerate() {
            let render_info = Self::get_voxel_render_info(&index_map, voxel)?;
            voxel_info_buf.write(ctx, index, render_info.as_bytes());
        }

        let texture_metadata_buf = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "texture metadata",
                instance_len: std::mem::size_of::<VoxelTextureMetadata>(),
                // n_instances: ctx.textures.textures().len(),
                n_instances: 1,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            },
        );

        for (face_tex, &index) in index_map.iter() {
            let texture_metadata = Self::get_texture_metadata(face_tex);
            texture_metadata_buf.write(ctx, index as _, texture_metadata.as_bytes())
        }

        Ok(Self {
            // texture_arr_views,
            // sampler,
            voxel_info_buf,
            texture_metadata_buf,
        })
    }

    // pub fn texture_array(&self) -> &[wgpu::TextureView] {
    //     self.texture_arr_views.as_slice()
    // }

    fn get_voxel_render_info(
        index_map: &HashMap<voxel::FaceTexture, u32>,
        voxel: &voxel::VoxelInfo,
    ) -> Result<VoxelRenderInfo> {
        let to_index = |_| -> Result<_> { Ok(0) };
        // let to_index = |face_tex| {
        //     index_map
        //         .get(face_tex)
        //         .ok_or_else(|| anyhow!("missing index for face tex {:?}", face_tex))
        //         .map(|&ix| ix)
        // };

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
        VoxelTextureMetadata {
            x_periodicity: face_tex.x_periodicity.unwrap_or(1),
            y_periodicity: face_tex.y_periodicity.unwrap_or(1),
        }
    }
}
