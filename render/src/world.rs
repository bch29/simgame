use std::collections::HashMap;

use anyhow::Result;
// use cgmath::{Angle, Rad};
use cgmath::{Deg, ElementWise, EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use log::debug;
use std::iter;
use zerocopy::AsBytes;

use simgame_core::block;
use simgame_core::block::index_utils;
use simgame_core::util::Bounds;
use simgame_core::world::{UpdatedWorldState, World};

use crate::buffer_util::{
    BufferSyncHelper, BufferSyncHelperDesc, BufferSyncedData, IntoBufferSynced,
};
use crate::mesh;

const LOOK_AT_DIR: Vector3<f32> = Vector3::new(1., 1., -2.);

pub struct ViewParams {
    pub camera_pos: Point3<f32>,
    pub z_level: i32,
}

pub struct WorldRenderInit<RV, RF> {
    pub vert_shader_spirv_bytes: RV,
    pub frag_shader_spirv_bytes: RF,
    pub aspect_ratio: f32,
    pub width: u32,
    pub height: u32,
}

pub struct WorldRenderState {
    render_pipeline: wgpu::RenderPipeline,
    cube_vertex_buf: wgpu::Buffer,
    cube_index_buf: wgpu::Buffer,
    cube_index_count: usize,

    uniforms: BufferSyncedData<Uniforms, f32>,

    bind_group_layout: wgpu::BindGroupLayout,
    rotation: Matrix4<f32>,
    depth_texture: wgpu::TextureView,

    chunk_batch: ChunkBatchRenderState,

    z_level: i32,
    // /// Contains textures for each block type.
    // /// Dimensions are 16x16xN, where N is number of block types.
    // block_master_texture: wgpu::TextureView,
}

#[repr(C)]
struct Uniforms {
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    model: Matrix4<f32>,
    camera_pos: Point3<f32>,
    visible_box_origin: Point3<f32>,
    visible_box_limit: Point3<f32>,
}

impl WorldRenderState {
    pub fn set_view(&mut self, params: &ViewParams) {
        self.uniforms.camera_pos = params.camera_pos;
        self.z_level = params.z_level;
    }

    pub fn new<RV, RF>(init: WorldRenderInit<RV, RF>, device: &wgpu::Device) -> Result<Self>
    where
        RV: std::io::Seek + std::io::Read,
        RF: std::io::Seek + std::io::Read,
    {
        let vs_module =
            device.create_shader_module(&wgpu::read_spirv(init.vert_shader_spirv_bytes)?);
        let fs_module =
            device.create_shader_module(&wgpu::read_spirv(init.frag_shader_spirv_bytes)?);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    // Uniforms
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                    },
                    // Block type buffer
                    wgpu::BindGroupLayoutBinding {
                        binding: 1,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        },
                    },
                    // Chunk offset buffer
                    wgpu::BindGroupLayoutBinding {
                        binding: 2,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: true,
                        },
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let cube_mesh = mesh::cube::create();
        let cube_vertex_buf = cube_mesh.vertex_buffer(device);
        let cube_index_buf = cube_mesh.index_buffer(device);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: init.width,
                height: init.height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });

        let uniforms = Uniforms {
            proj: OPENGL_TO_WGPU_MATRIX
                * cgmath::perspective(Deg(70f32), init.aspect_ratio, 1.0, 1000.0),
            view: Matrix4::look_at_dir(Point3::new(0., 0., 0.), LOOK_AT_DIR, Vector3::unit_z()),
            model: Matrix4::<f32>::identity(),
            camera_pos: Point3::origin(),
            visible_box_origin: Point3::origin(),
            visible_box_limit: Point3::origin(),
        }
        .buffer_synced(device);

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0u32,
                stencil_write_mask: 0u32,
            }),
            index_format: cube_mesh.index_format(),
            vertex_buffers: &[cube_mesh.vertex_buffer_descriptor()],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Ok(WorldRenderState {
            render_pipeline,
            cube_vertex_buf,
            cube_index_buf,
            cube_index_count: cube_mesh.indices.len(),
            uniforms,
            bind_group_layout,
            rotation: Matrix4::identity(),
            depth_texture: depth_texture.create_default_view(),
            chunk_batch: ChunkBatchRenderState::new(device),
            z_level: 0,
        })
    }

    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        frame: &wgpu::SwapChainOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let background_color = wgpu::Color::BLACK;

        self.uniforms.model = self.rotation;
        self.uniforms
            .sync_with(device, encoder, |data| data.as_slices());

        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                bindings: &[
                    self.uniforms.as_binding(0),
                    self.chunk_batch.block_type_binding(1),
                    self.chunk_batch.chunk_metadata_binding(2),
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: background_color,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.depth_texture,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_index_buffer(&self.cube_index_buf, 0);
            rpass.set_vertex_buffers(0, &[(&self.cube_vertex_buf, 0)]);
            rpass.draw_indexed(
                0..self.cube_index_count as u32,
                0,
                0..(index_utils::chunk_size_total() * self.chunk_batch.count_chunks()) as u32,
            );
        }
    }

    pub fn init(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
    ) {
        let view_box = {
            let mut center = self.uniforms.camera_pos + 60. * LOOK_AT_DIR;
            center.z = self.z_level as f32 - 0.5;
            let size = Vector3::new(128.0f32, 128., 5.);

            let float_bounds = Bounds::new(center - 0.5 * size, size);
            let positive_box =
                Bounds::from_limit(Point3::origin(), p_to_f32(world.blocks.bounds().limit()));
            float_bounds
                .intersection(positive_box)
                .map(|bounds| Bounds::new(p_from_f32(bounds.origin()), v_from_f32(bounds.size())))
        };

        if let Some(view_box) = view_box {
            self.uniforms.visible_box_origin = p_to_f32(view_box.origin());
            self.uniforms.visible_box_limit = p_to_f32(view_box.limit());
        }

        fn make_neighbor_indices(
            by_chunk_loc: &HashMap<Point3<i32>, i32>,
            chunk_loc: Point3<i32>,
        ) -> [i32; 6] {
            let directions = [
                Vector3::new(-1, 0, 0),
                Vector3::new(1, 0, 0),
                Vector3::new(0, -1, 0),
                Vector3::new(0, 1, 0),
                Vector3::new(0, 0, -1),
                Vector3::new(0, 0, 1),
            ];

            let mut result = [0i32; 6];
            directions
                .iter()
                .map(|dir| {
                    let neighbor_loc = chunk_loc + dir;
                    by_chunk_loc.get(&neighbor_loc).unwrap_or(&-1)
                })
                .zip(&mut result)
                .for_each(|(src, dst)| *dst = *src);

            result
        }

        let iter_chunks = || {
            view_box
                .iter()
                .flat_map(|&view_box| world.blocks.iter_chunks_in_bounds(view_box))
        };

        let indices_by_chunk_loc: HashMap<Point3<i32>, i32> = iter_chunks()
            .enumerate()
            .map(|(index, (chunk_loc, _))| (p_to_i32(chunk_loc), index as i32))
            .collect();

        let chunks = iter_chunks().map(|(chunk_loc, chunk)| {
            let offset =
                p_to_f32(chunk_loc.mul_element_wise(Point3::origin() + index_utils::chunk_size()));

            let neighbor_indices =
                make_neighbor_indices(&indices_by_chunk_loc, p_to_i32(chunk_loc));

            let meta = ChunkMeta {
                offset,
                _padding0: [0f32],
                neighbor_indices,
                _padding1: [0, 0],
            };

            (meta, chunk)
        });

        self.chunk_batch.update(device, encoder, chunks);
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
        _diff: &UpdatedWorldState,
    ) {
        // TODO: make use of diff instead of doing the whole init again
        self.init(device, encoder, world);
        // self.rotation = self.rotation * Matrix4::<f32>::from_angle_z(Rad::full_turn() / 1000.);
    }
}

struct ChunkBatchRenderState {
    count_chunks: usize,
    chunk_metadata_helper: BufferSyncHelper<u8>,
    chunk_metadata_buf: wgpu::Buffer,
    block_type_helper: BufferSyncHelper<u16>,
    block_type_buf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Debug, Clone)]
struct ChunkMeta {
    offset: Point3<f32>,
    // padding to align offset and neighbor indices to 16 bytes
    _padding0: [f32; 1],
    neighbor_indices: [i32; 6],
    // padding to align struct to 16 bytes
    _padding1: [i32; 2],
}

impl ChunkBatchRenderState {
    const fn max_batch_chunks() -> usize {
        // 16 MB of video memory holds a batch
        (1024 * 1024 * 8) / index_utils::chunk_size_total()
    }

    const fn max_batch_blocks() -> usize {
        Self::max_batch_chunks() * index_utils::chunk_size_total()
    }

    fn count_chunks(&self) -> usize {
        self.count_chunks
    }

    fn new(device: &wgpu::Device) -> Self {
        let block_type_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: Self::max_batch_blocks(),
            max_chunk_len: index_utils::chunk_size_total(),
            gpu_usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ,
        });

        let chunk_metadata_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: 4 * 12 * Self::max_batch_chunks(),
            max_chunk_len: 1024,
            gpu_usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ,
        });

        ChunkBatchRenderState {
            count_chunks: 0,
            block_type_buf: block_type_helper.make_buffer(device),
            block_type_helper,
            chunk_metadata_buf: chunk_metadata_helper.make_buffer(device),
            chunk_metadata_helper,
        }
    }

    #[allow(clippy::cast_lossless)]
    fn update<'a, Chunks>(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        chunks: Chunks,
    ) where
        Chunks: IntoIterator<Item = (ChunkMeta, &'a block::Chunk)>,
    {
        let mut fill_block_types =
            self.block_type_helper
                .begin_fill_buffer(device, &self.block_type_buf, 0);

        let mut fill_chunk_metadatas =
            self.chunk_metadata_helper
                .begin_fill_buffer(device, &self.chunk_metadata_buf, 0);

        self.count_chunks = 0;
        for (meta, chunk) in chunks {
            debug!("Filling chunk idx={} meta={:?}", self.count_chunks, meta);
            self.count_chunks += 1;
            assert!(self.count_chunks <= Self::max_batch_chunks());
            fill_block_types.advance(encoder, block::blocks_to_u16(&chunk.blocks));

            // Offset vector
            let offset_vec = meta.offset.as_ref() as &[f32; 3];
            fill_chunk_metadatas.advance(encoder, offset_vec.as_bytes());
            fill_chunk_metadatas.advance(encoder, meta._padding0.as_bytes());
            fill_chunk_metadatas.advance(encoder, meta.neighbor_indices.as_bytes());
            fill_chunk_metadatas.advance(encoder, meta._padding1.as_bytes());
        }

        fill_block_types.finish(encoder);
        fill_chunk_metadatas.finish(encoder);
    }

    fn block_type_binding(&self, index: u32) -> wgpu::Binding {
        self.block_type_helper
            .as_binding(index, &self.block_type_buf, 0)
    }

    fn chunk_metadata_binding(&self, index: u32) -> wgpu::Binding {
        self.chunk_metadata_helper
            .as_binding(index, &self.chunk_metadata_buf, 0)
    }
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl Uniforms {
    #[inline]
    fn as_slices(&self) -> impl Iterator<Item = &[f32]> {
        let proj: &[f32; 16] = self.proj.as_ref();
        let view: &[f32; 16] = self.view.as_ref();
        let model: &[f32; 16] = self.model.as_ref();
        let camera_pos: &[f32; 3] = self.camera_pos.as_ref();
        let visible_box_origin: &[f32; 3] = self.visible_box_origin.as_ref();
        let visible_box_limit: &[f32; 3] = self.visible_box_limit.as_ref();

        iter::once(proj as &[f32])
            .chain(iter::once(view as &[f32]))
            .chain(iter::once(model as &[f32]))
            .chain(iter::once(camera_pos as &[f32]))
            .chain(iter::once(&[0f32] as &[f32]))
            .chain(iter::once(visible_box_origin as &[f32]))
            .chain(iter::once(&[0f32] as &[f32]))
            .chain(iter::once(visible_box_limit as &[f32]))
    }
}

impl IntoBufferSynced for Uniforms {
    type Item = f32;

    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc {
        BufferSyncHelperDesc {
            buffer_len: 16 + 16 + 16 + 4 + 4 + 3,
            max_chunk_len: 128,
            gpu_usage: wgpu::BufferUsage::UNIFORM,
        }
    }
}

fn p_to_i32(p: Point3<usize>) -> Point3<i32> {
    Point3 {
        x: p.x as i32,
        y: p.y as i32,
        z: p.z as i32,
    }
}

fn p_to_f32(p: Point3<usize>) -> Point3<f32> {
    Point3 {
        x: p.x as f32,
        y: p.y as f32,
        z: p.z as f32,
    }
}

fn p_from_f32(p: Point3<f32>) -> Point3<usize> {
    Point3 {
        x: f32::max(0.0, p.x) as usize,
        y: f32::max(0.0, p.y) as usize,
        z: f32::max(0.0, p.z) as usize,
    }
}

fn v_from_f32(v: Vector3<f32>) -> Vector3<usize> {
    Vector3 {
        x: f32::max(0.0, v.x) as usize,
        y: f32::max(0.0, v.y) as usize,
        z: f32::max(0.0, v.z) as usize,
    }
}
