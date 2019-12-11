use anyhow::Result;
// use cgmath::{Angle, Rad};
use cgmath::{Deg, ElementWise, EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use std::iter;
use zerocopy::AsBytes;

use simgame_core::{
    block::{self, index_utils},
    convert_point, convert_vec,
    util::{Bounds, DivUp},
    world::{UpdatedWorldState, World},
};

use crate::buffer_util::{
    BufferSyncHelper, BufferSyncHelperDesc, BufferSyncedData, IntoBufferSynced,
};
use crate::mesh;
use crate::stable_map::StableMap;

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

    fn update_view_box(&mut self, world: &World) -> Option<Bounds<usize>> {
        let mut center = self.uniforms.camera_pos + 60. * LOOK_AT_DIR;
        center.z = self.z_level as f32 - 0.5;
        let size = Vector3::new(128.0f32, 128., 5.);

        let float_bounds = Bounds::new(center - 0.5 * size, size);
        let world_bounds_limit = world.blocks.bounds().limit();
        let positive_box =
            Bounds::from_limit(Point3::origin(), convert_point!(world_bounds_limit, f32));
        let view_box = float_bounds.intersection(positive_box).map(|bounds| {
            Bounds::new(
                convert_point!(bounds.origin(), usize),
                convert_vec!(bounds.size(), usize),
            )
        });

        if let Some(view_box) = view_box {
            self.uniforms.visible_box_origin = convert_point!(view_box.origin(), f32);
            self.uniforms.visible_box_limit = convert_point!(view_box.limit(), f32);
        }

        view_box
    }

    pub fn init(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: &World,
    ) {
        let view_box = self.update_view_box(world);

        let chunks = view_box.iter().flat_map(|view_box| {
            let bounds = Bounds::new(
                convert_point!(view_box.origin(), usize),
                convert_vec!(view_box.size(), usize),
            );
            world
                .blocks
                .iter_chunks_in_bounds(bounds)
                .map(|(p, chunk)| (convert_point!(p, i32), chunk))
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
    active_chunks: StableMap<Point3<i32>, ()>,
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

    const fn metadata_size() -> usize {
        4 * 12
    }

    fn count_chunks(&self) -> usize {
        self.active_chunks.len()
    }

    fn new(device: &wgpu::Device) -> Self {
        let visible_size: Vector3<i32> = Vector3::new(128, 128, 5);
        let visible_chunk_size = visible_size
            .div_up(&convert_vec!(index_utils::chunk_size(), i32))
            + Vector3::new(1, 1, 1);
        let max_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;

        assert!(max_visible_chunks <= Self::max_batch_chunks() as i32);

        let active_chunks = StableMap::new(max_visible_chunks as usize);

        let block_type_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: Self::max_batch_blocks(),
            max_chunk_len: index_utils::chunk_size_total(),
            gpu_usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ,
        });

        let chunk_metadata_helper = BufferSyncHelper::new(BufferSyncHelperDesc {
            buffer_len: Self::metadata_size() * Self::max_batch_chunks(),
            max_chunk_len: 1024,
            gpu_usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::STORAGE_READ,
        });

        ChunkBatchRenderState {
            active_chunks,
            block_type_buf: block_type_helper.make_buffer(device),
            block_type_helper,
            chunk_metadata_buf: chunk_metadata_helper.make_buffer(device),
            chunk_metadata_helper,
        }
    }

    fn clear_active_chunks(&mut self) {
        self.active_chunks.clear();
    }

    fn delete_chunks<Deleted>(&mut self, deleted: Deleted)
    where
        Deleted: IntoIterator<Item = Point3<i32>>,
    {
        for p in deleted.into_iter() {
            self.active_chunks.remove(&p);
        }
    }

    fn update<'a, Deleted, Chunks>(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        updated_chunks: Chunks,
    ) where
        Chunks: IntoIterator<Item = (Point3<i32>, &'a block::Chunk)>,
    {
        let mut fill_block_types =
            self.block_type_helper
                .begin_fill_buffer(device, &self.block_type_buf, 0);

        let mut fill_chunk_metadatas =
            self.chunk_metadata_helper
                .begin_fill_buffer(device, &self.chunk_metadata_buf, 0);

        for (p, chunk) in updated_chunks.into_iter() {
            let (index, _) = self.active_chunks.update(p, ());

            fill_block_types.seek(encoder, index * index_utils::chunk_size_total());
            fill_chunk_metadatas.seek(encoder, index * Self::metadata_size());

            fill_block_types.advance(encoder, block::blocks_to_u16(&chunk.blocks));

            let meta = Self::make_chunk_meta(&self.active_chunks, p);
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

    fn make_chunk_meta(active_chunks: &StableMap<Point3<i32>, ()>, p: Point3<i32>) -> ChunkMeta {
        fn make_neighbor_indices(
            map: &StableMap<Point3<i32>, ()>,
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
                    map.get(&neighbor_loc).map_or(-1, |(index, _)| index as i32)
                })
                .zip(&mut result)
                .for_each(|(src, dst)| *dst = src);

            result
        }

        let offset = convert_point!(p, f32)
            .mul_element_wise(Point3::origin() + convert_vec!(index_utils::chunk_size(), f32));

        let neighbor_indices = make_neighbor_indices(&active_chunks, p);

        ChunkMeta {
            offset,
            _padding0: [0f32],
            neighbor_indices,
            _padding1: [0, 0],
        }
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
