pub mod buffer_util;
mod pipelines;
pub mod resource;
pub mod shaders;
mod view;

use std::{collections::HashMap, sync::Arc};

use anyhow::{anyhow, bail, Result};
use cgmath::{SquareMatrix, Vector2, Vector3};
use raw_window_handle::HasRawWindowHandle;

use simgame_types::{Directory, MeshKey, ModelRenderData};
use simgame_util::{convert_vec, DivUp};
use simgame_voxels::{index_utils, VoxelData, VoxelDelta};

pub(crate) use pipelines::{FrameRenderContext, GraphicsContext};
use pipelines::{Pipeline, State as PipelineState};
use resource::{ResourceLoader, TextureLoader};
pub use view::ViewParams;
pub(crate) use view::ViewState;

const COUNT_TIMESTAMP_QUERIES: u32 = 6;

pub struct RendererBuilder {
    pub resource_loader: ResourceLoader,
    pub texture_loader: TextureLoader,
    pub directory: Arc<Directory>,
}

pub struct RenderStateInputs<'a, W> {
    pub window: &'a W,
    pub physical_win_size: Vector2<u32>,
    pub max_visible_chunks: usize,
    pub view_params: ViewParams,
    pub voxels: &'a VoxelData,
}

#[derive(Debug, Clone)]
pub struct RenderParams<'a> {
    pub trace_path: Option<&'a std::path::Path>,
}

/// Holds all state involved in rendering the game.
pub struct RenderState {
    swapchain: wgpu::SwapChain,
    world_voxels: pipelines::voxels::VoxelRenderState,
    meshes: HashMap<MeshKey, pipelines::mesh::MeshRenderState>,
    gui: pipelines::gui::GuiRenderState,
    view: ViewState,
    surface: wgpu::Surface,
    timestamp_query_results_buf: wgpu::Buffer,
}

/// Object responsible for rendering the game.
pub struct Renderer {
    ctx: GraphicsContext,
    pipelines: Pipelines,
    instance: wgpu::Instance,
    directory: Arc<Directory>,
}

pub fn visible_size_to_chunks(visible_size: Vector3<i32>) -> Vector3<i32> {
    visible_size.div_up(convert_vec!(index_utils::chunk_size(), i32)) + Vector3::new(1, 1, 1)
}

struct Pipelines {
    voxel: pipelines::voxels::VoxelRenderPipeline,
    mesh: pipelines::mesh::MeshRenderPipeline,
    gui: pipelines::gui::GuiRenderPipeline,
}

struct DeviceResult {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub multi_draw_enabled: bool,
}

impl RendererBuilder {
    pub async fn build(self, params: RenderParams<'_>) -> Result<Renderer> {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow!("Failed to request wgpu::Adaptor"))?;

        let device_result: DeviceResult = request_device(&params, &adapter).await?;

        let textures = self
            .texture_loader
            .load(&device_result.device, &device_result.queue)?;

        let timestamp_query_set =
            device_result
                .device
                .create_query_set(&wgpu::QuerySetDescriptor {
                    ty: wgpu::QueryType::Timestamp,
                    count: COUNT_TIMESTAMP_QUERIES,
                });

        let ctx = GraphicsContext {
            device: device_result.device,
            queue: device_result.queue,
            textures,
            timestamp_query_set,
            multi_draw_enabled: device_result.multi_draw_enabled,
        };

        let directory = self.directory;
        let resource_loader = self.resource_loader;

        let pipelines = {
            let voxel = Pipeline::create_pipeline(&ctx, &*directory, &resource_loader)?;
            let mesh = Pipeline::create_pipeline(&ctx, &*directory, &resource_loader)?;
            let gui = Pipeline::create_pipeline(&ctx, &*directory, &resource_loader)?;
            Pipelines { voxel, mesh, gui }
        };

        Ok(Renderer {
            ctx,
            pipelines,
            instance,
            directory,
        })
    }
}

impl Renderer {
    pub fn create_state<W: HasRawWindowHandle>(
        &self,
        input: RenderStateInputs<W>,
    ) -> Result<RenderState> {
        let surface = unsafe { self.instance.create_surface(input.window) };

        let swapchain_descriptor = swapchain_descriptor(input.physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&surface, &swapchain_descriptor);

        let view = ViewState::new(input.view_params, input.physical_win_size);
        let depth_texture = make_depth_texture(&self.ctx.device, input.physical_win_size);
        let pipeline_params = pipelines::Params {
            physical_win_size: input.physical_win_size,
            depth_texture: &depth_texture,
        };

        let world_voxels = self.pipelines.voxel.create_state(
            &self.ctx,
            pipeline_params,
            pipelines::voxels::VoxelRenderInput {
                model: SquareMatrix::identity(),
                voxels: &input.voxels,
                max_visible_chunks: input.max_visible_chunks,
                view_state: &view,
            },
        )?;

        let meshes = self
            .directory
            .model
            .meshes()
            .into_iter()
            .map(|(key, mesh)| {
                let state = self.pipelines.mesh.create_state(
                    &self.ctx,
                    pipeline_params,
                    pipelines::mesh::MeshRenderInput {
                        mesh,
                        instances: &[],
                        view_state: &view,
                    },
                )?;
                Ok((key, state))
            })
            .collect::<Result<_>>()?;

        let gui = self
            .pipelines
            .gui
            .create_state(&self.ctx, pipeline_params, ())?;

        let timestamp_query_results_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp query results"),
            size: COUNT_TIMESTAMP_QUERIES as wgpu::BufferAddress * 8,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(RenderState {
            swapchain,
            world_voxels,
            meshes,
            gui,
            view,
            surface,
            timestamp_query_results_buf,
        })
    }

    pub fn update_win_size(
        &self,
        state: &mut RenderState,
        physical_win_size: Vector2<u32>,
    ) -> Result<()> {
        let swapchain_descriptor = swapchain_descriptor(physical_win_size);
        let swapchain = self
            .ctx
            .device
            .create_swap_chain(&state.surface, &swapchain_descriptor);

        state.swapchain = swapchain;

        state.view.update_display_size(physical_win_size);

        let depth_texture = make_depth_texture(&self.ctx.device, physical_win_size);
        let pipeline_params = pipelines::Params {
            physical_win_size,
            depth_texture: &depth_texture,
        };
        state.world_voxels.update_window(&self.ctx, pipeline_params);
        for mesh in state.meshes.values_mut() {
            mesh.update_window(&self.ctx, pipeline_params);
        }
        state.gui.update_window(&self.ctx, pipeline_params);
        Ok(())
    }

    pub fn set_world_view(&self, state: &mut RenderState, params: ViewParams) {
        state.view.params = params;
    }

    pub fn render_frame(&self, state: &mut RenderState) -> Result<()> {
        let frame = state.swapchain.get_current_frame()?;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.write_timestamp(&self.ctx.timestamp_query_set, 0);

        let mut render = FrameRenderContext { frame, encoder };

        self.pipelines.voxel.render_frame(
            &self.ctx,
            pipelines::LoadAction::Clear,
            &mut render,
            &mut state.world_voxels,
        );
        for mesh in state.meshes.values_mut() {
            self.pipelines.mesh.render_frame(
                &self.ctx,
                pipelines::LoadAction::Load,
                &mut render,
                mesh,
            );
        }
        self.pipelines.gui.render_frame(
            &self.ctx,
            pipelines::LoadAction::Load,
            &mut render,
            &mut state.gui,
        );

        render
            .encoder
            .write_timestamp(&self.ctx.timestamp_query_set, 5);
        render.encoder.resolve_query_set(
            &self.ctx.timestamp_query_set,
            0..COUNT_TIMESTAMP_QUERIES,
            &state.timestamp_query_results_buf,
            0,
        );

        self.ctx
            .queue
            .submit(std::iter::once(render.encoder.finish()));

        // Collect device timings
        {
            let timestamp_period = self.ctx.queue.get_timestamp_period() as f64;
            let ts: Vec<i64>;

            {
                // we can ignore the future as we're about to wait for the device
                let _ = state
                    .timestamp_query_results_buf
                    .slice(..)
                    .map_async(wgpu::MapMode::Read);
                self.ctx.device.poll(wgpu::Maintain::Wait);
                // this is guaranteed to be ready
                let view = state
                    .timestamp_query_results_buf
                    .slice(..)
                    .get_mapped_range();
                let view_layout_verified: zerocopy::LayoutVerified<&[u8], [u64]> =
                    zerocopy::LayoutVerified::new_slice(&*view).unwrap();
                let query_results = view_layout_verified.into_slice();
                ts = query_results
                    .iter()
                    .copied()
                    .map(|x| (x as f64 * timestamp_period) as i64)
                    .collect();
            }
            state.timestamp_query_results_buf.unmap();

            let all_begin = ts[0];
            let voxel_begin = ts[1];
            let voxel_buffers = ts[2];
            let voxel_compute = ts[3];
            let voxel_render = ts[4];
            let all_end = ts[5];

            metrics::timing!("render.render", (all_end - all_begin) as u64);
            metrics::timing!(
                "render.pipelines.voxels.update_buffers",
                (voxel_buffers - voxel_begin) as u64
            );
            metrics::timing!(
                "render.pipelines.voxels.compute",
                (voxel_compute - voxel_buffers) as u64
            );
            metrics::timing!(
                "render.pipelines.voxels.render",
                (voxel_render - voxel_compute) as u64
            );
        }

        Ok(())
    }

    pub fn update(
        &self,
        state: &mut RenderState,
        voxels: &VoxelData,
        voxel_diff: &VoxelDelta,
        models: &[ModelRenderData],
    ) -> Result<()> {
        state.world_voxels.update(
            &self.ctx,
            pipelines::voxels::VoxelRenderInputDelta {
                model: SquareMatrix::identity(),
                voxels,
                view_state: &state.view,
                voxel_diff,
            },
        );

        use pipelines::mesh::MeshInstance;

        let mut instances: HashMap<MeshKey, Vec<MeshInstance>> = HashMap::new();
        for model in models {
            let model_data = self.directory.model.model_data(model.model)?;

            let mut face_tex_ids = [0; 16];
            let count_faces = model_data.face_texture_ids.len().min(16);
            face_tex_ids[..count_faces].clone_from_slice(model_data.face_texture_ids.as_slice());

            let instance = MeshInstance {
                face_tex_ids,
                transform: model.transform,
            };

            instances
                .entry(model_data.mesh)
                .or_insert_with(Vec::new)
                .push(instance);
        }

        for (key, mesh) in &mut state.meshes {
            if let Some(instances) = instances.get(key) {
                mesh.update(
                    &self.ctx,
                    pipelines::mesh::MeshRenderInputDelta {
                        instances,
                        view_state: &state.view,
                    },
                );
            }
        }
        state.gui.update(&self.ctx, ());

        Ok(())
    }
}

async fn request_device(
    params: &RenderParams<'_>,
    adapter: &wgpu::Adapter,
) -> Result<DeviceResult> {
    let required_features = wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
        | wgpu::Features::UNSIZED_BINDING_ARRAY
        | wgpu::Features::TIMESTAMP_QUERY;

    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MULTI_DRAW_INDIRECT | required_features,
                limits: wgpu::Limits {
                    max_bind_groups: 6,
                    max_storage_buffers_per_shader_stage: 7,
                    max_storage_textures_per_shader_stage: 6,
                    max_sampled_textures_per_shader_stage: 1024, // TODO: use proper texture arrays then reduce this limit
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024,
                    ..Default::default()
                },
            },
            params.trace_path,
        )
        .await
    {
        Ok(res) => res,
        Err(err) => {
            bail!("Failed to request graphics device: {:?}", err);
        }
    };

    if !device.features().contains(required_features) {
        bail!("Graphics device does not support required features");
    }

    let multi_draw_enabled = device
        .features()
        .contains(wgpu::Features::MULTI_DRAW_INDIRECT);
    if !multi_draw_enabled {
        log::warn!("Graphics device does not support MULTI_DRAW_INDIRECT");
    }

    Ok(DeviceResult {
        device,
        queue,
        multi_draw_enabled,
    })
}

fn make_depth_texture(device: &wgpu::Device, size: Vector2<u32>) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world depth texture"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
    })
}

fn swapchain_descriptor(win_size: Vector2<u32>) -> wgpu::SwapChainDescriptor {
    wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: win_size.x,
        height: win_size.y,
        present_mode: wgpu::PresentMode::Fifo,
    }
}
