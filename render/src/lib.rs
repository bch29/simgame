pub mod buffer_util;
mod mesh;
mod pipelines;
pub mod resource;
pub mod shaders;
mod view;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use cgmath::{SquareMatrix, Vector2, Vector3};
use raw_window_handle::HasRawWindowHandle;

use simgame_types::{entity, ActiveEntityModel, Directory, World, WorldDelta};
use simgame_util::{convert_vec, DivUp};
use simgame_voxels::index_utils;

pub(crate) use pipelines::{FrameRenderContext, GraphicsContext};
use pipelines::{Pipeline, State as PipelineState};
use resource::{ResourceLoader, TextureLoader};
pub use view::ViewParams;
pub(crate) use view::ViewState;

pub struct RendererBuilder<'a> {
    pub resource_loader: ResourceLoader,
    pub texture_loader: TextureLoader,
    pub directory: &'a Directory,
}

pub struct RenderStateInputs<'a, W> {
    pub window: &'a W,
    pub physical_win_size: Vector2<u32>,
    pub max_visible_chunks: usize,
    pub directory: &'a Directory,
    pub view_params: ViewParams,
    pub world: &'a World,
}

#[derive(Debug, Clone)]
pub struct RenderParams<'a> {
    pub trace_path: Option<&'a std::path::Path>,
}

/// Holds all state involved in rendering the game.
pub struct RenderState {
    swapchain: wgpu::SwapChain,
    world_voxels: pipelines::voxels::VoxelRenderState,
    meshes: HashMap<entity::config::ModelKind, pipelines::mesh::MeshRenderState>,
    gui: pipelines::gui::GuiRenderState,
    view: ViewState,
    surface: wgpu::Surface,
}

/// Object responsible for rendering the game.
pub struct Renderer {
    ctx: GraphicsContext,
    pipelines: Pipelines,
    instance: wgpu::Instance,
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
impl<'a> RendererBuilder<'a> {
    pub async fn build(self, params: RenderParams<'a>) -> Result<Renderer> {
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

        let ctx = GraphicsContext {
            device: device_result.device,
            queue: device_result.queue,
            multi_draw_enabled: device_result.multi_draw_enabled,
            resource_loader: self.resource_loader,
            textures,
        };

        let pipelines = {
            let voxel = {
                let voxel_info_manager =
                    pipelines::voxels::VoxelInfoManager::new(&self.directory, &ctx)?;
                pipelines::voxels::VoxelRenderPipeline::new(&ctx, voxel_info_manager)?
            };
            let mesh = pipelines::mesh::MeshRenderPipeline::new(&ctx)?;
            let gui = pipelines::gui::GuiRenderPipeline::new(&self.directory, &ctx)?;
            Pipelines { voxel, mesh, gui }
        };

        Ok(Renderer {
            ctx,
            pipelines,
            instance,
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
                voxels: &input.world.voxels,
                max_visible_chunks: input.max_visible_chunks,
                view_state: &view,
            },
        )?;

        let meshes = {
            let square = self.pipelines.mesh.create_state(
                &self.ctx,
                pipeline_params,
                pipelines::mesh::MeshRenderInput {
                    mesh: &crate::mesh::cube::Cube::new().mesh(),
                    instances: &[],
                    view_state: &view,
                },
            )?;

            let sphere = self.pipelines.mesh.create_state(
                &self.ctx,
                pipeline_params,
                pipelines::mesh::MeshRenderInput {
                    mesh: &crate::mesh::sphere::UnitSphere { detail: 6 }.mesh(),
                    instances: &[],
                    view_state: &view,
                },
            )?;

            vec![
                (entity::config::ModelKind::Cube, square),
                (entity::config::ModelKind::Sphere, sphere),
            ]
            .into_iter()
            .collect()
        };

        let gui = self
            .pipelines
            .gui
            .create_state(&self.ctx, pipeline_params, ())?;
        Ok(RenderState {
            swapchain,
            world_voxels,
            meshes,
            gui,
            view,
            surface,
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
        let encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render = FrameRenderContext { encoder, frame };

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

        let ts_begin = Instant::now();
        self.ctx
            .queue
            .submit(std::iter::once(render.encoder.finish()));
        let ts_submit = Instant::now();

        metrics::timing!("render.submit", ts_submit.duration_since(ts_begin));

        Ok(())
    }

    pub fn update(
        &self,
        state: &mut RenderState,
        world: &World,
        diff: &WorldDelta,
        entities: &[ActiveEntityModel],
    ) {
        state
            .world_voxels
            .update(pipelines::voxels::VoxelRenderInputDelta {
                model: SquareMatrix::identity(),
                voxels: &world.voxels,
                view_state: &state.view,
                voxel_diff: &diff.voxels,
            });

        use pipelines::mesh::MeshInstance;

        let mut instances: HashMap<entity::config::ModelKind, Vec<MeshInstance>> = HashMap::new();
        for entity in entities {
            let mut face_tex_ids = [0; 16];
            let count_faces = entity.face_tex_ids.len().min(16);
            face_tex_ids[..count_faces].clone_from_slice(entity.face_tex_ids);

            let instance = MeshInstance {
                face_tex_ids,
                transform: entity.transform,
            };

            instances
                .entry(entity.model_kind.clone())
                .or_insert_with(Vec::new)
                .push(instance);
        }

        for (kind, mesh) in &mut state.meshes {
            if let Some(instances) = instances.get(kind) {
                mesh.update(pipelines::mesh::MeshRenderInputDelta {
                    instances,
                    view_state: &state.view,
                });
            }
        }
        state.gui.update(());
    }
}

async fn request_device(
    params: &RenderParams<'_>,
    adapter: &wgpu::Adapter,
) -> Result<DeviceResult> {
    let required_features = wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
        | wgpu::Features::UNSIZED_BINDING_ARRAY;

    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::MULTI_DRAW_INDIRECT | required_features,
                limits: wgpu::Limits {
                    max_bind_groups: 6,
                    max_storage_buffers_per_shader_stage: 6,
                    max_storage_textures_per_shader_stage: 6,
                    max_sampled_textures_per_shader_stage: 1024, // TODO: use proper texture arrays then reduce this limit
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
            depth: 1,
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
