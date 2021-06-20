use anyhow::Result;
use bevy::{
    app::App,
    asset::{AddAsset, AssetLoader, AssetServer, BoxedFuture, LoadContext, LoadedAsset},
    ecs::system::{Commands, IntoSystem, Res},
    math::Vec3,
    render2::{
        camera::PerspectiveCameraBundle,
        shader::{Shader, ShaderStage},
    },
    transform::components::Transform,
    wgpu2::{WgpuFeature, WgpuFeatures, WgpuLimits, WgpuOptions},
};
use structopt::StructOpt;

use crate::{controls, files::FileContext, settings::Settings};
use simgame_types::VoxelDirectory;

#[derive(Debug, Clone, StructOpt)]
pub struct BevyOpts {
    #[structopt(short, long)]
    save_name: Option<String>,
}

#[derive(Default)]
struct ShaderLoader;

impl AssetLoader for ShaderLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), anyhow::Error>> {
        Box::pin(async move {
            let ext = load_context.path().extension().unwrap().to_str().unwrap();

            log::info!("Loading shader");

            let shader = match ext {
                "vert_glsl" => Shader::from_glsl(ShaderStage::Vertex, std::str::from_utf8(bytes)?),
                "frag_glsl" => {
                    Shader::from_glsl(ShaderStage::Fragment, std::str::from_utf8(bytes)?)
                }
                "compute_glsl" => {
                    Shader::from_glsl(ShaderStage::Compute, std::str::from_utf8(bytes)?)
                }
                #[cfg(not(target_arch = "wasm32"))]
                "spv" => Shader::from_spirv(bytes)?,
                #[cfg(target_arch = "wasm32")]
                "spv" => panic!("cannot load .spv file on wasm"),
                _ => panic!("unhandled extension: {}", ext),
            };

            load_context.set_default_asset(LoadedAsset::new(shader));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["vert_glsl", "frag_glsl", "compute_glsl", "spv"]
    }
}

pub fn run_bevy(ctx: FileContext, options: BevyOpts) -> Result<()> {
    let settings = ctx.load_settings()?;

    let visible_size = controls::restrict_visible_size(
        settings.render_test_params.max_visible_chunks,
        settings.render_test_params.initial_visible_size,
    );

    let voxel_directory = VoxelDirectory::new(&ctx.core_settings.voxel_config)?;

    let mut app = App::new();

    app.insert_resource(WgpuOptions {
        features: WgpuFeatures {
            features: vec![
                WgpuFeature::SampledTextureBindingArray,
                WgpuFeature::SampledTextureArrayDynamicIndexing,
                WgpuFeature::UnsizedBindingArray,
                WgpuFeature::TimestampQuery,
                WgpuFeature::MultiDrawIndirect,
            ],
        },
        limits: WgpuLimits {
            max_bind_groups: 6,
            max_storage_buffers_per_shader_stage: 7,
            max_storage_textures_per_shader_stage: 6,
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_sampled_textures_per_shader_stage: 1024,
            ..Default::default()
        },
        ..Default::default()
    });

    app.add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::asset::AssetPlugin::default())
        .add_plugin(bevy::scene::ScenePlugin::default())
        .add_plugin(bevy::render2::RenderPlugin::default())
        .add_plugin(bevy::wgpu2::WgpuPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default());
    // .add_plugin(bevy::wgpu2::diagnostic::WgpuResourceDiagnosticsPlugin::default());

    app.insert_resource(simgame_render::assets::SimgameAssetsParams {
        config: ctx.core_settings.asset_config.clone(),
    })
    .add_plugin(simgame_render::assets::SimgameAssetsPlugin);

    app.insert_resource(ctx)
        .insert_resource(options)
        .insert_resource(voxel_directory)
        .insert_resource(simgame_render::ViewParams {
            camera_pos: settings.render_test_params.initial_camera_pos,
            z_level: settings.render_test_params.initial_z_level,
            visible_size,
            look_at_dir: settings.render_test_params.look_at_dir,
        })
        .add_asset_loader(ShaderLoader)
        .add_plugin(load_world::LoadWorldPlugin)
        .add_plugin(controls::ControlsPlugin);

    app.insert_resource(simgame_render::voxels::Params {
        max_visible_chunks: settings.render_test_params.max_visible_chunks,
    })
    .add_plugin(simgame_render::voxels::VoxelRenderPlugin);

    app.insert_resource(settings)
        .add_startup_system(startup_system.system());

    app.run();

    Ok(())
}

fn startup_system(
    mut commands: Commands,
    settings: Res<Settings>,
    asset_server: Res<AssetServer>,
) {
    asset_server.watch_for_changes().unwrap();

    // camera
    commands
        .spawn()
        .insert_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(-50.0, -100.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(crate::controls::PlayerCamera(controls::ControlState::new(
            &settings.render_test_params,
        )));
}

mod load_world {
    use std::sync::Arc;

    use anyhow::Result;
    use bevy::{
        app::{App, Plugin},
        ecs::system::{Commands, In, IntoChainSystem, IntoSystem, Res},
    };
    use parking_lot::Mutex;

    use super::BevyOpts;
    use crate::files::FileContext;
    use simgame_voxels::{SharedVoxelData, VoxelDelta};

    pub struct LoadWorldPlugin;

    impl Plugin for LoadWorldPlugin {
        fn build(&self, app: &mut App) {
            app.add_startup_system(load_world.system().chain(handler_system.system()));
        }
    }

    fn load_world(
        ctx: Res<FileContext>,
        options: Res<BevyOpts>,
        mut commands: Commands,
    ) -> Result<()> {
        let voxels = match &options.save_name {
            Some(save_name) => ctx.load_world_voxels(save_name)?,
            None => FileContext::load_debug_world_voxels()?,
        };
        log::info!("Loaded world: {:?}", voxels.debug_summary());

        commands.insert_resource(SharedVoxelData {
            data: Arc::new(Mutex::new(voxels)),
        });

        commands.insert_resource(VoxelDelta::default());

        Ok(())
    }

    fn handler_system(In(result): In<Result<()>>) {
        match result {
            Ok(_) => {}
            Err(err) => {
                for error in err.chain() {
                    log::error!("{}", error);
                    log::error!("========");
                }
                panic!("Exiting due to error");
            }
        }
    }
}
