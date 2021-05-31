use std::{collections::HashSet, env, path::PathBuf};

use anyhow::{anyhow, Result};
use structopt::StructOpt;

use simgame::files::FileContext;
use simgame_render::resource::{self, ResourceLoader};
use simgame_types::{Directory, ModelDirectory, VoxelDirectory};
use simgame_world::worldgen;

#[derive(Debug, StructOpt)]
struct Opts {
    #[structopt(short, long)]
    data_root: Option<PathBuf>,
    #[structopt(subcommand)]
    action: Action,
}

#[derive(Debug, StructOpt)]
struct RenderWorldOpts {
    #[structopt(short, long)]
    save_name: Option<String>,

    #[structopt(short = "t", long)]
    graphics_trace_path: Option<PathBuf>,

    #[structopt(flatten)]
    resource_options: ResourceOptions,
}

#[derive(Debug, Clone, StructOpt)]
struct ResourceOptions {
    #[structopt(long)]
    recompile_shaders: Vec<String>,
}

#[derive(Debug, StructOpt)]
enum Action {
    GenerateWorld {
        #[structopt(flatten)]
        options: GenerateWorldOpts,
    },
    RenderWorld {
        #[structopt(flatten)]
        options: RenderWorldOpts,
    },
}

#[derive(Debug, StructOpt)]
struct GenerateWorldOpts {
    #[structopt(short, long)]
    save_name: String,
    #[structopt(short, long)]
    config: PathBuf,
}

fn run(opt: Opts) -> Result<()> {
    let data_root: PathBuf = match opt.data_root {
        None => {
            let exe_fp = env::current_exe()?;
            let mut res: PathBuf = exe_fp
                .parent()
                .ok_or_else(|| anyhow!("Expected absolute directory for current exe"))?
                .into();
            res.push("data");
            res
        }
        Some(data_root) => data_root,
    };

    let ctx = FileContext::load(data_root)?;
    ctx.ensure_directories()?;

    match &opt.action {
        Action::GenerateWorld { options } => run_generate(&ctx, options),
        Action::RenderWorld { options } => run_render_world(&ctx, &options),
    }
}

fn run_render_world(ctx: &FileContext, options: &RenderWorldOpts) -> Result<()> {
    let settings = ctx.load_settings()?;

    let resource_options = {
        let recompile_shaders: HashSet<_> = options
            .resource_options
            .recompile_shaders
            .iter()
            .cloned()
            .collect();

        let recompile_option = if recompile_shaders.is_empty() {
            resource::RecompileOption::None
        } else if recompile_shaders.contains("all") {
            resource::RecompileOption::All
        } else {
            resource::RecompileOption::Subset(recompile_shaders)
        };
        resource::ResourceOptions { recompile_option }
    };

    let resource_loader = ResourceLoader::new(
        ctx.data_root.as_path(),
        ctx.core_settings.resources.clone(),
        resource_options,
    )?;

    let voxels = match &options.save_name {
        Some(save_name) => ctx.load_world_voxels(save_name)?,
        None => FileContext::load_debug_world_voxels()?,
    };
    log::info!("Loaded world: {:?}", voxels.debug_summary());

    let render_params = simgame::RenderParams {
        trace_path: options.graphics_trace_path.as_deref(),
    };

    let texture_loader = resource_loader.texture_loader()?;

    let directory = {
        let voxel = VoxelDirectory::new(&ctx.core_settings.voxel_config)?;
        let texture = texture_loader.directory();
        let model = ModelDirectory::new(&ctx.core_settings.entity_config.models, &texture)?;
        Directory {
            voxel,
            texture,
            model,
        }
    };

    let entities = hecs::World::new();

    simgame::run_game(simgame::GameArgs {
        voxels,
        test_params: settings.render_test_params,
        entities,
        render_params,
        directory,
        resource_loader,
        texture_loader,
        metrics_controller: ctx.metrics_controller.clone(),
    })
}

fn run_generate(ctx: &FileContext, options: &GenerateWorldOpts) -> Result<()> {
    let mut metrics_exporter = metrics_runtime::exporters::LogExporter::new(
        ctx.metrics_controller.clone(),
        metrics_runtime::observers::YamlBuilder::new(),
        log::Level::Info,
        std::time::Duration::from_secs(10),
    );

    let voxel_directory = VoxelDirectory::new(&ctx.core_settings.voxel_config)?;

    let config_file = std::fs::File::open(&options.config)?;
    let config = serde_yaml::from_reader(config_file)?;
    let voxels = worldgen::generate_world(&config, &voxel_directory)?;

    log::info!("Saving");
    ctx.save_world_voxels(options.save_name.as_str(), &voxels)?;

    metrics_exporter.turn();

    Ok(())
}

fn main() {
    env_logger::init();

    match run(Opts::from_args()) {
        Ok(()) => (),
        Err(end_error) => {
            for error in end_error.chain() {
                log::error!("{}", error);
                log::error!("========");
            }
        }
    }
}
