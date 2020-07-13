use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use log::{error, info};
use structopt::StructOpt;

use simgame::files::FileContext;
use simgame_core::block::BlockConfigHelper;
use simgame_core::world::World;
use simgame_render::resource::{self, ResourceLoader};
use simgame_worldgen as worldgen;

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

#[derive(Debug, StructOpt)]
struct ResourceOptions {
    #[structopt(long)]
    force_recompile_all_shaders: bool,

    #[structopt(long)]
    force_recompile_shaders: Vec<String>,
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
        Action::RenderWorld { options } => smol::run(run_render_world(&ctx, &options)),
    }
}

async fn run_render_world(ctx: &FileContext, options: &RenderWorldOpts) -> Result<()> {
    let settings = ctx.load_settings()?;

    let force_recompile_shaders = if options.resource_options.force_recompile_all_shaders {
        resource::ForceRecompileOption::All
    } else if !options.resource_options.force_recompile_shaders.is_empty() {
        resource::ForceRecompileOption::Subset(
            options
                .resource_options
                .force_recompile_shaders
                .iter()
                .cloned()
                .collect(),
        )
    } else {
        resource::ForceRecompileOption::None
    };

    let resource_loader = ResourceLoader::new(
        ctx.data_root.as_path(),
        ctx.core_settings.resources.clone(),
        resource::ResourceOptions {
            force_recompile_shaders,
        },
    )?;

    let blocks = match &options.save_name {
        Some(save_name) => ctx.load_world_blocks(save_name)?,
        None => FileContext::load_debug_world_blocks()?,
    };
    info!("Loaded world: {:?}", blocks.debug_summary());

    let world = World::new(
        blocks,
        BlockConfigHelper::new(&ctx.core_settings.block_config),
    );

    let params = simgame::RenderParams {
        trace_path: options.graphics_trace_path.as_ref().map(|p| p.as_path()),
    };

    simgame::test_render(world, settings.render_test_params, params, resource_loader).await
}

fn run_generate(ctx: &FileContext, options: &GenerateWorldOpts) -> Result<()> {
    let block_helper = BlockConfigHelper::new(&ctx.core_settings.block_config);

    let config_file = std::fs::File::open(&options.config)?;
    let config = serde_yaml::from_reader(config_file)?;
    let blocks = worldgen::generate_world(&config, &block_helper)?;

    info!("Saving");
    ctx.save_world_blocks(options.save_name.as_str(), &blocks)?;

    Ok(())
}

fn main() {
    env_logger::init();

    match run(Opts::from_args()) {
        Ok(()) => (),
        Err(end_error) => {
            for error in end_error.chain() {
                error!("{}", error);
                error!("========");
            }
        }
    }
}
