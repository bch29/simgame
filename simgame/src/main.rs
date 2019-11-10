use anyhow::{anyhow, Result};
use log::{error, info};
use std::env;
use std::path::PathBuf;
use structopt::StructOpt;

use simgame_core::files::FileContext;
use simgame_core::worldgen;
// use simgame::settings::{CoreSettings, Settings};

#[derive(Debug, StructOpt)]
struct Opts {
    #[structopt(short, long)]
    data_root: Option<PathBuf>,
    #[structopt(subcommand)]
    action: Action,
}

#[derive(Debug, StructOpt)]
enum Action {
    GenerateWorld {
        #[structopt(flatten)]
        options: GenerateWorldOptions,
    },
    LoadWorld {
        #[structopt(short, long)]
        save_name: String,
    },
    TestRender,
}

#[derive(Debug, StructOpt)]
struct GenerateWorldOptions {
    #[structopt(short, long)]
    save_name: String,
    #[structopt(short = "x", long)]
    size_x: usize,
    #[structopt(short = "y", long)]
    size_y: usize,
    #[structopt(short = "z", long, default_value = "128")]
    size_z: usize,
}

fn run(opt: Opts) -> Result<()> {
    let data_root: PathBuf = match opt.data_root {
        None => {
            let exe_fp = env::current_exe()?;
            exe_fp
                .parent()
                .ok_or_else(|| anyhow!("Expected absolute directory for current exe"))?
                .into()
        }
        Some(data_root) => data_root,
    };

    let ctx = FileContext::load(data_root)?;
    ctx.ensure_directories()?;

    match &opt.action {
        Action::GenerateWorld { options } => run_generate(&ctx, options),
        Action::LoadWorld { save_name } => run_load_world(&ctx, save_name),
        Action::TestRender => run_render(),
    }
}

fn run_render() -> Result<()> {
    let vert_shader: &[u8] = simgame_shaders::VERT_SHADER;
    let frag_shader: &[u8] = simgame_shaders::FRAG_SHADER;

    simgame_render::test::test_render(vert_shader, frag_shader)
}

fn run_load_world(ctx: &FileContext, save_name: &str) -> Result<()> {
    let blocks = ctx.load_world_blocks(save_name)?;
    info!("Loaded world: {:?}", blocks.debug_summary());

    Ok(())
}

fn run_generate(ctx: &FileContext, options: &GenerateWorldOptions) -> Result<()> {
    let config = worldgen::GenerateWorldConfig {
        size: cgmath::Vector3::new(options.size_x, options.size_y, options.size_z),
    };

    let blocks = worldgen::generate_world(&config)?;

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
