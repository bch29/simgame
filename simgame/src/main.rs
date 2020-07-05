use anyhow::{anyhow, Result};
use log::{error, info};
use std::env;
use std::path::PathBuf;
use structopt::StructOpt;

use simgame_core::block::index_utils;
use simgame_core::files::FileContext;
use simgame_core::world::World;
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

        #[structopt(short = "t", long)]
        graphics_trace_path: Option<PathBuf>,
    },
    DebugRender,
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

    let ctx = || -> Result<FileContext> {
        let ctx = FileContext::load(data_root)?;
        ctx.ensure_directories()?;
        Ok(ctx)
    };

    match &opt.action {
        Action::GenerateWorld { options } => run_generate(&ctx()?, options),
        Action::LoadWorld {
            save_name,
            graphics_trace_path,
        } => smol::run(run_load_world(
            &ctx()?,
            save_name,
            graphics_trace_path.as_ref().map(|p| p.as_path()),
        )),
        Action::DebugRender => {
            let blocks = FileContext::load_debug_world_blocks()?;
            info!("Loaded debug world: {:?}", blocks.debug_summary());
            smol::run(run_world(World::from_blocks(blocks), None))
        }
    }
}

async fn run_load_world(
    ctx: &FileContext,
    save_name: &str,
    graphics_trace_path: Option<&std::path::Path>,
) -> Result<()> {
    let blocks = ctx.load_world_blocks(save_name)?;
    info!("Loaded world: {:?}", blocks.debug_summary());

    let world = World::from_blocks(blocks);

    run_world(world, graphics_trace_path).await
}

async fn run_world(world: World, graphics_trace_path: Option<&std::path::Path>) -> Result<()> {
    let mut shader_compiler = simgame_shaders::Compiler::new(simgame_shaders::CompileParams {
        chunk_size: index_utils::chunk_size().into(),
    })?;

    let vert_shader = shader_compiler.compile_vert("shaders/vert_partial.glsl".as_ref())?;
    let frag_shader = shader_compiler.compile_frag("shaders/frag.glsl".as_ref())?;
    let comp_shader =
        shader_compiler.compile_compute("shaders/comp_block_vertices.glsl".as_ref())?;

    let shaders = simgame_render::WorldShaders {
        vert: vert_shader.as_ref(),
        frag: frag_shader.as_ref(),
        comp: comp_shader.as_ref(),
    };

    let params = simgame_render::RenderParams {
        trace_path: graphics_trace_path,
    };

    simgame_render::test::test_render(world, params, shaders).await
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
