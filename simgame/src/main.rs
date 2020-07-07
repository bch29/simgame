use anyhow::{anyhow, Result};
use log::{error, info};
use std::env;
use std::path::{PathBuf, Path};
use structopt::StructOpt;

use simgame_core::files::FileContext;
use simgame_core::world::World;
use simgame_core::worldgen;

#[derive(Debug, StructOpt)]
struct Opts {
    #[structopt(short, long)]
    data_root: Option<PathBuf>,
    #[structopt(subcommand)]
    action: Action,
}

#[cfg(feature = "shader-compiler")]
#[derive(Debug, StructOpt)]
struct ShaderOpts {
    #[structopt(long)]
    force_shader_compile: bool,
}

#[cfg(not(feature = "shader-compiler"))]
#[derive(Debug, StructOpt)]
struct ShaderOpts {}

#[derive(Debug, StructOpt)]
struct RenderWorldOpts {
    #[structopt(short, long)]
    save_name: Option<String>,

    #[structopt(short = "t", long)]
    graphics_trace_path: Option<PathBuf>,

    #[structopt(flatten)]
    shader_opts: ShaderOpts,
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
    #[cfg(feature = "shader-compiler")]
    CompileShaders,
}

#[derive(Debug, StructOpt)]
struct GenerateWorldOpts {
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
        Action::RenderWorld { options } => smol::run(run_render_world(&ctx, &options)),
        #[cfg(feature = "shader-compiler")]
        Action::CompileShaders => run_compile_shaders(&ctx),
    }
}

async fn run_render_world(ctx: &FileContext, options: &RenderWorldOpts) -> Result<()> {
    let shaders = load_shaders(&ctx, &options.shader_opts)?;

    let settings = ctx.load_settings()?;

    let blocks = match &options.save_name {
        Some(save_name) => ctx.load_world_blocks(save_name)?,
        None => FileContext::load_debug_world_blocks()?,
    };
    info!("Loaded world: {:?}", blocks.debug_summary());

    let world = World::from_blocks(blocks);

    let ref_shaders: simgame_render::WorldShaders<&[u32]> = shaders.map(|x| &x[..]);

    let params = simgame_render::RenderParams {
        trace_path: options.graphics_trace_path.as_ref().map(|p| p.as_path()),
    };

    simgame_render::test::test_render(world, settings.render_test_params, params, ref_shaders)
        .await
}

fn run_generate(ctx: &FileContext, options: &GenerateWorldOpts) -> Result<()> {
    let config = worldgen::GenerateWorldConfig {
        size: cgmath::Vector3::new(options.size_x, options.size_y, options.size_z),
    };

    let blocks = worldgen::generate_world(&config)?;

    info!("Saving");
    ctx.save_world_blocks(options.save_name.as_str(), &blocks)?;

    Ok(())
}

#[cfg(feature = "shader-compiler")]
fn run_compile_shaders(ctx: &FileContext) -> Result<()> {
    load_shaders(
        &ctx,
        &ShaderOpts {
            force_shader_compile: true,
        },
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum LoadShaderAction {
    CompileOnly,
    CachedOrCompile,
    CachedOnly,
}

impl ShaderOpts {
    #[cfg(feature = "shader-compiler")]
    fn load_action(&self) -> LoadShaderAction {
        if self.force_shader_compile {
            LoadShaderAction::CompileOnly
        } else {
            LoadShaderAction::CachedOrCompile
        }
    }

    #[cfg(not(feature = "shader-compiler"))]
    fn load_action(&self) -> LoadShaderAction {
        LoadShaderAction::CachedOnly
    }
}

#[allow(unused_variables)]
#[allow(dead_code)]
fn load_shaders(
    ctx: &FileContext,
    shader_opts: &ShaderOpts,
) -> Result<simgame_render::WorldShaderData> {
    use simgame_shaders::{CompileParams, Compiler, ShaderKind};

    let mut shader_compiler = Compiler::new(CompileParams {
        chunk_size: simgame_core::block::index_utils::chunk_size().into(),
    })?;

    let mut compile: Box<dyn for<'p> FnMut(&'p Path, ShaderKind) -> Result<Vec<u32>>> =
        Box::new(|p, t| shader_compiler.compile(p, t));

    let mut action = match shader_opts.load_action() {
        LoadShaderAction::CompileOnly => {
            simgame_core::files::ShaderLoadAction::CompileOnly(&mut compile)
        }
        LoadShaderAction::CachedOrCompile => {
            simgame_core::files::ShaderLoadAction::CachedOrCompile(&mut compile)
        }
        LoadShaderAction::CachedOnly => simgame_core::files::ShaderLoadAction::CachedOnly,
    };

    Ok(simgame_render::WorldShaderData {
        vert: ctx.load_shader("vert_partial", ShaderKind::Vertex, &mut action)?,
        frag: ctx.load_shader("frag", ShaderKind::Fragment, &mut action)?,
        comp: ctx.load_shader("comp_block_vertices", ShaderKind::Compute, &mut action)?,
    })
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
