use anyhow::{anyhow, Result};
use std::env;
use std::path::PathBuf;
use structopt::StructOpt;

use simgame_core::files::FileContext;
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
        config: GenerateWorldConfig,
    },
    LoadWorld {
        #[structopt(short, long)]
        save_name: String,
    },
    TestRender
}

use generate_world::{run_generate, GenerateWorldConfig};

fn run_load_world(ctx: &FileContext, save_name: &str) -> Result<()> {
    let blocks = ctx.load_world_blocks(save_name)?;
    println!("Loaded world: {:?}", blocks.debug_summary());

    Ok(())
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
        Action::GenerateWorld { config } => run_generate(&ctx, config),
        Action::LoadWorld { save_name } => run_load_world(&ctx, save_name),
        Action::TestRender => simgame_render::test_render()
    }
}

fn main() {
    env_logger::init();

    match run(Opts::from_args()) {
        Ok(()) => (),
        Err(end_error) => {
            println!("Error:");
            for error in end_error.chain() {
                println!("{}", error);
                println!("========");
            }
        }
    }
}

mod generate_world {
    use anyhow::{bail, Result};
    use cgmath::Vector3;
    use structopt::StructOpt;
    use rand::Rng;

    use simgame_core::block::{Block, WorldBlocks, CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z};
    use simgame_core::files::FileContext;

    #[derive(Debug, StructOpt)]
    pub struct GenerateWorldConfig {
        #[structopt(short, long)]
        save_name: String,
        #[structopt(short = "x", long)]
        size_x: usize,
        #[structopt(short = "y", long)]
        size_y: usize,
        #[structopt(short = "z", long, default_value = "128")]
        size_z: usize,
    }

    pub fn run_generate(ctx: &FileContext, config: &GenerateWorldConfig) -> Result<()> {
        if config.size_x % CHUNK_SIZE_X != 0 {
            bail!("size_x must be a multiple of {}", CHUNK_SIZE_X);
        }

        if config.size_y % CHUNK_SIZE_Y != 0 {
            bail!("size_y must be a multiple of {}", CHUNK_SIZE_Y);
        }

        if config.size_z % CHUNK_SIZE_Z != 0 {
            bail!("size_z must be a multiple of {}", CHUNK_SIZE_Z);
        }

        let count_chunks = Vector3 {
            x: config.size_x / CHUNK_SIZE_X,
            y: config.size_y / CHUNK_SIZE_Y,
            z: config.size_z / CHUNK_SIZE_Z,
        };

        println!("Creating empty world");
        let mut blocks = WorldBlocks::empty(count_chunks);

        let world_size = Vector3 {
            x: blocks.size().x as f64,
            y: blocks.size().y as f64,
            z: blocks.size().z as f64,
        };

        println!("Generating terrain");
        let base_z = world_size.z / 2.0;
        let terrain_height = world_size.z / 8.0;

        let mut blocks_done = 0;
        let total_blocks = blocks.size().x * blocks.size().y * blocks.size().z;

        let progress_count = 10;
        let mut next_progress_step = 1;

        let mut rng = rand::thread_rng();
        for (loc, block) in blocks.iter_blocks_with_loc_mut() {
            // Normalized point with coords in range (0, 1)
            let p = Vector3 {
                x: loc.x as f64 / world_size.x,
                y: loc.y as f64 / world_size.y,
                z: loc.z as f64 / world_size.z,
            };

            // Empty above curve, filled below curve
            use std::f64::consts::PI;
            // val in range [0, 1]
            let val = 0.5 * (1.0 + (p.x * PI * 2.0).sin() * (p.y * PI * 4.0).cos());

            // this is the height of the terrain at current x, y coordinate
            let height_here = base_z + val * terrain_height;

            // fill with air if current z is above height_here, else rock
            if p.z * world_size.z > height_here {
                *block = Block::air();
            } else {
                let block_val = rng.gen::<u16>() % 7;
                *block = Block::from_u16(block_val);
            }

            blocks_done += 1;
            if blocks_done == (next_progress_step * total_blocks) / progress_count {
                println!("{}%", next_progress_step * 10);
                next_progress_step += 1;
            }
        }

        println!("Generated world: {:?}", blocks.debug_summary());

        println!("Saving");
        ctx.save_world_blocks(config.save_name.as_str(), &blocks)?;

        Ok(())
    }
}
