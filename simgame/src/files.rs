use anyhow::{anyhow, bail, Context, Result};
use byteorder::{ReadBytesExt, WriteBytesExt};
use directories::UserDirs;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};

use simgame_core::block::WorldBlockData;
use simgame_core::util::Bounds;

use crate::settings::{CoreSettings, Settings};

const CORE_SETTINGS_FILE_NAME: &str = "core_config.yaml";
const USER_SETTINGS_FILE_NAME: &str = "settings.yaml";
const SAVE_DIR_NAME: &str = "saves";
const SETTINGS_DIR_NAME: &str = "settings";
const SHADER_SPIRV_DIR_NAME: &str = "shader_spirv";
const SHADER_SRC_DIR_NAME: &str = "shader_src";

pub struct FileContext {
    /// Game engine settings.
    pub core_settings: CoreSettings,
    /// Root directory for saved games.
    pub save_root: PathBuf,
    /// Root directory for user settings.
    pub settings_root: PathBuf,
    /// Root directory for game data files, which are not meant to change except in
    /// development or modding.
    pub data_root: PathBuf,
    /// Root directory for compiled shaders.
    pub shader_spirv_root: PathBuf,
    /// Root directory for shader source code.
    pub shader_src_root: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldMeta {
    world_bounds: Bounds<i64>,
}

#[derive(Debug)]
pub enum ShaderLoadAction<F> {
    /// Load cached shaders. Return failure if there is no cached shader.
    CachedOnly,
    /// Try to load cached shaders. If there is no cached shader, compile from source and write to
    /// cache.
    CachedOrCompile(F),
    /// Ignore any cached shaders. Compile from source and write to cache.
    CompileOnly(F),
}

impl FileContext {
    pub fn new(core_settings: CoreSettings, data_root: PathBuf, user_data_root: PathBuf) -> Self {
        let mut save_root = user_data_root.clone();
        save_root.push(SAVE_DIR_NAME);

        let mut settings_root = user_data_root;
        settings_root.push(SETTINGS_DIR_NAME);

        let mut shader_spirv_root = data_root.clone();
        shader_spirv_root.push(SHADER_SPIRV_DIR_NAME);

        let mut shader_src_root = data_root.clone();
        shader_src_root.push(SHADER_SRC_DIR_NAME);

        FileContext {
            core_settings,
            data_root,
            save_root,
            settings_root,
            shader_spirv_root,
            shader_src_root,
        }
    }

    pub fn load(data_root: PathBuf) -> Result<Self> {
        log::info!("working directory is {:?}", std::env::current_dir()?);

        let core_settings: CoreSettings = {
            let mut core_settings_path = data_root.clone();
            core_settings_path.push(CORE_SETTINGS_FILE_NAME);
            log::info!("loading core settings from {:?}", core_settings_path);

            let file = std::fs::File::open(&core_settings_path).context(format!(
                "Opening core settings file {}",
                core_settings_path.to_string_lossy()
            ))?;
            serde_yaml::from_reader(file)?
        };

        let user_dirs =
            UserDirs::new().ok_or_else(|| anyhow!("Could not find user directories."))?;
        let docs_dir = user_dirs
            .document_dir()
            .ok_or_else(|| anyhow!("Could not find document directory."))?;

        let mut user_data_root = PathBuf::new();
        user_data_root.push(docs_dir);
        user_data_root.push(&core_settings.path_name);

        Ok(FileContext::new(core_settings, data_root, user_data_root))
    }

    pub fn load_shader<T, F>(
        &self,
        name: &'static str,
        shader_type: T,
        load_action: &mut ShaderLoadAction<F>,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(&Path, T) -> Result<Vec<u32>>,
    {
        let src_name = format!("{}.glsl", name);
        let spirv_name = format!("{}.glsl.spirv", name);

        let mut src_path = PathBuf::new();
        src_path.push(&self.shader_src_root);
        src_path.push(src_name);

        let mut spirv_path = PathBuf::new();
        spirv_path.push(&self.shader_spirv_root);
        spirv_path.push(spirv_name);

        let load_spirv = || -> Result<Vec<u32>> {
            let mut file = std::fs::File::open(&spirv_path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;

            let mut words = vec![0; bytes.len() / 4];
            bytes
                .as_slice()
                .read_u32_into::<byteorder::LittleEndian>(&mut words)?;
            log::info!("Loaded precompiled shader: {:?}", spirv_path);
            Ok(words)
        };

        let save_spirv = |words| -> Result<()> {
            log::info!("Saving compiled shader: {:?}", spirv_path);
            let mut file = std::fs::File::create(&spirv_path)?;
            for &word in words {
                file.write_u32::<byteorder::LittleEndian>(word)?;
            }
            Ok(())
        };

        match load_action {
            ShaderLoadAction::CachedOnly => load_spirv(),
            ShaderLoadAction::CompileOnly(compile) => {
                log::info!("Compiling shader from source: {:?}", src_path);
                let result = compile(src_path.as_path(), shader_type)?;
                save_spirv(&result[..])?;
                Ok(result)
            }
            ShaderLoadAction::CachedOrCompile(compile) => match load_spirv() {
                Ok(result) => Ok(result),
                Err(_) => {
                    log::info!("Compiling shader from source: {:?}", src_path);
                    let result = compile(src_path.as_path(), shader_type)?;
                    save_spirv(&result[..])?;
                    Ok(result)
                }
            },
        }
    }

    pub fn ensure_directories(&self) -> Result<()> {
        std::fs::create_dir_all(&self.save_root)?;
        std::fs::create_dir_all(&self.settings_root)?;
        if !self.data_root.is_dir() {
            bail!(
                "Could not find data root at {}",
                self.data_root.to_string_lossy()
            );
        }

        std::fs::create_dir_all(&self.shader_spirv_root)?;

        Ok(())
    }

    pub fn load_settings(&self) -> Result<Settings> {
        let mut fp = self.settings_root.clone();
        fp.push(USER_SETTINGS_FILE_NAME);

        if fp.is_file() {
            log::info!("Loading settings from {:?}", fp);
            let file = std::fs::File::open(&fp).context("Opening settings file")?;
            Ok(serde_yaml::from_reader(file).context("Parsing settings file")?)
        } else {
            log::info!("Creating default settings file {:?}", fp);
            let settings = Settings::default();
            let file = std::fs::File::create(&fp).context("Creating settings file")?;
            serde_yaml::to_writer(file, &settings).context("Serializing settings")?;
            Ok(settings)
        }
    }

    fn get_save_dir(&self, save_name: &str) -> PathBuf {
        let mut res = self.save_root.clone();
        res.push(save_name);
        res
    }

    pub fn save_world_blocks(&self, save_name: &str, blocks: &WorldBlockData) -> Result<()> {
        let save_dir = self.get_save_dir(save_name);
        if !save_dir.is_dir() {
            println!("Creating save directory at {}", save_dir.to_string_lossy());
            std::fs::create_dir_all(&save_dir)?;
        } else {
            println!(
                "Overwriting existing save at {}",
                save_dir.to_string_lossy()
            );
        }

        {
            let file_path = {
                let mut path = save_dir.clone();
                path.push(&self.core_settings.world_meta_file_name);
                path
            };
            let meta = WorldMeta {
                world_bounds: blocks.bounds(),
            };
            let file = std::fs::File::create(&file_path)
                .context("Creating world meta file for saved game")?;
            serde_yaml::to_writer(file, &meta)
                .context("Writing world meta file for saved game")?;
        }

        {
            let file_path = {
                let mut path = save_dir;
                path.push(&self.core_settings.block_data_file_name);
                path
            };
            let file = std::fs::File::create(&file_path)
                .context("Creating world block data file for saved game")?;
            let mut encoder = lz4::EncoderBuilder::new()
                .build(file)
                .context("Initializing encoder for block data file")?;
            blocks
                .serialize_blocks(&mut encoder)
                .context("Writing out world block data for saved game")?;
            let (_file, res) = encoder.finish();
            res?;
        }

        Ok(())
    }

    pub fn load_debug_world_blocks() -> Result<WorldBlockData> {
        let meta_bytes = include_bytes!("data/debug_saves/small/world_meta.yaml");
        let block_data_bytes = include_bytes!("data/debug_saves/small/world_block_data.dat");

        let meta = Self::load_world_meta(&meta_bytes[..])?;
        let result = Self::load_world_blocks_data(&meta, &block_data_bytes[..])?;
        Ok(result)
    }

    pub fn load_world_blocks(&self, save_name: &str) -> Result<WorldBlockData> {
        let save_dir = self.get_save_dir(save_name);
        if !save_dir.is_dir() {
            return Err(anyhow!(
                "Saved game {} does not exist: expected {} to be a directory",
                save_name,
                save_dir.to_string_lossy()
            ));
        }

        let meta_path = {
            let mut path = save_dir.clone();
            path.push(&self.core_settings.world_meta_file_name);
            path
        };

        if !meta_path.is_file() {
            return Err(anyhow!(
                "Block meta file for saved game {} does not exist: expected {} to be a file",
                save_name,
                meta_path.to_string_lossy()
            ));
        }

        let data_path = {
            let mut path = save_dir;
            path.push(&self.core_settings.block_data_file_name);
            path
        };

        if !data_path.is_file() {
            return Err(anyhow!(
                "Block data file for saved game {} does not exist: expected {} to be a file",
                save_name,
                data_path.to_string_lossy()
            ));
        }

        log::info!("loading world metadata from {:?}", meta_path);
        let meta_file = std::fs::File::open(meta_path.as_path())
            .context("Opening world meta file for saved game")?;
        let meta = Self::load_world_meta(meta_file)?;
        log::info!("loading world blocks from {:?}", data_path);
        let data_file = std::fs::File::open(data_path.as_path())
            .context("Opening block data file for saved game")?;
        let result = Self::load_world_blocks_data(&meta, data_file)?;
        Ok(result)
    }

    pub fn load_world_meta<R: std::io::Read>(file: R) -> Result<WorldMeta> {
        Ok(serde_yaml::from_reader(file).context("Parsing world meta file for saved game")?)
    }

    pub fn load_world_blocks_data<R>(meta: &WorldMeta, file: R) -> Result<WorldBlockData>
    where
        R: std::io::Read,
    {
        let mut decompressed =
            lz4::Decoder::new(file).context("Initializing decoder for block data file")?;

        let mut world_blocks = WorldBlockData::empty(meta.world_bounds);
        world_blocks
            .deserialize_blocks(&mut decompressed)
            .context("Deserializing block data")?;
        Ok(world_blocks)
    }
}
