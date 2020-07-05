use anyhow::{anyhow, bail, Context, Result};
use directories::UserDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::block::WorldBlockData;
use crate::settings::{CoreSettings, Settings};
use crate::util::Bounds;

const CORE_SETTINGS_FILE_NAME: &str = "core_config.yaml";
const USER_SETTINGS_FILE_NAME: &str = "settings.yaml";
const SAVE_DIR_NAME: &str = "saves";
const SETTINGS_DIR_NAME: &str = "settings";

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldMeta {
    world_bounds: Bounds<usize>,
}

impl FileContext {
    pub fn new(core_settings: CoreSettings, data_root: PathBuf, user_data_root: PathBuf) -> Self {
        let mut save_root = user_data_root.clone();
        save_root.push(SAVE_DIR_NAME);

        let mut settings_root = user_data_root.clone();
        settings_root.push(SETTINGS_DIR_NAME);

        FileContext {
            core_settings,
            data_root,
            save_root,
            settings_root,
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

    pub fn ensure_directories(&self) -> Result<()> {
        std::fs::create_dir_all(&self.save_root)?;
        std::fs::create_dir_all(&self.settings_root)?;
        if !self.data_root.is_dir() {
            bail!(
                "Could not find data root at {}",
                self.data_root.to_string_lossy()
            );
        }

        Ok(())
    }

    pub fn load_settings(&self) -> Result<Settings> {
        let mut fp = self.settings_root.clone();
        fp.push(USER_SETTINGS_FILE_NAME);
        let file = std::fs::File::open(&fp).context("Opening settings file")?;
        Ok(serde_yaml::from_reader(file).context("Parsing settings file")?)
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
                let mut path = save_dir.clone();
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
            let mut path = save_dir.clone();
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
