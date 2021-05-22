use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use directories::UserDirs;
use serde::{Deserialize, Serialize};

use simgame_util::Bounds;
use simgame_voxels::VoxelData;

use crate::settings::{CoreSettings, Settings};

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

    pub metrics_controller: metrics_runtime::Controller,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldMeta {
    world_bounds: Bounds<i64>,
}

impl FileContext {
    pub fn new(
        core_settings: CoreSettings,
        data_root: PathBuf,
        user_data_root: PathBuf,
    ) -> Result<Self> {
        let mut save_root = user_data_root.clone();
        save_root.push(SAVE_DIR_NAME);

        let mut settings_root = user_data_root;
        settings_root.push(SETTINGS_DIR_NAME);

        let metrics_receiver = metrics_runtime::Receiver::builder().build()?;
        let metrics_controller = metrics_receiver.controller();
        metrics_receiver.install();

        Ok(FileContext {
            core_settings,
            save_root,
            settings_root,
            data_root,

            metrics_controller,
        })
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

        FileContext::new(core_settings, data_root, user_data_root)
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

    pub fn save_world_voxels(&self, save_name: &str, voxels: &VoxelData) -> Result<()> {
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
                world_bounds: voxels.bounds(),
            };
            let file = std::fs::File::create(&file_path)
                .context("Creating world meta file for saved game")?;
            serde_yaml::to_writer(file, &meta)
                .context("Writing world meta file for saved game")?;
        }

        {
            let file_path = {
                let mut path = save_dir;
                path.push(&self.core_settings.voxel_data_file_name);
                path
            };
            let file = std::fs::File::create(&file_path)
                .context("Creating world voxel data file for saved game")?;
            let mut encoder = lz4::EncoderBuilder::new()
                .build(file)
                .context("Initializing encoder for voxel data file")?;
            voxels
                .serialize_voxels(&mut encoder)
                .context("Writing out world voxel data for saved game")?;
            let (_file, res) = encoder.finish();
            res?;
        }

        Ok(())
    }

    pub fn load_debug_world_voxels() -> Result<VoxelData> {
        let meta_bytes = include_bytes!("data/debug_saves/small/world_meta.yaml");
        let voxel_data_bytes = include_bytes!("data/debug_saves/small/world_voxel_data.dat");

        let meta = Self::load_world_meta(&meta_bytes[..])?;
        let result = Self::load_world_voxels_data(&meta, &voxel_data_bytes[..])?;
        Ok(result)
    }

    pub fn load_world_voxels(&self, save_name: &str) -> Result<VoxelData> {
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
                "Voxel meta file for saved game {} does not exist: expected {} to be a file",
                save_name,
                meta_path.to_string_lossy()
            ));
        }

        let data_path = {
            let mut path = save_dir;
            path.push(&self.core_settings.voxel_data_file_name);
            path
        };

        if !data_path.is_file() {
            return Err(anyhow!(
                "Voxel data file for saved game {} does not exist: expected {} to be a file",
                save_name,
                data_path.to_string_lossy()
            ));
        }

        log::info!("loading world metadata from {:?}", meta_path);
        let meta_file = std::fs::File::open(meta_path.as_path())
            .context("Opening world meta file for saved game")?;
        let meta = Self::load_world_meta(meta_file)?;
        log::info!("loading world voxels from {:?}", data_path);
        let data_file = std::fs::File::open(data_path.as_path())
            .context("Opening voxel data file for saved game")?;
        let result = Self::load_world_voxels_data(&meta, data_file)?;
        Ok(result)
    }

    pub fn load_world_meta<R: std::io::Read>(file: R) -> Result<WorldMeta> {
        serde_yaml::from_reader(file).context("Parsing world meta file for saved game")
    }

    pub fn load_world_voxels_data<R>(meta: &WorldMeta, file: R) -> Result<VoxelData>
    where
        R: std::io::Read,
    {
        let mut decompressed =
            lz4::Decoder::new(file).context("Initializing decoder for voxel data file")?;

        let mut world_voxels = VoxelData::empty(meta.world_bounds);
        world_voxels
            .deserialize_voxels(&mut decompressed)
            .context("Deserializing voxel data")?;
        Ok(world_voxels)
    }
}
