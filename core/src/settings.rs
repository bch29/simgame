use crate::block::BlockConfig;
use serde::{Serialize, Deserialize};

/// Settings specific to this particular game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
}

/// Settings specific to the game engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSettings {
    pub game_name: String,
    pub path_name: String,
    pub block_data_file_name: String,
    pub world_meta_file_name: String,
    pub block_config: BlockConfig,
}
