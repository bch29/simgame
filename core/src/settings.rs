use crate::block::BlockConfig;
use serde::{Serialize, Deserialize};
use cgmath::{Vector3, Point3};

/// Settings specific to this particular game.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    pub render_test_params: RenderTestParams
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderTestParams {
    pub game_step_millis: u64,
    pub fixed_refresh_rate: Option<u64>,
    pub initial_visible_size: Vector3<i32>,
    pub initial_camera_pos: Point3<f32>,
    pub initial_z_level: i32,
    pub max_visible_chunks: usize,
    pub look_at_dir: Vector3<f32>,
}

impl Default for RenderTestParams {
    fn default() -> Self {
        Self {
            game_step_millis: 10,
            fixed_refresh_rate: None,
            initial_visible_size: Vector3::new(128, 128, 32),
            initial_camera_pos: Point3::new(-20.0, -20.0, 20.0),
            initial_z_level: 20,
            max_visible_chunks: 1024 * 16,
            look_at_dir: Vector3::new(1., 1., -6.),
        }
    }
}
