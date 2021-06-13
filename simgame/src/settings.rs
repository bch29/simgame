use cgmath::{Point3, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use simgame_types::{
    config::{AssetConfig, EntityConfig},
    Behavior,
};
use simgame_voxels::VoxelConfig;
use simgame_world::tree::TreeConfig;

/// Settings specific to this particular game.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Settings {
    pub render_test_params: RenderTestParams,
}

/// Settings specific to the game engine.
#[derive(Debug, Serialize, Deserialize)]
pub struct CoreSettings {
    pub game_name: String,
    pub path_name: String,
    pub voxel_data_file_name: String,
    pub world_meta_file_name: String,
    pub voxel_config: VoxelConfig,
    pub entity_config: EntityConfig,
    pub asset_config: AssetConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RenderTestParams {
    pub game_step_millis: u64,
    pub fixed_refresh_rate: Option<u64>,
    pub initial_visible_size: Vector3<i32>,
    pub initial_camera_pos: Point3<f32>,
    pub initial_z_level: i32,
    pub max_visible_chunks: usize,
    pub look_at_dir: Vector3<f32>,
    pub video_settings: VideoSettings,

    pub tree: Option<TreeConfig>,

    pub entity_archetypes: Vec<EntityArchetype>,
    pub entities: Vec<RenderTestEntity>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VideoMode {
    Borderless,
    Fullscreen,
    Windowed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSettings {
    pub win_dimensions: Vector2<f64>,
    pub video_mode: VideoMode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RenderTestEntity {
    pub archetype: String,
    pub location: Point3<f64>,
    pub behaviors: Option<Vec<Box<dyn Behavior>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EntityArchetype {
    pub name: String,
    pub clone_from: Option<String>,
    pub model: Option<String>,
    pub behaviors: Option<Vec<Box<dyn Behavior>>>,
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
            video_settings: VideoSettings {
                win_dimensions: Vector2::new(1920., 1080.),
                video_mode: VideoMode::Windowed,
            },
            tree: None,
            entity_archetypes: Vec::new(),
            entities: Vec::new(),
        }
    }
}
