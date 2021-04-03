use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, Matrix4, Point3, Quaternion, Vector3, Zero};
use rand::SeedableRng;

use simgame_types::{Directory, ModelRenderData};
use simgame_util::{convert_point, convert_vec, ray::Ray, Bounds};
use simgame_voxels::{
    index_utils,
    primitive::{self, Primitive},
    Voxel, VoxelData, VoxelDelta, VoxelRaycastHit, VoxelUpdater,
};

use crate::{
    background_object::{self, BackgroundObject},
    component,
    tree::{TreeConfig, TreeSystem},
};

pub struct WorldStateBuilder<'a> {
    pub voxels: Arc<Mutex<VoxelData>>,
    pub entities: Arc<Mutex<hecs::World>>,
    pub directory: Arc<Directory>,
    pub tree_config: Option<&'a TreeConfig>,
}

/// Keeps track of world state. Responds immediately to user input and hands off updates to the
/// background state.
pub struct WorldState {
    entities: Arc<Mutex<hecs::World>>,
    connection: Option<background_object::Connection<BackgroundState>>,
}

/// Handles background updates. While locked on the voxel mutex, the rendering thread will be
/// blocked, so care should be taken not to lock it for long. Other blocking in this object will
/// not voxel the rendering thread.
struct BackgroundState {
    voxels: Arc<Mutex<VoxelData>>,
    voxel_delta: VoxelDelta,
    #[allow(unused)]
    entities: Arc<Mutex<hecs::World>>,
    rng: rand::rngs::StdRng,
    updating: bool,
    filled_voxels: i32,

    tree_system: Option<TreeSystem>,
}

#[derive(Debug, Clone)]
struct Tick {
    pub elapsed: f64,
}

#[derive(Debug, Clone)]
enum WorldUpdateAction {
    SpawnTree { raycast_hit: VoxelRaycastHit },
    ModifyFilledVoxels { delta: i32 },
    ToggleUpdates,
}

impl<'a> WorldStateBuilder<'a> {
    pub fn build(self) -> Result<WorldState> {
        let tree_system = match self.tree_config {
            Some(tree_config) => Some(TreeSystem::new(tree_config, &self.directory.voxel)?),
            None => None,
        };

        let world_state = BackgroundState {
            voxels: self.voxels,
            voxel_delta: Default::default(),
            entities: self.entities.clone(),
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_voxels: (16 * 16 * 4) / 8,
            tree_system,
        };

        let connection = background_object::Connection::new(world_state, Default::default())?;

        Ok(WorldState {
            entities: self.entities,
            connection: Some(connection),
        })
    }
}

impl WorldState {
    /// Moves the world forward by one tick.
    pub fn tick(&mut self, elapsed: f64) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("sending tick on a closed connection"))?;

        connection.send_tick(Tick { elapsed })
    }

    pub fn modify_filled_voxels(&mut self, delta: i32) -> Result<()> {
        self.send_action(WorldUpdateAction::ModifyFilledVoxels { delta })
    }

    pub fn on_click(
        &mut self,
        directory: &Directory,
        voxels: &Mutex<VoxelData>,
        camera_pos: Point3<f64>,
        camera_facing: Vector3<f64>,
    ) -> Result<()> {
        let ray = Ray {
            origin: camera_pos,
            dir: camera_facing,
        };

        let raycast_hit = {
            let voxels = voxels.lock().map_err(|_| anyhow!("voxel mutex poisoned"))?;
            match voxels.cast_ray(&ray, &directory.voxel) {
                None => return Ok(()),
                Some(raycast_hit) => raycast_hit,
            }
        };

        self.send_action(WorldUpdateAction::SpawnTree { raycast_hit })
    }

    pub fn toggle_updates(&mut self) -> Result<()> {
        self.send_action(WorldUpdateAction::ToggleUpdates)
    }

    pub fn voxel_delta(&mut self) -> Result<&mut VoxelDelta> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("collecting diffs on a closed connection"))?;
        connection.current_response()
    }

    fn send_action(&mut self, action: WorldUpdateAction) -> Result<()> {
        let connection = self
            .connection
            .as_mut()
            .ok_or_else(|| anyhow!("sending action on a closed connection: {:?}", action))?;
        connection.send_user(action)
    }

    /// Returns entity models within the given bounds, for rendering.
    pub fn model_render_data<'a>(
        &self,
        directory: &'a Directory,
        search_bounds: Option<Bounds<f64>>,
        result: &mut Vec<ModelRenderData<'a>>,
    ) -> Result<()> {
        let mut entities = self.entities.lock().unwrap();
        let entities = entities.query_mut::<(
            &component::Model,
            &component::Bounds,
            Option<&component::Position>,
            Option<&component::Orientation>,
        )>();

        for (_entity, (model, bounds, position, orientation)) in entities {
            let offset = position
                .map(|position| position.point - Point3::origin())
                .unwrap_or_else(Vector3::zero);

            if let Some(search_bounds) = search_bounds {
                if !search_bounds.contains_bounds(bounds.translate(offset)) {
                    continue;
                }
            }

            let mut transform = model.transform;

            if let Some(orientation) = orientation {
                let quat = Quaternion {
                    v: convert_vec!(orientation.quat.v, f32),
                    s: orientation.quat.s as f32,
                };
                let rotation_matrix: Matrix4<f32> = quat.into();

                transform = rotation_matrix * transform;
            }

            transform = Matrix4::from_translation(convert_vec!(offset, f32)) * transform;

            let model_data = directory.model.model_data(model.key)?;

            result.push({
                ModelRenderData {
                    mesh: model_data.mesh,
                    face_tex_ids: model_data.face_texture_ids.as_slice(),
                    transform,
                }
            })
        }

        Ok(())
    }
}

impl BackgroundState {
    fn bounce_system(&self, elapsed: f64, entities: &mut hecs::World) {
        let entities = entities.query_mut::<(&mut component::Bounce, &mut component::Position)>();

        for (_entity, (bounce, position)) in entities {
            bounce.progress += elapsed * 2.0;
            if bounce.progress >= 1.0 {
                bounce.progress -= 2.0;
            }

            position.point += Vector3 {
                x: 0.0,
                y: 0.0,
                z: 10.0 * bounce.progress * elapsed,
            }
        }
    }
}

impl BackgroundState {
    /// Moves the world forward by one tick.
    #[allow(clippy::unnecessary_wraps)]
    fn tick(&mut self, elapsed: f64) -> Result<()> {
        let mut entities = self.entities.lock().unwrap();

        self.bounce_system(elapsed, &mut *entities);

        if !self.updating {
            return Ok(());
        }

        // let mut updater = VoxelUpdater::new(&mut self.world.voxels, &mut self.voxel_delta.voxels);
        // shape.draw(&mut updater);

        self.updating = false;
        Ok(())
    }

    fn modify_filled_voxels(&mut self, delta: i32) -> Result<()> {
        self.filled_voxels += delta * 6;
        if self.filled_voxels < 1 {
            self.filled_voxels = 1
        } else if self.filled_voxels >= index_utils::chunk_size_total() as i32 {
            self.filled_voxels = index_utils::chunk_size_total() as i32
        }

        let bounds: Bounds<i64> = Bounds::new(Point3::new(32, 32, 0), Vector3::new(16, 16, 1024));

        let step = index_utils::chunk_size_total() / self.filled_voxels as i64;
        let mut count_filled = 0;

        {
            let mut voxels = self
                .voxels
                .lock()
                .map_err(|_| anyhow!("voxel mutex poisoned"))?;
            for p in bounds.iter_points() {
                if (p.x + p.y + p.z) % step == 0 {
                    voxels.set_voxel(p, Voxel::from_u16(1));
                    count_filled += 1;
                } else {
                    voxels.set_voxel(p, Voxel::from_u16(0));
                }
                let (chunk_pos, _) = index_utils::to_chunk_pos(p);
                self.voxel_delta.record_chunk_update(chunk_pos);
            }
        }

        log::debug!(
            "Setting number of filled voxels to {}/{}",
            count_filled as f64 * index_utils::chunk_size_total() as f64 / bounds.volume() as f64,
            index_utils::chunk_size_total()
        );
        Ok(())
    }

    fn spawn_tree(&mut self, raycast_hit: VoxelRaycastHit) -> Result<()> {
        log::debug!(
            "Clicked on voxel {:?} at pos {:?} with intersection {:?}",
            raycast_hit.voxel,
            raycast_hit.voxel_pos,
            raycast_hit.intersection
        );

        let mut tree_system = match &self.tree_system {
            None => return Ok(()),
            Some(system) => system.clone(),
        };

        let tree = tree_system.generate(&mut self.rng)?;

        let root_pos =
            convert_point!(raycast_hit.voxel_pos, f64) + raycast_hit.intersection.normal;

        let draw_transform = Matrix4::from_translation(root_pos - Point3::origin());
        self.draw_shape(tree, draw_transform)
    }

    /// Draws a complex shape while trying to avoid voxeling the rendering thread due to world
    /// updates.
    fn draw_shape(&mut self, shape: primitive::Shape, draw_transform: Matrix4<f64>) -> Result<()> {
        for (fill_voxel, primitive) in shape.iter_transformed_primitives(draw_transform) {
            // lock the mutex inside the loop so that the rendering thread isn't voxeled while we
            // update
            let mut voxels = self
                .voxels
                .lock()
                .map_err(|_| anyhow!("voxel mutex poisoned"))?;
            let mut updater = VoxelUpdater::new(&mut voxels, &mut self.voxel_delta);
            primitive.draw(&mut updater, fill_voxel);
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn toggle_updates(&mut self) -> Result<()> {
        self.updating = !self.updating;
        Ok(())
    }
}

impl BackgroundObject for BackgroundState {
    type UserAction = WorldUpdateAction;
    type TickAction = Tick;
    type Response = VoxelDelta;

    fn receive_user(&mut self, action: WorldUpdateAction) -> Result<()> {
        match action {
            WorldUpdateAction::SpawnTree { raycast_hit } => self.spawn_tree(raycast_hit),
            WorldUpdateAction::ToggleUpdates => self.toggle_updates(),
            WorldUpdateAction::ModifyFilledVoxels { delta } => self.modify_filled_voxels(delta),
        }
    }

    fn receive_tick(&mut self, tick: Tick) -> Result<()> {
        let Tick { elapsed } = tick;
        self.tick(elapsed)
    }

    fn produce_response(&mut self) -> Result<Option<VoxelDelta>> {
        if self.voxel_delta.is_empty() {
            return Ok(None);
        }
        let mut delta = VoxelDelta::default();
        std::mem::swap(&mut self.voxel_delta, &mut delta);
        Ok(Some(delta))
    }
}

impl background_object::Cumulative for VoxelDelta {
    fn empty() -> Self {
        VoxelDelta::default()
    }

    fn append(&mut self, mut other: Self) -> Result<()> {
        self.update_from(&mut other);
        Ok(())
    }
}
