use std::{
    default::Default,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Vector3, Zero};
use rand::SeedableRng;

use simgame_types::{Directory, ModelRenderData};
use simgame_util::{
    background_object::{BackgroundObject, Cumulative},
    convert_point, convert_vec,
    ray::{Intersection, Ray},
    Bounds,
};
use simgame_voxels::{
    index_utils,
    primitive::{self, Primitive},
    Voxel, VoxelData, VoxelDelta, VoxelRaycastHit, VoxelUpdater,
};

use crate::{behavior, component, tree::TreeSystem};

#[derive(Debug, Clone, Default)]
pub(super) struct WorldResponse {
    pub voxel_delta: VoxelDelta,
    pub models: Vec<ModelRenderData>,
}

#[derive(Debug, Clone)]
pub(super) struct Tick {
    pub elapsed: f64,
}

#[derive(Debug, Clone)]
pub(super) enum WorldUpdateAction {
    SpawnTree { raycast_hit: VoxelRaycastHit },
    ModifyFilledVoxels { delta: i32 },
    ToggleUpdates,
}

/// Handles background updates. While locked on the voxel mutex, the rendering thread will be
/// blocked, so care should be taken not to lock it for long. Other blocking in this object will
/// not block the rendering thread.
pub(super) struct WorldState {
    directory: Arc<Directory>,
    voxels: Arc<Mutex<VoxelData>>,
    entities: hecs::World,

    rng: rand::rngs::StdRng,
    updating: bool,
    filled_voxels: i32,
    response: WorldResponse,

    tree_system: Option<TreeSystem>,
}

impl WorldState {
    pub fn new(
        directory: Arc<Directory>,
        voxels: Arc<Mutex<VoxelData>>,
        entities: hecs::World,
        tree_system: Option<TreeSystem>,
    ) -> Self {
        Self {
            directory,
            voxels,
            response: Default::default(),
            entities,
            rng: SeedableRng::from_entropy(),
            updating: false,
            filled_voxels: (16 * 16 * 4) / 8,
            tree_system,
        }
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
                self.response.voxel_delta.record_chunk_update(chunk_pos);
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
            let mut updater = VoxelUpdater::new(&mut voxels, &mut self.response.voxel_delta);
            primitive.draw(&mut updater, fill_voxel);
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn toggle_updates(&mut self) -> Result<()> {
        self.updating = !self.updating;
        Ok(())
    }

    /// Moves the world forward by one tick.
    #[allow(clippy::unnecessary_wraps)]
    fn tick(&mut self, elapsed: f64) -> Result<()> {
        // update velocity
        self.run_friction(elapsed);
        self.run_fall(elapsed);
        self.run_accelerate(elapsed);
        self.run_impulse(elapsed);

        // update position
        Motion::run(self, elapsed);

        // rendering
        self.run_render_mesh(elapsed);

        if self.updating {
            // let mut updater = VoxelUpdater::new(&mut self.world.voxels, &mut self.voxel_delta.voxels);
            // shape.draw(&mut updater);

            self.updating = false;
        }

        Ok(())
    }

    fn run_fall(&mut self, elapsed: f64) {
        let gravity_vec = Vector3 {
            x: 0.0,
            y: 0.0,
            z: -20.0,
        };

        let entities = self
            .entities
            .query_mut::<(&mut component::Velocity, &behavior::Fall)>();

        for (_entity, (component::Velocity(velocity), behavior::Fall)) in entities {
            *velocity += elapsed * gravity_vec;
        }
    }

    fn run_accelerate(&mut self, elapsed: f64) {
        let entities = self
            .entities
            .query_mut::<(&mut component::Velocity, &behavior::Accelerate)>();

        for (_entity, (component::Velocity(velocity), behavior::Accelerate(accel))) in entities {
            *velocity += elapsed * *accel;
        }
    }

    fn run_impulse(&mut self, _elapsed: f64) {
        let entities = self
            .entities
            .query_mut::<(&mut component::Velocity, &mut behavior::Impulse)>();

        for (_entity, (component::Velocity(velocity), behavior::Impulse(impulse))) in entities {
            *velocity += *impulse;
            *impulse = Zero::zero();
        }
    }

    fn run_friction(&mut self, elapsed: f64) {
        let epsilon = 0.01 * elapsed;

        let entities = self.entities.query_mut::<(
            &mut component::Velocity,
            Option<&component::Ground>,
            &behavior::Friction,
        )>();

        for (_entity, (component::Velocity(velocity), ground_state, friction)) in entities {
            let coeff = match ground_state {
                Some(component::Ground::OnGround) => friction.ground,
                _ => friction.air,
            };
            *velocity += elapsed * coeff * -*velocity;
            if velocity.magnitude() <= epsilon {
                *velocity = Zero::zero();
            }
        }
    }

    fn run_render_mesh(&mut self, _elapsed: f64) {
        let entities = self.entities.query_mut::<(
            &component::Model,
            Option<&component::Position>,
            Option<&component::Orientation>,
        )>();

        for (_entity, (model, position, orientation)) in entities {
            let offset = position
                .map(|position| position.0 - Point3::origin())
                .unwrap_or_else(Vector3::zero);

            let mut transform = model.transform;

            if let Some(component::Orientation(quat)) = orientation {
                let quat = Quaternion {
                    v: convert_vec!(quat.v, f32),
                    s: quat.s as f32,
                };
                let rotation_matrix: Matrix4<f32> = quat.into();

                transform = rotation_matrix * transform;
            }

            transform = Matrix4::from_translation(convert_vec!(offset, f32)) * transform;

            self.response.models.push({
                ModelRenderData {
                    model: model.key,
                    transform,
                }
            })
        }
    }
}

/// Encapsulates motion computation for a single entity.
struct Motion<'a> {
    entity: hecs::Entity,
    position: &'a mut Point3<f64>,
    velocity: &'a mut Vector3<f64>,
    ground: Option<&'a mut component::Ground>,
    bounds: Option<&'a component::Bounds>,
    bounce: Option<&'a behavior::Bounce>,
}

impl<'a> Motion<'a> {
    fn run(world: &mut WorldState, elapsed: f64) {
        let entities = world.entities.query_mut::<(
            &mut component::Position,
            &mut component::Velocity,
            Option<&mut component::Ground>,
            Option<&component::Bounds>,
            Option<&behavior::Bounce>,
        )>();

        let voxels = world.voxels.lock().expect("unable to lock voxel data");

        for (
            entity,
            (component::Position(position), component::Velocity(velocity), ground, bounds, bounce),
        ) in entities
        {
            let mut motion = Motion {
                entity,
                position,
                velocity,
                ground,
                bounds,
                bounce,
            };
            motion.run_entity(&*voxels, &*world.directory, elapsed);
        }
    }

    fn run_entity(&mut self, voxels: &VoxelData, directory: &Directory, elapsed: f64) {
        let vel = *self.velocity;

        let approx_radius = if let Some(component::Bounds(bounds)) = self.bounds {
            (bounds.size() / 2.0).magnitude()
        } else {
            0.0
        };

        // approximate bounds as a sphere for collision
        let collision_offset = approx_radius * vel.normalize();

        // collision with voxels
        let ray = Ray {
            origin: *self.position + collision_offset - 0.01 * vel.normalize(),
            dir: elapsed * vel,
        };

        match voxels.cast_ray(&ray, &directory.voxel) {
            Some(hit) if hit.intersection.t <= 1.0 => {
                self.handle_collision(elapsed, hit.intersection)
            }
            _ => *self.position += elapsed * vel,
        }

        let new_ground = self.handle_rest(elapsed, voxels, directory, approx_radius);

        if let Some(ground) = self.ground.as_deref_mut() {
            if new_ground != *ground {
                log::info!(
                    "Entity {:?} change rest state to {:?}",
                    self.entity,
                    new_ground
                );
            }
            *ground = new_ground;
        }
    }

    fn handle_collision(&mut self, elapsed: f64, hit: Intersection<f64>) {
        let bounce_coeff = self.bounce.copied().map(|b| b.coeff).unwrap_or(0.0);
        let vel = *self.velocity;

        let t = hit.t;
        let impulse_size = -(1.0 + bounce_coeff) * hit.normal.dot(vel);
        let vel_after_collision = vel + impulse_size * hit.normal;

        // interpolate motion based on how far from the surface the entity is
        *self.position += elapsed * (t * vel + (1.0 - t) * vel_after_collision);
        *self.velocity = vel_after_collision;
    }

    fn handle_rest(
        &mut self,
        elapsed: f64,
        voxels: &VoxelData,
        directory: &Directory,
        radius: f64,
    ) -> component::Ground {
        let epsilon = 0.01 * elapsed;

        let up_vec = Vector3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

        let down_vec = -up_vec;
        let top_point = *self.position + radius * up_vec;

        let ray = Ray {
            origin: top_point,
            dir: down_vec,
        };

        let hit = match voxels.cast_ray(&ray, &directory.voxel) {
            Some(hit) if hit.intersection.t <= 2.0 * radius + epsilon => hit.intersection,
            _ => return component::Ground::NotOnGround,
        };

        let ground_pos = ray.origin + hit.t * ray.dir;

        if self.velocity.dot(up_vec) <= epsilon {
            // hit the ground and at rest

            self.velocity.z = 0.0;
            *self.position = ground_pos + radius * up_vec;
            component::Ground::OnGround
        } else {
            // hit the ground but not at rest

            if hit.t <= 2.0 * radius {
                *self.position = ground_pos + radius * up_vec;
            }
            component::Ground::NotOnGround
        }
    }
}

impl BackgroundObject for WorldState {
    type UserAction = WorldUpdateAction;
    type TickAction = Tick;
    type Response = WorldResponse;

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

    fn produce_response(&mut self) -> Result<Option<WorldResponse>> {
        if self.response.is_empty() {
            return Ok(None);
        }
        let mut response = WorldResponse::default();
        std::mem::swap(&mut self.response, &mut response);
        Ok(Some(response))
    }
}

impl Cumulative for WorldResponse {
    fn empty() -> Self {
        Self::default()
    }

    fn append(&mut self, mut other: Self) -> Result<()> {
        // voxel data gets accumulated
        self.voxel_delta.update_from(&mut other.voxel_delta);

        // models to render get overwritten
        if !other.models.is_empty() {
            self.models.clear();
            self.models.extend(other.models.into_iter());
        }
        Ok(())
    }
}

impl WorldResponse {
    fn is_empty(&self) -> bool {
        self.voxel_delta.is_empty() && self.models.is_empty()
    }
}
