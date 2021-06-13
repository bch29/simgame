use std::collections::{HashMap, HashSet};

use bevy::{
    render2::render_resource::{BufferUsage, RenderResourceBinding},
    wgpu2::WgpuRenderResourceContext,
};
use cgmath::{ElementWise, EuclideanSpace, Point3, Vector3};
use zerocopy::{AsBytes, FromBytes};

use simgame_util::{convert_point, convert_vec, stable_map::StableMap, Bounds};
use simgame_voxels::{index_utils, Chunk, VoxelData, VoxelDelta};

use crate::buffer_util::{InstancedBuffer, InstancedBufferDesc};

type ActiveChunks = StableMap<Point3<i32>, Chunk>;

use super::ChunkMeta;

pub struct ChunkState {
    active_chunks: ActiveChunks,
    meta_tracker: ChunkMetaTracker,

    compute_commands_buf: InstancedBuffer,
    chunk_metadata_buf: InstancedBuffer,
    voxel_type_buf: InstancedBuffer,
    active_view_box: Option<Bounds<i32>>,
}

struct ChunkMetaTracker {
    touched: HashSet<usize>,
    chunk_metas: EndlessVec<ChunkMeta>,
    new_metas: HashMap<usize, ChunkMeta>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, Default, PartialEq)]
struct ComputeCommand {
    chunk_meta_index: u32,
    vertex_data_start: u32,
}

struct EndlessVec<T> {
    data: Vec<T>,
}

impl ChunkState {
    pub fn new(
        ctx: &WgpuRenderResourceContext,
        max_visible_chunks: usize,
        visible_size: Vector3<i32>,
    ) -> Self {
        let visible_chunk_size = crate::visible_size_to_chunks(visible_size);

        let active_visible_chunks =
            visible_chunk_size.x * visible_chunk_size.y * visible_chunk_size.z;
        assert!(active_visible_chunks <= max_visible_chunks as i32);

        let active_chunks = ActiveChunks::new(max_visible_chunks as usize);

        let voxel_type_buf = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "voxel types",
                instance_len: std::mem::size_of::<Chunk>(),
                n_instances: max_visible_chunks,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            },
        );

        let chunk_metadata_buf = InstancedBuffer::new(
            ctx,
            InstancedBufferDesc {
                label: "chunk metadata",
                instance_len: std::mem::size_of::<ChunkMeta>(),
                n_instances: max_visible_chunks,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
            },
        );

        let compute_commands_buf = {
            let result = InstancedBuffer::new(
                ctx,
                InstancedBufferDesc {
                    label: "voxel compute commands",
                    instance_len: std::mem::size_of::<ComputeCommand>(),
                    n_instances: max_visible_chunks,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_DST,
                },
            );

            let mut fill_buffer = result
                .sync_helper()
                .begin_fill_buffer(ctx, result.buffer(), 0);

            for i in 0..max_visible_chunks {
                let command = ComputeCommand {
                    chunk_meta_index: i as u32,
                    vertex_data_start: 3 * index_utils::chunk_size_total() as u32 * i as u32,
                };
                fill_buffer.advance(command.as_bytes());
            }

            fill_buffer.finish();
            result
        };

        ChunkState {
            meta_tracker: ChunkMetaTracker::with_capacity(active_chunks.capacity()),
            active_chunks,
            compute_commands_buf,
            voxel_type_buf,
            chunk_metadata_buf,
            active_view_box: None,
        }
    }

    pub fn count_chunks(&self) -> usize {
        self.active_chunks.capacity()
    }

    pub fn iter_chunk_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.active_chunks.iter().map(|(_, index, _)| index)
    }

    #[allow(dead_code)]
    pub fn iter_chunk_positions(&self) -> impl Iterator<Item = Point3<i32>> + '_ {
        self.active_chunks.iter().map(|(&point, _, _)| point)
    }

    fn update_box_chunks(&mut self, view_box: Bounds<i32>, voxels: &VoxelData) {
        assert!(self.active_chunks.is_empty());
        let bounds = Bounds::new(
            convert_point!(view_box.origin(), i64),
            convert_vec!(view_box.size(), i64),
        );
        for (p, chunk) in voxels
            .iter_chunks_in_bounds(bounds)
            .map(|(p, chunk)| (convert_point!(p, i32), chunk))
        {
            self.active_chunks.update(p, chunk.clone());
        }
    }

    pub fn update_view_box(&mut self, active_view_box: Bounds<i32>, voxels: &VoxelData) -> bool {
        let old_view_box = self.active_view_box;
        self.active_view_box = Some(active_view_box);

        let old_view_box = match old_view_box {
            Some(x) => x,
            None => {
                // no chunks in previous view; insert all
                self.update_box_chunks(active_view_box, voxels);
                return true;
            }
        };

        if active_view_box == old_view_box {
            return false; // nothing changed at all
        }

        let new_chunk_box = view_box_to_chunks(active_view_box);
        let old_chunk_box = view_box_to_chunks(old_view_box);

        if new_chunk_box == old_chunk_box {
            return true; // set of active chunks didn't change but visible voxels at edges did
        }

        // if we get here we need to update the set of active chunks

        // 1. delete chunks that have left the view
        for pos in old_chunk_box.diff(new_chunk_box) {
            self.active_chunks.remove(&pos);
        }

        // 2. insert chunks that are newly in the view
        for pos in new_chunk_box.diff(old_chunk_box) {
            if let Some(chunk) = voxels.chunks().get(convert_point!(pos, i64)) {
                self.active_chunks.update(pos, chunk.clone());
            }
        }

        for &pos in self.active_chunks.keys() {
            assert!(
                new_chunk_box.contains_point(pos),
                "Position {:?} is out of active view box: new={:?}/{:?} old={:?}/{:?}",
                pos,
                new_chunk_box,
                active_view_box,
                old_chunk_box,
                old_view_box
            );
        }

        true
    }

    pub fn apply_chunk_diff(&mut self, voxels: &VoxelData, diff: &VoxelDelta) {
        let active_chunk_box = match self.active_view_box {
            Some(active_view_box) => view_box_to_chunks(active_view_box),
            None => return,
        };

        for &pos in &diff.modified_chunks {
            let pos_i32 = convert_point!(pos, i32);
            if !active_chunk_box.contains_point(pos_i32) {
                continue;
            }

            if let Some(chunk) = voxels.chunks().get(pos) {
                self.active_chunks.update(pos_i32, chunk.clone());
            } else {
                self.active_chunks.remove(&pos_i32);
            }
        }
    }

    // Returns the number of compute work groups that need to be run
    pub fn update_buffers(&mut self, ctx: &WgpuRenderResourceContext) -> u32 {
        let mut fill_voxel_types = self.voxel_type_buf.sync_helper().begin_fill_buffer(
            ctx,
            self.voxel_type_buf.buffer(),
            0,
        );

        let mut fill_chunk_metadatas = self.chunk_metadata_buf.sync_helper().begin_fill_buffer(
            ctx,
            self.chunk_metadata_buf.buffer(),
            0,
        );

        self.meta_tracker.reset();

        {
            let diff = self.active_chunks.take_diff();
            let active_chunks = diff.inner();

            // Copy chunk data to GPU buffers for only the chunks that have changed since last time
            // buffers were updated.
            for (index, opt_point) in diff.changed_entries().into_iter() {
                if let Some((&point, chunk)) = opt_point {
                    self.meta_tracker.modify(point, index, active_chunks);

                    let chunk_data = simgame_voxels::voxels_to_u16(&chunk.voxels);
                    fill_voxel_types.seek(index * std::mem::size_of::<Chunk>());
                    fill_voxel_types.advance(chunk_data.as_bytes());
                } else {
                    self.meta_tracker.remove(index)
                }
            }

            for (index, meta) in self.meta_tracker.update(&active_chunks) {
                fill_chunk_metadatas.seek(index * std::mem::size_of::<ChunkMeta>());
                fill_chunk_metadatas.advance(meta.as_bytes());
            }

            fill_voxel_types.finish();
            fill_chunk_metadatas.finish();
        }

        self.count_chunks() as u32
    }

    pub fn voxel_type_binding(&self) -> RenderResourceBinding {
        self.voxel_type_buf.as_binding()
    }

    pub fn chunk_metadata_binding(&self) -> RenderResourceBinding {
        self.chunk_metadata_buf.as_binding()
    }

    pub fn compute_commands_binding(&self) -> RenderResourceBinding {
        self.compute_commands_buf.as_binding()
    }

    #[allow(dead_code)]
    pub fn debug_get_active_key_ixes(&self) -> Vec<(usize, Point3<i32>)> {
        let mut active_keys: Vec<_> = self
            .active_chunks
            .keys_ixes()
            .map(|(&p, i)| (i, p))
            .collect();
        active_keys.sort_by_key(|&(i, p)| (i, p.x, p.y, p.z));
        active_keys
    }
}

impl<T> EndlessVec<T>
where
    T: Default + Clone,
{
    pub fn new(initial_len: usize) -> Self {
        Self {
            data: Vec::with_capacity(initial_len),
        }
    }

    pub fn get(&self, index: usize) -> T {
        if index < self.data.len() {
            self.data[index].clone()
        } else {
            T::default()
        }
    }

    pub fn set(&mut self, index: usize, value: T) {
        if index >= self.data.len() {
            self.data
                .extend(std::iter::repeat(T::default()).take(1 + index - self.data.len()));
        }

        self.data[index] = value;
    }
}

impl ChunkMetaTracker {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            touched: HashSet::with_capacity(capacity),
            chunk_metas: EndlessVec::new(capacity),
            new_metas: HashMap::with_capacity(capacity),
        }
    }

    fn reset(&mut self) {
        self.touched.clear();
        self.new_metas.clear();
    }

    fn touch_neighbors(touched: &mut HashSet<usize>, meta: &ChunkMeta) {
        if meta.active == 0 {
            return;
        }

        for &neighbor_index in &meta.neighbor_indices {
            if neighbor_index != -1 {
                touched.insert(neighbor_index as usize);
            }
        }
    }

    fn modify(&mut self, point: Point3<i32>, index: usize, active_chunks: &ActiveChunks) {
        let old_meta = self.chunk_metas.get(index);
        if old_meta.active == 1 {
            let old_point: Point3<i32> = old_meta.offset.into();

            if old_point == point {
                // an updated chunk whose location has not changed does no trigger any metadata
                // updates
                return;
            }

            Self::touch_neighbors(&mut self.touched, &old_meta);
        } else {
            let new_meta = Self::make_chunk_meta(active_chunks, point);
            Self::touch_neighbors(&mut self.touched, &new_meta);
            self.new_metas.insert(index, new_meta);
        }
    }

    fn remove(&mut self, index: usize) {
        let old_meta = self.chunk_metas.get(index);
        Self::touch_neighbors(&mut self.touched, &old_meta);
        self.new_metas.insert(index, ChunkMeta::default());
    }

    fn update<'a>(
        &'a mut self,
        active_chunks: &ActiveChunks,
    ) -> impl Iterator<Item = (usize, ChunkMeta)> + 'a {
        for &index in &self.touched {
            if index > active_chunks.capacity() || self.new_metas.contains_key(&index) {
                continue;
            }

            let (&point, _) = match active_chunks.index(index) {
                Some(x) => x,
                None => continue,
            };
            self.new_metas
                .insert(index, Self::make_chunk_meta(active_chunks, point));
        }

        for (&index, &chunk_meta) in &self.new_metas {
            self.chunk_metas.set(index, chunk_meta);
        }

        self.new_metas.iter().map(|(index, meta)| (*index, *meta))
    }

    fn make_chunk_meta(active_chunks: &ActiveChunks, p: Point3<i32>) -> ChunkMeta {
        fn make_neighbor_indices(map: &ActiveChunks, chunk_loc: Point3<i32>) -> [i32; 6] {
            let directions = [
                Vector3::new(-1, 0, 0),
                Vector3::new(1, 0, 0),
                Vector3::new(0, -1, 0),
                Vector3::new(0, 1, 0),
                Vector3::new(0, 0, -1),
                Vector3::new(0, 0, 1),
            ];

            let mut result = [0i32; 6];
            for (src, dst) in directions
                .iter()
                .map(|dir| {
                    let neighbor_loc = chunk_loc + dir;
                    map.get(&neighbor_loc).map_or(-1, |(index, _)| index as i32)
                })
                .zip(&mut result)
            {
                *dst = src;
            }

            result
        }

        let offset =
            p.mul_element_wise(Point3::origin() + convert_vec!(index_utils::chunk_size(), i32));

        let neighbor_indices = make_neighbor_indices(&active_chunks, p);

        ChunkMeta {
            offset: offset.into(),
            _padding0: [0],
            neighbor_indices,
            active: 1,
            _padding1: 0,
        }
    }
}

fn view_box_to_chunks(view_box: Bounds<i32>) -> Bounds<i32> {
    view_box.quantize_down(convert_vec!(index_utils::chunk_size(), i32))
}
