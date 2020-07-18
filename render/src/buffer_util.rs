use std::ops::{Deref, DerefMut};

use wgpu::{Buffer, BufferDescriptor, BufferUsage, Device, Queue};
use zerocopy::{AsBytes, FromBytes};

pub struct BufferSyncedData<Data, Item> {
    data: Data,
    helper: BufferSyncHelper<Item>,
    buffer: Buffer,
}

#[derive(Debug, Clone)]
pub struct BufferSyncHelper<Item> {
    desc: BufferSyncHelperDesc,
    _marker: std::marker::PhantomData<Item>,
}

#[derive(Debug, Clone)]
pub struct BufferSyncHelperDesc {
    pub label: &'static str,
    pub buffer_len: usize,
    pub max_chunk_len: usize,
    pub usage: BufferUsage,
}

pub trait BufferSyncable {
    type Item;

    fn sync<'a>(&self, fill_buffer: &mut FillBuffer<'a, Self::Item>);
}

pub trait IntoBufferSynced: BufferSyncable {
    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc;

    fn into_buffer_synced(self, device: &wgpu::Device) -> BufferSyncedData<Self, Self::Item>
    where
        Self: Sized,
    {
        let desc = self.buffer_sync_desc();
        BufferSyncedData::new(device, self, desc)
    }
}

impl<Data, Item> BufferSyncedData<Data, Item> {
    pub fn new(device: &Device, data: Data, desc: BufferSyncHelperDesc) -> Self {
        assert!(desc.usage.contains(wgpu::BufferUsage::COPY_DST));

        let helper = BufferSyncHelper::new(desc);
        let buffer = helper.make_buffer(device);
        BufferSyncedData {
            data,
            helper,
            buffer,
        }
    }

    #[allow(dead_code)]
    pub fn from_buffer(data: Data, desc: BufferSyncHelperDesc, buffer: Buffer) -> Self {
        BufferSyncedData {
            data,
            helper: BufferSyncHelper::new(desc),
            buffer,
        }
    }

    #[inline]
    pub fn sync<'a>(&'a self, queue: &Queue)
    where
        Data: BufferSyncable<Item = Item>,
        Item: Copy + AsBytes + FromBytes + 'static,
    {
        let mut fill_buffer = self.helper.begin_fill_buffer(queue, &self.buffer, 0);
        self.data.sync(&mut fill_buffer);
        fill_buffer.finish();
    }

    #[inline]
    #[allow(dead_code)]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    #[allow(dead_code)]
    pub fn sync_helper(&self) -> &BufferSyncHelper<Item> {
        &self.helper
    }

    #[inline]
    pub fn as_binding(&self, index: u32) -> wgpu::BindGroupEntry {
        self.helper.as_binding(index, &self.buffer, 0)
    }
}

impl<Data, Item> Deref for BufferSyncedData<Data, Item> {
    type Target = Data;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<Data, Item> DerefMut for BufferSyncedData<Data, Item> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<Item> BufferSyncHelper<Item> {
    pub fn make_buffer(&self, device: &Device) -> Buffer {
        log::info!(
            "Creating buffer \"{}\" of size {} MB",
            self.desc().label,
            self.desc().buffer_len * std::mem::size_of::<Item>() / (1024 * 1024)
        );

        device.create_buffer(&BufferDescriptor {
            label: Some(self.desc().label),
            size: (self.desc.buffer_len * std::mem::size_of::<Item>()) as u64,
            usage: self.desc.usage,
            mapped_at_creation: false,
        })
    }

    pub fn new(desc: BufferSyncHelperDesc) -> Self {
        let item_size = std::mem::size_of::<Item>();
        let byte_len = desc.buffer_len * item_size;
        let chunk_byte_len = desc.max_chunk_len * item_size;

        assert!(
            byte_len % 4 == 0,
            "buffer helper byte length {} should be a multiple of 4",
            byte_len
        );
        assert!(
            chunk_byte_len % 4 == 0,
            "buffer helper max chunk byte length {} should be a multiple of 4",
            chunk_byte_len
        );

        BufferSyncHelper {
            desc,
            _marker: std::marker::PhantomData,
        }
    }

    /// Creates a `FillBuffer` object which can be used to copy chunks of data to the buffer one by
    /// one, for example to split the copy over multiple command submissions or to copy chunks to
    /// multiple buffers at the same time.
    ///
    /// Once you are done with the `FillBuffer` object, you must consume it e.g. with `finish()`.
    /// The `Drop` impl may panic if you fail to do so.
    pub fn begin_fill_buffer<'a>(
        &'a self,
        queue: &'a Queue,
        target: &'a Buffer,
        start_pos: usize,
    ) -> FillBuffer<'a, Item>
    where
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        FillBuffer {
            queue,
            target,
            mapped_buffer: None,
            pos: start_pos,
            batch_len: 0,
            sync_helper: self,
        }
    }

    /// Fills the buffer with chunks of data from the given iterator.
    pub fn fill_buffer<Chunks>(
        &self,
        queue: &Queue,
        target: &Buffer,
        start_pos: usize,
        chunks: Chunks,
    ) where
        Chunks: IntoIterator,
        Chunks::Item: AsRef<[Item]>,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        let mut fill_buffer = self.begin_fill_buffer(queue, target, start_pos);

        for chunk in chunks {
            fill_buffer.advance(chunk);
        }

        fill_buffer.finish();
    }

    #[inline]
    pub fn desc(&self) -> &BufferSyncHelperDesc {
        &self.desc
    }

    #[inline]
    pub fn buffer_byte_len(&self) -> wgpu::BufferAddress {
        (self.desc.buffer_len * std::mem::size_of::<Item>()) as wgpu::BufferAddress
    }

    #[inline]
    pub fn as_binding<'a>(
        &self,
        index: u32,
        buffer: &'a Buffer,
        start_offset: wgpu::BufferAddress,
    ) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding: index,
            resource: wgpu::BindingResource::Buffer(
                buffer.slice(start_offset..start_offset + self.buffer_byte_len()),
            ),
        }
    }
}

pub struct FillBuffer<'a, Item> {
    queue: &'a Queue,
    target: &'a Buffer,
    mapped_buffer: Option<Buffer>,
    pos: usize,
    batch_len: usize,
    sync_helper: &'a BufferSyncHelper<Item>,
}

impl<'a, Item> FillBuffer<'a, Item> {
    /// Copy a single chunk of data to the buffer.
    #[inline]
    pub fn advance<Chunk>(&mut self, chunk: Chunk)
    where
        Chunk: AsRef<[Item]>,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        let chunk_slice = chunk.as_ref();
        let chunk_len = chunk_slice.len();

        let byte_len = chunk_len * std::mem::size_of::<Item>();

        assert!(
            byte_len % 4 == 0,
            "chunk byte len {} should be a multiple of 4",
            byte_len
        );
        assert!(chunk_len <= self.sync_helper.desc.max_chunk_len);

        if self.batch_len + chunk_len > self.sync_helper.desc.max_chunk_len {
            self.end_batch();
        }

        let item_size = std::mem::size_of::<Item>();
        let begin = item_size * (self.pos + self.batch_len);

        self.queue
            .write_buffer(self.target, begin as _, chunk_slice.as_bytes());
        self.batch_len += chunk_len;
    }

    /// Finalizes the copy, copying any remaining chunks.
    pub fn finish(mut self)
    where
        Item: Copy,
    {
        self.end_batch()
    }

    fn end_batch(&mut self)
    where
        Item: Copy,
    {
        // if let Some(mapped_buffer) = self.mapped_buffer.take() {
        //     let item_size = std::mem::size_of::<Item>();
        //     encoder.copy_buffer_to_buffer(
        //         &mapped_buffer,
        //         0,
        //         self.target,
        //         (item_size * self.pos) as BufferAddress,
        //         (item_size * self.batch_len) as BufferAddress,
        //     );

        //     self.pos += self.batch_len;
        // }

        self.pos += self.batch_len;
        self.batch_len = 0;
    }

    /// Subsequent writes will begin at the given position. May end the current batch.
    #[inline]
    pub fn seek(&mut self, new_pos: usize)
    where
        Item: Copy,
    {
        if self.pos + self.batch_len != new_pos {
            self.end_batch();
            self.pos = new_pos;
        }
    }
}

impl<'a, Item> Drop for FillBuffer<'a, Item> {
    fn drop(&mut self) {
        if self.mapped_buffer.is_some() && !std::thread::panicking() {
            panic!("must call finish on FillBuffer object!");
        }
    }
}

pub struct InstancedBuffer {
    desc: InstancedBufferDesc,
    helper: BufferSyncHelper<u8>,
    buffer: Buffer,
}

#[derive(Debug, Clone)]
pub struct InstancedBufferDesc {
    pub label: &'static str,
    pub n_instances: usize,
    pub instance_len: usize,
    pub usage: BufferUsage,
}

impl InstancedBuffer {
    pub fn new(device: &wgpu::Device, desc: InstancedBufferDesc) -> Self {
        let helper_desc = BufferSyncHelperDesc {
            label: desc.label,
            buffer_len: desc.n_instances * desc.instance_len,
            max_chunk_len: desc.instance_len,
            usage: desc.usage,
        };
        let helper = BufferSyncHelper::new(helper_desc);

        let buffer = helper.make_buffer(device);
        Self {
            desc,
            helper,
            buffer,
        }
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    #[allow(dead_code)]
    pub fn sync_helper(&self) -> &BufferSyncHelper<u8> {
        &self.helper
    }

    #[inline]
    pub fn as_binding(&self, index: u32) -> wgpu::BindGroupEntry {
        self.helper.as_binding(index, &self.buffer, 0)
    }

    #[inline]
    pub fn instance_offset(&self, instance: usize) -> wgpu::BufferAddress {
        (instance * self.desc.instance_len) as _
    }

    #[inline]
    pub fn size(&self) -> wgpu::BufferAddress {
        self.helper.desc().buffer_len as _
    }

    #[inline]
    pub fn clear(&self, queue: &Queue) {
        self.helper.fill_buffer(
            queue,
            &self.buffer,
            0,
            std::iter::repeat(&[0u8, 0, 0, 0]).take(self.size() as usize / 4),
        );
    }

    #[inline]
    pub fn write(&self, queue: &Queue, instance: usize, data: &[u8]) {
        let mut fill = self.helper.begin_fill_buffer(
            queue,
            &self.buffer,
            self.instance_offset(instance) as usize,
        );
        fill.advance(data);
        fill.finish();
    }
}
