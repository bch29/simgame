use std::ops::{Deref, DerefMut};

use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsage, CommandEncoder, CreateBufferMapped,
    Device,
};
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
    pub buffer_len: usize,
    pub max_chunk_len: usize,
    pub usage: BufferUsage,
}

pub trait BufferSyncable {
    type Item;

    fn sync<'a>(&self, fill_buffer: &mut FillBuffer<'a, Self::Item>, encoder: &mut CommandEncoder);
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

    pub fn from_buffer(data: Data, desc: BufferSyncHelperDesc, buffer: Buffer) -> Self {
        BufferSyncedData {
            data,
            helper: BufferSyncHelper::new(desc),
            buffer,
        }
    }

    #[inline]
    pub fn sync<'a>(&'a self, device: &Device, encoder: &mut CommandEncoder)
    where
        Data: BufferSyncable<Item=Item>,
        Item: Copy + AsBytes + FromBytes + 'static
    {
        let mut fill_buffer = self.helper.begin_fill_buffer(device, &self.buffer, 0);
        self.data.sync(&mut fill_buffer, encoder);
        fill_buffer.finish(encoder);
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    pub fn sync_helper(&self) -> &BufferSyncHelper<Item> {
        &self.helper
    }

    #[inline]
    pub fn as_binding(&self, index: u32) -> wgpu::Binding {
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
        device.create_buffer(&BufferDescriptor {
            label: None,
            size: (self.desc.buffer_len * std::mem::size_of::<Item>()) as u64,
            usage: self.desc.usage,
        })
    }

    pub fn new(desc: BufferSyncHelperDesc) -> Self {
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
        device: &'a Device,
        target: &'a Buffer,
        start_pos: usize,
    ) -> FillBuffer<'a, Item>
    where
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        FillBuffer {
            device,
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
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        start_pos: usize,
        chunks: Chunks,
    ) where
        Chunks: IntoIterator,
        Chunks::Item: AsRef<[Item]>,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        let mut fill_buffer = self.begin_fill_buffer(device, target, start_pos);

        for chunk in chunks {
            fill_buffer.advance(encoder, chunk);
        }

        fill_buffer.finish(encoder);
    }

    #[inline]
    fn make_src_buffer<'a>(&self, device: &'a Device) -> CreateBufferMapped<'a>
    where
        Item: Copy + 'static,
    {
        device.create_buffer_mapped(&wgpu::BufferDescriptor {
            label: None,
            size: (std::mem::size_of::<Item>() * self.desc.max_chunk_len) as u64,
            usage: BufferUsage::COPY_SRC,
        })
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
    ) -> wgpu::Binding<'a> {
        wgpu::Binding {
            binding: index,
            resource: wgpu::BindingResource::Buffer {
                buffer,
                range: start_offset..start_offset + self.buffer_byte_len(),
            },
        }
    }
}

pub struct FillBuffer<'a, Item> {
    device: &'a Device,
    target: &'a Buffer,
    mapped_buffer: Option<CreateBufferMapped<'a>>,
    pos: usize,
    batch_len: usize,
    sync_helper: &'a BufferSyncHelper<Item>,
}

pub struct FillBufferWithIter<'a, Iter, Item> {
    fill_buffer: FillBuffer<'a, Item>,
    iter: Iter,
}

impl<'a, Item> FillBuffer<'a, Item> {
    /// By attaching an iterator to the object, it gains the ability to `advance` by consuming from
    /// its internal iterator instead of having the caller supply chunks directly.
    #[inline]
    pub fn attach_iter<Iter>(self, iter: Iter) -> FillBufferWithIter<'a, Iter::IntoIter, Item>
    where
        Iter: IntoIterator,
    {
        FillBufferWithIter {
            fill_buffer: self,
            iter: iter.into_iter(),
        }
    }

    /// Copy a single chunk of data to the buffer.
    #[inline]
    pub fn advance<Chunk>(&mut self, encoder: &mut CommandEncoder, chunk: Chunk)
    where
        Chunk: AsRef<[Item]>,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        let chunk_slice = chunk.as_ref();
        let chunk_len = chunk_slice.len();

        assert!(chunk_len <= self.sync_helper.desc.max_chunk_len);

        if self.batch_len + chunk_len > self.sync_helper.desc.max_chunk_len {
            self.end_batch(encoder);
        }

        let sync_helper = &self.sync_helper;
        let device = &self.device;

        let mapped_buffer = self
            .mapped_buffer
            .get_or_insert_with(|| sync_helper.make_src_buffer(device));

        let item_size = std::mem::size_of::<Item>();
        let begin = item_size * self.batch_len;
        let end = begin + item_size * chunk_len;
        mapped_buffer.data[begin..end].copy_from_slice(chunk_slice.as_bytes());

        self.batch_len += chunk_len;
    }

    /// Copies each chunk from the given iterator to the buffer.
    #[inline]
    pub fn advance_iter<Iter>(&mut self, encoder: &mut CommandEncoder, chunks: Iter)
    where
        Iter::Item: AsRef<[Item]>,
        Iter: IntoIterator,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        for chunk in chunks {
            self.advance(encoder, chunk)
        }
    }

    /// Copies each chunk from the given iterator to the buffer, then finalizes the copy.
    #[inline]
    pub fn finish_with_iter<Iter>(self, encoder: &mut CommandEncoder, chunks: Iter)
    where
        Iter::Item: AsRef<[Item]>,
        Iter: IntoIterator,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        self.attach_iter(chunks).drain(encoder)
    }

    /// Finalizes the copy, copying any remaining chunks.
    pub fn finish(mut self, encoder: &mut CommandEncoder)
    where
        Item: Copy,
    {
        self.end_batch(encoder)
    }

    fn end_batch(&mut self, encoder: &mut CommandEncoder)
    where
        Item: Copy,
    {
        if let Some(mapped_buffer) = self.mapped_buffer.take() {
            let item_size = std::mem::size_of::<Item>();
            let src = mapped_buffer.finish();
            encoder.copy_buffer_to_buffer(
                &src,
                0,
                self.target,
                (item_size * self.pos) as BufferAddress,
                (item_size * self.batch_len) as BufferAddress,
            );

            self.pos += self.batch_len;
        }

        self.batch_len = 0;
    }

    /// Subsequent writes will begin at the given position. May end the current batch.
    #[inline]
    pub fn seek(&mut self, encoder: &mut CommandEncoder, new_pos: usize)
    where
        Item: Copy,
    {
        if self.pos + self.batch_len != new_pos {
            self.end_batch(encoder);
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

impl<'a, Iter, Item> FillBufferWithIter<'a, Iter, Item>
where
    Iter: Iterator,
    Iter::Item: AsRef<[Item]>,
    Item: 'static + Copy + AsBytes + FromBytes,
{
    #[inline]
    pub fn advance(mut self, encoder: &mut CommandEncoder) -> Option<Self> {
        match self.iter.next() {
            Some(chunk) => {
                self.fill_buffer.advance(encoder, chunk);
                Some(self)
            }
            None => {
                self.fill_buffer.finish(encoder);
                None
            }
        }
    }

    #[inline]
    pub fn drain(mut self, encoder: &mut CommandEncoder) {
        loop {
            self = match self.advance(encoder) {
                Some(x) => x,
                None => break,
            }
        }
    }

    #[inline]
    pub fn detach(self) -> FillBuffer<'a, Item> {
        self.fill_buffer
    }
}

pub struct OpaqueBuffer {
    helper: BufferSyncHelper<u8>,
    buffer: Buffer
}

impl OpaqueBuffer {
    pub fn new(device: &Device, desc: BufferSyncHelperDesc) -> Self {
        let helper = BufferSyncHelper::new(desc);
        let buffer = helper.make_buffer(device);
        Self {
            helper,
            buffer,
        }
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    pub fn sync_helper(&self) -> &BufferSyncHelper<u8> {
        &self.helper
    }

    #[inline]
    pub fn as_binding(&self, index: u32) -> wgpu::Binding {
        self.helper.as_binding(index, &self.buffer, 0)
    }
}

pub struct InstancedBuffer {
    desc: InstancedBufferDesc,
    helper: BufferSyncHelper<u8>,
    buffer: Buffer
}

#[derive(Debug, Clone)]
pub struct InstancedBufferDesc {
    pub n_instances: usize,
    pub instance_len: usize,
    pub usage: BufferUsage
}

impl InstancedBuffer {
    pub fn new(device: &wgpu::Device, desc: InstancedBufferDesc) -> Self {
        let helper_desc = BufferSyncHelperDesc {
            buffer_len: desc.n_instances * desc.instance_len,
            max_chunk_len: 0,
            usage: desc.usage
        };
        let helper = BufferSyncHelper::new(helper_desc);
        let buffer = helper.make_buffer(device);
        Self {
            desc,
            helper,
            buffer
        }
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    pub fn sync_helper(&self) -> &BufferSyncHelper<u8> {
        &self.helper
    }

    #[inline]
    pub fn as_binding(&self, index: u32) -> wgpu::Binding {
        self.helper.as_binding(index, &self.buffer, 0)
    }

    #[inline]
    pub fn instance_offset(&self, instance: usize) -> wgpu::BufferAddress {
        (instance * self.desc.instance_len) as _
    }
}

pub struct Swappable<T> {
    active: T,
    inactive: T
}

impl<T> Swappable<T> {
    pub fn new(active: T, inactive: T) -> Self {
        Self { active, inactive }
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.active, &mut self.inactive)
    }

    pub fn active(&self) -> &T {
        &self.active
    }

    pub fn inactive(&self) -> &T {
        &self.inactive
    }
}
