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
    pub gpu_usage: BufferUsage,
}

pub trait IntoBufferSynced {
    type Item;

    fn buffer_sync_desc(&self) -> BufferSyncHelperDesc;

    fn buffer_synced(self, device: &wgpu::Device) -> BufferSyncedData<Self, Self::Item>
    where
        Self: Sized,
    {
        let desc = self.buffer_sync_desc();
        BufferSyncedData::new(device, self, desc)
    }
}

impl<Data, Item> BufferSyncedData<Data, Item> {
    pub fn new(device: &Device, data: Data, desc: BufferSyncHelperDesc) -> Self {
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
        &'a Data: IntoIterator,
        <&'a Data as IntoIterator>::Item: AsRef<[Item]>,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        self.sync_with(device, encoder, |x| x.into_iter())
    }

    #[inline]
    pub fn sync_with<'a, Iter, F>(
        &'a self,
        device: &Device,
        encoder: &mut CommandEncoder,
        into_iter: F,
    ) where
        Iter: Iterator,
        Iter::Item: AsRef<[Item]>,
        F: FnOnce(&'a Data) -> Iter,
        Item: 'static + Copy + AsBytes + FromBytes,
    {
        self.helper
            .fill_buffer(device, encoder, &self.buffer, 0, into_iter(&self.data));
    }

    // #[inline]
    // pub fn start_sync_with<'a, Iter, F>(
    //     &'a self,
    //     device: &'a Device,
    //     into_iter: F,
    // ) -> FillBuffer<'a, Item>
    //     where
    //     Iter: Iterator,
    //     Iter::Item: AsRef<[Item]>,
    //     F: FnOnce(&'a Data) -> Iter,
    //     Item: 'static + Copy + AsBytes + FromBytes,
    // {
    //     self.helper.begin_fill_buffer(device, &self.buffer, 0)
    //     // self.helper
    //     //     .fill_buffer(device, encoder, &self.buffer, 0, into_iter(&self.data));
    // }

    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    #[inline]
    pub fn sync_helper(&self) -> &BufferSyncHelper<Item> {
        &self.helper
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
            size: (self.desc.buffer_len * std::mem::size_of::<Item>()) as u64,
            usage: BufferUsage::COPY_DST | self.desc.gpu_usage,
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
            total_len: start_pos,
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
    fn make_src_buffer<'a>(&self, device: &'a Device) -> CreateBufferMapped<'a, Item>
    where
        Item: Copy + 'static,
    {
        device.create_buffer_mapped(self.desc.max_chunk_len, BufferUsage::COPY_SRC)
    }

    #[inline]
    pub fn desc(&self) -> &BufferSyncHelperDesc {
        &self.desc
    }

    #[inline]
    pub fn buffer_byte_len(&self) -> wgpu::BufferAddress {
        (self.desc.buffer_len * std::mem::size_of::<Item>()) as wgpu::BufferAddress
    }
}

pub struct FillBufferWithIter<'a, Iter, Item> {
    fill_buffer: FillBuffer<'a, Item>,
    iter: Iter,
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

pub struct FillBuffer<'a, Item> {
    device: &'a Device,
    target: &'a Buffer,
    mapped_buffer: Option<CreateBufferMapped<'a, Item>>,
    total_len: usize,
    batch_len: usize,
    sync_helper: &'a BufferSyncHelper<Item>,
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
        mapped_buffer.data[self.batch_len..self.batch_len + chunk_len]
            .copy_from_slice(chunk_slice);

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
                (item_size * self.total_len) as BufferAddress,
                (item_size * self.batch_len) as BufferAddress,
            );

            self.total_len += self.batch_len;
        }
    }
}

impl<'a, Item> Drop for FillBuffer<'a, Item> {
    fn drop(&mut self) {
        if self.mapped_buffer.is_some() {
            panic!("must call finish on FillBuffer object!");
        }
    }
}
