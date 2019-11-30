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
        let mut iter_chunks = chunks.into_iter().peekable();

        let mut total_len = start_pos;
        let item_size = std::mem::size_of::<Item>();

        while iter_chunks.peek().is_some() {
            let mut batch_len = 0;
            let cpu_buffer = {
                let mapped_buffer = self.make_src_buffer(device);

                while let Some(chunk) = iter_chunks.peek() {
                    let chunk_len = chunk.as_ref().len();
                    assert!(chunk_len <= self.desc.max_chunk_len);

                    if batch_len + chunk_len > self.desc.max_chunk_len {
                        break;
                    }

                    let chunk_slice = chunk.as_ref();
                    let chunk_len = chunk_slice.len();

                    mapped_buffer.data[batch_len..batch_len + chunk_len]
                        .copy_from_slice(chunk_slice);

                    batch_len += chunk_len;
                    iter_chunks.next();
                }

                assert!(total_len + batch_len <= self.desc.buffer_len);
                mapped_buffer.finish()
            };

            encoder.copy_buffer_to_buffer(
                &cpu_buffer,
                0,
                target,
                (item_size * total_len) as BufferAddress,
                (item_size * batch_len) as BufferAddress,
            );

            total_len += batch_len;
        }
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
