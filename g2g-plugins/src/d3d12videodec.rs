// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{size_of, size_of_val, ManuallyDrop};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList, ID3D12Resource, D3D12_BOX, D3D12_COMMAND_LIST_TYPE_COPY,
    D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE,
    D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_VIDEO_DECODE_READ, D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    D3D12_TEXTURE_LAYOUT_UNKNOWN,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_NV12, DXGI_FORMAT_UNKNOWN, DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};
use windows::Win32::Media::MediaFoundation::{
    ID3D12VideoDecodeCommandList, ID3D12VideoDecoder, ID3D12VideoDecoderHeap, ID3D12VideoDevice,
    D3D12_BITSTREAM_ENCRYPTION_TYPE_NONE, D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT,
    D3D12_FEATURE_VIDEO_DECODE_SUPPORT, D3D12_VIDEO_DECODER_DESC, D3D12_VIDEO_DECODER_HEAP_DESC,
    D3D12_VIDEO_DECODE_ARGUMENT_TYPE_INVERSE_QUANTIZATION_MATRIX,
    D3D12_VIDEO_DECODE_ARGUMENT_TYPE_PICTURE_PARAMETERS,
    D3D12_VIDEO_DECODE_ARGUMENT_TYPE_SLICE_CONTROL, D3D12_VIDEO_DECODE_COMPRESSED_BITSTREAM,
    D3D12_VIDEO_DECODE_CONFIGURATION,
    D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED,
    D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_REFERENCE_ONLY_ALLOCATIONS_REQUIRED,
    D3D12_VIDEO_DECODE_FRAME_ARGUMENT, D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS,
    D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS, D3D12_VIDEO_DECODE_PROFILE_H264,
    D3D12_VIDEO_DECODE_REFERENCE_FRAMES, D3D12_VIDEO_DECODE_SUPPORT_FLAG_SUPPORTED,
    D3D12_VIDEO_DECODE_TIER_2, D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE_NONE,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

pub use crate::displayorder::PictureMeta;
use crate::dxva::{
    h264_flat_scaling_matrices, h264_picture_parameters, DxvaH264Picture, DxvaSliceH264Short,
};
use crate::gpu::{GpuContext, WgpuNv12Texture};
use crate::h264dpb::{
    chroma_rectangle, h264_dpb_slots, h264_visible_picture, H264DecodeSession, H264Picture,
};
use crate::h264hardwaredec::{H264HardwareDec, H264HardwareDecoder};
use crate::h264parse::{H264Sps, H264StreamRefused};
use crate::texturepool::take_from_pool;
use alloc::boxed::Box;
use g2g_core::g2g_error;
use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedWgpuTexture, SystemSlice};
use g2g_core::meta::CropRectangle;
use g2g_core::runtime::block_on;
use g2g_core::{ElementMetadata, G2gError, MemoryDomain};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum D3D12VideoError {
    NoD3D12Adapter,
    DecodeUnsupported(&'static str),
    StreamRefused(&'static str),
    CallFailed(i32),
    DeviceLost,
}

impl From<windows::core::Error> for D3D12VideoError {
    fn from(error: windows::core::Error) -> Self {
        D3D12VideoError::CallFailed(error.code().0)
    }
}

impl From<H264StreamRefused> for D3D12VideoError {
    fn from(refused: H264StreamRefused) -> Self {
        D3D12VideoError::StreamRefused(refused.0)
    }
}

// windows events are kernel handles that stay open until closed
#[derive(Debug)]
struct FenceEvent(HANDLE);

// SAFETY: a windows event handle is a process-wide kernel object, valid on any thread.
unsafe impl Send for FenceEvent {}

impl Drop for FenceEvent {
    fn drop(&mut self) {
        // SAFETY: the handle came from `CreateEventW` and is closed once, here.
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}

pub struct D3D12DecodeDevice {
    device: ID3D12Device,
    video_device: ID3D12VideoDevice,
    wgpu_queue: ID3D12CommandQueue,
    decode_queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    command_list: ID3D12VideoDecodeCommandList,
    copy_queue: ID3D12CommandQueue,
    copy_allocator: ID3D12CommandAllocator,
    copy_list: ID3D12GraphicsCommandList,
    completion_fence: ID3D12Fence,
    completion_fence_value: u64,
    wgpu_fence: ID3D12Fence,
    wgpu_fence_value: u64,
    fence_event: FenceEvent,
    // wgpu-hal unloads d3d12.dll when its last device drops
    gpu: GpuContext,
}

impl core::fmt::Debug for D3D12DecodeDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("D3D12DecodeDevice")
            .field("adapter", &self.gpu.adapter.get_info().name)
            .finish_non_exhaustive()
    }
}

pub async fn open_d3d12_decode_device() -> Result<D3D12DecodeDevice, D3D12VideoError> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|_| D3D12VideoError::NoD3D12Adapter)?;
    if !adapter
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        return Err(D3D12VideoError::NoD3D12Adapter);
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("g2g-d3d12-decode"),
            required_features: wgpu::Features::TEXTURE_FORMAT_NV12,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|_| D3D12VideoError::NoD3D12Adapter)?;
    // SAFETY: the handles are cloned out (an AddRef each) and the guard dropped
    // before any wgpu call, and nothing here destroys them.
    let (raw_device, wgpu_queue) = unsafe {
        let hal = device
            .as_hal::<wgpu_hal::api::Dx12>()
            .ok_or(D3D12VideoError::NoD3D12Adapter)?;
        (hal.raw_device().clone(), hal.raw_queue().clone())
    };
    let video_device: ID3D12VideoDevice = raw_device.cast()?;
    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
        ..Default::default()
    };
    let copy_queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_COPY,
        ..Default::default()
    };
    // SAFETY: each call gets valid descriptors and returns a new object.
    unsafe {
        let allocator: ID3D12CommandAllocator =
            raw_device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE)?;
        let command_list: ID3D12VideoDecodeCommandList = raw_device.CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
            &allocator,
            None,
        )?;
        command_list.Close()?;
        let copy_allocator: ID3D12CommandAllocator =
            raw_device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
        let copy_list: ID3D12GraphicsCommandList =
            raw_device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &copy_allocator, None)?;
        copy_list.Close()?;
        Ok(D3D12DecodeDevice {
            gpu: GpuContext::from_wgpu(instance, adapter, device, queue),
            decode_queue: raw_device.CreateCommandQueue(&queue_desc)?,
            allocator,
            command_list,
            copy_queue: raw_device.CreateCommandQueue(&copy_queue_desc)?,
            copy_allocator,
            copy_list,
            completion_fence: raw_device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
            completion_fence_value: 0,
            wgpu_fence: raw_device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
            wgpu_fence_value: 0,
            fence_event: FenceEvent(CreateEventW(None, false, false, None)?),
            device: raw_device,
            video_device,
            wgpu_queue,
        })
    }
}

// the rate D3D12 checks support at, since the stream's own rate is not known yet
const SUPPORT_QUERY_FRAME_RATE: DXGI_RATIONAL = DXGI_RATIONAL {
    Numerator: 30,
    Denominator: 1,
};

const H264_CONFIGURATION: D3D12_VIDEO_DECODE_CONFIGURATION = D3D12_VIDEO_DECODE_CONFIGURATION {
    DecodeProfile: D3D12_VIDEO_DECODE_PROFILE_H264,
    BitstreamEncryption: D3D12_BITSTREAM_ENCRYPTION_TYPE_NONE,
    InterlaceType: D3D12_VIDEO_FRAME_CODED_INTERLACE_TYPE_NONE,
};

// D3D12 video decode reads the first slice from a 128-byte-aligned buffer offset
const BITSTREAM_ALIGNMENT: usize = 128;
const ANNEX_B_START_CODE: [u8; 3] = [0, 0, 1];

impl D3D12DecodeDevice {
    pub fn gpu_context(&self) -> &GpuContext {
        &self.gpu
    }

    fn h264_texture_extent(&self, width: u32, height: u32) -> Result<(u32, u32), D3D12VideoError> {
        let mut support = D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT {
            NodeIndex: 0,
            Configuration: H264_CONFIGURATION,
            Width: width,
            Height: height,
            DecodeFormat: DXGI_FORMAT_NV12,
            FrameRate: SUPPORT_QUERY_FRAME_RATE,
            ..Default::default()
        };
        // SAFETY: the pointer and size describe `support`, which outlives the call.
        unsafe {
            self.video_device.CheckFeatureSupport(
                D3D12_FEATURE_VIDEO_DECODE_SUPPORT,
                (&mut support as *mut D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT).cast(),
                size_of_val(&support) as u32,
            )?;
        }
        if !support
            .SupportFlags
            .contains(D3D12_VIDEO_DECODE_SUPPORT_FLAG_SUPPORTED)
        {
            return Err(D3D12VideoError::DecodeUnsupported(
                "H.264 decode to NV12 at this size",
            ));
        }
        // tier 1 keeps every reference in one texture array, which the per-picture texture pool cannot use
        if support.DecodeTier.0 < D3D12_VIDEO_DECODE_TIER_2.0 {
            return Err(D3D12VideoError::DecodeUnsupported(
                "decode tier 2 (references in separate textures)",
            ));
        }
        if support
            .ConfigurationFlags
            .contains(D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_REFERENCE_ONLY_ALLOCATIONS_REQUIRED)
        {
            return Err(D3D12VideoError::DecodeUnsupported(
                "decoding without reference-only allocations",
            ));
        }
        let height = if support
            .ConfigurationFlags
            .contains(D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED)
        {
            height.next_multiple_of(32)
        } else {
            height
        };
        Ok((width, height))
    }

    fn create_nv12_texture(
        &self,
        (width, height): (u32, u32),
    ) -> Result<ID3D12Resource, D3D12VideoError> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        self.create_resource(D3D12_HEAP_TYPE_DEFAULT, &desc, D3D12_RESOURCE_STATE_COMMON)
    }

    fn create_upload_buffer(&self, size: usize) -> Result<ID3D12Resource, D3D12VideoError> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        // d3d12 upload heap resources start and stay in GENERIC_READ
        self.create_resource(
            D3D12_HEAP_TYPE_UPLOAD,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
    }

    fn create_resource(
        &self,
        heap_type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE,
        desc: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, D3D12VideoError> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: the descriptors outlive the call, which writes `resource`.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                desc,
                state,
                None,
                &mut resource,
            )?;
        }
        resource.ok_or(D3D12VideoError::CallFailed(0))
    }

    // orders the next work on `queue` after every wgpu submission so far, which covers the reads of recycled textures
    fn wait_for_wgpu(&mut self, queue: &ID3D12CommandQueue) -> Result<(), D3D12VideoError> {
        self.wgpu_fence_value += 1;
        // SAFETY: both queues and the fence belong to `self.device`.
        unsafe {
            self.wgpu_queue
                .Signal(&self.wgpu_fence, self.wgpu_fence_value)?;
            queue.Wait(&self.wgpu_fence, self.wgpu_fence_value)?;
        }
        Ok(())
    }

    fn execute_and_wait(
        &mut self,
        queue: &ID3D12CommandQueue,
        list: ID3D12CommandList,
    ) -> Result<(), D3D12VideoError> {
        self.completion_fence_value += 1;
        // SAFETY: the list was closed by the caller, and the event outlives the wait.
        unsafe {
            queue.ExecuteCommandLists(&[Some(list)]);
            queue.Signal(&self.completion_fence, self.completion_fence_value)?;
            if self.completion_fence.GetCompletedValue() < self.completion_fence_value {
                self.completion_fence
                    .SetEventOnCompletion(self.completion_fence_value, self.fence_event.0)?;
                WaitForSingleObject(self.fence_event.0, INFINITE);
            }
        }
        Ok(())
    }

    fn copy_picture(
        &mut self,
        source: &ID3D12Resource,
        destination: &ID3D12Resource,
        picture: CropRectangle,
    ) -> Result<(), D3D12VideoError> {
        let queue = self.copy_queue.clone();
        self.wait_for_wgpu(&queue)?;
        // SAFETY: the allocator's previous list finished (`execute_and_wait`), the copy
        // locations borrow resources that outlive the recording, and copy queues
        // promote textures in COMMON to the copy states and decay them back after.
        unsafe {
            self.copy_allocator.Reset()?;
            self.copy_list.Reset(&self.copy_allocator, None)?;
            for (plane, region) in [(0, picture), (1, chroma_rectangle(picture))] {
                let bounds = D3D12_BOX {
                    left: region.left,
                    top: region.top,
                    front: 0,
                    right: region.left + region.width,
                    bottom: region.top + region.height,
                    back: 1,
                };
                self.copy_list.CopyTextureRegion(
                    &plane_location(destination, plane),
                    0,
                    0,
                    0,
                    &plane_location(source, plane),
                    Some(&bounds),
                );
            }
            self.copy_list.Close()?;
        }
        let list: ID3D12CommandList = self.copy_list.cast()?;
        self.execute_and_wait(&queue, list)
    }
}

// borrows `resource` without a reference count, so the location must not outlive it
fn plane_location(resource: &ID3D12Resource, plane: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a COM pointer copied without AddRef, never released, see the function comment.
        pResource: unsafe { core::mem::transmute_copy(resource) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: plane,
        },
    }
}

// borrows `resource` into a barrier without a reference count, so the barrier must not outlive it
fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a COM pointer copied without AddRef, never released, see the function comment.
                pResource: unsafe { core::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[derive(Debug)]
pub struct DecodeTexture {
    pub resource: ID3D12Resource,
    pub extent: (u32, u32),
    // wgpu-hal unloads d3d12.dll when its last device drops
    _wgpu_device: wgpu::Device,
}

#[derive(Debug)]
pub struct DecodedPicture {
    pub meta: PictureMeta,
    pub texture: Arc<DecodeTexture>,
}

fn decode_texture(
    device: &D3D12DecodeDevice,
    extent: (u32, u32),
) -> Result<DecodeTexture, D3D12VideoError> {
    Ok(DecodeTexture {
        resource: device.create_nv12_texture(extent)?,
        extent,
        _wgpu_device: device.gpu.device.clone(),
    })
}

pub struct D3D12H264Decoder {
    decoder: ID3D12VideoDecoder,
    heap: ID3D12VideoDecoderHeap,
    texture_extent: (u32, u32),
    picture: CropRectangle,
    cropped_pool: Vec<Arc<DecodeTexture>>,
    session: H264DecodeSession,
    slot_textures: Vec<Option<Arc<DecodeTexture>>>,
    pool: Vec<Arc<DecodeTexture>>,
    bitstream: Option<(ID3D12Resource, usize)>,
    // wgpu-hal unloads d3d12.dll when its last device drops
    device: D3D12DecodeDevice,
}

impl core::fmt::Debug for D3D12H264Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("D3D12H264Decoder")
            .field("texture_extent", &self.texture_extent)
            .field("pool", &self.pool.len())
            .finish_non_exhaustive()
    }
}

impl D3D12H264Decoder {
    pub fn new(device: D3D12DecodeDevice, sps: &H264Sps) -> Result<Self, D3D12VideoError> {
        // D3D12_VIDEO_DECODE_PROFILE_H264 decodes 8-bit 4:2:0 only
        if sps.chroma_format_idc != 1 || sps.bit_depth_luma_minus8 != 0 {
            return Err(D3D12VideoError::StreamRefused(
                "H.264 other than 8-bit 4:2:0",
            ));
        }
        let (width, height) = sps.coded_extent();
        let crop = h264_visible_picture(sps)?;
        let texture_extent = device.h264_texture_extent(width, height)?;
        let slots = h264_dpb_slots(sps);
        let decoder_desc = D3D12_VIDEO_DECODER_DESC {
            NodeMask: 0,
            Configuration: H264_CONFIGURATION,
        };
        let heap_desc = D3D12_VIDEO_DECODER_HEAP_DESC {
            NodeMask: 0,
            Configuration: H264_CONFIGURATION,
            DecodeWidth: texture_extent.0,
            DecodeHeight: texture_extent.1,
            Format: DXGI_FORMAT_NV12,
            FrameRate: SUPPORT_QUERY_FRAME_RATE,
            BitRate: 0,
            MaxDecodePictureBufferCount: slots as u32,
        };
        // SAFETY: the descriptors outlive the calls.
        let (decoder, heap) = unsafe {
            let decoder: ID3D12VideoDecoder =
                device.video_device.CreateVideoDecoder(&decoder_desc)?;
            let heap: ID3D12VideoDecoderHeap =
                device.video_device.CreateVideoDecoderHeap(&heap_desc)?;
            (decoder, heap)
        };
        Ok(Self {
            device,
            decoder,
            heap,
            texture_extent,
            picture: crop,
            cropped_pool: Vec::new(),
            session: H264DecodeSession::new(sps),
            slot_textures: alloc::vec![None; slots],
            pool: Vec::new(),
            bitstream: None,
        })
    }

    pub fn reset(&mut self) {
        self.session.reset();
        self.slot_textures.fill(None);
    }

    pub fn gpu_context(&self) -> &GpuContext {
        self.device.gpu_context()
    }

    pub fn picture_extent(&self) -> (u32, u32) {
        (self.picture.width, self.picture.height)
    }

    pub fn wgpu_texture(&self, texture: &DecodeTexture) -> wgpu::Texture {
        let (width, height) = texture.extent;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::NV12;
        // SAFETY: the resource is an NV12 texture of `size` on this device, decoded
        // and back in COMMON state, which `UNINITIALIZED` maps to on D3D12.
        unsafe {
            let hal_texture = wgpu_hal::dx12::Device::texture_from_raw(
                texture.resource.clone(),
                format,
                wgpu::TextureDimension::D2,
                size,
                1,
                1,
            );
            self.gpu_context()
                .device
                .create_texture_from_hal::<wgpu_hal::api::Dx12>(
                    hal_texture,
                    &wgpu::TextureDescriptor {
                        label: Some("d3d12-decode-nv12"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                    wgpu::TextureUses::UNINITIALIZED,
                )
        }
    }

    // tight NV12, luma rows then interleaved chroma rows
    pub fn read_nv12(&self, texture: &DecodeTexture) -> Result<Vec<u8>, D3D12VideoError> {
        let wgpu_texture = self.wgpu_texture(texture);
        let gpu = self.gpu_context();
        let plane = |aspect, region, bytes_per_texel| {
            crate::gpu::read_texture_plane(
                &gpu.device,
                &gpu.queue,
                &wgpu_texture,
                aspect,
                region,
                bytes_per_texel,
            )
            .map_err(|_| D3D12VideoError::DeviceLost)
        };
        let mut bytes = plane(wgpu::TextureAspect::Plane0, self.picture, 1)?;
        bytes.extend(plane(
            wgpu::TextureAspect::Plane1,
            chroma_rectangle(self.picture),
            2,
        )?);
        Ok(bytes)
    }

    pub fn picture_texture(
        &mut self,
        decoded: Arc<DecodeTexture>,
    ) -> Result<WgpuNv12Texture, D3D12VideoError> {
        let gpu = self.gpu_context().clone();
        if (self.picture.left, self.picture.top) == (0, 0) {
            let texture = self.wgpu_texture(&decoded);
            return Ok(WgpuNv12Texture::with_recycle(
                gpu.device,
                gpu.queue,
                texture,
                Box::new(decoded),
            ));
        }
        // wgpu NV12 textures must have an even size
        let cropped_extent = (
            self.picture.width.next_multiple_of(2),
            self.picture.height.next_multiple_of(2),
        );
        let cropped = take_from_pool(&mut self.cropped_pool, || {
            decode_texture(&self.device, cropped_extent)
        })?;
        self.device
            .copy_picture(&decoded.resource, &cropped.resource, self.picture)?;
        let texture = self.wgpu_texture(&cropped);
        Ok(WgpuNv12Texture::with_recycle(
            gpu.device,
            gpu.queue,
            texture,
            Box::new(cropped),
        ))
    }

    pub fn decode_access_unit(
        &mut self,
        au: &[u8],
    ) -> Result<Vec<DecodedPicture>, D3D12VideoError> {
        let pictures = self.session.split_pictures(au)?;
        pictures
            .iter()
            .map(|picture| self.decode_picture(picture))
            .collect()
    }

    fn decode_picture(
        &mut self,
        picture: &H264Picture<'_>,
    ) -> Result<DecodedPicture, D3D12VideoError> {
        let slots = self.session.begin_picture(picture)?;
        let output = self.take_texture()?;
        self.slot_textures[slots.target] = Some(output.clone());

        let parameters = h264_picture_parameters(&DxvaH264Picture {
            sps: &picture.sps,
            pps: &picture.pps,
            header: &picture.header,
            order: &slots.order,
            target_slot: slots.target,
            references: &slots.references,
        });
        self.submit(&parameters, &picture.slices, &output.resource)?;
        let meta = self.session.end_picture(picture, &slots)?;
        let held = self.session.held_slots();
        for (slot, texture) in self.slot_textures.iter_mut().enumerate() {
            if !held.contains(&slot) {
                *texture = None;
            }
        }
        Ok(DecodedPicture {
            meta,
            texture: output,
        })
    }

    fn take_texture(&mut self) -> Result<Arc<DecodeTexture>, D3D12VideoError> {
        take_from_pool(&mut self.pool, || {
            decode_texture(&self.device, self.texture_extent)
        })
    }

    fn upload_bitstream(
        &mut self,
        slices: &[&[u8]],
    ) -> Result<(ID3D12Resource, usize, Vec<DxvaSliceH264Short>), D3D12VideoError> {
        let mut bytes = Vec::new();
        let mut slice_controls = Vec::with_capacity(slices.len());
        for slice in slices {
            let location = bytes.len();
            bytes.extend_from_slice(&ANNEX_B_START_CODE);
            bytes.extend_from_slice(slice);
            slice_controls.push(DxvaSliceH264Short {
                bitstream_data_location: location as u32,
                slice_bytes_in_buffer: (bytes.len() - location) as u32,
                bad_slice_chopping: 0,
            });
        }
        let length = bytes.len();
        bytes.resize(length.next_multiple_of(BITSTREAM_ALIGNMENT), 0);
        let fits = self
            .bitstream
            .as_ref()
            .is_some_and(|(_, capacity)| *capacity >= bytes.len());
        if !fits {
            let buffer = self.device.create_upload_buffer(bytes.len())?;
            self.bitstream = Some((buffer, bytes.len()));
        }
        let (buffer, _) = self
            .bitstream
            .as_ref()
            .expect("bitstream buffer allocated above");
        let mut mapped: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: the upload buffer holds at least `bytes.len()` bytes, the previous
        // decode finished before this one started, and the mapping is released here.
        unsafe {
            buffer.Map(0, None, Some(&mut mapped))?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            buffer.Unmap(0, None);
        }
        Ok((buffer.clone(), length, slice_controls))
    }

    fn submit(
        &mut self,
        parameters: &crate::dxva::DxvaPictureParametersH264,
        slices: &[&[u8]],
        output: &ID3D12Resource,
    ) -> Result<(), D3D12VideoError> {
        let (bitstream, bitstream_length, slice_controls) = self.upload_bitstream(slices)?;
        let scaling_matrices = h264_flat_scaling_matrices();
        // slots without a picture, such as frames a frame_num gap inferred, point at the output texture
        let mut reference_textures: Vec<Option<ID3D12Resource>> = self
            .slot_textures
            .iter()
            .map(|texture| {
                Some(
                    texture
                        .as_ref()
                        .map_or_else(|| output.clone(), |texture| texture.resource.clone()),
                )
            })
            .collect();
        let mut reference_subresources = alloc::vec![0u32; reference_textures.len()];
        let mut distinct_references: Vec<&ID3D12Resource> = Vec::new();
        for texture in self.slot_textures.iter().flatten() {
            let resource = &texture.resource;
            if resource != output && !distinct_references.contains(&resource) {
                distinct_references.push(resource);
            }
        }

        let mut frame_arguments: [D3D12_VIDEO_DECODE_FRAME_ARGUMENT; 10] = Default::default();
        frame_arguments[0] = D3D12_VIDEO_DECODE_FRAME_ARGUMENT {
            Type: D3D12_VIDEO_DECODE_ARGUMENT_TYPE_PICTURE_PARAMETERS,
            Size: size_of_val(parameters) as u32,
            pData: (parameters as *const crate::dxva::DxvaPictureParametersH264)
                .cast_mut()
                .cast(),
        };
        frame_arguments[1] = D3D12_VIDEO_DECODE_FRAME_ARGUMENT {
            Type: D3D12_VIDEO_DECODE_ARGUMENT_TYPE_INVERSE_QUANTIZATION_MATRIX,
            Size: size_of_val(&scaling_matrices) as u32,
            pData: (&scaling_matrices as *const crate::dxva::DxvaScalingMatricesH264)
                .cast_mut()
                .cast(),
        };
        frame_arguments[2] = D3D12_VIDEO_DECODE_FRAME_ARGUMENT {
            Type: D3D12_VIDEO_DECODE_ARGUMENT_TYPE_SLICE_CONTROL,
            Size: (size_of::<DxvaSliceH264Short>() * slice_controls.len()) as u32,
            pData: slice_controls.as_ptr().cast_mut().cast(),
        };
        let input = D3D12_VIDEO_DECODE_INPUT_STREAM_ARGUMENTS {
            NumFrameArguments: 3,
            FrameArguments: frame_arguments,
            ReferenceFrames: D3D12_VIDEO_DECODE_REFERENCE_FRAMES {
                NumTexture2Ds: reference_textures.len() as u32,
                ppTexture2Ds: reference_textures.as_mut_ptr(),
                pSubresources: reference_subresources.as_mut_ptr(),
                ppHeaps: core::ptr::null_mut(),
            },
            CompressedBitstream: D3D12_VIDEO_DECODE_COMPRESSED_BITSTREAM {
                pBuffer: ManuallyDrop::new(Some(bitstream)),
                Offset: 0,
                Size: bitstream_length as u64,
            },
            pHeap: ManuallyDrop::new(Some(self.heap.clone())),
        };
        let output_arguments = D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS {
            pOutputTexture2D: ManuallyDrop::new(Some(output.clone())),
            OutputSubresource: 0,
            ..Default::default()
        };

        let mut before = alloc::vec![transition(
            output,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE
        )];
        before.extend(distinct_references.iter().map(|reference| {
            transition(
                reference,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_DECODE_READ,
            )
        }));
        let after: Vec<D3D12_RESOURCE_BARRIER> = before
            .iter()
            .map(|barrier| {
                // SAFETY: every barrier built above is a transition.
                let transition_barrier = unsafe { &barrier.Anonymous.Transition };
                D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                            // SAFETY: the same borrowed pointer as the barrier it reverses.
                            pResource: unsafe {
                                core::mem::transmute_copy(&transition_barrier.pResource)
                            },
                            Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            StateBefore: transition_barrier.StateAfter,
                            StateAfter: transition_barrier.StateBefore,
                        }),
                    },
                }
            })
            .collect();

        let decode_queue = self.device.decode_queue.clone();
        self.device.wait_for_wgpu(&decode_queue)?;
        // SAFETY: the allocator's previous list finished (`execute_and_wait`), and
        // every pointer in the arguments outlives the recording and the wait.
        let recorded = unsafe {
            self.device.allocator.Reset()?;
            self.device.command_list.Reset(&self.device.allocator)?;
            self.device.command_list.ResourceBarrier(&before);
            self.device
                .command_list
                .DecodeFrame(&self.decoder, &output_arguments, &input);
            self.device.command_list.ResourceBarrier(&after);
            self.device.command_list.Close()
        };
        let result = recorded.map_err(D3D12VideoError::from).and_then(|()| {
            let list: ID3D12CommandList = self.device.command_list.cast()?;
            self.device.execute_and_wait(&decode_queue, list)
        });
        // release the AddRefs the argument structs took
        drop(ManuallyDrop::into_inner(input.CompressedBitstream.pBuffer));
        drop(ManuallyDrop::into_inner(input.pHeap));
        drop(ManuallyDrop::into_inner(output_arguments.pOutputTexture2D));
        drop(reference_textures);
        result
    }
}

fn decode_refused(error: D3D12VideoError) -> G2gError {
    let target = Target::category(D3D12Backend::LOG_CATEGORY);
    match error {
        D3D12VideoError::StreamRefused(reason) => {
            g2g_error!(target, "cannot decode this stream: {reason}")
        }
        D3D12VideoError::DecodeUnsupported(missing) => {
            g2g_error!(target, "the GPU has no {missing}")
        }
        D3D12VideoError::NoD3D12Adapter => g2g_error!(
            target,
            "no D3D12 GPU with NV12 textures was found, which d3d12h264dec needs"
        ),
        other => g2g_error!(target, "the GPU refused to decode this stream: {other:?}"),
    }
    G2gError::CapsMismatch
}

// the coded size of a 1080p H.264 picture
const SUPPORT_PROBE_CODED_EXTENT: (u32, u32) = (1920, 1088);

#[derive(Debug)]
pub struct D3D12Backend {
    device: Option<D3D12DecodeDevice>,
    decoder: Option<D3D12H264Decoder>,
}

impl D3D12Backend {
    fn gpu_context(&self) -> Option<&GpuContext> {
        match (&self.decoder, &self.device) {
            (Some(decoder), _) => Some(decoder.gpu_context()),
            (None, Some(device)) => Some(device.gpu_context()),
            (None, None) => None,
        }
    }

    fn session(&mut self) -> &mut D3D12H264Decoder {
        self.decoder
            .as_mut()
            .expect("H264HardwareDec decodes only once a session has started")
    }
}

impl H264HardwareDecoder for D3D12Backend {
    type Picture = Arc<DecodeTexture>;
    type Error = D3D12VideoError;
    const LOG_CATEGORY: &'static str = "D3D12H264Dec";
    const OUTPUT_DOMAINS: DomainSet =
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System);

    fn metadata() -> ElementMetadata {
        ElementMetadata::new(
            "Direct3D12 H.264 decoder",
            "Codec/Decoder/Video/Hardware",
            "Decodes H.264 on the GPU with the D3D12 video decode API",
            crate::AARON_BOXER,
        )
    }

    fn decodes_h264() -> bool {
        static DECODES_H264: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DECODES_H264.get_or_init(|| {
            block_on(open_d3d12_decode_device())
                .and_then(|device| {
                    let (width, height) = SUPPORT_PROBE_CODED_EXTENT;
                    device.h264_texture_extent(width, height)
                })
                .is_ok()
        })
    }

    fn open() -> Result<Self, D3D12VideoError> {
        let device = block_on(open_d3d12_decode_device())?;
        crate::gpu::publish_producer_context(device.gpu_context().clone());
        Ok(Self {
            device: Some(device),
            decoder: None,
        })
    }

    fn start_session(&mut self, sps: &H264Sps) -> Result<(), D3D12VideoError> {
        if self.decoder.is_some() {
            return Ok(());
        }
        // a failed session start drops the device it was given
        let device = self.device.take().ok_or(D3D12VideoError::StreamRefused(
            "an earlier SPS failed to start a decode session",
        ))?;
        self.decoder = Some(D3D12H264Decoder::new(device, sps)?);
        Ok(())
    }

    fn has_session(&self) -> bool {
        self.decoder.is_some()
    }

    fn decode_access_unit(
        &mut self,
        access_unit: &[u8],
    ) -> Result<Vec<(PictureMeta, Arc<DecodeTexture>)>, D3D12VideoError> {
        Ok(self
            .session()
            .decode_access_unit(access_unit)?
            .into_iter()
            .map(|picture| (picture.meta, picture.texture))
            .collect())
    }

    fn reset(&mut self) {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.reset();
        }
    }

    fn picture_extent(&self) -> (u32, u32) {
        self.decoder
            .as_ref()
            .expect("H264HardwareDec emits pictures only once a session has started")
            .picture_extent()
    }

    fn output(
        &mut self,
        picture: Arc<DecodeTexture>,
        domain: MemoryDomainKind,
    ) -> Result<MemoryDomain, D3D12VideoError> {
        let decoder = self.session();
        let (width, height) = decoder.picture_extent();
        if domain == MemoryDomainKind::WgpuTexture {
            // the pool reuses a texture only once this frame drops its reference
            let keep = decoder.picture_texture(picture)?;
            return Ok(MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                width,
                height,
                Arc::new(keep),
            )));
        }
        let bytes = decoder.read_nv12(&picture)?;
        Ok(MemoryDomain::System(SystemSlice::from_boxed(
            bytes.into_boxed_slice(),
        )))
    }

    fn stream_refusal(error: &D3D12VideoError) -> Option<&'static str> {
        match error {
            D3D12VideoError::StreamRefused(reason) => Some(reason),
            _ => None,
        }
    }

    fn refused(error: D3D12VideoError) -> G2gError {
        decode_refused(error)
    }
}

pub type D3D12H264Dec = H264HardwareDec<D3D12Backend>;

impl D3D12H264Dec {
    pub fn gpu_context(&self) -> Option<&GpuContext> {
        self.decoder().and_then(D3D12Backend::gpu_context)
    }
}
