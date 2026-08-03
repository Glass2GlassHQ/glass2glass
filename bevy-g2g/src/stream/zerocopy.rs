//! The NVIDIA zero-copy encode path: Bevy renders on g2g's interop device,
//! the target texture is copied device->device into a CUDA surface
//! (`WgpuToCuda`) and encoded by the native `NvEnc`. Only H.264 access units
//! leave the GPU.

use std::sync::Arc;

use bevy::render::{
    renderer::{
        RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue, WgpuWrapper,
    },
    settings::{RenderCreation, RenderResources},
};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat,
};
use g2g_plugins::cudawgpu::{create_interop_device_full, WgpuToCuda};
use g2g_plugins::nvenc::NvEnc;

use super::StreamSettings;

/// Create g2g's interop device (Vulkan + VK_KHR_external_memory_fd, opened
/// with the adapter's full features so Bevy's renderer is happy on it) and
/// wrap it for `RenderCreation::Manual`. Bevy adopting it means every texture
/// it renders is exportable to CUDA on this exact device, the prerequisite
/// for the zero-copy bridge.
pub(super) fn interop_render_creation() -> Result<RenderCreation, G2gError> {
    let interop = pollster::block_on(create_interop_device_full())?;
    let resources = RenderResources(
        RenderDevice::from(interop.device.clone()),
        RenderQueue(Arc::new(WgpuWrapper::new(interop.queue.clone()))),
        RenderAdapterInfo(WgpuWrapper::new(interop.adapter.get_info())),
        RenderAdapter(Arc::new(WgpuWrapper::new(interop.adapter.clone()))),
        RenderInstance(Arc::new(WgpuWrapper::new(interop.instance.clone()))),
    );
    // Bevy holds its own (reference-counted) clones now; drop our handle.
    drop(interop);
    Ok(RenderCreation::Manual(resources))
}

/// The render-world encode state: the NVENC encoder and the wgpu->CUDA
/// bridge, both living on Bevy's (= the interop) device, plus the running
/// frame index for presentation timestamps.
///
/// Field order matters for `Drop`: `nvenc` is declared first so it drops
/// first. NVENC's session lives in the CUDA primary context the `bridge`
/// retains, so the session must be destroyed before the bridge releases that
/// context, else teardown destroys a session on a freed context (an
/// intermittent exit segfault).
pub(super) struct EncodeState {
    nvenc: NvEnc,
    bridge: WgpuToCuda,
    fps: u32,
    keyframe_interval: u64,
    frame_no: u64,
}

impl EncodeState {
    /// Build the bridge + encoder on `device` (Bevy's interop device).
    pub(super) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        settings: &StreamSettings,
    ) -> Result<Self, G2gError> {
        // SAFETY: `device` is the VK_KHR_external_memory_fd interop device
        // created by `create_interop_device_full` and handed to Bevy, so the
        // bridge's exportable-image allocation and CUDA import are valid on it.
        let bridge = unsafe { WgpuToCuda::new(device, queue, settings.width, settings.height) }?;
        let mut nvenc = NvEnc::new().with_bitrate(settings.bitrate);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(settings.width),
            height: Dim::Fixed(settings.height),
            framerate: Rate::Fixed(settings.fps << 16),
        };
        AsyncElement::configure_pipeline(&mut nvenc, &caps)?;
        Ok(Self {
            nvenc,
            bridge,
            fps: settings.fps,
            keyframe_interval: settings.keyframe_interval.max(1) as u64,
            frame_no: 0,
        })
    }

    /// Copy `texture` (Bevy's just-rendered target) into the bridge's CUDA
    /// surface and encode it, returning any ready H.264 access units with
    /// their timestamps. No device->host copy.
    pub(super) fn encode(
        &mut self,
        texture: &wgpu::Texture,
    ) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let pts_ns = self.frame_no * 1_000_000_000 / self.fps as u64;
        if self.frame_no.is_multiple_of(self.keyframe_interval) {
            self.nvenc.force_keyframe();
        }
        self.frame_no += 1;
        self.bridge.ingest_texture(texture)?;
        let frame = self.bridge.to_cuda_frame(pts_ns)?;
        let mut cap = CaptureAus::default();
        // NVENC sync-mode encode; the capture sink resolves immediately, so
        // the block_on returns this frame's access unit without a reactor.
        let fut =
            AsyncElement::process(&mut self.nvenc, PipelinePacket::DataFrame(frame), &mut cap);
        pollster::block_on(fut)?;
        Ok(cap.aus)
    }
}

/// Render-world sink that captures `NvEnc`'s emitted H.264 access units
/// (System memory) and their timestamps.
#[derive(Default)]
struct CaptureAus {
    aus: Vec<(Vec<u8>, u64)>,
}

impl OutputSink for CaptureAus {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<PushOutcome, G2gError>> + 'a>>
    {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push((s.to_vec(), f.timing.pts_ns));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
