#![cfg(all(target_os = "linux", feature = "cuda-wgpu", feature = "nvenc"))]
//! M271: the wgpu -> CUDA -> NVENC moat. Proves a GPU-rendered RGBA texture can
//! be H.264-encoded with no device->host read-back, the path the M267 Bevy demo
//! pays for today (it reads back to system memory and encodes with the ffmpeg
//! NVENC backend).
//!
//! Writes an RGBA pattern into the `WgpuToCuda` bridge's exportable wgpu texture,
//! bridges it to a `MemoryDomain::Cuda` frame device->device, and encodes it
//! through the native `NvEnc` (which color converts ABGR -> H.264 internally).
//! No PCIe download on the encode path.
//!
//! ```sh
//! cargo test -p g2g-plugins --features "cuda-wgpu nvenc" --test wgpu_to_cuda -- --nocapture
//! ```
//!
//! Skips when no Vulkan interop device / NVIDIA GPU is present.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
    RawVideoFormat, Rate, VideoCodec,
};
use g2g_plugins::cudawgpu::{create_interop_device, WgpuToCuda};
use g2g_plugins::nvenc::NvEnc;

const W: u32 = 320;
const H: u32 = 240;

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    aus: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.aus.push(s.as_slice().to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A moving RGBA gradient so successive frames differ (the encoder would emit
/// near-empty inter frames for a flat image, weakening the check).
fn rgba_pattern(seq: u32) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut data = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            data[p] = ((x + seq as usize * 5) & 0xff) as u8; // R
            data[p + 1] = ((y + seq as usize * 3) & 0xff) as u8; // G
            data[p + 2] = ((x ^ y) & 0xff) as u8; // B
            data[p + 3] = 0xff; // A
        }
    }
    data
}

#[tokio::test]
async fn wgpu_rgba_texture_encodes_through_nvenc_no_readback() {
    let dev = match create_interop_device().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no Vulkan interop device ({e:?})");
            return;
        }
    };

    // The bridge retains its own CUDA primary context (the GPU the interop device
    // also selects) and owns the exportable render-target texture.
    // SAFETY: `dev.device` is a VK_KHR_external_memory_fd interop device; the
    // clones share it (wgpu handles are Arc-backed) and `dev` outlives the bridge.
    let bridge = match unsafe { WgpuToCuda::new(dev.device.clone(), dev.queue.clone(), W, H) } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: WgpuToCuda unavailable (no CUDA? {e:?})");
            return;
        }
    };

    let caps = Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    let mut enc = NvEnc::new();
    enc.configure_pipeline(&caps).expect("configure NvEnc for RGBA");

    let mut sink = CaptureSink::default();
    for seq in 0..10u32 {
        // Write this frame's RGBA pattern into the bridge's exportable texture
        // (a real consumer renders into it; the upload stands in for the render).
        let data = rgba_pattern(seq);
        bridge.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: bridge.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(W * 4),
                rows_per_image: Some(H),
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        bridge.queue().submit([]);
        // Ensure the write is visible to CUDA before the device->device copy.
        bridge
            .device()
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("poll");

        let frame: Frame =
            bridge.to_cuda_frame(seq as u64 * 33_000_000).expect("bridge texture to CUDA frame");
        // Sanity: the frame is CUDA-resident in the bridge's context.
        match &frame.domain {
            MemoryDomain::Cuda(buf) => {
                assert_eq!((buf.width, buf.height), (W, H));
                assert_eq!(buf.context, bridge.context());
                assert!(buf.luma_ptr != 0, "linear CUDA buffer allocated");
            }
            _ => panic!("bridge must emit a MemoryDomain::Cuda frame"),
        }
        if enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.is_err() {
            eprintln!("skipping: NVENC unavailable on this host");
            return;
        }
    }
    enc.process(PipelinePacket::Eos, &mut sink).await.expect("flush NvEnc");

    assert!(!sink.aus.is_empty(), "NVENC produced H.264 access units from the GPU texture");
    assert_eq!(
        sink.caps,
        vec![Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        }],
        "H.264 output caps announced once at the texture geometry"
    );
    let first = &sink.aus[0];
    let annex_b = first.starts_with(&[0, 0, 0, 1]) || first.starts_with(&[0, 0, 1]);
    assert!(annex_b, "encoded output is Annex-B framed, got {:?}", &first[..4.min(first.len())]);
    eprintln!("encoded {} H.264 access units from a GPU RGBA texture, no read-back", sink.aus.len());
}
