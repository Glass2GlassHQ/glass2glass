//! M1157: `WgpuSink` presents the two-plane `TextureFormat::NV12` texture
//! `VulkanVideoDec` hands out when `NV12` is pinned on the `WgpuTexture` domain,
//! sampling the decoder's picture through its `Plane0` / `Plane1` views with no
//! copy. The picture it renders must match the same clip taken through the
//! packed-NV12 path (the decoder's system-memory output uploaded into the sink's
//! own R8Uint plane), which is the same colour math over the same samples.
//!
//! One sink, one negotiated set of NV12 caps, both layouts: which blit runs is
//! decided per frame from the texture. Runs on the RTX 3060; skips when the GPU
//! lacks the H.264 decode profile, a compute queue, or the wgpu NV12 feature.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video",
    feature = "wgpu-sink"
))]

use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::block_on;
use g2g_core::{AsyncElement, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome};
use g2g_plugins::gpu::{texture_layout, texture_of, WgpuTextureLayout};
use g2g_plugins::wgpusink::WgpuSink;

mod vulkan_nv12_common;
use vulkan_nv12_common::{
    decode, nv12_caps, reference_nv12, skip_reason, system_frame, CLIP_FRAMES, H, W,
};

/// Per-channel bound on the difference between the two-plane render and the
/// packed one. Both stages fetch the same texels and run the same matrix, and
/// only differ in that the plane views hand the shader `byte / 255` where the
/// packed R8Uint plane hands it the integer, so the render target can quantize
/// at most one code differently. Measured 0 on the RTX 3060.
const MAX_CHANNEL_DELTA: i32 = 1;
/// Bound on the mean absolute per-channel difference over a whole frame: at most
/// one channel in a hundred may sit one code out. A wrong matrix, a swapped
/// plane or a dropped range shift moves every pixel, so it cannot hide here.
const MAX_MEAN_DELTA: f64 = 0.01;

struct NullSink;
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Present one frame and read the offscreen RGBA target back.
fn present(sink: &mut WgpuSink, frame: Frame) -> Vec<u8> {
    block_on(sink.process(PipelinePacket::DataFrame(frame), &mut NullSink)).expect("present");
    sink.read_target().expect("read offscreen target")
}

/// Max and mean absolute per-channel difference between two RGBA targets,
/// ignoring alpha (both blits write it opaque).
fn compare(two_plane: &[u8], packed: &[u8]) -> (i32, f64) {
    assert_eq!(two_plane.len(), packed.len(), "same target size");
    let mut max = 0i32;
    let mut total = 0u64;
    let mut counted = 0u64;
    for (a, b) in two_plane
        .as_chunks::<4>()
        .0
        .iter()
        .zip(packed.as_chunks::<4>().0)
    {
        for channel in 0..3 {
            let delta = (a[channel] as i32 - b[channel] as i32).abs();
            max = max.max(delta);
            total += delta as u64;
            counted += 1;
        }
    }
    (max, total as f64 / counted as f64)
}

/// The darkest and brightest code in a render. A flat pair means a cleared
/// target rather than a picture.
fn code_span(target: &[u8]) -> (u8, u8) {
    (
        *target.iter().min().expect("non-empty target"),
        *target.iter().max().expect("non-empty target"),
    )
}

// One device-creating test in the file: libtest runs test functions on parallel
// threads, and two threads building a `wgpu::Instance` at once fault inside the
// Vulkan loader.
#[test]
fn two_plane_nv12_texture_presents_like_the_packed_path() {
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }

    // The packed reference: the same decoder's system-memory NV12 output, which
    // the sink uploads into its own R8Uint plane.
    let reference = reference_nv12();
    let (dec, frames) = decode(MemoryDomainKind::WgpuTexture, Some(nv12_caps()));
    assert_eq!(frames.len(), CLIP_FRAMES);

    // The sink shares the decoder's device, so the two-plane texture is presented
    // where it lies. Its target matches the picture, so both paths address the
    // same texels.
    let ctx = dec.gpu_context().expect("device open after configure");
    let mut sink = WgpuSink::offscreen(ctx, W, H);
    sink.configure_pipeline(&nv12_caps())
        .expect("sink configure");

    let mut worst = (0i32, 0.0f64);
    for (i, frame) in frames.into_iter().enumerate() {
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("frame {i}: expected a WgpuTexture frame");
        };
        let texture = texture_of(owned).expect("recover the texture");
        assert_eq!(
            texture_layout(texture),
            Some(WgpuTextureLayout::MultiplanarNv12),
            "frame {i}: pinning NV12 on the texture domain gives a two-plane texture"
        );
        assert_eq!((texture.width(), texture.height()), (W, H));

        let two_plane = present(&mut sink, frame);
        let packed = present(&mut sink, system_frame(&reference[i]));
        // Both renders would compare equal if the sink only ever cleared, so the
        // reference has to carry a picture of its own.
        let (dark, bright) = code_span(&packed);
        assert!(
            dark < bright,
            "frame {i}: the packed render is flat at {dark}, not a picture"
        );

        let (max, mean) = compare(&two_plane, &packed);
        assert!(
            max <= MAX_CHANNEL_DELTA,
            "frame {i}: two-plane render is {max} codes off the packed render"
        );
        assert!(
            mean <= MAX_MEAN_DELTA,
            "frame {i}: two-plane render averages {mean} codes off the packed render"
        );
        if max > worst.0 {
            worst = (max, mean);
        }
    }
    assert_eq!(sink.presented_count(), (CLIP_FRAMES * 2) as u64);
    eprintln!(
        "two-plane vs packed NV12 present: worst frame max {} codes, mean {:.3}",
        worst.0, worst.1
    );
}
