//! M1021: a display sink presents on the device the decoder already opened.
//!
//! `VulkanVideoDec` opens its own Vulkan device (video decode needs queues and
//! extensions wgpu never asks for), and a `wgpu::Texture` binds to no other
//! device, so a sink that opened its own cannot present the decoded frames at
//! all. The decoder publishes its context and the sink adopts it. This pins the
//! two halves a windowed run cannot assert headlessly: that the decoder publishes
//! a present-capable context, and that it keeps one device across the repeated
//! `configure_pipeline` a launch line always does (a second device would leave
//! the sink presenting from one nothing produces on).
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_core::{AsyncElement, Caps, Dim, Rate, VideoCodec};
use g2g_plugins::gpu::{present_on_producer_device, producer_context};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

/// Geometry a launch line starts from (a placeholder) and the real one that
/// arrives once the stream is read.
const PLACEHOLDER: (u32, u32) = (16, 16);
const REAL: (u32, u32) = (640, 480);

fn h264(geometry: (u32, u32)) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(geometry.0),
        height: Dim::Fixed(geometry.1),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[test]
fn the_decoder_publishes_one_device_for_the_sink_to_present_on() {
    match block_on(open_h264_decode_device()) {
        Ok(device) if !device.present_capable() => {
            eprintln!("skipping: this GPU's decode device has no swapchain support");
            return;
        }
        Ok(_) => {}
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("probe failed: {e:?}"),
    }

    let mut dec = VulkanVideoDec::new();
    dec.configure_pipeline(&h264(PLACEHOLDER))
        .expect("configure opens the decode device");
    let published = producer_context().expect("the decoder offers its device to a display sink");
    // Identity is read off the instance: `wgpu::Device` and `wgpu::Adapter`
    // compare by backend id, which two separately opened devices share, while
    // each opened device brings its own instance.
    let decoder_instance = dec
        .gpu_context()
        .expect("device open after configure")
        .instance;
    assert_eq!(
        published.instance, decoder_instance,
        "the offered device is the one the decoder decodes on"
    );

    // The real geometry arrives mid-stream and configures the element again. The
    // device does not depend on geometry, so the sink's adopted device stays the
    // one producing the textures.
    dec.configure_pipeline(&h264(REAL))
        .expect("configure again on the real caps");
    assert_eq!(
        dec.gpu_context().expect("device still open").instance,
        decoder_instance,
        "re-configuring keeps the open decode device"
    );
    assert_eq!(
        producer_context().expect("still offered").instance,
        decoder_instance,
        "and the sink is still pointed at it"
    );

    // A sink that cannot build its surface on the offered instance opens its own
    // device instead of failing.
    assert!(
        present_on_producer_device::<()>(|_instance| Err(()), REAL.0, REAL.1).is_none(),
        "an unusable surface falls back rather than adopting"
    );
}
