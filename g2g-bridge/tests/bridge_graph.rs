//! `BridgeGraph`: an embedded g2g sub-graph driven from synchronous code, the
//! cross-thread push/pull path a GStreamer `chain` function uses (design/README.md).
//!
//! `default_registry` (and the bridge) are `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use g2g_bridge::{frame_bytes, BridgeError, BridgeGraph};

const CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=30/1";

// Opening wgpu devices concurrently crashes some drivers.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// The buffers an embedder pushes flow through the sub-graph and come back out,
/// with timestamps preserved, across the thread boundary: the graph runs on its
/// own OS thread while the test pushes and drains from this one.
#[test]
fn round_trips_buffers_across_the_thread_boundary() {
    let bridge = BridgeGraph::new("identity", CAPS).expect("appsrc ! identity ! appsink builds");

    // Push three distinct 2x2 RGBA buffers from this thread.
    for i in 0u8..3 {
        assert!(
            bridge.push(&[i; 16], u64::from(i) * 1_000),
            "feed accepted buffer {i}"
        );
    }
    bridge.end_of_stream();

    // Drain them back on this thread; the graph produced them on its own.
    let mut out = Vec::new();
    while let Some(frame) = bridge.pull_blocking() {
        let bytes = frame_bytes(&frame).expect("system-memory frame").to_vec();
        out.push((bytes, frame.timing.pts_ns));
    }

    assert_eq!(out.len(), 3, "every pushed buffer came back");
    assert_eq!(
        out[0].0,
        vec![0u8; 16],
        "bytes round-tripped through the sub-graph"
    );
    assert_eq!(out[1].1, 1_000, "presentation timestamp carried through");

    let stats = bridge.finish().expect("clean shutdown");
    assert_eq!(stats.frames_consumed, 3, "sink consumed every frame");
}

/// A real caps-driven transform (not just a pass-through) runs inside the
/// sub-graph and its output reaches the drain. This exercises the path where the
/// runner cascades caps a second time through the embedded graph (a format/size
/// transform), which must not strand the frame at the `appsink`.
#[test]
fn caps_driven_transform_delivers_output() {
    let bridge = BridgeGraph::new("videoconvert", CAPS).expect("appsrc ! videoconvert ! appsink");
    assert!(bridge.push(&[42u8; 16], 0));
    bridge.end_of_stream();

    let mut frames = 0;
    while let Some(frame) = bridge.pull_blocking() {
        assert!(frame_bytes(&frame).is_some(), "system-memory output");
        frames += 1;
    }
    assert_eq!(frames, 1, "the transformed frame reached the drain");
}

/// A rescaling fragment changes the buffer size: `with_output_caps` pins the
/// sub-graph's output, and the drained frame is the smaller output size, not the
/// input size. (The GStreamer shell relies on this to allocate output buffers.)
#[test]
fn rescaling_fragment_changes_output_size() {
    let in_caps = "video/x-raw,format=RGBA,width=8,height=8,framerate=30/1"; // 8*8*4 = 256
    let out_caps = "video/x-raw,format=RGBA,width=4,height=4,framerate=30/1"; // 4*4*4 = 64
    let bridge =
        BridgeGraph::with_output_caps("videoscale", in_caps, out_caps).expect("scale sub-graph");
    assert!(bridge.push(&[9u8; 256], 0));
    bridge.end_of_stream();

    let mut out_lens = Vec::new();
    while let Some(frame) = bridge.pull_blocking() {
        out_lens.push(frame_bytes(&frame).expect("system memory").len());
    }
    assert_eq!(
        out_lens,
        vec![64],
        "the downscaled frame is 4x4 RGBA, not the 8x8 input"
    );
}

/// An imported DMABUF frame is ingested by the embedded graph and travels
/// through it in the `DmaBuf` memory domain (no copy to system memory). This is
/// the zero-copy import foundation; a fragment that *consumes* dma-buf (a GPU
/// import) is separate future work, so this uses an `identity` passthrough and a
/// placeholder fd, asserting the domain rather than reading pixels.
///
/// dma-buf is a Unix fd concept (`std::os::fd`, `/dev/null`); the Windows CI
/// build has neither, so gate it to Unix.
#[cfg(unix)]
#[test]
fn imports_a_dmabuf_frame_zero_copy() {
    use g2g_core::memory::{MemoryDomain, OwnedDmaBuf};
    use std::os::fd::IntoRawFd;

    let bridge = BridgeGraph::new("identity", CAPS).expect("builds");

    // A real, closeable fd stands in for a dma-buf descriptor; `identity` does
    // not read it, so no mapping is needed. `OwnedDmaBuf` closes it on drop.
    let fd = std::fs::File::open("/dev/null")
        .expect("open /dev/null")
        .into_raw_fd();
    // SAFETY: `fd` is a fresh fd this test solely owns; ownership transfers to
    // the OwnedDmaBuf, which closes it once.
    let dmabuf = unsafe {
        OwnedDmaBuf::from_raw(fd, /*stride*/ 8, /*offset*/ 0)
    };
    assert!(
        bridge.push_dmabuf(dmabuf, 0),
        "feed accepted the dma-buf frame"
    );
    bridge.end_of_stream();

    let mut domains = Vec::new();
    while let Some(frame) = bridge.pull_blocking() {
        domains.push(matches!(frame.domain, MemoryDomain::DmaBuf(_)));
    }
    assert_eq!(
        domains,
        vec![true],
        "the frame stayed in the DmaBuf domain end to end"
    );
}

/// A fragment that names an element g2g lacks fails construction with a parse
/// error (carrying the launch diagnostics / porting hint), not a panic or a hung
/// thread. This is the feedback an app developer gets while porting.
#[test]
fn unknown_element_fails_to_build() {
    // A name no feature ever registers ("x264enc" is a real alias under the
    // ffmpeg feature).
    let err = BridgeGraph::new("nosuchelement", CAPS).expect_err("element unknown to g2g");
    assert!(
        matches!(err, BridgeError::Parse(_)),
        "surfaced as a parse error: {err}"
    );
}

/// Dropping a `BridgeGraph` without draining must not deadlock: releasing the
/// pull handle lets the sink discard undeliverable frames so the run thread can
/// reach EOS and be joined. (If this regressed, the test would hang.)
#[test]
fn drop_without_draining_does_not_deadlock() {
    let bridge = BridgeGraph::new("identity", CAPS).expect("builds");
    for i in 0u8..3 {
        bridge.push(&[i; 16], 0);
    }
    bridge.end_of_stream();
    drop(bridge); // joins the run thread in Drop; must return.
}

#[test]
fn gpu_fragment_comes_back_as_system_bytes() {
    use g2g_plugins::wgpu;

    // 37 RGBA pixels are 148 bytes, short of the 256-byte row a texture copy pads to.
    const WIDTH: usize = 37;
    const HEIGHT: usize = 5;
    const RGBA_BYTES_PER_PIXEL: usize = 4;
    const OPAQUE: u8 = 255;

    let _gpu = GPU_LOCK.lock().unwrap();
    let has_adapter = g2g_core::runtime::block_on(
        wgpu::Instance::default().request_adapter(&wgpu::RequestAdapterOptions::default()),
    )
    .is_ok();
    if !has_adapter {
        eprintln!("no wgpu adapter; skipping");
        return;
    }

    let caps = format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1");
    let fragment = format!("wgpucompositor width={WIDTH} height={HEIGHT} gpu-output=true");
    let bridge = BridgeGraph::new(&fragment, &caps).expect("builds");

    // Opaque pixels come through compositing unchanged.
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * RGBA_BYTES_PER_PIXEL)
        .map(
            |i| match i % RGBA_BYTES_PER_PIXEL == RGBA_BYTES_PER_PIXEL - 1 {
                true => OPAQUE,
                false => (i.wrapping_mul(7).wrapping_add(3)) as u8,
            },
        )
        .collect();
    assert!(bridge.push(&pixels, 0));
    bridge.end_of_stream();

    let frame = bridge.pull_blocking().expect("a frame came back");
    assert_eq!(
        frame_bytes(&frame).expect("system-memory frame"),
        pixels.as_slice()
    );
}

// compositor labels its output 30/1, which the 1/1 caps pinned on the appsink reject
#[test]
fn failed_negotiation_ends_the_drain() {
    const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
    let caps = "video/x-raw,format=RGBA,width=2,height=2,framerate=1/1";
    let bridge = Arc::new(BridgeGraph::new("compositor width=2 height=2", caps).expect("builds"));
    assert!(bridge.push(&[0u8; 16], 0));

    let (ended_sender, ended_receiver) = mpsc::channel();
    let drain = Arc::clone(&bridge);
    let drain_thread = std::thread::spawn(move || {
        let _ = ended_sender.send(drain.pull_blocking().is_none());
    });
    let ended = ended_receiver
        .recv_timeout(DRAIN_TIMEOUT)
        .expect("pull_blocking returned instead of waiting forever");
    assert!(ended, "a graph that did not negotiate produces no frame");
    drain_thread.join().expect("drain thread");

    let bridge = Arc::into_inner(bridge).expect("the drain thread released its handle");
    assert!(bridge.finish().is_err(), "the run reports the failure");
}

// GStreamer exported this from a GL texture here, the suffix is a tiled NVIDIA layout.
const TILED_DRM_FORMAT: &str = "AB24:0x020000001056bb03";

#[test]
fn tiled_dmabuf_caps_fail_to_build() {
    let caps = format!(
        "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format={TILED_DRM_FORMAT},width=2,height=2,framerate=30/1"
    );
    let err = BridgeGraph::new("dmabuftowgpu", &caps).expect_err("a tiled dma-buf");
    assert!(
        matches!(err, BridgeError::UnsupportedDmaBufCaps(_)),
        "surfaced as unsupported dma-buf caps: {err}"
    );
}

#[cfg(target_os = "linux")]
mod dmabuf {
    use g2g_core::memory::{MemoryDomain, OwnedDmaBuf};
    use g2g_core::{
        AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, OutputSink, PipelinePacket,
        PushOutcome, Rate, RawVideoFormat,
    };
    use g2g_plugins::wgpu;
    use g2g_plugins::wgpudmabuf::WgpuToDmaBuf;

    use super::*;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 16;
    const RGBA_BYTES_PER_PIXEL: usize = 4;
    // GStreamer's drm-format for RGBA
    const RGBA_DRM_FORMAT: &str = "AB24";

    #[derive(Default)]
    struct Capture {
        frame: Option<Frame>,
    }

    impl OutputSink for Capture {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                self.frame = Some(frame);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    async fn export_dmabuf(pixels: &[u8]) -> Option<OwnedDmaBuf> {
        let mut export = WgpuToDmaBuf::new();
        let (device, queue) = export.gpu().await.ok()?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bridge-dmabuf-source"),
            size: pixels.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(pixels);
        buffer.unmap();
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        export.configure_pipeline(&caps).expect("configures");
        let domain = MemoryDomain::WgpuBuffer(WgpuToDmaBuf::wrap_buffer(
            &device,
            &queue,
            buffer,
            pixels.len(),
        ));
        let mut capture = Capture::default();
        export
            .process(
                PipelinePacket::DataFrame(Frame::new(domain, FrameTiming::default(), 0)),
                &mut capture,
            )
            .await
            .expect("exports");
        match capture.frame.expect("the export pushed a frame").domain {
            MemoryDomain::DmaBuf(dmabuf) => Some(dmabuf),
            other => panic!("the export emitted {:?}", other.kind()),
        }
    }

    #[test]
    fn dmabuf_caps_fragment_round_trips_a_gpu_frame() {
        let _gpu = GPU_LOCK.lock().unwrap();
        let dmabuf_caps = format!(
            "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format={RGBA_DRM_FORMAT},width={WIDTH},height={HEIGHT},framerate=30/1"
        );
        let system_caps =
            format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate=30/1");
        let pixels: Vec<u8> = (0..WIDTH as usize * HEIGHT as usize * RGBA_BYTES_PER_PIXEL)
            .map(|i| (i.wrapping_mul(7).wrapping_add(3)) as u8)
            .collect();
        let Some(dmabuf) = g2g_core::runtime::block_on(export_dmabuf(&pixels)) else {
            eprintln!("no Vulkan export device; skipping");
            return;
        };

        let bridge = BridgeGraph::new("dmabuftowgpu ! wgputodmabuf", &dmabuf_caps).expect("builds");
        assert!(bridge.push_dmabuf(dmabuf, 0));
        bridge.end_of_stream();
        let frame = bridge.pull_blocking().expect("a frame came back");
        let MemoryDomain::DmaBuf(exported) = frame.domain else {
            panic!("expected a dma-buf, got {:?}", frame.domain.kind());
        };
        bridge.finish().expect("clean shutdown");

        let readback = BridgeGraph::with_output_caps(
            "dmabuftowgpu ! wgpudownload",
            &dmabuf_caps,
            &system_caps,
        )
        .expect("builds");
        assert!(readback.push_dmabuf(exported, 0));
        readback.end_of_stream();
        let frame = readback.pull_blocking().expect("a frame came back");
        assert_eq!(
            frame_bytes(&frame).expect("system-memory frame"),
            pixels.as_slice()
        );
    }
}
