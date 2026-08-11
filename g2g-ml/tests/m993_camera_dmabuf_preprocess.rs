#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
//! M993: packed YUYV, the format a webcam exports, through the M990 dma-buf
//! handshake into `WgpuPreprocess`, and the live camera that produces it.
//!
//! Two halves. The synthetic cases export a YUYV dma-buf from the GPU (as M990
//! does for NV12) and check the YUYV compute shader against the host BT.601
//! reference, padded row stride included; they run wherever a GPU can export a
//! dma-buf. The live case is the real producer: `v4l2src io-mode=dmabuf` hands over
//! the driver's capture buffer, which is CPU-backed, so the import runs on an
//! integrated GPU (a discrete one refuses to bind such an fd) and the tensor is
//! compared against the *same* frame's bytes taken through the copy path, a live
//! camera never giving the same picture twice.
//!
//! ```sh
//! cargo test -p g2g-ml --features dmabuf-wgpu --test m993_camera_dmabuf_preprocess -- --nocapture
//! cargo test -p g2g-ml --features v4l2-dmabuf-wgpu --test m993_camera_dmabuf_preprocess \
//!     -- --ignored --nocapture
//! ```

#[cfg(feature = "v4l2-dmabuf-wgpu")]
use core::future::Future;
#[cfg(feature = "v4l2-dmabuf-wgpu")]
use core::pin::Pin;
use g2g_ml::wgpupreprocess::{gpu_available, yuyv_to_rgb_tensor};

include!("util/dmabuf_preprocess.rs");

/// Frame geometry the synthetic tensor is built at.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
/// Extra pixels per row for the padded-stride case (4 pixels = 8 bytes of YUYV).
const PIXEL_PADDING: u32 = 4;

fn yuyv_caps(w: u32, h: u32) -> Caps {
    raw_caps(RawVideoFormat::Yuyv, w, h)
}

/// A YUYV frame `stride_pixels * 2` bytes per row, of which only the first
/// `width * 2` are the picture: `Y0 Cb Y1 Cr` per pixel pair, the padding 0xFF so
/// the tensor shifts if the shader ever reads it. Luma differs within a pair and
/// chroma varies per pair, so a wrong byte order or a swapped Cb/Cr changes the
/// output.
fn padded_yuyv(stride_pixels: u32, width: u32, height: u32) -> Vec<u8> {
    let stride = stride_pixels as usize * 2;
    let (width, height) = (width as usize, height as usize);
    let mut out = vec![0xFFu8; stride * height];
    for y in 0..height {
        for pair in 0..width / 2 {
            let at = y * stride + pair * 4;
            out[at] = (16 + (pair * 26 + y * 29) % 200) as u8;
            out[at + 1] = ((pair * 7 + y * 11) % 256) as u8;
            out[at + 2] = (16 + (pair * 26 + y * 29 + 13) % 200) as u8;
            out[at + 3] = ((pair * 23 + y * 5) % 256) as u8;
        }
    }
    out
}

/// The tensor must match the host BT.601 reference for `yuyv` (tightly packed at
/// `width x height`) and be non-grayscale, proving the chroma bytes were read.
fn assert_matches_reference(got: &[f32], yuyv: &[u8], width: u32, height: u32) {
    let area = (width * height) as usize;
    let expect = yuyv_to_rgb_tensor(yuyv, width as usize, height as usize);
    assert_eq!(got.len(), 3 * area, "NCHW tensor length");
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "tensor[{i}] = {g}, reference {e}: imported pixels differ"
        );
    }
    assert!(
        (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3),
        "coloured chroma must break R == G, else the chroma bytes were not read"
    );
}

#[tokio::test]
async fn yuyv_dmabuf_import_matches_system_upload() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let caps = yuyv_caps(WIDTH, HEIGHT);
    let yuyv = padded_yuyv(WIDTH, WIDTH, HEIGHT);
    let Some(frame) = export_dmabuf(&yuyv, &caps).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), false).await
    else {
        return;
    };

    let got = frame_f32(&tensor);
    assert_matches_reference(&got, &yuyv, WIDTH, HEIGHT);
    assert_eq!(
        got,
        preprocess_system(&caps, yuyv).await,
        "an imported YUYV dma-buf and the uploaded bytes give the same tensor, bit for bit"
    );
    eprintln!(
        "PASS: YUYV dma-buf import == system upload, {} values",
        got.len()
    );
}

#[tokio::test]
async fn padded_stride_yuyv_dmabuf_needs_no_repack() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let stride_pixels = WIDTH + PIXEL_PADDING;
    let padded = padded_yuyv(stride_pixels, WIDTH, HEIGHT);
    // The export derives the row stride from the caps width, so a wider caps pads
    // every row; the preprocess still runs at the real WIDTH x HEIGHT.
    let Some(frame) = export_dmabuf(&padded, &yuyv_caps(stride_pixels, HEIGHT)).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let caps = yuyv_caps(WIDTH, HEIGHT);
    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), false).await
    else {
        return;
    };

    let got = frame_f32(&tensor);
    let tight = repack_tight(
        &padded,
        stride_pixels as usize * 2,
        WIDTH as usize * 2,
        HEIGHT as usize,
    );
    assert_matches_reference(&got, &tight, WIDTH, HEIGHT);
    assert_eq!(
        got,
        preprocess_system(&caps, tight).await,
        "a padded YUYV dma-buf gives the same tensor as the repacked pixels"
    );
    eprintln!(
        "PASS: YUYV stride {} bytes for width {WIDTH} read in place, no repack",
        stride_pixels * 2
    );
}

/// The live proof: one webcam dma-buf frame, imported into the compute pass on an
/// integrated GPU, gives the tensor that same frame's bytes give through the copy
/// path. Skips cleanly with no camera, no bindable adapter, or a busy device.
#[cfg(feature = "v4l2-dmabuf-wgpu")]
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE) + an integrated GPU"]
async fn camera_dmabuf_frame_reaches_the_tensor() {
    use g2g_core::memory::MemoryDomainKind;

    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let device = camera::device();
    if !std::path::Path::new(&device).exists() {
        eprintln!("skipping: no camera at {device} (set G2G_V4L2_DEVICE)");
        return;
    }
    let Some((frame, caps)) = camera::capture_one_dmabuf(&device).await else {
        return;
    };
    let (format, width, height) = raw_geometry(&caps);
    assert_eq!(
        format,
        RawVideoFormat::Yuyv,
        "dmabuf mode carries raw YUYV, never the camera's MJPEG mode"
    );
    let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
        panic!(
            "io-mode=dmabuf must emit a DmaBuf frame, got {:?}",
            frame.domain
        );
    };
    assert_eq!(
        MemoryDomainKind::DmaBuf,
        frame.domain.kind(),
        "the frame's domain kind is what the auto-plug sees"
    );
    let stride = dmabuf.stride as usize;
    // Read this frame's pixels while it still holds the fd: the copy path's input
    // has to be the very frame the import binds, since a camera never repeats one.
    let padded = camera::read_dmabuf(dmabuf.as_raw(), stride * height as usize);
    let tight = repack_tight(&padded, stride, width as usize * 2, height as usize);

    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::Integrated, false).await
    else {
        eprintln!("skipping: no GPU on this host binds a CPU-backed capture dma-buf");
        return;
    };
    let got = frame_f32(&tensor);
    // The two routes run on different GPUs (the import on the integrated one, the
    // upload on the default adapter), which may contract the colour math
    // differently, so this is a float match rather than bit equality.
    assert_matches_reference(&got, &tight, width, height);
    let uploaded = preprocess_system(&caps, tight.clone()).await;
    let worst = got
        .iter()
        .zip(&uploaded)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-3,
        "imported camera dma-buf and the same frame uploaded differ by {worst}"
    );

    // How bright the frame was and how far its luma spread are the scene's
    // business (a lens against a dark desk is nearly black and nearly flat), so
    // they are reported. What is asserted is only that the buffer is not one
    // constant value, which a zero-filled or unmapped one would be.
    let mut luma: Vec<u8> = tight.chunks(2).map(|pair| pair[0]).collect();
    let luma_mean = luma.iter().map(|y| *y as f64).sum::<f64>() / luma.len() as f64;
    let mean = got.iter().sum::<f32>() / got.len() as f32;
    luma.sort_unstable();
    luma.dedup();
    eprintln!(
        "PASS: {device} {width}x{height} YUYV dma-buf (stride {stride}) -> tensor on the \
         integrated GPU, {} values, tensor mean {mean:.3}, luma mean {luma_mean:.1} over {} \
         distinct values, worst delta vs the copy path {worst:e}",
        got.len(),
        luma.len()
    );
    assert!(
        luma.len() >= 2,
        "the captured buffer is one flat value ({luma_mean:.1}), so nothing was captured into it"
    );
}

/// v4l2 capture, only for the live test.
#[cfg(feature = "v4l2-dmabuf-wgpu")]
mod camera {
    use super::*;

    use g2g_core::runtime::{run_simple_pipeline, LatencyProfile, SourceLoop};
    use g2g_core::{CapsConstraint, ConfigureOutcome, PipelineClock};
    use g2g_plugins::v4l2src::{IoMode, V4l2Src};

    /// Frames to capture before keeping one: the first frames off a UVC camera are
    /// still auto-exposing.
    const FRAMES: u64 = 8;
    /// Geometry the camera is asked for.
    const CAPTURE_WIDTH: u32 = 640;
    const CAPTURE_HEIGHT: u32 = 480;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    pub(super) fn device() -> String {
        std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string())
    }

    /// The bytes behind an exported dma-buf fd. A dma-buf has no `read(2)`, so
    /// `mmap(2)` is the only way to look at one from the CPU, and vb2's exporters
    /// implement the mmap op for exactly that.
    pub(super) fn read_dmabuf(fd: i32, len: usize) -> Vec<u8> {
        // SAFETY: `fd` is a live dma-buf fd (the frame that shares it is still
        // alive), and a read-only shared mapping is what vb2 exports it for.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(
            ptr,
            libc::MAP_FAILED,
            "mmap of the exported dma-buf failed: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: mmap just returned `len` readable bytes at `ptr`.
        let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), len) }.to_vec();
        // SAFETY: unmapping exactly the region mapped above.
        unsafe { libc::munmap(ptr, len) };
        bytes
    }

    /// Capture a short run in `io-mode=dmabuf` and keep the last frame, whose
    /// dma-buf fd stays open because the frame does. `None` when the camera cannot
    /// be driven (absent, busy, no dmabuf export), which is a skip, not a failure.
    pub(super) async fn capture_one_dmabuf(device: &str) -> Option<(Frame, Caps)> {
        let mut src = V4l2Src::new(device.to_string())
            .with_size(CAPTURE_WIDTH, CAPTURE_HEIGHT)
            .with_fps(30)
            .with_frame_limit(FRAMES)
            .with_io_mode(IoMode::DmaBuf);
        assert_eq!(
            SourceLoop::output_memory(&src),
            g2g_core::MemoryDomainKind::DmaBuf
        );
        let mut sink = HoldLast::default();
        let run = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_simple_pipeline(
                &mut src,
                &mut sink,
                &ZeroClock,
                LatencyProfile::Live.link_capacity(),
            ),
        )
        .await;
        match run {
            Err(_) => {
                eprintln!("skipping: {device} produced no frames within 30s");
                return None;
            }
            Ok(Err(e)) => {
                eprintln!("skipping: {device} capture failed ({e:?}): camera absent or busy");
                return None;
            }
            Ok(Ok(stats)) => assert_eq!(stats.frames_emitted, FRAMES),
        }
        Some((
            sink.last?,
            sink.caps.expect("the source announced its caps"),
        ))
    }

    /// Keeps the most recent frame (and the negotiated caps), so the test decides
    /// when the capture buffer goes back to the driver.
    #[derive(Default)]
    struct HoldLast {
        caps: Option<Caps>,
        last: Option<Frame>,
    }

    impl AsyncElement for HoldLast {
        type ProcessFuture<'a>
            = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
        where
            Self: 'a;

        fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream_caps.clone())
        }

        fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
            CapsConstraint::AcceptsAny
        }

        fn configure_pipeline(
            &mut self,
            absolute_caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            self.caps = Some(absolute_caps.clone());
            Ok(ConfigureOutcome::Accepted)
        }

        fn input_domains(&self) -> g2g_core::DomainSet {
            g2g_core::DomainSet::ALL
        }

        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(caps) => self.caps = Some(caps),
                    // Replacing drops the previous frame, which hands its capture
                    // buffer back to the driver.
                    PipelinePacket::DataFrame(frame) => self.last = Some(frame),
                    _ => {}
                }
                Ok(())
            })
        }
    }
}
