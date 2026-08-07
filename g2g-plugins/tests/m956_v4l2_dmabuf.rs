//! M956 `v4l2src io-mode=dmabuf`: the driver's MMAP capture buffers are exported
//! once as dma-buf fds and handed downstream with no copy, so a GPU consumer can
//! import the camera buffer directly.
//!
//! The tests need a real `/dev/videoN` device the running user can open, so they
//! are ignored by default like the other v4l2 smoke tests. Override the device
//! with `G2G_V4L2_DEVICE`. Run with `--test-threads=1`: the tests share the
//! camera, and a parallel open fails with EBUSY.
//!
//! ```sh
//! cargo test -p g2g-plugins --features v4l2 \
//!     --test m956_v4l2_dmabuf -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(target_os = "linux", feature = "v4l2"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, MemoryDomain, MemoryDomainKind,
    OutputSink, PipelineClock, PipelinePacket, PropValue, RawVideoFormat,
};
use g2g_plugins::v4l2src::{IoMode, V4l2Src};

/// The mmap buffer-ring depth `v4l2src` requests, i.e. how many distinct dma-buf
/// fds can exist for one stream. Not public, so it is restated here: the recycling
/// test needs to capture more frames than there are buffers.
const BUFFER_COUNT: usize = 4;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn device() -> String {
    std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string())
}

/// The bytes behind an exported dma-buf fd. A dma-buf has no `read(2)`, so
/// `mmap(2)` is the only way to look at one from the CPU, and vb2's exporters
/// implement the mmap op for exactly that.
fn read_dmabuf(fd: i32, len: usize) -> Vec<u8> {
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

/// What a file descriptor points at, via `/proc/self/fd`. A dma-buf shows up as
/// `/dmabuf:...` or `anon_inode:dmabuf`. Only meaningful while the fd is open:
/// a closed number may already have been handed to something else.
fn fd_target(fd: i32) -> String {
    std::fs::read_link(format!("/proc/self/fd/{fd}"))
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("<unreadable: {e}>"))
}

/// Sink that records what each dma-buf frame carried and keeps the most recent
/// `hold` frames alive, so a test controls when a capture buffer is released back
/// to the driver.
struct DmaBufProbeSink {
    hold: usize,
    held: Vec<Frame>,
    /// `(fd, stride)` of every frame, in arrival order.
    exported: Vec<(i32, u32)>,
    /// What each frame's fd pointed at, read while the frame still held it.
    fd_targets: Vec<String>,
    /// Frames that arrived in a domain other than DmaBuf (must stay zero).
    wrong_domain: u64,
    caps: Option<Caps>,
    /// The first frame's bytes, read back through its dma-buf fd.
    first_bytes: Option<Vec<u8>>,
}

impl DmaBufProbeSink {
    fn new(hold: usize) -> Self {
        Self {
            hold,
            held: Vec::new(),
            exported: Vec::new(),
            fd_targets: Vec::new(),
            wrong_domain: 0,
            caps: None,
            first_bytes: None,
        }
    }

    fn distinct_fds(&self) -> Vec<i32> {
        let mut fds: Vec<i32> = self.exported.iter().map(|(fd, _)| *fd).collect();
        fds.sort_unstable();
        fds.dedup();
        fds
    }
}

impl AsyncElement for DmaBufProbeSink {
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                PipelinePacket::DataFrame(frame) => {
                    match &frame.domain {
                        MemoryDomain::DmaBuf(buffer) => {
                            self.exported.push((buffer.as_raw(), buffer.stride));
                            self.fd_targets.push(fd_target(buffer.as_raw()));
                            if self.first_bytes.is_none() {
                                // Read it while this frame still holds the fd:
                                // that is the proof the payload is the picture,
                                // not just a plausible descriptor.
                                self.first_bytes = Some(read_dmabuf(
                                    buffer.as_raw(),
                                    buffer.stride as usize * HEIGHT as usize,
                                ));
                            }
                        }
                        _ => self.wrong_domain += 1,
                    }
                    // Oldest first out: dropping a frame is what hands its
                    // capture buffer back to the driver.
                    self.held.push(frame);
                    while self.held.len() > self.hold {
                        self.held.remove(0);
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Every frame arrives in `MemoryDomain::DmaBuf` with the driver's stride, and
/// what the fd maps to really is the captured picture.
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn dmabuf_mode_exports_the_camera_buffers() {
    let dev = device();
    let frames: u64 = 8;
    let mut src = V4l2Src::new(dev.clone())
        .with_size(WIDTH, HEIGHT)
        .with_fps(30)
        .with_frame_limit(frames);
    // through the property, the way a launch line sets it.
    SourceLoop::set_property(&mut src, "io-mode", PropValue::Str("dmabuf".into()))
        .expect("io-mode=dmabuf");
    assert_eq!(SourceLoop::output_memory(&src), MemoryDomainKind::DmaBuf);

    let mut sink = DmaBufProbeSink::new(1);
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture finishes within 30s")
    .expect("dmabuf capture pipeline runs");

    assert_eq!(stats.frames_emitted, frames, "source emitted every frame");
    assert_eq!(
        sink.exported.len() as u64,
        frames,
        "every frame must reach the sink as a dma-buf"
    );
    assert_eq!(sink.wrong_domain, 0, "a frame arrived outside DmaBuf");
    // dmabuf export carries no payload length, so only raw formats are offered:
    // the link must be raw YUYV, never the camera's MJPEG mode.
    assert!(
        matches!(
            sink.caps,
            Some(Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                ..
            })
        ),
        "unexpected caps {:?}",
        sink.caps
    );

    // Plausible descriptors: real fds (0..2 are the std streams), and the
    // driver's bytesperline for packed YUYV is at least two bytes per pixel.
    for (fd, stride) in &sink.exported {
        assert!(*fd > 2, "implausible dma-buf fd {fd}");
        assert!(
            *stride >= WIDTH * 2,
            "stride {stride} is short for {WIDTH}px YUYV"
        );
    }
    // What VIDIOC_EXPBUF handed over really is a dma-buf, not some other
    // descriptor that happens to map.
    for target in &sink.fd_targets {
        assert!(target.contains("dmabuf"), "exported fd points at {target}");
    }
    // The buffers rotate, so more than one of them was exported.
    let distinct = sink.distinct_fds();
    assert!(
        distinct.len() > 1 && distinct.len() <= BUFFER_COUNT,
        "{} distinct fds over {frames} frames: {distinct:?}",
        distinct.len()
    );

    // The payload is the picture: YUYV luma is every other byte, and a camera
    // frame is never one flat value.
    let bytes = sink.first_bytes.expect("first frame read back");
    let stride = sink.exported[0].1 as usize;
    assert_eq!(bytes.len(), stride * HEIGHT as usize);
    let luma: Vec<u8> = bytes.chunks(2).map(|pair| pair[0]).collect();
    let mean = luma.iter().map(|y| *y as f64).sum::<f64>() / luma.len() as f64;
    let variance = luma.iter().map(|y| (*y as f64 - mean).powi(2)).sum::<f64>() / luma.len() as f64;
    let mut values: Vec<u8> = luma.clone();
    values.sort_unstable();
    values.dedup();
    eprintln!(
        "{dev}: {frames} dma-buf frames, fds {distinct:?} -> {}, stride {stride}, \
         luma mean {mean:.1} variance {variance:.1} over {} distinct values",
        sink.fd_targets[0],
        values.len()
    );
    // sensor noise alone spreads luma over many values; a zero-filled or
    // constant-garbage mapping cannot. variance is scene-dependent (a lens
    // against a dark desk is nearly flat), so it is reported, not asserted.
    assert!(
        values.len() >= 8,
        "the mapped buffer does not look like sensor data: variance {variance:.3}, \
         {} distinct luma values",
        values.len()
    );

    record_hardware_evidence(
        &dev,
        &format!("io-mode=dmabuf: {frames} exported frames, stride {stride}"),
    );
}

/// The recycling invariant: capturing more frames than there are buffers only
/// works if a released buffer goes back to the driver, and the fds must stay open
/// across that reuse (they are exported once, not per frame).
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn buffers_recycle_once_the_consumer_releases_them() {
    let frames: u64 = 3 * BUFFER_COUNT as u64;
    let mut src = V4l2Src::new(device())
        .with_size(WIDTH, HEIGHT)
        .with_fps(30)
        .with_frame_limit(frames)
        .with_io_mode(IoMode::DmaBuf);
    // Holds two frames at a time, so buffers are only released once newer ones
    // arrive: the capture loop has to wait for them and then re-queue.
    let mut sink = DmaBufProbeSink::new(2);

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture finishes within 30s: a held buffer must not stall the loop")
    .expect("dmabuf capture pipeline runs");

    assert_eq!(stats.frames_emitted, frames);
    assert_eq!(sink.exported.len() as u64, frames);
    let distinct = sink.distinct_fds();
    assert!(
        distinct.len() <= BUFFER_COUNT,
        "{frames} frames used {} fds, so buffers were not reused: {distinct:?}",
        distinct.len()
    );
    // Every buffer appeared more than once, or nothing was recycled.
    assert!(
        frames as usize > distinct.len(),
        "no buffer was captured into twice"
    );

    // The frames the sink still holds keep their fds open past the end of the
    // stream: nothing was closed when the buffer was re-queued or the capture
    // thread stopped.
    assert_eq!(sink.held.len(), 2);
    for frame in &sink.held {
        let MemoryDomain::DmaBuf(buffer) = &frame.domain else {
            panic!("a held frame lost its dma-buf");
        };
        let bytes = read_dmabuf(buffer.as_raw(), buffer.stride as usize * HEIGHT as usize);
        assert!(
            bytes.iter().any(|byte| *byte != 0),
            "a held buffer read back empty"
        );
    }
    eprintln!(
        "recycled {} buffers over {frames} frames: fds {distinct:?}",
        distinct.len()
    );
}

/// The exported buffer holds the same picture the copy path produces: capture the
/// same scene both ways moments apart and the frames must agree, which no
/// mis-mapped or stale buffer would.
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn an_exported_frame_matches_a_copied_one() {
    let copied = capture_one_luma(IoMode::Mmap).await;
    let exported = capture_one_luma(IoMode::DmaBuf).await;
    assert_eq!(
        copied.len(),
        exported.len(),
        "the exported buffer is a different size than the copied frame"
    );

    let mean = |luma: &[u8]| luma.iter().map(|y| *y as f64).sum::<f64>() / luma.len() as f64;
    let (copied_mean, exported_mean) = (mean(&copied), mean(&exported));
    eprintln!("copied luma mean {copied_mean:.1}, exported luma mean {exported_mean:.1}");
    // Neither may be a stuck all-black or all-white buffer.
    for (label, value) in [("copied", copied_mean), ("exported", exported_mean)] {
        assert!(
            value > 1.0 && value < 254.0,
            "{label} frame is flat at {value:.1}"
        );
    }
    // Auto-exposure drifts between the two captures, so this is a loose match on
    // the same scene rather than an equality.
    assert!(
        (copied_mean - exported_mean).abs() < 40.0,
        "exported frame does not show the scene the copied one did: \
         {copied_mean:.1} vs {exported_mean:.1}"
    );
}

/// The luma plane of one frame captured in `mode`: for the copy path the bytes
/// come off the frame directly, for the export path through the dma-buf fd.
async fn capture_one_luma(mode: IoMode) -> Vec<u8> {
    let frames = 4;
    let mut src = V4l2Src::new(device())
        .with_size(WIDTH, HEIGHT)
        .with_fps(30)
        // The first frames off a UVC camera are still auto-exposing, so take the
        // last of a short run.
        .with_frame_limit(frames)
        .with_io_mode(mode);
    let mut sink = LastFrameSink::default();
    tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture finishes within 30s")
    .expect("capture pipeline runs");
    let bytes = sink.last.expect("a frame arrived");
    // YUYV packs one luma byte per pixel, interleaved with chroma.
    bytes.chunks(2).map(|pair| pair[0]).collect()
}

/// Sink that keeps the last frame's bytes whichever domain it arrived in, so the
/// copy path and the export path can be compared.
#[derive(Default)]
struct LastFrameSink {
    last: Option<Vec<u8>>,
}

impl AsyncElement for LastFrameSink {
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

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
            if let PipelinePacket::DataFrame(frame) = packet {
                self.last = match &frame.domain {
                    MemoryDomain::System(slice) => Some(slice.as_slice().to_vec()),
                    MemoryDomain::DmaBuf(buffer) => Some(read_dmabuf(
                        buffer.as_raw(),
                        buffer.stride as usize * HEIGHT as usize,
                    )),
                    other => panic!("unexpected domain {other:?}"),
                };
            }
            Ok(())
        })
    }
}

/// The capture ran against a real camera: persist camera-tagged `Hardware`
/// evidence so `g2g-inspect --maturity` derives v4l2src as HardwareValidated.
fn record_hardware_evidence(device: &str, detail: &str) {
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    use g2g_plugins::conformance::persist;
    persist::record_evidence(
        "v4l2src",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(persist::v4l2_platform_tag(device))
            .codec("yuyv")
            .detail(detail),
    )
    .expect("record hardware evidence");
}
