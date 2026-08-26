//! M1059: the capture / decode producers that declare a `PlaneLayout` instead
//! of repacking a driver's padded rows.
//!
//! Two legs run here. The pure one drives `v4l2src`'s repack-vs-declare decision
//! with a fabricated `bytesperline`, so the request gate and both copies are
//! exercised without a camera. The live one opens `/dev/video0` (override with
//! `G2G_V4L2_DEVICE`), captures one frame behind a sink that requests a
//! `PlaneLayout`, and proves a consumer reading the declared rows gets the same
//! pixels the repack would have produced. It skips with an `eprintln!` when
//! there is no device or when the device's rows are already tight, since then
//! there is no padding to declare.
//!
//! The `pipewirevideosrc` and `vaapidec` legs of the milestone are
//! compile-checked only: PipeWire's capture needs a running daemon with a shared
//! node, and VAAPI needs an Intel / AMD GPU this host does not have. Their
//! derivations are unit-tested in the crate (`pwvideo`, `pipewirevideosrc`,
//! `vaapidec`).

#![cfg(all(target_os = "linux", feature = "v4l2", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{MetaRequests, PlaneLayout};
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, RawVideoFormat,
};
use g2g_plugins::capturepixelformat::CapturePixelFormat;
use g2g_plugins::v4l2src::{row_handling, take_rows, RowHandling, V4l2Src};
use g2g_plugins::videoconvert::VideoConvert;

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

/// A 4x2 NV12 frame the driver wrote at a 7-byte pitch: 4 bytes of picture and 3
/// of padding per row, luma rows then the interleaved chroma one. Returned
/// padded, then tight.
fn padded_nv12() -> (Vec<u8>, Vec<u8>) {
    let mut padded = Vec::new();
    let mut tight = Vec::new();
    for row in 0..3u8 {
        let picture = [row * 4 + 1, row * 4 + 2, row * 4 + 3, row * 4 + 4];
        tight.extend_from_slice(&picture);
        padded.extend_from_slice(&picture);
        padded.extend_from_slice(&[0xee; 3]);
    }
    (padded, tight)
}

/// The pure leg: the real decision function, with the driver's stride fabricated
/// rather than read off a device. Nobody asking means the capture is repacked
/// exactly as it always was; a request means the driver's buffer travels as it
/// is with its pitch reported, and the rows read back out of it are the ones the
/// repack would have produced.
#[test]
fn the_request_decides_whether_a_padded_capture_is_repacked() {
    let (padded, tight) = padded_nv12();

    let repack = row_handling(CapturePixelFormat::Nv12, 4, 7, false);
    assert_eq!(
        repack,
        RowHandling::Repack {
            format: RawVideoFormat::Nv12,
            first_stride: 7
        }
    );
    let (bytes, stride) = take_rows(&padded, repack, 4, 2).expect("the buffer holds a frame");
    assert_eq!(bytes, tight, "the repack packs every plane");
    assert_eq!(stride, 0, "packed rows carry no pitch");

    let declare = row_handling(CapturePixelFormat::Nv12, 4, 7, true);
    assert_eq!(
        declare,
        RowHandling::Declare {
            format: RawVideoFormat::Nv12,
            first_stride: 7
        }
    );
    let (bytes, stride) = take_rows(&padded, declare, 4, 2).expect("the buffer holds a frame");
    assert_eq!(bytes, padded, "the driver's buffer travels untouched");
    assert_eq!(stride, 7);

    // What a consumer reads off the declared rows is what the repack produced.
    let layout = PlaneLayout::new(&[
        g2g_core::meta::Plane {
            offset: 0,
            stride: 7,
        },
        g2g_core::meta::Plane {
            offset: 14,
            stride: 7,
        },
    ])
    .expect("a two-plane layout");
    let mut read_back = Vec::new();
    for row in 0..2 {
        read_back.extend_from_slice(&bytes[layout.row_range(0, row, 4).expect("a luma row")]);
    }
    read_back.extend_from_slice(&bytes[layout.row_range(1, 0, 4).expect("the chroma row")]);
    assert_eq!(read_back, tight);

    // A tight capture has nothing to declare, whoever asks.
    for requested in [false, true] {
        assert_eq!(
            row_handling(CapturePixelFormat::Nv12, 4, 4, requested),
            RowHandling::AsIs
        );
    }
}

/// Sink that asks every producer upstream for a `PlaneLayout` and keeps the
/// first frame it is handed, plus the caps the link settled on.
#[derive(Default)]
struct LayoutDemandingSink {
    caps: Option<Caps>,
    first: Option<Frame>,
}

impl AsyncElement for LayoutDemandingSink {
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

    fn meta_requests(&self) -> MetaRequests {
        MetaRequests::new().request_from_every_consumer::<PlaneLayout>()
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(caps) => self.caps = Some(caps),
                PipelinePacket::DataFrame(frame) if self.first.is_none() => {
                    self.first = Some(frame);
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Collects whatever a transform pushes, so a test can run one frame through a
/// real element and look at the result.
#[derive(Default)]
struct FrameSink {
    frames: Vec<Frame>,
}

impl OutputSink for FrameSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            self.frames.push(frame);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Byte width of one row of a capture format's first plane, or `None` for a
/// format this test does not unpack. The camera formats `v4l2src` maps are
/// 8-bit: YUYV carries two bytes per pixel, the YUV planar ones one luma byte.
fn plane0_row_bytes(format: RawVideoFormat, width: usize) -> Option<usize> {
    match format {
        RawVideoFormat::Yuyv => Some(width * 2),
        RawVideoFormat::Nv12 | RawVideoFormat::I420 => Some(width),
        _ => None,
    }
}

/// Convert `frame` to I420 through the real element, returning its output bytes.
/// `None` when the convert will not take these caps at all.
async fn convert_to_i420(frame: Frame, caps: &Caps) -> Option<Vec<u8>> {
    let mut convert = VideoConvert::new(RawVideoFormat::I420);
    convert.configure_pipeline(caps).ok()?;
    let mut sink = FrameSink::default();
    convert
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("the convert accepts the frame");
    Some(
        sink.frames
            .pop()
            .expect("one converted frame")
            .domain
            .as_system_slice()
            .expect("system memory out")
            .to_vec(),
    )
}

/// The live leg: a real camera behind a sink that asks for the layout. When the
/// driver pads its rows, the frame carries them with the pitch declared, and
/// converting it gives the same pixels as converting the rows packed by hand.
#[tokio::test]
async fn a_padded_capture_declares_its_rows_and_converts_identically() {
    let dev = device();
    if !std::path::Path::new(&dev).exists() {
        eprintln!("no {dev}; skipping the M1059 live v4l2 leg");
        return;
    }
    let mut src = V4l2Src::new(dev.clone())
        .with_size(WIDTH, HEIGHT)
        .with_frame_limit(1);
    let mut sink = LayoutDemandingSink::default();
    let run = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await;
    let Ok(Ok(stats)) = run else {
        eprintln!("{dev} did not capture ({run:?}); skipping the M1059 live v4l2 leg");
        return;
    };
    assert_eq!(stats.frames_emitted, 1, "the source emitted its one frame");

    let frame = sink.first.take().expect("a frame reached the sink");
    let caps = sink.caps.clone().expect("the link settled on caps");
    let Caps::RawVideo { format, .. } = &caps else {
        eprintln!("{dev} captured {caps:?}, not raw rows; skipping the M1059 live v4l2 leg");
        return;
    };
    let Some(row) = plane0_row_bytes(*format, WIDTH as usize) else {
        eprintln!("{dev} captured {format:?}, which this test cannot unpack; skipping");
        return;
    };
    let Some(layout) = frame.meta.get::<PlaneLayout>().copied() else {
        eprintln!(
            "{dev} packs its rows tight (no bytesperline padding); \
             skipping the M1059 live v4l2 leg"
        );
        return;
    };

    let bytes = frame
        .domain
        .as_system_slice()
        .expect("the capture is system memory")
        .to_vec();
    let stride = layout.plane(0).expect("plane 0").stride;
    assert!(stride > row, "a declared layout means real padding");
    assert!(
        bytes.len() >= stride * (HEIGHT as usize - 1) + row,
        "the padded buffer travelled whole"
    );

    // Pack the declared rows by hand, then convert both frames: the consumer's
    // pixels must not depend on which shape it was handed.
    let mut packed = Vec::new();
    for y in 0..HEIGHT as usize {
        packed.extend_from_slice(&bytes[layout.row_range(0, y, row).expect("a row")]);
    }
    let packed_frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(packed.into_boxed_slice())),
        frame.timing,
        frame.sequence,
    );
    let (Some(from_declared), Some(from_packed)) = (
        convert_to_i420(frame, &caps).await,
        convert_to_i420(packed_frame, &caps).await,
    ) else {
        eprintln!("videoconvert does not take {caps:?}; skipping the M1059 live v4l2 leg");
        return;
    };
    assert!(!from_packed.is_empty());
    assert_eq!(
        from_declared, from_packed,
        "reading the declared rows where they lie converts to the same pixels"
    );
    eprintln!("{dev} pads its rows to {stride} bytes; the M1059 live v4l2 leg ran");
}
