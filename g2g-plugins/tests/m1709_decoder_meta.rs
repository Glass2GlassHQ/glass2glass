//! M1709: every decoder puts the metadata and arrival time of an access unit on
//! the picture that unit decodes to, matched by pts.
//!
//! Each unit keeps the pts its demuxer read and carries a `WallClockMeta` and an
//! `arrival_ns` derived from that pts, so a picture is correct when it carries
//! the values derived from its own pts. The H.264 fixture has B-frames, so the
//! units arrive out of pts order and the decoder emits in pts order: a
//! first-in first-out pairing of stamps to pictures fails the check.
#![cfg(all(
    feature = "metadata",
    any(
        all(target_os = "linux", any(feature = "ffmpeg", feature = "nvdec")),
        all(
            any(target_os = "linux", target_os = "windows"),
            feature = "vulkan-video"
        ),
        feature = "dav1d",
        feature = "rav1d",
    )
))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, WallClockMeta};

const WALL_CLOCK_BASE_NS: i64 = 1_700_000_000_000_000_000;
const ARRIVAL_BASE_NS: u64 = 5_000_000_000;

// both GPU tests open the same device, one at a time
#[cfg(any(
    all(target_os = "linux", feature = "nvdec"),
    all(
        any(target_os = "linux", target_os = "windows"),
        feature = "vulkan-video"
    ),
))]
static GPU_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn wall_clock_for(pts_ns: u64) -> i64 {
    WALL_CLOCK_BASE_NS + pts_ns as i64
}

fn arrival_for(pts_ns: u64) -> u64 {
    ARRIVAL_BASE_NS + pts_ns
}

#[derive(Debug)]
struct CodedUnit {
    bytes: Vec<u8>,
    pts_ns: u64,
}

#[derive(Debug)]
struct Demuxed {
    caps: Caps,
    units: Vec<CodedUnit>,
}

// runs a whole file through a demuxer element, keeping each unit's pts
async fn demux<E: AsyncElement>(
    mut demuxer: E,
    encoding: ByteStreamEncoding,
    file: &[u8],
) -> Demuxed {
    demuxer
        .configure_pipeline(&Caps::ByteStream { encoding })
        .expect("the demuxer accepts the file");
    let mut sink = CollectSink::default();
    let whole = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(file.into())),
        FrameTiming::default(),
        0,
    );
    demuxer
        .process(PipelinePacket::DataFrame(whole), &mut sink)
        .await
        .expect("demux the file");
    demuxer
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at EOS");
    let mut caps = None;
    let mut units = Vec::new();
    for packet in sink.packets {
        match packet {
            PipelinePacket::CapsChanged(c) => caps = Some(c),
            PipelinePacket::DataFrame(f) => units.push(CodedUnit {
                bytes: f
                    .domain
                    .require_system_slice("m1709")
                    .expect("system bytes")
                    .to_vec(),
                pts_ns: f.timing.pts_ns,
            }),
            _ => {}
        }
    }
    Demuxed {
        caps: caps.expect("the demuxer announces its stream"),
        units,
    }
}

fn stamped_unit(unit: CodedUnit, sequence: u64) -> Frame {
    let mut frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(unit.bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: unit.pts_ns,
            arrival_ns: arrival_for(unit.pts_ns),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    };
    frame.meta.attach(WallClockMeta {
        unix_nanos: wall_clock_for(unit.pts_ns),
    });
    frame
}

#[derive(Debug)]
struct DecodedStamp {
    pts_ns: u64,
    arrival_ns: u64,
    wall_clock: Option<i64>,
}

// drops each picture once read, so a decoder with a fixed surface pool keeps decoding
#[derive(Debug, Default)]
struct StampSink {
    pictures: Vec<DecodedStamp>,
}

impl OutputSink for StampSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.pictures.push(DecodedStamp {
                pts_ns: frame.timing.pts_ns,
                arrival_ns: frame.timing.arrival_ns,
                wall_clock: frame.meta.get::<WallClockMeta>().map(|w| w.unix_nanos),
            });
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

// decodes `stream` and checks every picture carries the stamp of the unit at its pts
async fn assert_each_picture_carries_its_own_stamp<E: AsyncElement>(
    decoder: &mut E,
    stream: Demuxed,
    expect_reorder: bool,
) {
    let mut fed: Vec<u64> = stream.units.iter().map(|u| u.pts_ns).collect();
    assert!(fed.len() > 1, "the fixture has more than one unit");
    if expect_reorder {
        assert!(
            fed.windows(2).any(|w| w[1] < w[0]),
            "the units arrive in pts order, so a FIFO pairing would pass: {fed:?}"
        );
    }
    let mut sink = StampSink::default();
    for (index, unit) in stream.units.into_iter().enumerate() {
        decoder
            .process(
                PipelinePacket::DataFrame(stamped_unit(unit, index as u64)),
                &mut sink,
            )
            .await
            .expect("decode a unit");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at EOS");

    for picture in &sink.pictures {
        assert_eq!(
            picture.wall_clock,
            Some(wall_clock_for(picture.pts_ns)),
            "picture at pts {} carries the wrong wall clock",
            picture.pts_ns
        );
        assert_eq!(
            picture.arrival_ns,
            arrival_for(picture.pts_ns),
            "picture at pts {} carries the wrong arrival time",
            picture.pts_ns
        );
    }
    let mut emitted: Vec<u64> = sink.pictures.iter().map(|p| p.pts_ns).collect();
    emitted.sort_unstable();
    fed.sort_unstable();
    assert_eq!(
        emitted, fed,
        "each unit decodes to exactly one picture at its own pts"
    );
}

#[cfg(any(
    all(target_os = "linux", any(feature = "ffmpeg", feature = "nvdec")),
    all(
        any(target_os = "linux", target_os = "windows"),
        feature = "vulkan-video"
    ),
))]
mod h264 {
    use super::*;
    use g2g_plugins::mp4demux::Mp4Demux;

    const BFRAMES_MP4: &[u8] = include_bytes!("fixtures/h264_64x48_bframes_frag.mp4");

    pub(crate) async fn stream() -> Demuxed {
        demux(Mp4Demux::new(), ByteStreamEncoding::Mp4, BFRAMES_MP4).await
    }
}

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
#[tokio::test]
async fn ffmpegdec_keeps_each_units_meta_across_reorder() {
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;
    let stream = h264::stream().await;
    let mut decoder = FfmpegVideoDec::new();
    decoder
        .configure_pipeline(&stream.caps)
        .expect("libavcodec opens the H.264 decoder");
    assert_each_picture_carries_its_own_stamp(&mut decoder, stream, true).await;
}

#[cfg(all(target_os = "linux", feature = "nvdec"))]
#[tokio::test]
async fn nvdec_keeps_each_units_meta_across_reorder() {
    use g2g_plugins::nvdec::NvDec;
    let _gpu = GPU_TEST_LOCK.lock().await;
    let stream = h264::stream().await;
    let mut decoder = NvDec::new();
    if decoder.configure_pipeline(&stream.caps).is_err() {
        eprintln!("skipping: NVDEC unavailable on this host");
        return;
    }
    assert_each_picture_carries_its_own_stamp(&mut decoder, stream, true).await;
}

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]
mod vulkan {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::{AllocationParams, DomainSet, MemoryDomainKind};
    use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoDec, VulkanVideoError};

    fn has_h264_decode() -> bool {
        match block_on(open_h264_decode_device()) {
            Ok(_) => true,
            Err(VulkanVideoError::NoVulkanAdapter)
            | Err(VulkanVideoError::ExtensionUnsupported)
            | Err(VulkanVideoError::NoDecodeQueue) => false,
            Err(e) => panic!("open the H.264 decode device: {e:?}"),
        }
    }

    fn decode(output_domain: Option<MemoryDomainKind>) {
        let _gpu = GPU_TEST_LOCK.blocking_lock();
        if !has_h264_decode() {
            eprintln!("skipping: no Vulkan H.264 decode adapter");
            return;
        }
        let stream = block_on(h264::stream());
        let mut decoder = VulkanVideoDec::new();
        if let Some(domain) = output_domain {
            decoder.configure_allocation(&AllocationParams {
                domain,
                accepts: DomainSet::only(domain),
                ..Default::default()
            });
        }
        decoder
            .configure_pipeline(&stream.caps)
            .expect("configure opens the decode device");
        block_on(assert_each_picture_carries_its_own_stamp(
            &mut decoder,
            stream,
            true,
        ));
    }

    #[test]
    fn system_output_keeps_each_units_meta_across_reorder() {
        decode(None);
    }

    #[test]
    fn texture_output_keeps_each_units_meta_across_reorder() {
        decode(Some(MemoryDomainKind::WgpuTexture));
    }
}

#[cfg(any(feature = "dav1d", feature = "rav1d"))]
mod av1 {
    use super::*;
    use g2g_plugins::ivfdemux::IvfDemux;

    const IVF: &[u8] = include_bytes!("fixtures/av1_321x181.ivf");

    pub(crate) async fn check<E: AsyncElement>(mut decoder: E) {
        let stream = demux(IvfDemux::new(), ByteStreamEncoding::Ivf, IVF).await;
        decoder
            .configure_pipeline(&stream.caps)
            .expect("the AV1 decoder opens");
        // AV1 shows pictures in unit order, so only a frame-threading delay can misplace a stamp
        assert_each_picture_carries_its_own_stamp(&mut decoder, stream, false).await;
    }
}

#[cfg(feature = "dav1d")]
#[tokio::test]
async fn dav1ddec_keeps_each_units_meta() {
    av1::check(g2g_plugins::dav1ddec::Dav1dDec::new()).await;
}

#[cfg(feature = "rav1d")]
#[tokio::test]
async fn rav1ddec_keeps_each_units_meta() {
    av1::check(g2g_plugins::rav1ddec::Rav1dDec::new()).await;
}
