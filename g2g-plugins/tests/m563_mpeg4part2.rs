//! M563 - MPEG-4 Part 2 (Visual) decode via the MP4 container + libavcodec.
//!
//! A whole-file MP4 whose video track is an `mp4v` sample entry (esds
//! objectTypeIndication 0x20) demuxes to `Caps::CompressedVideo { codec:
//! Mpeg4Part2 }` with the VOL header prepended in-band, and `FfmpegVideoDec`
//! (libavcodec `mpeg4`) decodes it to raw frames. This exercises the whole path
//! end to end: the container `mp4v`/esds detection, the verbatim VOL framing, and
//! the software decoder.
//!
//! The fixture (`tests/fixtures/mpeg4part2_640x480.mp4`) is a committed 10-frame
//! 640x480 MPEG-4 Part 2 clip, generated with
//! `ffmpeg -i h264_640x480.h264 -c:v mpeg4 -q:v 5 mpeg4part2_640x480.mp4`.
//! libav* (the `ffmpeg` cargo feature) is required to build the decode half.

#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{ByteStreamEncoding, Caps, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
use g2g_plugins::mp4demux::Mp4Demux;

const FIXTURE: &[u8] = include_bytes!("fixtures/mpeg4part2_640x480.mp4");

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
    /// The System-domain bytes of each emitted `DataFrame`. `Frame` is not `Clone`
    /// (routes, not broadcasts), so a downstream stage is fed rebuilt frames.
    fn frame_bytes(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }
    fn frame_count(&self) -> usize {
        self.packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
            .count()
    }
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

#[tokio::test]
async fn mpeg4_part2_mp4_demuxes_and_decodes() {
    // 1. Demux the whole file: the mp4v/esds track -> CompressedVideo{Mpeg4Part2}
    //    plus the VOL header prepended to the first access unit.
    let mut demux = Mp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("qtdemux accepts an MP4 byte stream");
    let mut demuxed = Collect::default();
    demux
        .process(data_frame(FIXTURE.to_vec()), &mut demuxed)
        .await
        .expect("buffer the file");
    demux
        .process(PipelinePacket::Eos, &mut demuxed)
        .await
        .expect("drain at EOS");

    let demux_caps = demuxed.caps_changes();
    assert!(
        demux_caps.iter().any(|c| matches!(
            c,
            Caps::CompressedVideo {
                codec: VideoCodec::Mpeg4Part2,
                ..
            }
        )),
        "demuxer must tag the mp4v track as MPEG-4 Part 2, got {demux_caps:?}"
    );
    let access_units = demuxed.frame_bytes();
    assert!(!access_units.is_empty(), "demuxer must emit access units");

    // 2. Decode the access units with libavcodec (mpeg4) to raw I420.
    let mut dec = FfmpegVideoDec::new().with_output_format(OutputFormat::I420);
    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::Mpeg4Part2,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let narrowed = dec
        .intercept_caps(&upstream)
        .expect("decoder accepts MPEG-4 Part 2");
    dec.configure_pipeline(&narrowed)
        .expect("libavcodec mpeg4 initialises");

    let mut decoded = Collect::default();
    for au in access_units {
        dec.process(data_frame(au), &mut decoded)
            .await
            .expect("decode AU");
    }
    dec.process(PipelinePacket::Eos, &mut decoded)
        .await
        .expect("flush decoder");

    assert!(
        decoded.frame_count() > 0,
        "expected at least one decoded frame"
    );
    match decoded
        .caps_changes()
        .first()
        .expect("decoder emits raw caps")
    {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => {
            assert_eq!((*w, *h), (640, 480), "decoded geometry matches the source");
        }
        other => panic!("expected fixed I420 raw caps, got {other:?}"),
    }
}
