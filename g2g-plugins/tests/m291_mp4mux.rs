//! M291 fragmented-MP4 muxer element: `Mp4Mux` wraps an H.264 elementary stream
//! into an ISO-BMFF byte stream forwarded downstream (the gst `mp4mux`/`qtmux`
//! analog: `... ! x264enc ! mp4mux ! filesink`). The strong test is a real
//! round-trip: mux access units through `Mp4Mux`, demux them back through
//! `Fmp4Demux`, and compare byte-exact (the same round-trip proof `m158` runs).
//! Also checks the launch registry resolves `mp4mux` / `qtmux` to ISO-BMFF, and
//! that a non-H.264/H.265 caps is rejected at configure.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, G2gError, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::fmp4demux::Fmp4Demux;
use g2g_plugins::mp4mux::Mp4Mux;

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
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
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// First AU carries SPS+PPS+IDR (so the moov gets its parameter sets); the rest
/// are P slices.
fn source_access_units() -> Vec<Vec<u8>> {
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, 0xBB],
    ]
    .concat();
    let p = |fill: u8| [&[0, 0, 0, 1][..], &[0x41, fill, fill]].concat();
    vec![idr, p(1), p(2), p(3)]
}

/// Mux the access units to an fMP4 byte buffer via the `Mp4Mux` element,
/// concatenating the byte-stream frames it forwards downstream.
async fn mux_via_element(aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(64, 48)).unwrap();
    let mut sink = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(
            PipelinePacket::DataFrame(frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut sink,
        )
        .await
        .unwrap();
    }
    mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    assert_eq!(
        mux.emitted(),
        aus.len() as u64,
        "one byte-stream frame per AU"
    );
    sink.frames.concat()
}

async fn demux(fmp4: &[u8]) -> CaptureSink {
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    dmx.process(
        PipelinePacket::DataFrame(frame(fmp4.to_vec(), 0, 0)),
        &mut sink,
    )
    .await
    .unwrap();
    dmx.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink
}

#[tokio::test]
async fn mp4mux_roundtrips_through_fmp4demux() {
    let aus = source_access_units();
    let fmp4 = mux_via_element(&aus).await;

    // The forwarded stream is ISO-BMFF: starts with an ftyp box.
    assert_eq!(&fmp4[4..8], b"ftyp", "byte stream starts with ftyp");

    let sink = demux(&fmp4).await;
    assert_eq!(
        sink.caps,
        vec![h264_caps(64, 48).with_framerate_any()],
        "the moov drives one CapsChanged with the muxed codec + geometry"
    );
    assert_eq!(
        sink.frames, aus,
        "every access unit recovered, in order, byte-exact"
    );
}

#[test]
fn rejects_non_h26x_caps() {
    // The muxer carries H.264 / H.265 only; raw video is refused at configure
    // (the coverage the removed Mp4Sink test held).
    let raw = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Rgba8,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
    };
    assert!(
        Mp4Mux::new().configure_pipeline(&raw).is_err(),
        "raw video is not a muxable codec"
    );
    assert!(
        Mp4Mux::new().configure_pipeline(&h264_caps(64, 48)).is_ok(),
        "H.264 is accepted"
    );
}

#[test]
fn registry_resolves_mp4mux_and_qtmux_alias() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();
    // Both the canonical name and the qtmux alias parse into a runnable graph.
    for name in ["mp4mux", "qtmux"] {
        let line =
            format!("appsrc caps=video/x-raw,format=RGBA,width=2,height=2 ! {name} ! fakesink");
        // The structural parse + element lookup must succeed (the appsrc->mux caps
        // link itself is rejected at negotiation, not at element resolution, so we
        // only assert the element name resolves, i.e. not "unknown element").
        let err = parse_launch(&reg, &line)
            .err()
            .map(|e| format!("{e}"))
            .unwrap_or_default();
        assert!(
            !err.contains("unknown element"),
            "{name} should resolve: {err}"
        );
    }
}

// Small helper so the caps comparison reads cleanly (the demuxer reports the
// framerate as Any since the moov has no frame-rate box).
trait FramerateAny {
    fn with_framerate_any(self) -> Caps;
}
impl FramerateAny for Caps {
    fn with_framerate_any(self) -> Caps {
        match self {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                ..
            } => Caps::CompressedVideo {
                codec,
                width,
                height,
                framerate: Rate::Any,
            },
            other => other,
        }
    }
}
