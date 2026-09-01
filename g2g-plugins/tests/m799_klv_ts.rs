//! M799/M800 - STANAG 4609 KLV metadata in MPEG-TS. An H.264 video stream and a
//! MISB ST 0601 KLV telemetry stream are muxed into one transport stream (the
//! KLV on a private PES with the 'KLVA' registration descriptor), then each is
//! recovered: the video AUs bit-exact, the KLV packets bit-exact and parsed back
//! to the telemetry they encode, and `klvdecode` turns them into text lines.

use std::pin::Pin;

use core::future::Future;

use g2g_core::element::{AsyncElement, BoxFuture, PushOutcome};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};

use g2g_plugins::klv::{split_klv_packets, KlvDecode, UasDatalink};
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::tsmuxn::TsMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}
fn ts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

fn telemetry(i: u64) -> UasDatalink {
    UasDatalink {
        timestamp_us: Some(1_700_000_000_000_000 + i * 40_000),
        sensor_lat_deg: Some(60.1768 + i as f64 * 0.0001),
        sensor_lon_deg: Some(24.8288),
        sensor_alt_m: Some(145.0),
        heading_deg: Some(87.3),
        version: Some(19),
        ..Default::default()
    }
}

/// Emits a script of (access-unit, pts_ns) for one elementary stream, then EOS.
struct AuSrc {
    caps: Caps,
    aus: Vec<(Vec<u8>, u64)>,
    configured: bool,
}

impl SourceLoop for AuSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let aus = self.aus.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures before run");
            for (i, (au, pts)) in aus.iter().enumerate() {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                    FrameTiming {
                        pts_ns: *pts,
                        ..FrameTiming::default()
                    },
                    i as u64,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(aus.len() as u64)
        })
    }
}

/// Collects the muxed TS byte frames.
#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
}
impl AsyncElement for CaptureSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;
    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(f) = packet {
            if let Some(s) = f.domain.as_system_slice() {
                self.bytes.extend_from_slice(s);
            }
        }
        Box::pin(async { Ok(()) })
    }
}

/// An `OutputSink` recording each recovered frame's bytes and PTS.
#[derive(Default)]
struct CaptureSinkAus {
    aus: Vec<(Vec<u8>, u64)>,
}
impl OutputSink for CaptureSinkAus {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push((s.to_vec(), f.timing.pts_ns));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Mux 3 video AUs + 2 KLV packets into one TS, returning the TS bytes and the
/// encoded KLV packets.
async fn muxed_ts() -> (Vec<u8>, Vec<Vec<u8>>) {
    let video_aus = vec![
        (vec![0u8, 0, 0, 1, 0x65, 0x11], 0),
        (vec![0u8, 0, 0, 1, 0x41, 0x22], 40_000_000),
        (vec![0u8, 0, 0, 1, 0x41, 0x33], 80_000_000),
    ];
    let klv_packets: Vec<Vec<u8>> = (0..2).map(|i| telemetry(i).encode()).collect();
    let klv_aus: Vec<(Vec<u8>, u64)> = klv_packets
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), 20_000_000 + i as u64 * 40_000_000))
        .collect();

    let mut video = AuSrc {
        caps: h264_caps(),
        aus: video_aus,
        configured: false,
    };
    let mut klv = AuSrc {
        caps: Caps::Klv,
        aus: klv_aus,
        configured: false,
    };
    let mut mux = TsMux::new(2); // input 0 = video, input 1 = klv
    let mut sink = CaptureSink::default();
    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut video, &mut klv];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("video + klv mux pipeline completes");
    }
    assert_eq!(mux.emitted(), 5, "all five AUs (3 video + 2 klv) muxed");
    (sink.bytes, klv_packets)
}

/// Drive a whole TS byte buffer through a single-output `TsDemux` selecting
/// `stream`, returning the recovered (bytes, pts_ns) pairs.
async fn demux_stream(ts: &[u8], stream: TsStream) -> Vec<(Vec<u8>, u64)> {
    let mut demux = TsDemux::new().with_stream(stream);
    demux.configure_pipeline(&ts_caps()).unwrap();
    let mut sink = CaptureSinkAus::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(ts.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.aus
}

/// The muxed multiplex carries the video and the KLV on separate PIDs, both
/// recovered bit-exact with their PTS, and the PMT carries the 'KLVA'
/// registration descriptor a third-party demuxer keys on.
#[tokio::test]
async fn klv_and_video_round_trip_one_ts() {
    let (ts, klv_packets) = muxed_ts().await;

    // The PMT (a PSI section in the raw TS bytes) names the KLV stream as a
    // private PES (0x06) with the 'KLVA' registration descriptor.
    let ts_str: &[u8] = &ts;
    assert!(
        ts_str.windows(4).any(|w| w == b"KLVA"),
        "PMT carries the KLVA registration descriptor"
    );

    let got_video = demux_stream(&ts, TsStream::H264).await;
    assert_eq!(got_video.len(), 3, "three video AUs recovered");
    assert_eq!(got_video[1].1, 40_000_000, "video PTS preserved");

    let got_klv = demux_stream(&ts, TsStream::Klv).await;
    let want: Vec<&[u8]> = klv_packets.iter().map(|p| &p[..]).collect();
    let got: Vec<&[u8]> = got_klv.iter().map(|(p, _)| &p[..]).collect();
    assert_eq!(got, want, "KLV packets recovered bit-exact");
    assert_eq!(got_klv[0].1, 20_000_000, "KLV PTS preserved");

    // The recovered packets parse back to the telemetry that was encoded.
    for (i, (pkt, _)) in got_klv.iter().enumerate() {
        let ls = UasDatalink::parse(pkt).expect("valid ST 0601 set");
        assert_eq!(ls.timestamp_us, telemetry(i as u64).timestamp_us);
        assert!((ls.sensor_lat_deg.unwrap() - (60.1768 + i as f64 * 0.0001)).abs() < 1e-6);
    }

    // A video selection must not see KLV and vice versa.
    assert!(
        got_video.iter().all(|(au, _)| !au.starts_with(&[0x06])),
        "no KLV leaked into the video port"
    );
}

/// `klvdecode` downstream of the demux turns each recovered local set into one
/// timed text line.
#[tokio::test]
async fn klvdecode_emits_text_lines() {
    let (ts, _) = muxed_ts().await;
    let klv_frames = demux_stream(&ts, TsStream::Klv).await;

    let mut dec = KlvDecode::new();
    dec.configure_pipeline(&Caps::Klv).unwrap();
    let mut sink = CaptureSinkAus::default();
    for (bytes, pts) in &klv_frames {
        // Sanity: what the demux emits is splittable KLV.
        assert_eq!(split_klv_packets(bytes).len(), 1);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming {
                pts_ns: *pts,
                ..FrameTiming::default()
            },
            0,
        );
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    assert_eq!(dec.emitted(), 2, "one line per local set");
    let line = String::from_utf8(sink.aus[0].0.clone()).unwrap();
    assert!(line.contains("lat=60.17"), "line carries latitude: {line}");
    assert!(
        line.contains("heading=87.3"),
        "line carries heading: {line}"
    );
    assert_eq!(sink.aus[0].1, 20_000_000, "text keeps the KLV PTS");
}
