//! M207 - multi-stream MPEG-TS muxing. A video and an audio source are muxed
//! into ONE MPEG-TS byte stream (interleaved by PTS), then each elementary
//! stream is recovered by demuxing it back out with the existing single-output
//! `TsDemux`. The everyday A+V container round-trip.

use std::pin::Pin;

use core::future::Future;

use g2g_core::element::{AsyncElement, BoxFuture, PushOutcome};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};

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
    }
}
fn aac_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }
}
fn ts_caps() -> Caps {
    Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
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
                    FrameTiming { pts_ns: *pts, ..FrameTiming::default() },
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
            if let MemoryDomain::System(s) = &f.domain {
                self.bytes.extend_from_slice(s.as_slice());
            }
        }
        Box::pin(async { Ok(()) })
    }
}

/// Drive a whole TS byte buffer through a single-output `TsDemux` selecting
/// `stream`, returning the access units it recovers.
async fn demux_stream(ts: &[u8], stream: TsStream) -> Vec<Vec<u8>> {
    let mut demux = TsDemux::new().with_stream(stream);
    demux.configure_pipeline(&ts_caps()).unwrap();
    let mut sink = CaptureSinkAus::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(ts.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.aus
}

/// An `OutputSink` that records each recovered access unit's bytes.
#[derive(Default)]
struct CaptureSinkAus {
    aus: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSinkAus {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.aus.push(s.as_slice().to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn av_muxed_into_one_ts_then_demuxed_back() {
    // Video AUs at 0/40/80 ms, audio AUs at 20/60 ms: time-interleaved.
    let video_aus = vec![
        (vec![0u8, 0, 0, 1, 0x65, 0x11], 0),
        (vec![0u8, 0, 0, 1, 0x41, 0x22], 40_000_000),
        (vec![0u8, 0, 0, 1, 0x41, 0x33], 80_000_000),
    ];
    let audio_aus = vec![
        (vec![0xFFu8, 0xF1, 0xAA], 20_000_000),
        (vec![0xFFu8, 0xF1, 0xBB], 60_000_000),
    ];

    let mut video = AuSrc { caps: h264_caps(), aus: video_aus.clone(), configured: false };
    let mut audio = AuSrc { caps: aac_caps(), aus: audio_aus.clone(), configured: false };
    let mut mux = TsMux::new(2); // input 0 = video, input 1 = audio
    let mut sink = CaptureSink::default();

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut video, &mut audio];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("A+V mux pipeline completes");
    }
    assert_eq!(mux.emitted(), 5, "all five AUs (3 video + 2 audio) muxed");
    assert!(!sink.bytes.is_empty(), "produced a TS byte stream");

    // Demux each elementary stream back out of the single multiplex.
    let got_video = demux_stream(&sink.bytes, TsStream::H264).await;
    let got_audio = demux_stream(&sink.bytes, TsStream::Aac).await;

    let want_video: Vec<Vec<u8>> = video_aus.iter().map(|(au, _)| au.clone()).collect();
    let want_audio: Vec<Vec<u8>> = audio_aus.iter().map(|(au, _)| au.clone()).collect();
    assert_eq!(got_video, want_video, "video AUs recovered from the multiplex");
    assert_eq!(got_audio, want_audio, "audio AUs recovered from the multiplex");
}
