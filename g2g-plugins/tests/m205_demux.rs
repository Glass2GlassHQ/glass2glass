//! M205 - content-based demultiplexer fan-out. One `StreamDemux` splits a
//! multiplexed input onto N typed output ports (the bounded-N "dark slots"),
//! each branch retyped to its elementary stream's caps via a per-port
//! `CapsChanged`. A single demuxer feeds a video branch and an audio branch in
//! one pipeline, the `pad-added` analog.

use std::pin::Pin;

use core::future::Future;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_fanout, SourceLoop};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, ConfigureOutcome, Dim, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, VideoCodec,
};

use g2g_plugins::streamdemux::StreamDemux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn ts_input() -> Caps {
    Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
}
fn video_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Range { min: 16, max: 65_535 },
        height: Dim::Range { min: 16, max: 65_535 },
        framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
    }
}
fn audio_caps() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}

/// Emits a fixed script of frames, each prefixed with a stream-id byte
/// (0 = video, 1 = audio), advertising the TS byte-stream input caps.
struct MuxedSrc {
    script: Vec<u8>, // one stream-id byte per frame, in emission order
    configured: bool,
}

impl SourceLoop for MuxedSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(ts_input()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let script = self.script.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures before run");
            for (i, &tag) in script.iter().enumerate() {
                // Frame payload: [stream-id, sequence]. The demux classifier
                // reads the leading stream-id byte.
                let bytes = vec![tag, i as u8];
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming { pts_ns: i as u64 * 1000, ..FrameTiming::default() },
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(script.len() as u64)
        })
    }
}

/// Records the caps it was retyped to and the stream-id byte of every frame.
#[derive(Default)]
struct BranchSink {
    caps_changes: Vec<Caps>,
    tags: Vec<u8>,
}

impl AsyncElement for BranchSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

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
        match packet {
            PipelinePacket::CapsChanged(c) => self.caps_changes.push(c),
            PipelinePacket::DataFrame(f) => {
                if let MemoryDomain::System(s) = &f.domain {
                    self.tags.push(s.as_slice()[0]);
                }
            }
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn one_demux_splits_into_two_typed_branches() {
    // Interleaved multiplex: V A V A V (3 video, 2 audio).
    let mut src = MuxedSrc { script: vec![0, 1, 0, 1, 0], configured: false };
    let mut demux = StreamDemux::new(
        ts_input(),
        vec![video_caps(), audio_caps()],
        |f| match &f.domain {
            MemoryDomain::System(s) => s.as_slice()[0] as usize,
            _ => 0,
        },
    );
    assert_eq!(demux.port_count(), 2);

    let mut video = BranchSink::default();
    let mut audio = BranchSink::default();

    {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut video, &mut audio];
        run_source_fanout(&mut src, &mut demux, sinks, &ZeroClock, 4)
            .await
            .expect("demux fan-out completes");
    }

    // Each branch was retyped to its elementary stream exactly once. The
    // per-branch re-solve fixates geometry, so match on the media discriminant.
    assert_eq!(video.caps_changes.len(), 1, "video port announced once");
    assert!(
        matches!(video.caps_changes[0], Caps::CompressedVideo { codec: VideoCodec::H264, .. }),
        "video port retyped to H264, got {:?}",
        video.caps_changes[0]
    );
    assert_eq!(audio.caps_changes.len(), 1, "audio port announced once");
    assert!(
        matches!(audio.caps_changes[0], Caps::Audio { format: AudioFormat::Aac, .. }),
        "audio port retyped to AAC, got {:?}",
        audio.caps_changes[0]
    );

    // Frames were routed by stream id: only tag-0 to video, only tag-1 to audio.
    assert_eq!(video.tags, vec![0, 0, 0], "all three video frames reached the video branch");
    assert_eq!(audio.tags, vec![1, 1], "both audio frames reached the audio branch");
}

#[tokio::test]
async fn dark_port_stays_silent_when_its_stream_is_absent() {
    // The multiplex carries only video; the audio port is a dark slot.
    let mut src = MuxedSrc { script: vec![0, 0], configured: false };
    let mut demux = StreamDemux::new(ts_input(), vec![video_caps(), audio_caps()], |f| {
        match &f.domain {
            MemoryDomain::System(s) => s.as_slice()[0] as usize,
            _ => 0,
        }
    });

    let mut video = BranchSink::default();
    let mut audio = BranchSink::default();
    {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut video, &mut audio];
        run_source_fanout(&mut src, &mut demux, sinks, &ZeroClock, 4)
            .await
            .expect("demux fan-out completes");
    }

    assert_eq!(video.tags, vec![0, 0]);
    assert!(audio.tags.is_empty(), "absent stream: the dark port never emits a frame");
    assert!(audio.caps_changes.is_empty(), "and never announces caps");
}
