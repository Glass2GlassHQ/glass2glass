//! M10 fan-in: a true muxer combines all inputs (vs the Merger selector),
//! negotiates each input independently, and aggregates EOS, driven through
//! `run_muxer_sink`.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, VideoFormat,
};
use g2g_plugins::mux::InterleaveMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn vcaps(width: Dim) -> Caps {
    Caps::Video { format: VideoFormat::Rgba8, width, height: Dim::Fixed(480), framerate: Rate::Fixed(30 << 16) }
}

fn make_frame(seq: u64, caps: Caps) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        caps,
        timing: FrameTiming { pts_ns: 0, dts_ns: 0, duration_ns: 0, capture_ns: 0 },
        sequence: seq,
    }
}

/// Source that advertises `advertise` caps and emits `count` frames numbered
/// from `start_seq`, then EOS.
struct CapSrc {
    advertise: Caps,
    start_seq: u64,
    count: u64,
    configured: bool,
}

impl SourceLoop for CapSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.advertise.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let start = self.start_seq;
        let count = self.count;
        let configured = self.configured;
        let frame_caps = self.advertise.clone();
        Box::pin(async move {
            assert!(configured, "runner must configure source before run");
            for i in 0..count {
                out.push(PipelinePacket::DataFrame(make_frame(start + i, frame_caps.clone())))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

/// Order-independent sink: records the set of sequences received and how many
/// EOS it saw. Interleaved arrival is not monotonic, so `FakeSink` won't do.
#[derive(Default)]
struct CollectingSink {
    seqs: Vec<u64>,
    eos_count: u64,
}

impl AsyncElement for CollectingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        match packet {
            PipelinePacket::DataFrame(f) => self.seqs.push(f.sequence),
            PipelinePacket::Eos => self.eos_count += 1,
            PipelinePacket::CapsChanged(_) => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn muxer_negotiates_each_input_independently() {
    // Each input advertises a different width range; per-input fixation must
    // land on each input's own minimum.
    let mut a = CapSrc { advertise: vcaps(Dim::Range { min: 640, max: 1920 }), start_seq: 0, count: 0, configured: false };
    let mut b = CapSrc { advertise: vcaps(Dim::Range { min: 1280, max: 3840 }), start_seq: 0, count: 0, configured: false };
    let mut mux = InterleaveMux::new(2, vcaps(Dim::Fixed(640)));
    let mut snk = CollectingSink::default();
    let clock = ZeroClock;

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut snk, &clock, 4)
            .await
            .expect("muxer pipeline should complete");
    }

    assert_eq!(mux.input_caps(0), Some(&vcaps(Dim::Fixed(640))), "input 0 fixated to its own min");
    assert_eq!(mux.input_caps(1), Some(&vcaps(Dim::Fixed(1280))), "input 1 fixated to its own min");
}

#[tokio::test]
async fn muxer_forwards_all_inputs_and_aggregates_eos() {
    let caps = vcaps(Dim::Fixed(64));
    let mut a = CapSrc { advertise: caps.clone(), start_seq: 0, count: 3, configured: false };
    let mut b = CapSrc { advertise: caps.clone(), start_seq: 100, count: 3, configured: false };
    let mut mux = InterleaveMux::new(2, caps.clone());
    let mut snk = CollectingSink::default();
    let clock = ZeroClock;

    let stats = {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut snk, &clock, 4)
            .await
            .expect("muxer pipeline should complete")
    };

    assert_eq!(stats.frames_emitted, 6);
    assert_eq!(stats.frames_consumed, 6, "all inputs forwarded, not just one");

    let mut got = snk.seqs.clone();
    got.sort_unstable();
    assert_eq!(got, vec![0, 1, 2, 100, 101, 102], "every input's frames reached the sink");
    assert_eq!(snk.eos_count, 1, "single EOS after both inputs ended");
}
