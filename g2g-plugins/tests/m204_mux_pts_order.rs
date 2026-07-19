//! M204 - PTS-ordered muxer fan-in. `InterleaveMux` buffers frames per input
//! and releases the globally earliest-PTS frame once every contributor has one
//! queued, so two time-skewed inputs merge in timestamp order rather than
//! arrival order (the GstAggregator collect-and-pick-earliest rule).

use std::pin::Pin;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

use g2g_plugins::mux::InterleaveMux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits frames at the given PTS values (ns), then EOS. The PTS is also stored
/// as the sequence so the sink can report the merged order directly.
struct PtsSrc {
    pts: Vec<u64>,
    configured: bool,
}

impl SourceLoop for PtsSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let pts = self.pts.clone();
        let configured = self.configured;
        Box::pin(async move {
            assert!(configured, "runner configures the source before run");
            for p in &pts {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; 4].into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: *p,
                        ..FrameTiming::default()
                    },
                    sequence: *p,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                // Yield so the two sources actually interleave their delivery.
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pts.len() as u64)
        })
    }
}

/// Records the PTS of each frame in arrival (output) order.
#[derive(Default)]
struct OrderSink {
    pts: Vec<u64>,
}

impl AsyncElement for OrderSink {
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
        if let PipelinePacket::DataFrame(f) = packet {
            self.pts.push(f.timing.pts_ns);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn merges_two_skewed_inputs_in_pts_order() {
    // Interleaved-in-time streams: A at 0,20,40; B at 10,30,50. Whatever the
    // arrival interleaving, the merged output must be globally PTS-ordered.
    let mut a = PtsSrc {
        pts: vec![0, 20, 40],
        configured: false,
    };
    let mut b = PtsSrc {
        pts: vec![10, 30, 50],
        configured: false,
    };
    let mut mux = InterleaveMux::new(2, caps());
    let mut sink = OrderSink::default();

    let stats = {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("muxer pipeline completes")
    };

    assert_eq!(stats.frames_consumed, 6, "all frames reached the sink");
    assert_eq!(
        sink.pts,
        vec![0, 10, 20, 30, 40, 50],
        "frames emerge in global PTS order, not arrival order"
    );
}

#[tokio::test]
async fn flushes_remaining_when_one_input_ends_early() {
    // A is short (ends at 5) while B continues (10,20,30). After A ends and
    // drains, B's later frames must still flow (A dropping out of the merge).
    let mut a = PtsSrc {
        pts: vec![5],
        configured: false,
    };
    let mut b = PtsSrc {
        pts: vec![10, 20, 30],
        configured: false,
    };
    let mut mux = InterleaveMux::new(2, caps());
    let mut sink = OrderSink::default();

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut mux, &mut sink, &ZeroClock, 4)
            .await
            .expect("muxer pipeline completes");
    }

    assert_eq!(
        sink.pts,
        vec![5, 10, 20, 30],
        "earliest first, then B drains after A ends"
    );
}
