//! M558: reconnection for the distributed-graph TCP transport. Two scenarios:
//!  A. sink connect-retry: a `RemoteSink::with_reconnect` tolerates a peer that is
//!     not up yet, retrying the connect until the `RemoteSrc` binds.
//!  B. source keep-listening: a `RemoteSrc::with_reconnect` accepts a replacement
//!     client when the first drops without a clean `Eos`, so the stream continues
//!     across a sender restart.
#![cfg(feature = "remote")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};

use g2g_plugins::remotesink::RemoteSink;
use g2g_plugins::remotesrc::RemoteSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

#[derive(Default)]
struct CollectSink {
    caps: Vec<Caps>,
    seqs: Vec<u64>,
    eos: bool,
}
impl AsyncElement for CollectSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps.push(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.seqs.push(frame.sequence),
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(
            vec![seq as u8; 16].into_boxed_slice(),
        )),
        timing: FrameTiming {
            pts_ns: seq * 1_000_000,
            ..FrameTiming::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

/// A: the sink retries the connect until the far side comes up.
#[tokio::test]
async fn sink_retries_connect_until_server_is_up() {
    const N: u64 = 4;

    // Reserve a port, then release it so the sink's first connect fails.
    let port = {
        let l = StdTcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dest = format!("127.0.0.1:{port}").parse().unwrap();

    // Near side: a reconnecting sink starts immediately (server not up yet).
    let sender = async move {
        let mut sink = RemoteSink::new(dest).with_reconnect(200);
        sink.configure_pipeline(&test_caps())
            .expect("configure defers connect");
        let mut null = NullOut;
        for i in 0..N {
            sink.process(PipelinePacket::DataFrame(frame(i)), &mut null)
                .await
                .expect("frame delivered after reconnect");
        }
        let _ = sink.process(PipelinePacket::Eos, &mut null).await;
    };

    // Far side: bind the same port only after a delay, so the sink must retry.
    let receiver = async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let listener = StdTcpListener::bind(("127.0.0.1", port)).expect("late bind");
        let mut src = RemoteSrc::from_listener(listener)
            .unwrap()
            .with_frame_limit(N);
        let mut sink = CollectSink::default();
        let clock = ZeroClock;
        let stats = run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        )
        .await
        .expect("receive ok");
        (stats.frames_emitted, sink)
    };

    let recv = tokio::time::timeout(Duration::from_secs(10), receiver);
    let (recv_res, ()) = tokio::join!(recv, sender);
    let (emitted, sink) = recv_res.expect("finishes within 10s");

    assert_eq!(
        emitted, N,
        "all frames crossed after the sink retried the connect"
    );
    assert_eq!(
        sink.caps.first(),
        Some(&test_caps()),
        "caps discovered from the wire"
    );
    assert_eq!(
        sink.seqs,
        (0..N).collect::<Vec<_>>(),
        "every frame in order"
    );
}

/// B: the source keeps listening and accepts a replacement client when the first
/// drops without a clean Eos, so the stream survives a sender restart.
#[tokio::test]
async fn source_reaccepts_after_sender_drops() {
    // 2 frames from the first sender, then it "crashes" (drops without Eos); 3
    // more from a replacement. frame_limit ends the stream at 5.
    const TOTAL: u64 = 5;

    let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dest = format!("127.0.0.1:{port}").parse().unwrap();
    let mut src = RemoteSrc::from_listener(listener)
        .unwrap()
        .with_reconnect()
        .with_frame_limit(TOTAL);
    let mut collect = CollectSink::default();
    let clock = ZeroClock;

    let sender = async move {
        let mut null = NullOut;
        // First sender: caps + 2 frames, then drop (no Eos) by going out of scope.
        {
            let mut s1 = RemoteSink::new(dest);
            s1.configure_pipeline(&test_caps()).expect("s1 connect");
            for i in 0..2u64 {
                s1.process(PipelinePacket::DataFrame(frame(i)), &mut null)
                    .await
                    .expect("s1 frame");
            }
            // s1 dropped here: its TCP connection closes (EOF, no Eos).
        }
        // Give the source a moment to observe the drop and re-accept.
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Replacement sender: caps + the remaining frames (+ a late Eos, which the
        // source may have already ended on its frame limit).
        let mut s2 = RemoteSink::new(dest).with_reconnect(50);
        s2.configure_pipeline(&test_caps()).expect("s2 connect");
        for i in 2..TOTAL {
            let _ = s2
                .process(PipelinePacket::DataFrame(frame(i)), &mut null)
                .await;
        }
        let _ = s2.process(PipelinePacket::Eos, &mut null).await;
    };

    let recv = tokio::time::timeout(
        Duration::from_secs(10),
        run_simple_pipeline(
            &mut src,
            &mut collect,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (recv_res, ()) = tokio::join!(recv, sender);
    let stats = recv_res.expect("finishes within 10s").expect("receive ok");

    assert_eq!(
        stats.frames_emitted, TOTAL,
        "all frames across both senders delivered"
    );
    assert_eq!(
        collect.seqs,
        (0..TOTAL).collect::<Vec<_>>(),
        "frames in order across the reconnect"
    );
    // The initial caps (configure) plus the replacement's re-sent caps.
    assert!(
        collect.caps.len() >= 2,
        "re-accepted client re-sent its caps: {:?}",
        collect.caps
    );
    assert!(collect.eos, "stream ended (frame limit)");
}
