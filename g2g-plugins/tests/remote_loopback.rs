//! Distributed-graph primitive end-to-end over loopback (M551): a graph edge
//! cut across a TCP boundary. The near side drives a `RemoteSink` (serializing
//! the `PipelinePacket` stream via the g2g-core wire codec); the far side runs a
//! `RemoteSrc` -> collecting sink pipeline that reconstructs the identical
//! stream. Proves the leading `CapsChanged` is discovered from the wire, every
//! `DataFrame`'s bytes / timing / sequence survive the round trip, and the
//! stream ends on `Eos` (g2g <-> g2g loopback; the codec's per-variant fidelity
//! is unit-tested in `g2g_core::wire`).
#![cfg(feature = "remote")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

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
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Collects the caps and each frame's (sequence, timing, bytes) so the test can
/// assert the whole stream crossed the boundary intact.
#[derive(Default)]
struct CollectSink {
    caps: Vec<Caps>,
    frames: Vec<(u64, FrameTiming, Vec<u8>)>,
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
        // The initial caps reach a sink via configure_pipeline, not process.
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
                PipelinePacket::DataFrame(frame) => {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.frames
                            .push((frame.sequence, frame.timing, slice.to_vec()));
                    }
                }
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
        width: Dim::Fixed(4),
        height: Dim::Fixed(4),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[tokio::test]
async fn remote_transport_carries_a_split_graph_edge() {
    const N: u8 = 8;
    const FRAME_LEN: usize = 4 * 4 * 4; // 4x4 RGBA

    // Far side: bind the listener up front (so the near side can connect before
    // accept() runs) and read the actual ephemeral port.
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    let mut src = RemoteSrc::from_listener(listener)
        .unwrap()
        .with_frame_limit(N as u64);
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Near side: drive a RemoteSink dialing the far-side port, sending the
    // negotiated caps then N tagged frames then Eos.
    let sender = async {
        let dest = format!("127.0.0.1:{port}").parse().unwrap();
        let mut remote = RemoteSink::new(dest);
        remote
            .configure_pipeline(&test_caps())
            .expect("connect + configure");
        let mut null = NullOut;
        for i in 0u8..N {
            let mut bytes = vec![i; FRAME_LEN];
            bytes[1] = 0xAB; // a second marker so a mis-framed read is obvious
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: i as u64 * 1_000_000,
                    dts_ns: i as u64 * 1_000_000,
                    duration_ns: 33_000,
                    keyframe: i == 0,
                    ..FrameTiming::default()
                },
                sequence: i as u64,
                meta: Default::default(),
            };
            if remote
                .process(PipelinePacket::DataFrame(frame), &mut null)
                .await
                .is_err()
            {
                break;
            }
        }
        // The far side stops at its frame limit; a late Eos send may fail once
        // it has closed, which is fine.
        let _ = remote.process(PipelinePacket::Eos, &mut null).await;
        remote.sent()
    };

    let recv = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );

    let (recv_res, sent) = tokio::join!(recv, sender);
    let stats = recv_res
        .expect("receiver finishes within 10s")
        .expect("receive pipeline ok");

    // Every frame delivered.
    assert_eq!(
        stats.frames_emitted, N as u64,
        "all frames crossed the boundary"
    );
    assert!(
        sent >= (N as u64 + 1),
        "sender emitted caps + {N} frames: {sent}"
    );

    // The far side discovered the sender's caps from the wire.
    assert!(!sink.caps.is_empty(), "caps were carried");
    assert_eq!(
        sink.caps[0],
        test_caps(),
        "discovered caps match the sender's"
    );

    // Each frame's sequence, timing, and bytes survived byte-for-byte.
    assert_eq!(sink.frames.len(), N as usize);
    for (i, (seq, timing, bytes)) in sink.frames.iter().enumerate() {
        assert_eq!(*seq, i as u64, "sequence preserved");
        assert_eq!(timing.pts_ns, i as u64 * 1_000_000, "pts preserved");
        assert_eq!(timing.keyframe, i == 0, "keyframe flag preserved");
        assert_eq!(bytes.len(), FRAME_LEN, "frame length preserved");
        assert_eq!(bytes[0], i as u8, "payload tag preserved (correct framing)");
        assert_eq!(bytes[1], 0xAB, "second marker preserved (no mis-framing)");
    }
    assert!(sink.eos, "the stream ended on Eos");
}
