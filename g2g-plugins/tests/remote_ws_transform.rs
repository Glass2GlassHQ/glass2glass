//! Remote-transform primitive over a WebSocket (M555): a middle graph stage
//! offloaded to a remote peer and its processed output brought back, in-graph.
//! A `CountSrc -> RemoteWsTransform -> CollectSink` pipeline runs against a test
//! WebSocket server that reads the wire stream, *inverts* each frame's bytes (a
//! stand-in for a real remote stage, e.g. inference), and replies one processed
//! frame per frame. Proves the round trip: the caps reach the peer, each frame is
//! sent and its processed reply emitted downstream in order, and the transform is
//! genuinely applied (bytes come back inverted). This is the bidirectional
//! generalization the browser detection offload needs; the codec's per-variant
//! fidelity is unit-tested in `g2g_core::wire`.
#![cfg(feature = "remote-ws")]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_transform_sink, LatencyProfile, SourceLoop};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

use g2g_plugins::remotewstransform::RemoteWsTransform;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
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

/// Emits `n` RGBA frames (each byte = frame index) then EOS.
struct CountSrc {
    n: u8,
    configured: bool,
}

impl SourceLoop for CountSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(test_caps()))
    }
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(test_caps()))))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(test_caps())).await?;
            const LEN: usize = 2 * 2 * 4;
            for i in 0..self.n {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i; LEN].into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: i as u64 * 1000,
                        ..FrameTiming::default()
                    },
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n as u64)
        })
    }
}

/// Collects each emitted frame's (sequence, first byte) so the test can assert
/// the processed bytes came back.
#[derive(Default)]
struct CollectSink {
    frames: Vec<(u64, u8)>,
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
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                    if let MemoryDomain::System(s) = &frame.domain {
                        self.frames.push((frame.sequence, s.as_slice()[0]));
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// The remote stage: read the wire stream, invert each frame's bytes, reply one
/// processed frame per frame. Ignores caps (config only) and ends on Eos; never
/// echoes control, so the client's per-frame read pairs with its frame.
async fn invert_server(listener: StdTcpListener) -> Result<u64, Box<dyn std::error::Error>> {
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let (tcp, _) = listener.accept().await?;
    let mut ws = tokio_tungstenite::accept_async(tcp).await?;
    let mut processed = 0u64;
    while let Some(msg) = ws.next().await {
        let Message::Binary(bytes) = msg? else {
            continue;
        };
        match decode_packet(&bytes).map_err(|e| format!("decode: {e:?}"))? {
            PipelinePacket::DataFrame(mut frame) => {
                if let MemoryDomain::System(s) = &mut frame.domain {
                    for b in s.as_mut_slice() {
                        *b = !*b; // the "processing": invert every byte
                    }
                }
                let out = encode_packet(&PipelinePacket::DataFrame(frame))
                    .map_err(|e| format!("encode: {e:?}"))?;
                ws.send(Message::Binary(out)).await?;
                processed += 1;
            }
            PipelinePacket::Eos => break,
            _ => {} // caps / segment: no reply
        }
    }
    Ok(processed)
}

#[tokio::test]
async fn remote_ws_transform_offloads_and_returns_processed_frames() {
    const N: u8 = 6;

    // The remote stage: bind up front so the transform can connect once running.
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();

    let mut src = CountSrc {
        n: N,
        configured: false,
    };
    let mut xform = RemoteWsTransform::new(format!("ws://127.0.0.1:{port}"));
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Drive the pipeline and the remote stage concurrently on this task (a
    // Box<dyn Error> server result is not Send, so join! not spawn).
    let run = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    );
    let (run_res, server_res) = tokio::join!(run, invert_server(listener));
    let stats = run_res.expect("finishes within 10s").expect("pipeline ok");

    // Every frame round-tripped through the remote stage.
    assert_eq!(
        stats.frames_emitted, N as u64,
        "all frames crossed and returned"
    );
    assert_eq!(xform.emitted(), N as u64, "transform emitted one per frame");
    assert_eq!(sink.frames.len(), N as usize);
    for (i, (seq, byte)) in sink.frames.iter().enumerate() {
        assert_eq!(*seq, i as u64, "order preserved (FIFO reply pairing)");
        // The source sent byte == i; the remote stage inverted it.
        assert_eq!(
            *byte,
            !(i as u8),
            "the remote stage's processing was applied"
        );
    }
    assert!(sink.eos, "stream ended on Eos");

    let processed = server_res.expect("server ok");
    assert_eq!(processed, N as u64, "server processed every frame");
}
