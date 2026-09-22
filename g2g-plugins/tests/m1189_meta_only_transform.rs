//! M1189 - a remote transform that brings back metadata alone.
//!
//! The generic round trip ships the whole frame both ways, which is waste for a
//! stage that only attaches metadata (inference, analytics): the pixels it
//! returns are the pixels that were sent. With `meta-only` the client keeps its
//! frame, the peer replies with an empty payload carrying the meta, and the
//! client emits its own frame with that meta on it.
//!
//! The peer here is a real WebSocket server speaking the wire codec, attaching
//! one detection per frame. The assertions are the two halves of the claim: the
//! detections arrive, and the frame the sink sees is the one the source sent
//! (pixels untouched, and no pixels on the wire coming back).
#![cfg(all(feature = "remote-ws", feature = "metadata"))]

use core::future::Future;
use core::pin::Pin;
use std::net::TcpListener as StdTcpListener;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection};
use g2g_core::runtime::{run_source_transform_sink, LatencyProfile, SourceLoop};
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, PropValue, Rate, RawVideoFormat,
};

use g2g_plugins::remotewstransform::RemoteWsTransform;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const FRAMES: u8 = 4;
/// A frame big enough that the pixels dwarf the metadata, which is the whole
/// point of the mode: 64x64 RGBA.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const PIXELS: usize = (WIDTH * HEIGHT * 4) as usize;
/// The label the peer attaches, so the sink's reading names its source.
const DETECTED_LABEL: u32 = 7;

fn test_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Emits `FRAMES` RGBA frames (every byte = the frame index) then EOS.
struct CountSrc {
    sent: u8,
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

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(test_caps()))
    }
    fn caps_constraint(
        &mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'_>, G2gError>> + '_ {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(test_caps()))))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(test_caps())).await?;
            for i in 0..FRAMES {
                out.push(PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i; PIXELS].into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: u64::from(i) * 1000,
                        ..FrameTiming::default()
                    },
                    u64::from(i),
                )))
                .await?;
                self.sent += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(u64::from(FRAMES))
        })
    }
}

/// Records each frame's first pixel byte and the label of its lone detection.
#[derive(Default)]
struct CollectSink {
    frames: Vec<(u8, Option<u32>)>,
}

impl AsyncElement for CollectSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
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
        if let PipelinePacket::DataFrame(frame) = packet {
            let pixel = frame.domain.as_system_slice().map(|s| s[0]).unwrap_or(0);
            let label = frame
                .meta
                .get::<AnalyticsMeta>()
                .and_then(|m| m.detections().next().map(|d| d.label));
            self.frames.push((pixel, label));
        }
        core::future::ready(Ok(()))
    }
}

/// The remote stage: attach one detection per frame and reply with the meta
/// alone (an empty payload), the pixels-unchanged contract. Returns the bytes
/// it sent back, so the test can assert no pixels crossed.
async fn detect_server(listener: StdTcpListener) -> Result<usize, Box<dyn std::error::Error>> {
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let (tcp, _) = listener.accept().await?;
    let mut ws = tokio_tungstenite::accept_async(tcp).await?;
    let mut reply_bytes = 0usize;
    while let Some(msg) = ws.next().await {
        let Message::Binary(bytes) = msg? else {
            continue;
        };
        match decode_packet(&bytes).map_err(|e| format!("decode: {e:?}"))? {
            PipelinePacket::DataFrame(frame) => {
                let mut meta = AnalyticsMeta::new();
                meta.nodes.push(AnalyticsNode::Detection(ObjectDetection {
                    bbox: BBox {
                        x: 0.0,
                        y: 0.0,
                        w: 1.0,
                        h: 1.0,
                    },
                    label: DETECTED_LABEL,
                    confidence: 0.5,
                }));
                let mut reply = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Vec::new().into_boxed_slice())),
                    frame.timing,
                    frame.sequence,
                );
                reply.meta.attach(meta);
                let out = encode_packet(&PipelinePacket::DataFrame(reply))
                    .map_err(|e| format!("encode: {e:?}"))?;
                reply_bytes += out.len();
                ws.send(Message::Binary(out)).await?;
            }
            PipelinePacket::Eos => break,
            _ => {}
        }
    }
    Ok(reply_bytes)
}

#[tokio::test]
async fn a_meta_only_peer_returns_detections_and_no_pixels() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();

    let mut src = CountSrc { sent: 0 };
    let mut xform = RemoteWsTransform::new(format!("ws://127.0.0.1:{port}")).with_meta_only(true);
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

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
    let (run_res, server_res) = tokio::join!(run, detect_server(listener));
    run_res.expect("finishes within 10s").expect("pipeline ok");

    assert_eq!(sink.frames.len(), FRAMES as usize);
    for (i, (pixel, label)) in sink.frames.iter().enumerate() {
        assert_eq!(
            *pixel, i as u8,
            "the frame the sink sees is the one the source sent"
        );
        assert_eq!(
            *label,
            Some(DETECTED_LABEL),
            "the peer's detection rode back on it"
        );
    }

    // Each reply is a header plus one detection, so the return trip is a
    // fraction of what the frames cost going out.
    let reply_bytes = server_res.expect("server ok");
    let sent_pixels = FRAMES as usize * PIXELS;
    assert!(
        reply_bytes * 10 < sent_pixels,
        "the replies carried no pixels: {reply_bytes} bytes against {sent_pixels} sent"
    );
}

/// `meta-only` is a runtime property, the path a launch line takes.
#[test]
fn meta_only_is_a_runtime_property() {
    let mut xform = RemoteWsTransform::new("ws://127.0.0.1:1");
    assert_eq!(
        xform.get_property("meta-only"),
        Some(PropValue::Bool(false))
    );
    xform
        .set_property("meta-only", PropValue::Bool(true))
        .expect("meta-only is settable");
    assert!(xform.meta_only());
    assert_eq!(xform.get_property("meta-only"), Some(PropValue::Bool(true)));
}

/// A peer that replies with pixels while the client expects metadata alone is a
/// disagreement about the protocol, so the run fails rather than dropping what
/// it returned.
#[tokio::test]
async fn a_payload_reply_in_meta_only_mode_fails_the_run() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();

    let mut src = CountSrc { sent: 0 };
    let mut xform = RemoteWsTransform::new(format!("ws://127.0.0.1:{port}")).with_meta_only(true);
    let mut sink = CollectSink::default();
    let clock = ZeroClock;

    // Echoes the frame back whole, the full-round-trip peer. Spawned rather than
    // joined: the failing client holds its socket open past the run, so the
    // server's read never ends on its own.
    let echo = tokio::spawn(async move {
        listener.set_nonblocking(true).ok();
        let listener = tokio::net::TcpListener::from_std(listener).expect("listener");
        let (tcp, _) = listener.accept().await.expect("accept");
        let mut ws = tokio_tungstenite::accept_async(tcp)
            .await
            .expect("handshake");
        while let Some(Ok(Message::Binary(bytes))) = ws.next().await {
            if let Ok(PipelinePacket::DataFrame(frame)) = decode_packet(&bytes) {
                let out = encode_packet(&PipelinePacket::DataFrame(frame)).expect("encode");
                if ws.send(Message::Binary(out)).await.is_err() {
                    break;
                }
            }
        }
    });

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_source_transform_sink(
            &mut src,
            &mut xform,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("finishes within 10s");
    echo.abort();
    assert!(
        outcome.is_err(),
        "a peer returning pixels in meta-only mode fails the run"
    );
}
