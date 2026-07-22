//! M380 - per-branch decoder negotiation. A multi-output demuxer declares each
//! port's elementary-stream caps (`MultiOutputElement::port_output_caps`), so the
//! graph solver negotiates each branch against its port's codec rather than
//! broadcasting the demux's byte-stream input. The payoff: a real decoder (one
//! that *requires* its codec caps and rejects the byte stream) configures at
//! startup, instead of having to be an accept-anything element that retypes only
//! at runtime.
//!
//! `BytesSrc(Matroska) -> MkvDemuxN(H.264, AAC) -> picky decoder -> sink` per
//! port. Each `PickyDecoder` rejects anything but its codec at negotiation, so the
//! graph only runs if the solver handed each branch its port caps (the M380 fix);
//! the test asserts each decoder was configured with its elementary caps (not the
//! `ByteStream` input) and that frames flow through.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    G2gError, Graph, MultiInputElement, OutputSink, PipelineClock, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A decoder that accepts ONLY its codec (rejecting the demux's byte-stream input)
/// and produces a raw stream. It records the caps it was configured with, so the
/// test can confirm per-branch negotiation handed it the elementary caps. Frames
/// pass through (the test cares about negotiation + flow, not real decoding).
struct PickyDecoder {
    accept_video: bool,
    seen: Arc<Mutex<Option<Caps>>>,
}
impl core::fmt::Debug for PickyDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PickyDecoder").finish_non_exhaustive()
    }
}
fn picky_accepts(accept_video: bool, c: &Caps) -> bool {
    if accept_video {
        matches!(
            c,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        )
    } else {
        matches!(
            c,
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        )
    }
}
impl AsyncElement for PickyDecoder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        // Requires its codec, rejects the byte stream: only per-branch negotiation
        // (handing it the port caps) lets it configure. Passthrough output (the
        // negotiation, not real decoding, is what M380 is about).
        if picky_accepts(self.accept_video, c) {
            Ok(c.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let accept_video = self.accept_video;
        CapsConstraint::DerivedOutput(Box::new(move |c: &Caps| {
            if picky_accepts(accept_video, c) {
                CapsSet::one(c.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }
    fn configure_pipeline(&mut self, c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !picky_accepts(self.accept_video, c) {
            return Err(G2gError::CapsMismatch);
        }
        *self.seen.lock().unwrap() = Some(c.clone());
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            // The runner forwards Eos; everything else passes through.
            if !matches!(packet, PipelinePacket::Eos) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

#[derive(Default)]
struct CountSink {
    frames: Arc<AtomicUsize>,
}
impl AsyncElement for CountSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(_) = packet {
            self.frames.fetch_add(1, Ordering::Relaxed);
        }
        Box::pin(async { Ok(()) })
    }
}

// --- container source + A/V mux fixture ---
#[derive(Debug)]
struct BytesSrc {
    bytes: Option<Vec<u8>>,
}
impl SourceLoop for BytesSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;
    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bytes = self.bytes.take().unwrap_or_default();
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}
async fn mux_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(
        0,
        &Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
        },
    )
    .unwrap();
    mux.configure_pipeline(
        1,
        &Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        },
    )
    .unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

#[tokio::test]
async fn picky_decoders_negotiate_through_demux_branches() {
    let file = mux_av().await;
    let seen_v = Arc::new(Mutex::new(None));
    let seen_a = Arc::new(Mutex::new(None));
    let frames_v = Arc::new(AtomicUsize::new(0));
    let frames_a = Arc::new(AtomicUsize::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(BytesSrc { bytes: Some(file) }));
    let demux = g.add_demux(
        GraphNode::demux(MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac])),
        2,
    );
    let dec_v = g.add_transform(GraphNode::element(PickyDecoder {
        accept_video: true,
        seen: seen_v.clone(),
    }));
    let dec_a = g.add_transform(GraphNode::element(PickyDecoder {
        accept_video: false,
        seen: seen_a.clone(),
    }));
    let s0 = g.add_sink(GraphNode::element(CountSink {
        frames: frames_v.clone(),
    }));
    let s1 = g.add_sink(GraphNode::element(CountSink {
        frames: frames_a.clone(),
    }));
    g.link(src, demux.input()).unwrap();
    g.link(demux.out(0), dec_v).unwrap();
    g.link(dec_v, s0).unwrap();
    g.link(demux.out(1), dec_a).unwrap();
    g.link(dec_a, s1).unwrap();

    // The run only succeeds if each picky decoder negotiated against its port's
    // elementary caps (a broadcast tee would have handed it the byte stream, which
    // its intercept rejects -> CapsMismatch).
    let stats = run_graph(g, &NullClock, 4)
        .await
        .expect("per-branch decoder negotiation runs");
    assert_eq!(
        stats.frames_consumed, 4,
        "all four access units decoded across the branches"
    );

    // Each decoder was configured with its elementary stream, not the byte stream.
    let v = seen_v
        .lock()
        .unwrap()
        .clone()
        .expect("video decoder configured");
    assert!(
        matches!(
            v,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ),
        "video branch got H.264, got {v:?}"
    );
    let a = seen_a
        .lock()
        .unwrap()
        .clone()
        .expect("audio decoder configured");
    assert!(
        matches!(
            a,
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        ),
        "audio branch got AAC, got {a:?}"
    );

    assert_eq!(
        frames_v.load(Ordering::Relaxed),
        2,
        "two decoded video frames"
    );
    assert_eq!(
        frames_a.load(Ordering::Relaxed),
        2,
        "two decoded audio frames"
    );
}
