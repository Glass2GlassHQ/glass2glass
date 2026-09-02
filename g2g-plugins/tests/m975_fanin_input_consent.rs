//! M975: an element consents to a runtime-added input. A source attached to a
//! running fan-in through `DynamicFaninHandle::add_input` is negotiated against
//! the pad it would feed and then offered to the element itself
//! (`MultiInputElement::accepts_runtime_input`), which may refuse what the pad
//! capacity and the pad's caps constraint cannot express (here: a frame too large
//! for the muxer's canvas). A refused or unnegotiable input fails that one add,
//! reported on its `PendingInput`, and the run carries on with the inputs it
//! already has.

#![cfg(feature = "std")]

use core::future::{ready, Future, Ready};
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{
    run_aggregator_dynamic, run_muxer_sink_dynamic, DynSourceLoop, SourceLoop,
};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    MultiInputElement, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

/// Width the muxer's canvas can hold. An input wider than this is refused by the
/// element even though the pad's caps constraint accepts it.
const CANVAS_WIDTH: u32 = 16;

/// Per-pad offset applied to a forwarded frame's sequence, so the merged output
/// names the input each frame came from.
const PAD_SEQUENCE_STRIDE: u64 = 1000;

fn video_caps(width: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Any RGBA video, at any size: what an input pad accepts. Wider than the canvas
/// is the element's call, not the pad's.
fn pad_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Source pushing `n` frames of `caps` then EOS.
struct CountedSource {
    n: u64,
    caps: Caps,
}

impl SourceLoop for CountedSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = Ready<Result<Caps, G2gError>>;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        ready(Ok(self.caps.clone()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.n {
                let frame = g2g_core::frame::Frame::new(
                    g2g_core::MemoryDomain::System(g2g_core::memory::SystemSlice::from_boxed(
                        std::vec![0u8; 4].into_boxed_slice(),
                    )),
                    g2g_core::FrameTiming {
                        pts_ns: seq,
                        ..Default::default()
                    },
                    seq,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Fan-in element with a fixed canvas: its pads accept any RGBA video, but it
/// only takes a runtime input whose frames fit the canvas. Frames are forwarded
/// downstream with their pad stamped into the sequence, and counted per pad.
struct CanvasMuxer {
    pads: usize,
    frames: Arc<Mutex<Vec<u64>>>,
    eos: Arc<Mutex<Vec<u64>>>,
}

impl CanvasMuxer {
    fn new(pads: usize) -> Self {
        Self {
            pads,
            frames: Arc::new(Mutex::new(std::vec![0; pads])),
            eos: Arc::new(Mutex::new(std::vec![0; pads])),
        }
    }
}

impl MultiInputElement for CanvasMuxer {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.pads
    }
    fn intercept_caps(&self, _i: usize, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_input(&self, _i: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(pad_caps()))
    }
    fn accepts_runtime_input(&self, _pad: usize, caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { width, .. } if *width == Dim::Fixed(CANVAS_WIDTH))
    }
    fn configure_pipeline(&mut self, _i: usize, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(video_caps(CANVAS_WIDTH))
    }
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    self.frames.lock().unwrap()[input] += 1;
                    frame.sequence += input as u64 * PAD_SEQUENCE_STRIDE;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::Eos => self.eos.lock().unwrap()[input] += 1,
                _ => {}
            }
            Ok(())
        })
    }
}

/// Records the merged sequence of every frame the muxer emitted, readable while
/// the run is in flight.
struct RecordingSink {
    merged: Arc<Mutex<Vec<u64>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
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
        if let PipelinePacket::DataFrame(frame) = packet {
            self.merged.lock().unwrap().push(frame.sequence);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn late_input_joins_a_running_muxer_and_reaches_its_output() {
    const N: u64 = 200;
    const ATTACH_AFTER: usize = 5;
    let mut mux = CanvasMuxer::new(3);
    let per_pad = mux.frames.clone();
    let merged = Arc::new(Mutex::new(Vec::new()));
    let mut sink = RecordingSink {
        merged: merged.clone(),
    };

    // Depth 2 so the sources back-pressure and the run spans many polls, leaving
    // the late input a real mid-stream window to attach in.
    let (handle, run) = run_muxer_sink_dynamic(&mut mux, &mut sink, 2);
    let first = handle
        .add_input(Box::new(CountedSource {
            n: N,
            caps: video_caps(CANVAS_WIDTH),
        }) as Box<dyn DynSourceLoop>)
        .expect("the first input queues");

    let control = {
        let merged = merged.clone();
        async move {
            let first = first.accepted().await;
            for _ in 0..1_000_000 {
                if merged.lock().unwrap().len() >= ATTACH_AFTER {
                    break;
                }
                tokio::task::yield_now().await;
            }
            let emitted_before = merged.lock().unwrap().len();
            let late = handle
                .add_input(Box::new(CountedSource {
                    n: N,
                    caps: video_caps(CANVAS_WIDTH),
                }) as Box<dyn DynSourceLoop>)
                .expect("the late input queues");
            let late_pad = late.pad();
            let late = late.accepted().await;
            drop(handle);
            (first, emitted_before, late_pad, late)
        }
    };

    let (stats, (first, emitted_before, late_pad, late)) = tokio::join!(run, control);
    let stats = stats.expect("dynamic muxer run");
    assert_eq!(first, Ok(()), "the element accepted the first input");
    assert_eq!(late, Ok(()), "the element accepted the late input");
    assert_eq!(late_pad, 1, "the late input took the next free pad");
    assert!(
        emitted_before >= ATTACH_AFTER,
        "the late input was added mid-stream, after {emitted_before} merged frames"
    );

    let merged = merged.lock().unwrap();
    let from_first = merged.iter().filter(|s| **s < PAD_SEQUENCE_STRIDE).count();
    let from_late = merged.iter().filter(|s| **s >= PAD_SEQUENCE_STRIDE).count();
    assert_eq!(
        from_first, N as usize,
        "the merged output carries the pre-existing input's frames"
    );
    assert!(
        from_late > 0,
        "the merged output carries the late input's frames"
    );
    assert_eq!(
        from_late, N as usize,
        "the late input's whole stream was merged"
    );
    assert!(
        merged[..ATTACH_AFTER]
            .iter()
            .all(|s| *s < PAD_SEQUENCE_STRIDE),
        "nothing from the late input appears before it attached"
    );
    assert_eq!(
        stats.frames_consumed,
        merged.len() as u64,
        "the sink's merged frames are the run's consumed count"
    );
    assert_eq!(
        *per_pad.lock().unwrap(),
        std::vec![N, N, 0],
        "per-pad routing: the third pad stayed dark"
    );
}

#[tokio::test]
async fn a_refused_input_fails_its_add_and_leaves_the_run_running() {
    const GOOD: u64 = 4;
    let mut agg = CanvasMuxer::new(2);
    let per_pad = agg.frames.clone();
    let (handle, run) = run_aggregator_dynamic(&mut agg, 4);

    let fitting = handle
        .add_input(Box::new(CountedSource {
            n: GOOD,
            caps: video_caps(CANVAS_WIDTH),
        }) as Box<dyn DynSourceLoop>)
        .expect("the fitting input queues");
    // Accepted by the pad's caps constraint (RGBA video), refused by the element:
    // too wide for its canvas.
    let oversized = handle
        .add_input(Box::new(CountedSource {
            n: 9,
            caps: video_caps(CANVAS_WIDTH * 4),
        }) as Box<dyn DynSourceLoop>)
        .expect("the oversized input queues");

    let control = async move {
        let fitting = fitting.accepted().await;
        let oversized = oversized.accepted().await;
        drop(handle);
        (fitting, oversized)
    };

    let (stats, (fitting, oversized)) = tokio::join!(run, control);
    let stats = stats.expect("the run survives a refused input");
    assert_eq!(fitting, Ok(()), "the fitting input was accepted");
    assert_eq!(
        oversized,
        Err(G2gError::InputRefused),
        "the element refused the oversized input"
    );
    assert_eq!(
        *per_pad.lock().unwrap(),
        std::vec![GOOD, 0],
        "only the accepted input's frames were aggregated"
    );
    assert_eq!(stats.frames_consumed, GOOD);
}

#[tokio::test]
async fn an_input_the_pad_cannot_carry_is_rejected_before_any_frame() {
    const GOOD: u64 = 3;
    let mut agg = CanvasMuxer::new(2);
    let per_pad = agg.frames.clone();
    let (handle, run) = run_aggregator_dynamic(&mut agg, 4);

    let video = handle
        .add_input(Box::new(CountedSource {
            n: GOOD,
            caps: video_caps(CANVAS_WIDTH),
        }) as Box<dyn DynSourceLoop>)
        .expect("the video input queues");
    let audio = handle
        .add_input(Box::new(CountedSource {
            n: 7,
            caps: audio_caps(),
        }) as Box<dyn DynSourceLoop>)
        .expect("the audio input queues");

    let control = async move {
        let video = video.accepted().await;
        let audio = audio.accepted().await;
        drop(handle);
        (video, audio)
    };

    let (stats, (video, audio)) = tokio::join!(run, control);
    let stats = stats.expect("the run survives a rejected input");
    assert_eq!(video, Ok(()));
    assert_eq!(
        audio,
        Err(G2gError::CapsMismatch),
        "the pad accepts only RGBA video, so the audio input never attached"
    );
    assert_eq!(
        *per_pad.lock().unwrap(),
        std::vec![GOOD, 0],
        "no frame of the rejected input reached the element"
    );
    assert_eq!(stats.frames_consumed, GOOD);
}
