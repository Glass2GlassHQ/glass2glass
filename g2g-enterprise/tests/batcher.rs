//! M22: `TensorBatcher` gather semantics, byte-exact stacking, EOS shrink,
//! and an end-to-end run through the real `run_muxer_sink` runner.

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, G2gError, MultiInputElement, OutputSink, PipelineClock,
    TensorDType, TensorLayout, TensorShape,
};
use g2g_enterprise::batcher::TensorBatcher;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One-slot caps: a U8 tensor of 4 elements, so each frame is 4 bytes and
/// byte-level stacking assertions stay readable.
fn slot() -> Caps {
    Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape(vec![1, 4]),
        layout: TensorLayout::Nchw,
    }
}

fn batched(n: u32) -> Caps {
    Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape(vec![n, 4]),
        layout: TensorLayout::Nchw,
    }
}

fn tensor_frame(fill: u8, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([fill; 4]))),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn frame_bytes(f: &Frame) -> &[u8] {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("batcher emits System frames");
    };
    slice.as_slice()
}

#[tokio::test]
async fn gathers_one_frame_per_input_and_stacks_bytes() {
    let mut b = TensorBatcher::new(2, slot()).unwrap();
    let mut out = Collect::default();

    // input 0 leads; nothing can batch until input 1 contributes.
    b.process(0, PipelinePacket::DataFrame(tensor_frame(0xA0, 10, 0)), &mut out)
        .await
        .unwrap();
    assert!(out.frames().is_empty(), "half a round must not emit");

    b.process(1, PipelinePacket::DataFrame(tensor_frame(0xB0, 20, 0)), &mut out)
        .await
        .unwrap();
    // a second round, arriving input-1-first.
    b.process(1, PipelinePacket::DataFrame(tensor_frame(0xB1, 40, 1)), &mut out)
        .await
        .unwrap();
    b.process(0, PipelinePacket::DataFrame(tensor_frame(0xA1, 30, 1)), &mut out)
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(frame_bytes(frames[0]), &[0xA0; 4].iter().chain(&[0xB0; 4]).copied().collect::<Vec<_>>()[..]);
    assert_eq!(frame_bytes(frames[1]), &[0xA1; 4].iter().chain(&[0xB1; 4]).copied().collect::<Vec<_>>()[..]);
    // batch pts is the newest constituent.
    assert_eq!(frames[0].timing.pts_ns, 20);
    assert_eq!(frames[1].timing.pts_ns, 40);
    assert_eq!(frames[0].sequence, 0);
    assert_eq!(frames[1].sequence, 1);
    assert!(
        out.caps_changes().is_empty(),
        "full batches match the startup-negotiated output, no CapsChanged"
    );
    assert_eq!(b.batches_emitted(), 2);
}

#[tokio::test]
async fn eos_shrinks_the_batch_and_emits_caps_changed() {
    let mut b = TensorBatcher::new(2, slot()).unwrap();
    let mut out = Collect::default();

    b.process(0, PipelinePacket::DataFrame(tensor_frame(1, 0, 0)), &mut out)
        .await
        .unwrap();
    b.process(1, PipelinePacket::DataFrame(tensor_frame(2, 0, 0)), &mut out)
        .await
        .unwrap();

    // input 1 ends; input 0 keeps flowing and must not stall.
    b.process(1, PipelinePacket::Eos, &mut out).await.unwrap();
    b.process(0, PipelinePacket::DataFrame(tensor_frame(3, 0, 1)), &mut out)
        .await
        .unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2);
    assert_eq!(frame_bytes(frames[0]).len(), 8, "full [2,4] batch");
    assert_eq!(frame_bytes(frames[1]), &[3; 4], "shrunken [1,4] batch");
    assert_eq!(
        out.caps_changes(),
        vec![batched(1)],
        "exactly one CapsChanged, before the first shrunken batch"
    );
}

#[tokio::test]
async fn queued_frames_of_an_ended_input_still_batch() {
    let mut b = TensorBatcher::new(2, slot()).unwrap();
    let mut out = Collect::default();

    // input 1 delivers two rounds worth, then ends, all before input 0 moves.
    b.process(1, PipelinePacket::DataFrame(tensor_frame(0xB0, 0, 0)), &mut out)
        .await
        .unwrap();
    b.process(1, PipelinePacket::DataFrame(tensor_frame(0xB1, 0, 1)), &mut out)
        .await
        .unwrap();
    b.process(1, PipelinePacket::Eos, &mut out).await.unwrap();

    b.process(0, PipelinePacket::DataFrame(tensor_frame(0xA0, 0, 0)), &mut out)
        .await
        .unwrap();
    b.process(0, PipelinePacket::DataFrame(tensor_frame(0xA1, 0, 1)), &mut out)
        .await
        .unwrap();
    b.process(0, PipelinePacket::Eos, &mut out).await.unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 2, "ended input's queue drains into full batches");
    assert_eq!(frame_bytes(frames[1]), &[0xA1, 0xA1, 0xA1, 0xA1, 0xB1, 0xB1, 0xB1, 0xB1]);
    assert!(out.caps_changes().is_empty(), "every batch stayed full-size");
}

/// Source emitting `count` 4-byte U8 tensor frames then EOS.
struct TensorSrc {
    fill: u8,
    count: u64,
    configured: bool,
}

impl SourceLoop for TensorSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(slot()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        let fill = self.fill;
        let count = self.count;
        Box::pin(async move {
            for i in 0..count {
                out.push(PipelinePacket::DataFrame(tensor_frame(fill, i * 10, i)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}

/// Sink recording batch byte lengths and EOS count.
#[derive(Default)]
struct BatchSink {
    lens: Vec<usize>,
    eos_count: u64,
}

impl AsyncElement for BatchSink {
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
            PipelinePacket::DataFrame(f) => self.lens.push(frame_bytes(&f).len()),
            PipelinePacket::Eos => self.eos_count += 1,
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn batches_two_streams_through_the_real_muxer_runner() {
    let mut a = TensorSrc { fill: 0xA0, count: 3, configured: false };
    let mut b = TensorSrc { fill: 0xB0, count: 3, configured: false };
    let mut batcher = TensorBatcher::new(2, slot()).unwrap();
    let mut sink = BatchSink::default();
    let clock = ZeroClock;

    {
        let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut a, &mut b];
        run_muxer_sink(sources, &mut batcher, &mut sink, &clock, 4)
            .await
            .expect("batcher pipeline completes");
    }

    assert_eq!(
        sink.lens,
        vec![8, 8, 8],
        "3 rounds of 2 inputs, each batch [2,4] = 8 bytes"
    );
    assert_eq!(sink.eos_count, 1, "single aggregated EOS from the runner");
    assert_eq!(batcher.batches_emitted(), 3);
}
