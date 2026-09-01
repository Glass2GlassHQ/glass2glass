//! M974: the two claims the dynamic broadcast tee (`run_source_tee_dynamic`)
//! rests on but never proved. A branch attached while frames are flowing gets the
//! sticky caps and then only the frames that follow its attach (nothing replayed,
//! nothing skipped), and the per-branch copies of a `DataFrame` are refcount
//! handles on one buffer, not byte copies: every branch's payload is the *same*
//! allocation.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_tee_dynamic, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// Payload bytes carry the sequence, so a branch's frame is traceable to the one
/// the source emitted.
fn make_frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new(seq.to_le_bytes()))),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    }
}

fn payload_address(frame: &Frame) -> usize {
    match &frame.domain {
        MemoryDomain::System(s) => s.as_slice().as_ptr() as usize,
        other => panic!("expected System memory, got {other:?}"),
    }
}

struct CountingSource {
    n: u64,
}

impl SourceLoop for CountingSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(caps()))))
    }

    fn configure_pipeline(&mut self, _absolute: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.n {
                out.push(PipelinePacket::DataFrame(make_frame(seq))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.n)
        })
    }
}

/// Keeps every frame it received. Holding them alive is what makes the payload
/// addresses comparable after the run: a freed buffer's address could be handed
/// straight back to a sibling branch's copy.
#[derive(Default)]
struct KeepingSink {
    frames: Arc<Mutex<Vec<Frame>>>,
    caps_changes: Arc<AtomicUsize>,
    seen: Arc<AtomicUsize>,
}

impl AsyncElement for KeepingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        match packet {
            PipelinePacket::DataFrame(frame) => {
                self.frames.lock().unwrap().push(frame);
                self.seen.fetch_add(1, Ordering::SeqCst);
            }
            PipelinePacket::CapsChanged(_) => {
                self.caps_changes.fetch_add(1, Ordering::SeqCst);
            }
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn late_tee_branch_gets_sticky_caps_then_the_live_stream() {
    const N: u64 = 200;
    const ATTACH_AFTER: usize = 5;
    let mut source = CountingSource { n: N };

    let early = KeepingSink::default();
    let (early_frames, early_seen) = (early.frames.clone(), early.seen.clone());
    let late = KeepingSink::default();
    let (late_frames, late_caps) = (late.frames.clone(), late.caps_changes.clone());

    // Depth 2 so the source back-pressures and the run spans many polls, leaving
    // the late branch a real mid-stream window to attach in.
    let (handle, run) = run_source_tee_dynamic(&mut source, 2);
    handle
        .add_branch(Box::new(early) as Box<dyn DynAsyncElement>)
        .expect("add the early branch");

    let attach = async move {
        for _ in 0..1_000_000 {
            if early_seen.load(Ordering::SeqCst) >= ATTACH_AFTER {
                break;
            }
            tokio::task::yield_now().await;
        }
        let added = handle
            .add_branch(Box::new(late) as Box<dyn DynAsyncElement>)
            .is_ok();
        drop(handle);
        added
    };

    let (stats, added) = tokio::join!(run, attach);
    let stats = stats.expect("dynamic tee run");
    assert!(added, "the late branch attached while frames were flowing");

    let early = early_frames.lock().unwrap();
    let late = late_frames.lock().unwrap();
    assert_eq!(
        early.len(),
        N as usize,
        "the branch present from the start saw the whole stream"
    );
    assert_eq!(
        late_caps.load(Ordering::SeqCst),
        1,
        "the late branch configured from the replayed sticky caps"
    );

    let late_sequences: Vec<u64> = late.iter().map(|f| f.sequence).collect();
    assert!(
        !late_sequences.is_empty() && late_sequences.len() < N as usize,
        "the late branch got part of the stream, not all of it: {} frames",
        late_sequences.len()
    );
    let first = late_sequences[0];
    assert!(first > 0, "nothing before the attach was replayed to it");
    assert_eq!(
        late_sequences,
        (first..N).collect::<Vec<_>>(),
        "from its attach on, the late branch saw every frame in order"
    );
    assert_eq!(
        stats.frames_consumed,
        early.len() as u64 + late.len() as u64,
        "the run accounts for both branches' consumption"
    );
}

#[tokio::test]
async fn tee_branches_share_one_payload_rather_than_copying_it() {
    const N: u64 = 6;
    let mut source = CountingSource { n: N };

    let first = KeepingSink::default();
    let second = KeepingSink::default();
    let (first_frames, second_frames) = (first.frames.clone(), second.frames.clone());

    let (handle, run) = run_source_tee_dynamic(&mut source, 8);
    for sink in [first, second] {
        handle
            .add_branch(Box::new(sink) as Box<dyn DynAsyncElement>)
            .expect("add branch");
    }
    drop(handle);

    run.await.expect("dynamic tee run");

    let first = first_frames.lock().unwrap();
    let second = second_frames.lock().unwrap();
    assert_eq!(first.len(), N as usize);
    assert_eq!(second.len(), N as usize);

    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.sequence, b.sequence, "both branches saw the same frame");
        assert_eq!(
            payload_address(a),
            payload_address(b),
            "frame {} was shared as one buffer, not copied per branch",
            a.sequence
        );
        match &a.domain {
            MemoryDomain::System(s) => assert_eq!(
                s.as_slice(),
                a.sequence.to_le_bytes(),
                "the shared buffer still carries the source's payload"
            ),
            other => panic!("expected System memory, got {other:?}"),
        }
    }
    // Distinct frames are distinct buffers, so the equality above is sharing and
    // not one address the allocator kept reusing.
    let addresses: Vec<usize> = first.iter().map(payload_address).collect();
    let mut unique = addresses.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        addresses.len(),
        "each frame has its own buffer: {addresses:?}"
    );
}
