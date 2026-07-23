//! M288 regression: stacked auto `videoconvert`s steered toward a format pin must
//! produce the SAME bytes as a single convert, not corrupt output.
//!
//! Bug (pre-fix): `videotestsrc(RGBA) ! videoconvert ! videoconvert ! NV12-pin`
//! emitted the raw RGBA gradient bytes mislabeled NV12. The startup solve set the
//! first convert to RGBA->RGBA and the second to RGBA->NV12, but at runtime the
//! first convert's own output `CapsChanged(RGBA)` reached the second convert,
//! whose transform arm pre-fixed the forward output (NV12) and pushed it as
//! `process(CapsChanged(NV12))`. The convert then adopted NV12 as its *input*
//! format (`accept_input`) and ran the next RGBA frame as a bogus NV12->NV12
//! passthrough. The fix: a convert's `process(CapsChanged)` forwards that caps
//! (its arm-fixed output) downstream and leaves the `configure_pipeline`-set
//! input untouched, matching the decoder elements' two-caller contract.
//!
//! Built with `run_linear_chain` + a custom capturing sink so the test owns the
//! output bytes directly (the existing m188 tests only count frames, so they
//! passed with corrupt data). The NV12 pin is the sink's declared `Accepts`.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::run_linear_chain;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// Sink that accepts NV12 at any geometry (the format pin that steers the auto
/// converts) and records the bytes of every frame it receives.
struct CapturingSink {
    frames: Arc<Mutex<Vec<Vec<u8>>>>,
}

impl AsyncElement for CapturingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(nv12_any()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let frames = Arc::clone(&self.frames);
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    frames.lock().unwrap().push(s.to_vec());
                }
            }
            Ok(())
        })
    }
}

/// Run `videotestsrc(RGBA 64x48) ! videoconvert{n} ! NV12-sink` and return the
/// captured NV12 frames.
async fn run_converts(num_converts: usize) -> Vec<Vec<u8>> {
    let mut src = VideoTestSrc::new(64, 48, 30, 3);
    let mut convs: Vec<VideoConvert> = (0..num_converts).map(|_| VideoConvert::auto()).collect();
    let frames = Arc::new(Mutex::new(Vec::new()));
    let mut sink = CapturingSink {
        frames: Arc::clone(&frames),
    };

    let transforms: Vec<&mut dyn DynAsyncElement> = convs
        .iter_mut()
        .map(|c| c as &mut dyn DynAsyncElement)
        .collect();
    run_linear_chain(&mut src, transforms, &mut sink, &ZeroClock, 4)
        .await
        .expect("chain runs");

    let out = std::mem::take(&mut *frames.lock().unwrap());
    out
}

#[tokio::test]
async fn stacked_converts_match_single_convert_byte_for_byte() {
    let single = run_converts(1).await;
    let double = run_converts(2).await;

    assert_eq!(single.len(), 3, "single-convert delivered all frames");
    assert_eq!(double.len(), 3, "double-convert delivered all frames");
    // NV12 at 64x48 is luma (64*48) + interleaved chroma (64*48/2).
    let nv12_len = 64 * 48 + 64 * 48 / 2;
    assert_eq!(single[0].len(), nv12_len, "single output is NV12-sized");
    assert_eq!(double[0].len(), nv12_len, "double output is NV12-sized");

    // The core regression: a redundant second auto-convert must not alter the
    // bytes. Pre-fix, `double` was the raw RGBA gradient mislabeled NV12.
    assert_eq!(double, single, "stacked converts corrupt the output");

    // Sanity that we are actually looking at converted NV12, not passed-through
    // RGBA: the gradient's chroma plane is near-neutral (BT.601 of a grey ramp),
    // never the steep 0,1,2,... ramp of the raw RGBA bytes.
    let luma = 64 * 48;
    let chroma = &single[0][luma..];
    assert!(
        chroma.iter().all(|&c| (0x70..=0x90).contains(&c)),
        "NV12 chroma must be near-neutral 0x80, got {:?}",
        &chroma[..8]
    );
}
