//! M180: zero-copy strided flip. A shared-memory `VideoTestSrc` hands each RGBA
//! frame to `VideoFlip` as an `Arc`-backed `SystemView`; the flip composes
//! strides on that same allocation instead of copying. The proof is twofold:
//! the backing pointers the source emitted reach the sink unchanged *through the
//! flip* (zero-copy witness), and the sink's materialized image is a correct
//! rotate-180 of the source's pattern (correctness).

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Split a contiguous RGBA8 byte buffer into per-pixel `[r,g,b,a]` chunks.
fn pixels(bytes: &[u8]) -> Vec<[u8; 4]> {
    bytes
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

#[tokio::test]
async fn rotate180_flips_through_shared_memory_with_zero_copies() {
    let (w, h, frames) = (8u32, 4u32, 3u64);
    let mut src = VideoTestSrc::new(w, h, 30, frames)
        .with_pattern(Pattern::SmpteBars)
        .with_shared_memory();
    let mut flip = VideoFlip::new(FlipMethod::Rotate180);
    let mut sink = FakeSink::new();

    run_source_transform_sink(&mut src, &mut flip, &mut sink, &NullClock, 4)
        .await
        .expect("shared RGBA -> zero-copy flip -> sink negotiates and flows");

    assert_eq!(sink.received(), frames, "every frame delivered");
    assert!(sink.eos_seen());

    // Geometry is preserved by a 180 rotation; the sink saw RGBA 8x4.
    assert!(
        sink.caps_changes().iter().any(|c| matches!(
            &c.caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(8),
                height: Dim::Fixed(4),
                framerate: Rate::Fixed(_),
            }
        )),
        "sink saw RGBA 8x4, got {:?}",
        sink.caps_changes()
    );

    // Zero-copy witness: the flip emitted frames backed by the *same*
    // allocations the source created. If VideoFlip had copied (as the legacy
    // owned-buffer path does), these pointers would differ.
    let emitted = src.emitted_ptrs();
    let received = sink.view_backing_ptrs();
    assert_eq!(
        emitted.len(),
        frames as usize,
        "source emitted shared frames"
    );
    assert_eq!(
        received, emitted,
        "every flipped frame aliases the source's buffer (zero copies)"
    );

    // Correctness: a 180 rotation maps output pixel p to input pixel N-1-p, so
    // reversing the materialized output's pixel order recovers the source
    // frame. Compare against the source's own pre-flip bytes.
    let out = sink.last_view_bytes().expect("a shared-view frame arrived");
    let input = src
        .emitted_frames()
        .last()
        .expect("source recorded its frames");
    let mut out_px = pixels(out);
    out_px.reverse();
    assert_eq!(
        out_px,
        pixels(input),
        "materialized rotate-180 is the source frame with pixel order reversed"
    );
}
