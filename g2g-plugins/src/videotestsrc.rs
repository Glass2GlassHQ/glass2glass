//! Synthetic video source. Emits a deterministic gradient pattern at a
//! fixed framerate in the system memory domain. CPU-only.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{SystemSlice, SystemView};
use g2g_core::runtime::SourceLoop;
use g2g_core::tensor::TensorView;
use g2g_core::{
    BufferPool, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, PushOutcome, Rate, RawVideoFormat, TensorDType,
};

/// Synthetic content drawn into each frame. Some animate with the frame index
/// (gradient, snow, moving-bar, ball, zone-plate); the calibration patterns
/// (smpte, checker) are static. Integer-only math, so every pattern works on the
/// `no_std` baseline (no `libm` / float transcendentals).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Pattern {
    /// Per-byte animated gradient: each byte is `(index + frame) & 0xFF`. Smooth
    /// and cheap, but its motion is a subtle scroll, easy to miss on a small
    /// inset. The historical default.
    #[default]
    Gradient,
    /// Full-frame per-pixel pseudo-random noise, reseeded every frame, so the
    /// whole image churns visibly: an unmistakable "is it live?" check.
    Snow,
    /// A bright vertical bar that sweeps left to right (wrapping), over a dark
    /// field: obvious, smooth motion that's easy to spot in a small overlay.
    MovingBar,
    /// The seven 75% SMPTE colour bars (white, yellow, cyan, green, magenta, red,
    /// blue) across the width. Static; the canonical calibration pattern.
    SmpteBars,
    /// A black/white checkerboard (square side ~1/8 of the width). Static; makes
    /// scaling and block-compression artefacts obvious.
    Checkerboard,
    /// A filled white ball bouncing over a dark field (reflecting off the edges).
    /// Smooth two-axis motion for a frame-rate / judder check.
    Ball,
    /// Concentric rings whose spacing tightens with radius (`(x^2 + y^2)` square
    /// wave, phase-animated): an integer zone plate, for resampling / aliasing
    /// tests. A square-wave approximation of the sinusoidal chirp (no `libm`).
    ZonePlate,
}

#[derive(Debug)]
pub struct VideoTestSrc {
    width: u32,
    height: u32,
    framerate_q16: u32,
    target_frames: u64,
    pattern: Pattern,
    configured: bool,
    pool: Option<BufferPool<Box<[u8]>>>,
    /// Frames skipped in response to upstream QoS (M174). Cumulative across the
    /// run; surfaced for observability / tests.
    skipped: u64,
    /// M180: emit frames in the shared-CPU [`MemoryDomain::SystemView`] domain
    /// (an `Arc`-backed contiguous `[H, W, 4]` view) instead of an owned
    /// `System` buffer, so a downstream stride transform (VideoFlip) can flip
    /// zero-copy. Off by default; opt in with [`VideoTestSrc::with_shared_memory`].
    /// Incompatible with the pool path (a pooled buffer isn't `Arc`-shared).
    shared: bool,
    /// Data pointers of the shared backings emitted (M180), in push order. Lets
    /// a test prove a downstream element aliased the source's buffer rather than
    /// copying it. Only populated in `shared` mode.
    emitted_ptrs: Vec<usize>,
    /// Contiguous bytes of each shared frame emitted (M180), in push order. A
    /// test verifies a downstream flip against these. Only populated in `shared`
    /// mode; test-only bookkeeping, independent of the zero-copy claim.
    emitted_frames: Vec<Vec<u8>>,
}

impl VideoTestSrc {
    /// `framerate` is in nominal fps; stored internally as Q16 fixed-point.
    pub fn new(width: u32, height: u32, framerate: u32, target_frames: u64) -> Self {
        Self {
            width,
            height,
            framerate_q16: framerate << 16,
            target_frames,
            pattern: Pattern::default(),
            configured: false,
            pool: None,
            skipped: 0,
            shared: false,
            emitted_ptrs: Vec::new(),
            emitted_frames: Vec::new(),
        }
    }

    /// Select the drawn [`Pattern`] (default [`Pattern::Gradient`]). Use
    /// [`Pattern::Snow`] or [`Pattern::MovingBar`] when you need motion that is
    /// obvious to the eye (e.g. confirming a live overlay is actually updating).
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Pool-backed variant: every emitted frame draws its `width * height * 4`
    /// bytes from the pool, and the buffer returns to the pool when the
    /// downstream `Frame` is dropped. The pool's buffer size MUST be at
    /// least `width * height * 4`; this is checked at run time.
    pub fn with_pool(
        width: u32,
        height: u32,
        framerate: u32,
        target_frames: u64,
        pool: BufferPool<Box<[u8]>>,
    ) -> Self {
        Self {
            width,
            height,
            framerate_q16: framerate << 16,
            target_frames,
            pattern: Pattern::default(),
            configured: false,
            pool: Some(pool),
            skipped: 0,
            shared: false,
            emitted_ptrs: Vec::new(),
            emitted_frames: Vec::new(),
        }
    }

    /// M180: emit frames in the shared-CPU [`MemoryDomain::SystemView`] domain
    /// so a downstream stride transform can flip / crop them zero-copy. See the
    /// [`shared`](Self::shared) field. Has no effect when a pool is configured.
    pub fn with_shared_memory(mut self) -> Self {
        self.shared = true;
        self
    }

    /// Frames skipped in response to upstream QoS over the run (M174).
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Data pointers of the shared backings emitted in `shared` mode (M180), in
    /// push order. A test asserts a downstream element's received pointers equal
    /// these to prove the frames flowed through with zero copies.
    pub fn emitted_ptrs(&self) -> &[usize] {
        &self.emitted_ptrs
    }

    /// Contiguous pre-transform bytes of each shared frame emitted (M180), in
    /// push order. A test verifies a downstream flip's output against these.
    pub fn emitted_frames(&self) -> &[Vec<u8>] {
        &self.emitted_frames
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }
}

impl SourceLoop for VideoTestSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    /// M16 step 5f: native `Produces` constraint. The chain is now
    /// fully-native when paired with `AcceptsAny`-migrated sinks
    /// (e.g. `FakeSink`, `syncsink`), exercising the all-native
    /// arc-consistency solver path instead of the mixed cascade.
    /// Synchronous override (no I/O), so we sidestep the default's
    /// `async move` indirection.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }

            let fps_q16 = u64::from(self.framerate_q16);
            let pts_step_ns: u64 = (1_000_000_000u64 << 16)
                .checked_div(fps_q16)
                .unwrap_or(0);

            let bytes_per_frame = (self.width as usize)
                .checked_mul(self.height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or(G2gError::CapsMismatch)?;

            let mut seq = 0u64;
            let mut pushed = 0u64;
            while seq < self.target_frames {
                let domain = if let Some(pool) = &self.pool {
                    let mut buf = pool.acquire().await;
                    if buf.len() < bytes_per_frame {
                        return Err(G2gError::CapsMismatch);
                    }
                    fill_pattern(self.pattern, &mut buf.as_mut()[..bytes_per_frame], self.width, seq);
                    MemoryDomain::System(SystemSlice::from_pool(buf, bytes_per_frame))
                } else if self.shared {
                    // M180: hand the frame out as Arc-shared bytes with a
                    // contiguous [H, W, 4] view, so a downstream flip composes
                    // strides on this same allocation. Record the data pointer
                    // and bytes for the zero-copy / correctness assertions.
                    let mut buf = vec![0u8; bytes_per_frame].into_boxed_slice();
                    fill_pattern(self.pattern, &mut buf, self.width, seq);
                    self.emitted_frames.push(buf.to_vec());
                    let backing: Arc<[u8]> = Arc::from(buf);
                    self.emitted_ptrs.push(backing.as_ptr() as usize);
                    let view = TensorView::contiguous(
                        TensorDType::U8,
                        &[self.height, self.width, 4],
                    );
                    MemoryDomain::SystemView(SystemView::new(backing, view))
                } else {
                    let mut buf = vec![0u8; bytes_per_frame].into_boxed_slice();
                    fill_pattern(self.pattern, &mut buf, self.width, seq);
                    MemoryDomain::System(SystemSlice::from_boxed(buf))
                };

                let pts = seq * pts_step_ns;
                // Source-side wall-clock stamp so downstream sinks can
                // record glass-to-glass latency via
                // `monotonic_ns() - arrival_ns`. Matches the convention
                // used by RtspSrc for production sources. Std-gated
                // because `monotonic_ns` lives behind g2g-core's `std`
                // feature; in no_std builds `arrival_ns` stays zero
                // and downstream sinks silently skip latency recording.
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: pts_step_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: true, // raw frames are each independently presentable
                    },
                    sequence: seq,
                    meta: Default::default(),
                };

                let outcome = out.push(PipelinePacket::DataFrame(frame)).await?;
                pushed += 1;
                // M174: react to upstream QoS by skipping ahead, so a downstream
                // that can't keep up sheds the source's frame-generation load.
                // The skipped frames advance `seq` (and thus PTS) without being
                // generated, so the timeline stays correct.
                match outcome {
                    PushOutcome::Qos(q) if q.jitter_ns > 0 => {
                        let period = pts_step_ns.max(1);
                        let skip = (q.jitter_ns as u64 / period).min(self.target_frames);
                        self.skipped += skip;
                        seq = seq.saturating_add(1).saturating_add(skip);
                    }
                    _ => seq += 1,
                }
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEOTESTSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Video test source",
            "Source/Video",
            "Generates a synthetic test pattern (SMPTE bars, snow, moving bar, ball, ...)",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "pattern" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.pattern = pattern_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                // GStreamer convention: -1 means "no limit".
                self.target_frames = if n < 0 { u64::MAX } else { n as u64 };
                Ok(())
            }
            "width" => {
                self.width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                let (n, d) = value.as_fraction().ok_or(PropError::Type)?;
                if n < 0 || d <= 0 {
                    return Err(PropError::Value);
                }
                // Store fps as Q16: (n/d) << 16.
                self.framerate_q16 = (((n as u64) << 16) / d as u64) as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "pattern" => Some(PropValue::Str(pattern_to_str(self.pattern).into())),
            "num-buffers" => Some(PropValue::Int(if self.target_frames == u64::MAX {
                -1
            } else {
                self.target_frames as i64
            })),
            "width" => Some(PropValue::Uint(self.width as u64)),
            "height" => Some(PropValue::Uint(self.height as u64)),
            // Report the integral fps numerator over /1 (Q16 stores fps*65536).
            "framerate" => Some(PropValue::Fraction((self.framerate_q16 >> 16) as i32, 1)),
            _ => None,
        }
    }
}

/// `VideoTestSrc`'s settable properties (M104).
static VIDEOTESTSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("pattern", PropKind::Str, "drawn pattern")
        .with_enum_values("gradient | snow | bar | smpte | checkers-8 | ball | zone-plate")
        .with_default("smpte"),
    PropertySpec::new("num-buffers", PropKind::Int, "frames to emit then EOS (-1 = forever)")
        .with_range("-1", "9223372036854775807")
        .with_default("-1"),
    PropertySpec::new("width", PropKind::Uint, "frame width in pixels").with_default("320"),
    PropertySpec::new("height", PropKind::Uint, "frame height in pixels").with_default("240"),
    PropertySpec::new("framerate", PropKind::Fraction, "frames per second (e.g. 30/1)")
        .with_default("30/1"),
];

/// Parse a `pattern` property string to a [`Pattern`]. Canonical names match
/// GStreamer's `videotestsrc` nicknames (`bar`, `checkers-8`); the historical
/// g2g spellings are accepted as aliases so both port. g2g's `checkers-8` is a
/// width-relative checkerboard, not gst's fixed 8px squares (nearest match).
fn pattern_from_str(s: &str) -> Option<Pattern> {
    match s {
        "gradient" => Some(Pattern::Gradient),
        "snow" => Some(Pattern::Snow),
        "bar" | "moving-bar" => Some(Pattern::MovingBar),
        "smpte" => Some(Pattern::SmpteBars),
        "checkers-8" | "checker" => Some(Pattern::Checkerboard),
        "ball" => Some(Pattern::Ball),
        "zone-plate" => Some(Pattern::ZonePlate),
        _ => None,
    }
}

/// The canonical (GStreamer) `pattern` property string for a [`Pattern`].
fn pattern_to_str(p: Pattern) -> &'static str {
    match p {
        Pattern::Gradient => "gradient",
        Pattern::Snow => "snow",
        Pattern::MovingBar => "bar",
        Pattern::SmpteBars => "smpte",
        Pattern::Checkerboard => "checkers-8",
        Pattern::Ball => "ball",
        Pattern::ZonePlate => "zone-plate",
    }
}

/// Draw `pattern` for frame `seq` into a `width`-wide RGBA8 buffer (`buf.len()`
/// is exactly the frame's `width * height * 4` bytes). Animated by `seq`.
fn fill_pattern(pattern: Pattern, buf: &mut [u8], width: u32, seq: u64) {
    match pattern {
        // Per-byte scrolling gradient, byte-for-byte the historical output.
        Pattern::Gradient => {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = ((i as u64).wrapping_add(seq) & 0xFF) as u8;
            }
        }
        // Per-pixel hash of (pixel index, frame): full-frame churn. Alpha stays
        // opaque so the noise is visible through a compositor's blend.
        Pattern::Snow => {
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                // Cheap integer hash, reseeded per frame via `seq`.
                let mut h = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seq.wrapping_mul(0x2545_F491_4F6C_DD1D);
                h ^= h >> 29;
                h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h ^= h >> 32;
                let v = (h & 0xFF) as u8;
                px[0] = v;
                px[1] = v;
                px[2] = v;
                px[3] = 255;
            }
        }
        // A bright bar sweeping left to right over a dark field. The bar is
        // ~1/8 of the width and advances a few pixels per frame, wrapping.
        Pattern::MovingBar => {
            let w = width.max(1) as usize;
            let bar_w = (w / 8).max(1);
            let bar_x = (seq as usize).wrapping_mul(4) % w;
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                let x = p % w;
                // Distance from the bar's left edge, modulo width (so it wraps).
                let within = x.wrapping_sub(bar_x) % w < bar_w;
                let v = if within { 255 } else { 32 };
                px[0] = v;
                px[1] = v;
                px[2] = v;
                px[3] = 255;
            }
        }
        // Seven 75% colour bars across the width. Static calibration pattern.
        Pattern::SmpteBars => {
            const BARS: [[u8; 3]; 7] = [
                [192, 192, 192], // white
                [192, 192, 0],   // yellow
                [0, 192, 192],   // cyan
                [0, 192, 0],     // green
                [192, 0, 192],   // magenta
                [192, 0, 0],     // red
                [0, 0, 192],     // blue
            ];
            let w = width.max(1) as usize;
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                let x = p % w;
                let [r, g, b] = BARS[(x * BARS.len() / w).min(BARS.len() - 1)];
                px[0] = r;
                px[1] = g;
                px[2] = b;
                px[3] = 255;
            }
        }
        // Black/white checkerboard, square side ~1/8 of the width. Static.
        Pattern::Checkerboard => {
            let w = width.max(1) as usize;
            let square = (w / 8).max(1);
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                let (x, y) = (p % w, p / w);
                let v = if ((x / square) + (y / square)) & 1 == 0 { 255 } else { 0 };
                px[0] = v;
                px[1] = v;
                px[2] = v;
                px[3] = 255;
            }
        }
        // A filled white ball bouncing (reflecting at the edges) over a dark
        // field. Center advances per frame; pixels inside the radius are white.
        Pattern::Ball => {
            let w = width.max(1) as usize;
            let h = ((buf.len() / 4) / w).max(1);
            let radius = (w.min(h) / 8).max(1) as i64;
            let cx = bounce(seq, w, 3) as i64;
            let cy = bounce(seq, h, 2) as i64;
            let r2 = radius * radius;
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                let (dx, dy) = ((p % w) as i64 - cx, (p / w) as i64 - cy);
                let v = if dx * dx + dy * dy <= r2 { 255 } else { 32 };
                px[0] = v;
                px[1] = v;
                px[2] = v;
                px[3] = 255;
            }
        }
        // Concentric rings about the centre. Ring frequency rises with radius
        // because `d2` is quadratic (the zone-plate aliasing property); `seq`
        // phase-shifts them so the rings pulse outward.
        Pattern::ZonePlate => {
            let w = width.max(1) as usize;
            let h = ((buf.len() / 4) / w).max(1);
            let (cx, cy) = ((w / 2) as i64, (h / 2) as i64);
            for (p, px) in buf.chunks_exact_mut(4).enumerate() {
                let (dx, dy) = ((p % w) as i64 - cx, (p / w) as i64 - cy);
                let d2 = (dx * dx + dy * dy) as u64;
                let v = if ((d2 >> 6).wrapping_add(seq)) & 1 == 0 { 255 } else { 0 };
                px[0] = v;
                px[1] = v;
                px[2] = v;
                px[3] = 255;
            }
        }
    }
}

/// A reflecting (triangle-wave) coordinate in `0..span`, advancing `speed` units
/// per `seq` step: the ball bounces off the edges instead of wrapping.
fn bounce(seq: u64, span: usize, speed: u64) -> usize {
    if span <= 1 {
        return 0;
    }
    let period = 2 * (span as u64 - 1);
    let t = seq.wrapping_mul(speed) % period;
    (if t < span as u64 { t } else { period - t }) as usize
}

impl PadTemplates for VideoTestSrc {
    /// Static superset: the type always produces RGBA at any geometry /
    /// framerate. A constructed instance narrows to its configured dims via
    /// `SourceLoop::caps_constraint`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 16x16 RGBA frame (wide enough for the bar to advance between frames).
    const W: u32 = 16;
    const BYTES: usize = (W * W * 4) as usize;

    #[test]
    fn gradient_is_byte_identical_to_the_historical_output() {
        let mut buf = [0u8; BYTES];
        fill_pattern(Pattern::Gradient, &mut buf, W, 7);
        for (i, &b) in buf.iter().enumerate() {
            assert_eq!(b, ((i as u64).wrapping_add(7) & 0xFF) as u8);
        }
    }

    #[test]
    fn snow_and_bar_change_between_frames_so_motion_is_visible() {
        for pattern in [Pattern::Snow, Pattern::MovingBar] {
            let mut a = [0u8; BYTES];
            let mut b = [0u8; BYTES];
            fill_pattern(pattern, &mut a, W, 0);
            fill_pattern(pattern, &mut b, W, 1);
            assert_ne!(a, b, "{pattern:?} must animate between consecutive frames");
            // Alpha stays opaque so the content survives a compositor blend.
            assert!(a.chunks_exact(4).all(|px| px[3] == 255), "{pattern:?} opaque");
        }
    }

    #[test]
    fn animated_patterns_change_between_frames() {
        for pattern in [Pattern::Ball, Pattern::ZonePlate] {
            let mut a = [0u8; BYTES];
            let mut b = [0u8; BYTES];
            fill_pattern(pattern, &mut a, W, 0);
            fill_pattern(pattern, &mut b, W, 1);
            assert_ne!(a, b, "{pattern:?} must animate between consecutive frames");
            assert!(a.chunks_exact(4).all(|px| px[3] == 255), "{pattern:?} opaque");
        }
    }

    #[test]
    fn smpte_bars_run_white_to_blue_and_are_static() {
        let mut f0 = [0u8; BYTES];
        let mut f5 = [0u8; BYTES];
        fill_pattern(Pattern::SmpteBars, &mut f0, W, 0);
        fill_pattern(Pattern::SmpteBars, &mut f5, W, 5);
        // Leftmost bar is 75% white, the last column falls in the blue bar.
        assert_eq!(&f0[0..4], &[192, 192, 192, 255], "first bar white");
        let last = ((W - 1) * 4) as usize;
        assert_eq!(&f0[last..last + 4], &[0, 0, 192, 255], "last column blue");
        assert_eq!(f0, f5, "smpte bars are a static calibration pattern");
    }

    #[test]
    fn checkerboard_alternates_and_is_static() {
        let mut f0 = [0u8; BYTES];
        let mut f9 = [0u8; BYTES];
        fill_pattern(Pattern::Checkerboard, &mut f0, W, 0);
        fill_pattern(Pattern::Checkerboard, &mut f9, W, 9);
        // Square side is W/8 = 2, so the first square is white and the next black.
        assert_eq!(f0[0], 255, "top-left square white");
        assert_eq!(f0[2 * 4], 0, "adjacent square black");
        assert_eq!(f0, f9, "checkerboard is static");
    }

    #[test]
    fn every_pattern_string_round_trips() {
        for pattern in [
            Pattern::Gradient,
            Pattern::Snow,
            Pattern::MovingBar,
            Pattern::SmpteBars,
            Pattern::Checkerboard,
            Pattern::Ball,
            Pattern::ZonePlate,
        ] {
            assert_eq!(pattern_from_str(pattern_to_str(pattern)), Some(pattern));
        }
    }
}
