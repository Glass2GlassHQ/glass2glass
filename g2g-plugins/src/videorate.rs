//! Software temporal resampler (P1.2). Drops or duplicates whole frames to
//! hit a configured target framerate: `30 fps -> 10 fps` for ML inference,
//! `30 fps -> 60 fps` for delivery. Pairs with `VideoScale` (spatial) ahead
//! of `WgpuPreprocess` / `OrtInference`, which want a fixed input rate the
//! source rarely matches.
//!
//! Format-agnostic: it never touches pixels, only forwards, drops, or
//! repeats the System-memory frame, so it preserves the pixel format and
//! geometry and replaces only the framerate. CPU-only, `no_std` baseline,
//! no feature gate.
//!
//! The cadence follows GStreamer's `videorate`: hold the previous frame,
//! and on each new frame emit the held frame for every output slot whose
//! timestamp is at least as close to the held frame as to the new one
//! (nearest-neighbour in time). This duplicates on upscale and drops on
//! downscale with one frame of latency, and re-stamps output PTS onto the
//! exact target grid. The drop/duplicate decision is the pure `emit_slots`
//! helper so it is host-testable without the runner.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PassthroughFields, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

#[derive(Debug)]
struct HeldFrame {
    bytes: Box<[u8]>,
    pts_ns: u64,
    arrival_ns: u64,
    capture_ns: u64,
}

#[derive(Debug)]
pub struct VideoRate {
    /// Target framerate in Q16 fps and the matching output inter-frame
    /// interval in ns. Zero marks an invalid (non-positive) target, which
    /// fails loud at negotiation / configure.
    rate_q16: u32,
    dt_ns: u64,
    /// Caps-driven (M290): take the target framerate from the negotiated output
    /// caps (a downstream capsfilter), the gst `videorate ! caps,framerate=N`
    /// idiom, instead of the `framerate` property. With no downstream pin it
    /// defaults to passthrough (the input rate, no retiming).
    auto: bool,
    input: Option<(RawVideoFormat, u32, u32)>,
    /// The most recent input frame, held one step so the next frame's PTS
    /// decides how many output slots it serves.
    prev: Option<HeldFrame>,
    /// PTS of the next output slot to fill.
    next_pts: Option<u64>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl VideoRate {
    /// `target_fps` is nominal frames per second; fractional rates
    /// (29.97) are fine. A non-positive rate is rejected at configure.
    pub fn new(target_fps: f64) -> Self {
        let rate_q16 = if target_fps > 0.0 {
            (target_fps * 65536.0 + 0.5) as u32
        } else {
            0
        };
        let dt_ns = if rate_q16 > 0 {
            (1_000_000_000u64 << 16) / rate_q16 as u64
        } else {
            0
        };
        Self {
            rate_q16,
            dt_ns,
            auto: false,
            input: None,
            prev: None,
            next_pts: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Caps-driven (M290): take the target framerate from the negotiated output
    /// caps (a downstream capsfilter). With no downstream pin it passes the
    /// input rate through unchanged.
    pub fn auto() -> Self {
        Self {
            rate_q16: 0,
            dt_ns: 0,
            auto: true,
            input: None,
            prev: None,
            next_pts: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Set the effective target framerate (Q16 fps) and matching frame interval.
    fn set_rate_q16(&mut self, rate_q16: u32) {
        self.rate_q16 = rate_q16;
        self.dt_ns = if rate_q16 > 0 {
            (1_000_000_000u64 << 16) / rate_q16 as u64
        } else {
            0
        };
    }

    fn accept_input(&self, caps: &Caps) -> Result<(RawVideoFormat, u32, u32), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h))
    }

    fn output_caps(&self, format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(self.rate_q16),
        }
    }

    /// Return the output caps to push if they differ from the last emitted,
    /// recording them. Synchronous so it doesn't hold a borrow across an
    /// await.
    fn caps_to_emit(&mut self, format: RawVideoFormat, w: u32, h: u32) -> Option<Caps> {
        let caps = self.output_caps(format, w, h);
        if self.last_caps.as_ref() != Some(&caps) {
            self.last_caps = Some(caps.clone());
            Some(caps)
        } else {
            None
        }
    }

    async fn emit_held(
        &mut self,
        held_bytes: &[u8],
        arrival_ns: u64,
        capture_ns: u64,
        slot_pts: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(held_bytes.into())),
            timing: FrameTiming {
                pts_ns: slot_pts,
                dts_ns: slot_pts,
                duration_ns: self.dt_ns,
                capture_ns,
                arrival_ns,
                keyframe: false, // raw-video rate conversion; no keyframe semantics
            },
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

/// Output slots the held frame fills before the new frame takes over, and
/// the advanced next-slot. A slot at `next` is served by the held frame
/// while `next` is at least as close to `prev_pts` as to `cur_pts`
/// (nearest-neighbour, ties to the held frame so an exact-grid input
/// duplicates rather than drops). `saturating_add` keeps a near-`u64::MAX`
/// PTS from overflowing.
fn emit_slots(prev_pts: u64, cur_pts: u64, next_pts: u64, dt_ns: u64) -> (Vec<u64>, u64) {
    let mut slots = Vec::new();
    let mut next = next_pts;
    while next <= cur_pts && dt_ns > 0 {
        if prev_pts.abs_diff(next) <= cur_pts.abs_diff(next) {
            slots.push(next);
            next = next.saturating_add(dt_ns);
        } else {
            break;
        }
    }
    (slots, next)
}

impl AsyncElement for VideoRate {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // input side: any raw video at the upstream geometry, unchanged.
        match upstream_caps {
            Caps::RawVideo { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: any raw input maps to the same format and
    /// geometry at the configured target framerate. An invalid target
    /// (non-positive fps) collapses to the empty set so the solve fails
    /// loud.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let rate = self.rate_q16;
        let auto = self.auto;
        // Passthrough format + geometry (retarget framerate only), so a
        // downstream geometry pin still couples back through this rate-only
        // element. Framerate is the changed field.
        let passthrough = PassthroughFields::NONE
            .with_format()
            .with_width()
            .with_height();
        let derive = Box::new(move |input: &Caps| match input {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
            } => {
                let mk = |fr: Rate| Caps::RawVideo {
                    format: *format,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: fr,
                };
                if auto {
                    // Caps-driven: default to passthrough (the input rate, no
                    // retiming), but advertise "any rate" so a downstream
                    // capsfilter pins the target. Passthrough is preferred (first).
                    CapsSet::from_alternatives(vec![mk(framerate.clone()), mk(Rate::Any)])
                } else if rate > 0 {
                    // Property-driven: the fixed target framerate.
                    CapsSet::one(mk(Rate::Fixed(rate)))
                } else {
                    // Invalid (non-positive property, not auto): fail loud.
                    CapsSet::from_alternatives(Vec::new())
                }
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        });
        CapsConstraint::DerivedCoupled {
            derive,
            passthrough,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // A non-auto instance needs a positive target; an auto instance gets its
        // rate from `configure_output` (the downstream pin), so 0 is fine here.
        if !self.auto && self.rate_q16 == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let (format, w, h) = self.accept_input(absolute_caps)?;
        self.input = Some((format, w, h));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// M290: in auto mode, take the target framerate from the negotiated output
    /// caps (a downstream capsfilter). The framerate is fixated by the solver, so
    /// it is concrete here. No-op for the property-driven instance.
    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        if !self.auto {
            return Ok(());
        }
        let Caps::RawVideo {
            framerate: Rate::Fixed(q),
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *q == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.set_rate_q16(*q);
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // An auto instance with no downstream framerate pin never
                    // resolved a target (dt_ns == 0); fail loud rather than hold
                    // every frame forever.
                    if self.rate_q16 == 0 {
                        return Err(G2gError::NotConfigured);
                    }
                    let (format, w, h) = self.input.ok_or(G2gError::NotConfigured)?;
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let cur_pts = frame.timing.pts_ns;
                    let cur_bytes: Box<[u8]> = slice.as_slice().into();

                    // fill the output slots the held frame still owns, then
                    // make the new frame the held one.
                    let (slots, new_next) = match &self.prev {
                        Some(p) => emit_slots(
                            p.pts_ns,
                            cur_pts,
                            self.next_pts.unwrap_or(cur_pts),
                            self.dt_ns,
                        ),
                        None => (Vec::new(), cur_pts),
                    };
                    if !slots.is_empty() {
                        let (bytes, arrival, capture) = {
                            let p = self.prev.as_ref().expect("slots imply a held frame");
                            (p.bytes.clone(), p.arrival_ns, p.capture_ns)
                        };
                        if let Some(caps) = self.caps_to_emit(format, w, h) {
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                        }
                        for slot in slots {
                            self.emit_held(&bytes, arrival, capture, slot, out).await?;
                        }
                    }
                    self.next_pts = Some(new_next);
                    self.prev = Some(HeldFrame {
                        bytes: cur_bytes,
                        pts_ns: cur_pts,
                        arrival_ns: frame.timing.arrival_ns,
                        capture_ns: frame.timing.capture_ns,
                    });
                }
                PipelinePacket::CapsChanged(c) => {
                    let new_input = self.accept_input(&c)?;
                    if self.input != Some(new_input) {
                        // geometry/format change: the held frame belongs to
                        // the old stream, so drop it and restart the grid.
                        self.input = Some(new_input);
                        self.prev = None;
                        self.next_pts = None;
                        self.last_caps = None;
                    }
                }
                PipelinePacket::Flush => {
                    self.prev = None;
                    self.next_pts = None;
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward unchanged, never into rate logic.
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {
                    // emit the held frame once so the stream's last frame is
                    // not dropped (it is otherwise emitted only when a later
                    // frame arrives).
                    if let Some((format, w, h)) = self.input {
                        if let Some(p) = self.prev.take() {
                            let slot = self.next_pts.unwrap_or(p.pts_ns);
                            if let Some(caps) = self.caps_to_emit(format, w, h) {
                                out.push(PipelinePacket::CapsChanged(caps)).await?;
                            }
                            self.emit_held(&p.bytes, p.arrival_ns, p.capture_ns, slot, out)
                                .await?;
                        }
                    }
                    // the transform arm forwards EOS; the element only flushes here.
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VIDEORATE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "framerate" => {
                let (n, d) = value.as_fraction().ok_or(PropError::Type)?;
                if n <= 0 || d <= 0 {
                    return Err(PropError::Value);
                }
                // An explicit framerate property overrides caps-driven mode.
                self.auto = false;
                self.set_rate_q16((((n as u64) << 16) / d as u64) as u32);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            // Report the stored Q16 rate as a reduced fraction, not the floored
            // integer, so a fractional target (e.g. 30000/1001) round-trips.
            "framerate" => {
                let g = gcd(self.rate_q16, 1 << 16).max(1);
                Some(PropValue::Fraction(
                    (self.rate_q16 / g) as i32,
                    ((1u32 << 16) / g) as i32,
                ))
            }
            _ => None,
        }
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `VideoRate`'s settable properties (M104).
static VIDEORATE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "framerate",
    PropKind::Fraction,
    "target output frames per second (e.g. 10/1)",
)];

#[cfg(test)]
mod tests {
    use super::*;

    // 30 fps grid (ns) and the target intervals used below.
    const STEP_30: u64 = 1_000_000_000 / 30;
    const DT_10: u64 = 1_000_000_000 / 10;
    const DT_60: u64 = 1_000_000_000 / 60;

    #[test]
    fn framerate_property_round_trips_a_fraction() {
        let mut vr = VideoRate::new(30.0);
        vr.set_property("framerate", PropValue::Fraction(30000, 1001))
            .unwrap();
        let Some(PropValue::Fraction(num, den)) = vr.get_property("framerate") else {
            panic!("framerate reads back as a fraction");
        };
        // Reads back at ~29.97 within Q16 precision, not floored to 29/1.
        let fps = num as f64 / den as f64;
        assert!(
            (fps - 30000.0 / 1001.0).abs() < 0.01,
            "got {num}/{den} = {fps}"
        );
    }

    #[test]
    fn downsample_keeps_one_in_three() {
        // walk a 30 fps stream against a 10 fps grid; each input frame is
        // one `emit_slots` call. Count emissions and check spacing.
        let mut next = 0u64;
        let mut prev = 0u64;
        let mut out = Vec::new();
        for k in 0..9u64 {
            let cur = k * STEP_30;
            if k > 0 {
                let (slots, nn) = emit_slots(prev, cur, next, DT_10);
                out.extend(slots);
                next = nn;
            }
            prev = cur;
        }
        // emitted slots land on the 10 fps grid, strictly increasing.
        assert!(out.windows(2).all(|w| w[1] == w[0] + DT_10));
        assert!(out.iter().all(|&t| t % DT_10 == 0));
        // ~1-in-3 of the 9 inputs (the final frame is flushed on EOS, not here).
        assert_eq!(out.len(), 3, "got {out:?}");
    }

    #[test]
    fn upsample_duplicates_to_two() {
        let mut next = 0u64;
        let mut prev = 0u64;
        let mut out = Vec::new();
        for k in 0..4u64 {
            let cur = k * STEP_30;
            if k > 0 {
                let (slots, nn) = emit_slots(prev, cur, next, DT_60);
                out.extend(slots);
                next = nn;
            }
            prev = cur;
        }
        // monotonic on the 60 fps grid; roughly two outputs per input.
        assert!(out.windows(2).all(|w| w[1] == w[0] + DT_60));
        assert!(out.len() >= 5, "expected ~2x upsampling, got {out:?}");
    }

    #[test]
    fn near_max_pts_does_not_overflow() {
        // a slot near u64::MAX must saturate, not panic, on the dt advance.
        let prev = u64::MAX - 5;
        let cur = u64::MAX;
        let (slots, next) = emit_slots(prev, cur, u64::MAX - 5, DT_60);
        assert_eq!(next, u64::MAX);
        assert!(!slots.is_empty());
    }

    #[test]
    fn backward_pts_emits_nothing() {
        // a backward jump (next ahead of cur) yields no slots and leaves
        // the grid where it was.
        let (slots, next) = emit_slots(1_000, 100, 1_000, DT_60);
        assert!(slots.is_empty());
        assert_eq!(next, 1_000);
    }

    #[test]
    fn new_rounds_target_fps_to_q16() {
        let r = VideoRate::new(30.0);
        assert_eq!(r.rate_q16, 30 << 16);
        assert_eq!(r.dt_ns, 1_000_000_000 / 30);
        // 29.97 rounds to the nearest Q16 tick.
        let r = VideoRate::new(29.97);
        assert_eq!(r.rate_q16, (29.97 * 65536.0 + 0.5) as u32);
        // non-positive is invalid (rejected at configure).
        assert_eq!(VideoRate::new(0.0).rate_q16, 0);
    }

    #[test]
    fn configure_rejects_invalid_rate_and_caps() {
        let mut bad = VideoRate::new(0.0);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(64),
            height: Dim::Fixed(32),
            framerate: Rate::Fixed(30 << 16),
        };
        assert_eq!(
            bad.configure_pipeline(&caps).expect_err("zero fps"),
            G2gError::CapsMismatch
        );
        // compressed input is rejected
        let mut r = VideoRate::new(10.0);
        let h264 = Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(32),
            framerate: Rate::Any,
        };
        assert_eq!(
            r.configure_pipeline(&h264).expect_err("compressed"),
            G2gError::CapsMismatch
        );
        assert!(r.configure_pipeline(&caps).is_ok());
    }

    fn nv12_320x240(rate: Rate) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: rate,
        }
    }

    #[test]
    fn derived_output_replaces_framerate_only() {
        let r = VideoRate::new(10.0);
        let CapsConstraint::DerivedCoupled {
            derive,
            passthrough,
        } = r.caps_constraint_as_transform()
        else {
            panic!("expected DerivedCoupled");
        };
        // format + geometry pass through; framerate is the retargeted field.
        assert_eq!(
            passthrough,
            PassthroughFields::NONE
                .with_format()
                .with_width()
                .with_height()
        );
        let out = derive(&nv12_320x240(Rate::Fixed(30 << 16)));
        assert_eq!(out.alternatives(), &[nv12_320x240(Rate::Fixed(10 << 16))]);
    }

    #[test]
    fn auto_advertises_passthrough_then_any_and_resolves_from_output() {
        // M290 caps-driven: with no property, the derive prefers the input rate
        // (passthrough) and also advertises `Rate::Any` so a downstream
        // framerate pin couples back.
        let mut r = VideoRate::auto();
        {
            let CapsConstraint::DerivedCoupled { derive, .. } = r.caps_constraint_as_transform()
            else {
                panic!("expected DerivedCoupled");
            };
            let out = derive(&nv12_320x240(Rate::Fixed(30 << 16)));
            assert_eq!(
                out.alternatives(),
                &[nv12_320x240(Rate::Fixed(30 << 16)), nv12_320x240(Rate::Any)],
                "passthrough preferred, then any-rate for a downstream pin"
            );
        }
        // The solver fixates the output to the pinned 15 fps; configure_output
        // captures it as the effective target.
        r.configure_output(&nv12_320x240(Rate::Fixed(15 << 16)))
            .expect("fixed output rate");
        assert_eq!(r.rate_q16, 15 << 16);
        assert_eq!(r.dt_ns, (1_000_000_000u64 << 16) / (15 << 16) as u64);
        // An un-fixated (Any) output rate is rejected loud.
        let mut r2 = VideoRate::auto();
        assert_eq!(
            r2.configure_output(&nv12_320x240(Rate::Any))
                .expect_err("any rate"),
            G2gError::CapsMismatch
        );
    }
}
