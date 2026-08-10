//! AV1 software encoder element (Av1Enc, `av1-encode` feature): `RawVideo{I420}`
//! in, `CompressedVideo{Av1}` out, via the pure-Rust `rav1e` encoder.
//!
//! rav1e has frame lookahead, so a frame sent in does not immediately produce a
//! packet: `process` drains whatever packets are ready after each `send_frame`,
//! and the EOS path flushes the encoder (a `None` frame) and drains the rest. Each
//! output packet is one encoded AV1 frame; its PTS is recovered from the input it
//! came from (`Packet::input_frameno`), since AV1 may reorder. Output is the
//! low-overhead OBU stream that [`crate::av1parse::Av1Parse`] reads.
//!
//! Scope: planar YUV at 8 / 10 / 12-bit in 4:2:0 (`I420`), 4:2:2 (`I422`), and
//! 4:4:4 (`I444`), geometry fixed at configure. rav1e is generic over the sample
//! type, so the encoder holds either a `Context<u8>` (8-bit) or a `Context<u16>`
//! (10/12-bit, samples little-endian) selected from the input format; one generic
//! `encode_frame` drives both. The speed preset is builder-configurable
//! (`with_speed`, 0..=10).
//!
//! Rate control is one of two mutually exclusive modes, since rav1e turns its
//! rate controller off exactly when `bitrate <= 0` and reads `quantizer` only
//! then: a target bitrate (`bitrate`, from a property or downstream congestion
//! control) or a fixed quantizer (`with_quantizer` / the `quantizer` property,
//! 1 best .. 255 worst) for constant quality. An explicit set of either one
//! clears the other, and a downstream bitrate estimate is ignored while an
//! explicit quantizer is in force. rav1e fixes both at `Context` construction,
//! so a change rebuilds the context (the next frame is a keyframe) after
//! flushing the running one.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use rav1e::prelude::{
    ChromaSampling, Config, Context, EncoderConfig, EncoderStatus, FrameParameters,
    FrameTypeOverride, Pixel, SpeedSettings,
};

/// A live rav1e context, monomorphized to the sample type the input format needs:
/// `u8` for 8-bit, `u16` for 10/12-bit (samples little-endian).
enum RavCtx {
    U8(Context<u8>),
    U16(Context<u16>),
}

/// rav1e speed preset (0 slowest/best .. 10 fastest); 9 is a fast default for a
/// real-time-ish software encode.
const DEFAULT_SPEED: u8 = 9;

/// rav1e's own default base quantizer, used when neither an explicit quantizer
/// nor a bitrate target is set.
const DEFAULT_QUANTIZER: usize = 100;

/// Encodes raw planar-YUV video into an AV1 elementary stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::av1enc::Av1Enc;
///
/// let encoder = Av1Enc::new().with_speed(9).with_quantizer(100);
/// assert_eq!(encoder.emitted(), 0);
/// ```
pub struct Av1Enc {
    speed: u8,
    width: u32,
    height: u32,
    /// The negotiated input format (planar `I420` / `I422` / `I444` at 8/10/12-bit);
    /// fixes the rav1e chroma sampling, bit depth, and the per-frame plane geometry.
    format: RawVideoFormat,
    framerate: Rate,
    ctx: Option<RavCtx>,
    /// Source PTS keyed by `Packet::input_frameno`. Entries are removed as their
    /// packet is emitted, so this stays bounded to the encoder's lookahead window
    /// rather than growing one slot per frame for the stream lifetime.
    pts_by_frameno: BTreeMap<u64, u64>,
    /// Next input frame number to assign (resets with the rav1e context).
    next_frameno: u64,
    emitted: u64,
    caps_sent: bool,
    /// A downstream element (e.g. a WebRTC sink on a remote PLI) asked for a
    /// keyframe; the next `encode` overrides the frame type to Key and clears it.
    force_keyframe: bool,
    /// Target bitrate (bits/second) from a property or downstream congestion
    /// control, or `None` when not rate-targeted. A change rebuilds the rav1e
    /// context. Mutually exclusive with `quantizer`.
    bitrate_bps: Option<u32>,
    /// Explicit base quantizer (1 best .. 255 worst) for constant-quality
    /// encoding, or `None` for rav1e's default. Mutually exclusive with
    /// `bitrate_bps`.
    quantizer: Option<u8>,
    /// Packets flushed out of the previous context by a property-driven rebuild.
    /// They are older than anything the new context produces, so the next `emit`
    /// leads with them rather than dropping them (a property set has no sink to
    /// push to).
    pending: Vec<(Vec<u8>, u64)>,
    configured: bool,
}

impl core::fmt::Debug for Av1Enc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // rav1e's Context is not Debug, so report the configuration instead.
        f.debug_struct("Av1Enc")
            .field("speed", &self.speed)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for Av1Enc {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1Enc {
    pub fn new() -> Self {
        Self {
            speed: DEFAULT_SPEED,
            width: 0,
            height: 0,
            format: RawVideoFormat::I420,
            framerate: Rate::Any,
            ctx: None,
            pts_by_frameno: BTreeMap::new(),
            next_frameno: 0,
            emitted: 0,
            caps_sent: false,
            force_keyframe: false,
            bitrate_bps: None,
            quantizer: None,
            pending: Vec::new(),
            configured: false,
        }
    }

    /// Set the rav1e speed preset (0 slowest/best quality .. 10 fastest).
    pub fn with_speed(mut self, speed: u8) -> Self {
        self.speed = speed.min(10);
        self
    }

    /// Encode at a fixed base quantizer (1 highest quality .. 255 lowest) for
    /// constant quality instead of a bitrate target. 0 is lossless, which rav1e
    /// does not implement, so it is raised to 1.
    pub fn with_quantizer(mut self, quantizer: u8) -> Self {
        self.apply_rate_control(Some(quantizer.max(1)), None);
        self
    }

    /// Count of AV1 frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    fn build_context(&mut self) -> Result<(), G2gError> {
        let depth = self.format.bit_depth() as usize;
        // rav1e reads these two together: rate control runs only while
        // `bitrate > 0`, and in that mode `quantizer` is reinterpreted as the
        // worst quantizer index the controller may pick (255 = unconstrained,
        // what rav1e's own CLI passes with a bitrate). Outside it, `quantizer`
        // is the flat base quantizer for every frame.
        let (bitrate, quantizer) = match (self.quantizer, self.bitrate_bps) {
            (Some(q), _) => (0, q as usize),
            (None, Some(bps)) => (bps.min(i32::MAX as u32) as i32, 255),
            (None, None) => (0, DEFAULT_QUANTIZER),
        };
        let enc = EncoderConfig {
            width: self.width as usize,
            height: self.height as usize,
            bit_depth: depth,
            chroma_sampling: chroma_for(self.format).ok_or(G2gError::CapsMismatch)?,
            speed_settings: SpeedSettings::from_preset(self.speed),
            // rav1e's `bitrate` is bits/second.
            bitrate,
            quantizer,
            ..Default::default()
        };
        let cfg = Config::new().with_encoder_config(enc);
        // rav1e packs 10/12-bit samples into `u16`; 8-bit uses `u8`.
        self.ctx = Some(if depth > 8 {
            RavCtx::U16(
                cfg.new_context::<u16>()
                    .map_err(|_| G2gError::CapsMismatch)?,
            )
        } else {
            RavCtx::U8(
                cfg.new_context::<u8>()
                    .map_err(|_| G2gError::CapsMismatch)?,
            )
        });
        self.pts_by_frameno.clear();
        self.next_frameno = 0;
        Ok(())
    }

    /// Encode one planar-YUV access unit, returning the ready packets as `(data, pts)`.
    /// The chroma plane size and per-sample byte width follow the configured format,
    /// so 4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 / 12-bit share this path.
    fn encode(&mut self, planar: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let bps = self.format.bytes_per_sample();
        // `chroma_shift` is `Some` for every supported (planar) input format.
        let (hs, vs) = self.format.chroma_shift().ok_or(G2gError::CapsMismatch)?;
        let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
        let plane_dims = [(w, h), (cw, ch), (cw, ch)];
        if planar.len() < (w * h + 2 * cw * ch) * bps {
            return Err(G2gError::CapsMismatch);
        }
        self.pts_by_frameno.insert(self.next_frameno, pts_ns);
        self.next_frameno += 1;
        // A pending keyframe request (downstream PLI) overrides this frame's type
        // to Key; consume the flag now.
        let force_keyframe = core::mem::take(&mut self.force_keyframe);
        let raw = match self.ctx.as_mut().ok_or(G2gError::NotConfigured)? {
            RavCtx::U8(ctx) => encode_frame(ctx, planar, plane_dims, bps, force_keyframe),
            RavCtx::U16(ctx) => encode_frame(ctx, planar, plane_dims, bps, force_keyframe),
        };
        Ok(self.map_pts(raw))
    }

    /// Flush the encoder at EOS and return the remaining packets.
    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let raw = match self.ctx.as_mut().ok_or(G2gError::NotConfigured)? {
            RavCtx::U8(ctx) => {
                let _ = ctx.send_frame(None);
                drain_ready(ctx)
            }
            RavCtx::U16(ctx) => {
                let _ = ctx.send_frame(None);
                drain_ready(ctx)
            }
        };
        Ok(self.map_pts(raw))
    }

    fn map_pts(&mut self, raw: Vec<(Vec<u8>, u64)>) -> Vec<(Vec<u8>, u64)> {
        raw.into_iter()
            .map(|(data, frameno)| (data, self.pts_by_frameno.remove(&frameno).unwrap_or(0)))
            .collect()
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = self.output_caps();
        // Anything a property-driven rebuild flushed out of the previous context
        // leads the batch: it is older than every packet the new one produced.
        let mut batch = core::mem::take(&mut self.pending);
        batch.extend(packets);
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            batch,
            &caps,
            out,
        )
        .await?;
        // A downstream keyframe request (PLI) latches here; the next `encode`
        // forces a Key frame.
        if feedback.force_keyframe {
            self.force_keyframe = true;
        }
        // A downstream bitrate estimate (WebRTC BWE) retargets the encoder. The
        // rebuild flushes the old context's lookahead; emit those frames too.
        if let Some(bps) = feedback.bitrate_bps {
            let drained = self.set_target_bitrate(bps);
            if !drained.is_empty() {
                crate::encoder_base::emit_packets(
                    &mut self.caps_sent,
                    &mut self.emitted,
                    drained,
                    &caps,
                    out,
                )
                .await?;
            }
        }
        Ok(())
    }

    /// Apply a target bitrate (bits/second) from downstream congestion control.
    /// rav1e fixes the rate at `Context` construction, so a change rebuilds the
    /// context (the next frame is then a keyframe). Hysteresis-gated (see
    /// `encoder_base::bitrate_change_is_significant`), so a
    /// jittery estimate near the frame rate does not thrash the encoder (each
    /// rebuild costs a keyframe). A bitrate drop is exactly when a fresh keyframe
    /// is wanted anyway. An explicit quantizer outranks the estimate: constant
    /// quality was asked for deliberately, congestion control only guesses.
    /// Returns the packets flushed from the old context so the caller can emit
    /// them; empty when no rebuild happened.
    fn set_target_bitrate(&mut self, bps: u32) -> Vec<(Vec<u8>, u64)> {
        if self.quantizer.is_some() {
            return Vec::new();
        }
        let bps = bps.max(1);
        let changed = match self.bitrate_bps {
            None => true,
            Some(cur) => crate::encoder_base::bitrate_change_is_significant(cur as u64, bps as u64),
        };
        if !changed {
            return Vec::new();
        }
        self.bitrate_bps = Some(bps);
        self.rebuild()
    }

    /// Install a rate-control setting explicitly (a builder or a property, not a
    /// downstream estimate), so it is applied ungated by the bitrate hysteresis:
    /// an explicit set is intent. At most one of the two is ever live. Packets
    /// flushed out of the old context are held for the next emit, there being no
    /// sink at hand.
    fn apply_rate_control(&mut self, quantizer: Option<u8>, bitrate_bps: Option<u32>) {
        if (self.quantizer, self.bitrate_bps) == (quantizer, bitrate_bps) {
            return;
        }
        self.quantizer = quantizer;
        self.bitrate_bps = bitrate_bps;
        let drained = self.rebuild();
        self.pending.extend(drained);
    }

    /// Rebuild the rav1e context onto the current rate-control setting, flushing
    /// the running one first so its in-flight lookahead is emitted rather than
    /// dropped, and returning those packets. Before configure there is nothing to
    /// flush: the first `build_context` picks the setting up. Rebuild failure
    /// leaves the current context running.
    fn rebuild(&mut self) -> Vec<(Vec<u8>, u64)> {
        if self.ctx.is_none() {
            return Vec::new();
        }
        let drained = self.flush().unwrap_or_default();
        let _ = self.build_context();
        drained
    }
}

/// The rav1e chroma sampling for a supported input format, or `None` if the format
/// is not a planar YUV the encoder accepts. Covers 8 / 10 / 12-bit (the sample
/// depth picks the `Context` pixel type separately); the subsampling is read from
/// the format itself so every depth of one chroma maps the same.
fn chroma_for(format: RawVideoFormat) -> Option<ChromaSampling> {
    Some(match format.chroma_shift()? {
        (1, 1) => ChromaSampling::Cs420,
        (1, 0) => ChromaSampling::Cs422,
        (0, 0) => ChromaSampling::Cs444,
        _ => return None,
    })
}

/// Fill a fresh frame from the tightly-packed planar `src` (Y, U, V planes of
/// `plane_dims` samples, `bps` bytes each), send it, and return the ready packets.
/// Generic over the rav1e sample type so the 8-bit (`u8`) and 10/12-bit (`u16`)
/// contexts share one body; `copy_from_raw_u8` reinterprets `src` per `bps`.
fn encode_frame<T: Pixel>(
    ctx: &mut Context<T>,
    src: &[u8],
    plane_dims: [(usize, usize); 3],
    bps: usize,
    force_keyframe: bool,
) -> Vec<(Vec<u8>, u64)> {
    let mut frame = ctx.new_frame();
    let mut off = 0;
    for (i, (pw, ph)) in plane_dims.iter().enumerate() {
        let len = pw * ph * bps;
        frame.planes[i].copy_from_raw_u8(&src[off..off + len], pw * bps, bps);
        off += len;
    }
    // Replicate each plane's edges into its allocation padding. rav1e pads in place
    // only when it can uniquely borrow the frame, but the retry-on-EnoughData loop
    // below holds a clone, so it cannot; it then asserts the padding is present.
    // Padding up front satisfies that for both the 8- and high-bit-depth paths.
    let (luma_w, luma_h) = plane_dims[0];
    for plane in frame.planes.iter_mut() {
        plane.pad(luma_w, luma_h);
    }
    let arc = Arc::new(frame);
    // `FrameParameters` is not `Clone`, so it is rebuilt per `send_frame` attempt.
    let frame_params = || {
        force_keyframe.then(|| FrameParameters {
            frame_type_override: FrameTypeOverride::Key,
            ..Default::default()
        })
    };
    let mut packets = Vec::new();
    // send_frame asks us to drain (EnoughData) when its lookahead is full.
    loop {
        match ctx.send_frame((arc.clone(), frame_params())) {
            Ok(()) => break,
            Err(EncoderStatus::EnoughData) => packets.extend(drain_ready(ctx)),
            Err(_) => break,
        }
    }
    packets.extend(drain_ready(ctx));
    packets
}

/// Drain the packets rav1e has ready. `Encoded` means a frame was consumed
/// without emitting a packet (keep going); any other status means nothing more is
/// ready right now (`NeedMoreData`) or the stream is finished (`LimitReached`).
fn drain_ready<T: Pixel>(ctx: &mut Context<T>) -> Vec<(Vec<u8>, u64)> {
    let mut out = Vec::new();
    loop {
        match ctx.receive_packet() {
            Ok(pkt) => out.push((pkt.data, pkt.input_frameno)),
            Err(EncoderStatus::Encoded) => continue,
            Err(_) => break,
        }
    }
    out
}

impl AsyncElement for Av1Enc {
    // M759: a re-encode to a compressed codec drops pixel-derived meta
    // (AnalyticsMeta); opaque side-data (BlobMeta) rides through.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    fn handles_keyframe_requests(&self) -> bool {
        true
    }

    fn handles_bitrate_requests(&self) -> bool {
        true
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Accept any supported planar-YUV input, narrowing only geometry; the format
        // itself is kept (the encoder configures its chroma sampling to match).
        if let Caps::RawVideo { format, .. } = upstream_caps {
            if chroma_for(*format).is_some() {
                return upstream_caps.intersect(&Caps::RawVideo {
                    format: *format,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: g2g_core::Interlace::Any,
                });
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                interlace: _,
            } if chroma_for(*format).is_some() => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            interlace: _,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if chroma_for(*format).is_none() {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        self.width = *w;
        self.height = *h;
        self.format = *format;
        self.framerate = framerate.clone();
        self.build_context()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AV1 encoder",
            "Codec/Encoder/Video",
            "Encodes raw planar YUV (I420/I422/I444) to AV1 via rav1e",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "bitrate",
                PropKind::Uint,
                "target bitrate, bits/second (0 = none); clears quantizer",
            )
            .with_default("0"),
            PropertySpec::new(
                "quantizer",
                PropKind::Uint,
                "constant-quality quantizer, 1 best .. 255 worst (0 = unset); clears bitrate",
            )
            .with_range("0", "255")
            .with_default("0"),
            PropertySpec::new(
                "speed",
                PropKind::Uint,
                "rav1e speed preset (0 slowest/best .. 10 fastest)",
            )
            .with_range("0", "10")
            .with_default("9"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // bits/second; 0 drops the rate target. A real target switches back
            // to rate control, dropping an explicit quantizer.
            "bitrate" => {
                let bps = value.as_uint().ok_or(PropError::Type)?;
                if bps > u32::MAX as u64 {
                    return Err(PropError::Value);
                }
                let bitrate = (bps != 0).then(|| (bps as u32).max(1));
                let quantizer = self.quantizer.filter(|_| bitrate.is_none());
                self.apply_rate_control(quantizer, bitrate);
                Ok(())
            }
            // 0 = unset (rav1e's default / whatever the bitrate target implies);
            // 1..=255 selects constant quality and drops the bitrate target.
            "quantizer" => {
                let q = value.as_uint().ok_or(PropError::Type)?;
                if q > 255 {
                    return Err(PropError::Value);
                }
                let quantizer = (q != 0).then_some(q as u8);
                let bitrate = self.bitrate_bps.filter(|_| quantizer.is_none());
                self.apply_rate_control(quantizer, bitrate);
                Ok(())
            }
            "speed" => {
                self.speed = (value.as_uint().ok_or(PropError::Type)? as u8).min(10);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps.unwrap_or(0) as u64)),
            "quantizer" => Some(PropValue::Uint(self.quantizer.unwrap_or(0) as u64)),
            "speed" => Some(PropValue::Uint(self.speed as u64)),
            _ => None,
        }
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let packets = self.encode(slice, frame.timing.pts_ns)?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the lookahead; the runner's transform arm forwards EOS.
                    let packets = self.flush()?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Av1Enc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let any = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let sink = CapsSet::from_alternatives(Vec::from([
            any(RawVideoFormat::I420),
            any(RawVideoFormat::I422),
            any(RawVideoFormat::I444),
        ]));
        Vec::from([
            PadTemplate::sink(sink),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1parse::Av1Parse;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{FrameTiming, PushOutcome};

    fn i420_grey(w: usize, h: usize) -> Vec<u8> {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut v = alloc::vec![128u8; w * h]; // mid-grey luma
        v.extend(alloc::vec![128u8; cw * ch]); // U
        v.extend(alloc::vec![128u8; cw * ch]); // V
        v
    }

    /// Deterministic pseudo-random (xorshift32) I420 frame. Detail the encoder
    /// cannot code for free is what makes the quantizer observable: a flat frame
    /// is a handful of bytes at any quality.
    fn i420_noise(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as u8
        };
        (0..w * h + 2 * cw * ch).map(|_| next()).collect()
    }

    fn i420_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
                        }
                    }
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[tokio::test]
    async fn encodes_i420_to_av1_that_av1parse_reads() {
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..5u64 {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    i420_grey(64, 64).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert!(!sink.frames.is_empty(), "the encoder produced AV1 frames");
        assert!(
            sink.frames.iter().all(|f| !f.is_empty()),
            "no empty packets"
        );
        assert_eq!(
            sink.caps,
            alloc::vec![Caps::CompressedVideo {
                codec: VideoCodec::Av1,
                width: Dim::Fixed(64),
                height: Dim::Fixed(64),
                framerate: Rate::Any,
            }]
        );

        // Round-trip: av1parse recovers the geometry from the encoded sequence
        // header, proving the output is a valid AV1 elementary stream.
        let mut parse = Av1Parse::new();
        parse.configure_pipeline(&sink.caps[0]).unwrap();
        let mut psink = CaptureSink::default();
        for data in &sink.frames {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            parse
                .process(PipelinePacket::DataFrame(f), &mut psink)
                .await
                .unwrap();
        }
        let geometry = psink.caps.iter().find_map(|c| match c {
            Caps::CompressedVideo {
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => Some((*w, *h)),
            _ => None,
        });
        assert_eq!(
            geometry,
            Some((64, 64)),
            "av1parse recovers the encoded 64x64 geometry"
        );
    }

    #[tokio::test]
    async fn pts_map_is_bounded_and_round_trips() {
        #[derive(Default)]
        struct PtsSink {
            pts: Vec<u64>,
        }
        impl OutputSink for PtsSink {
            fn poll_push(
                &mut self,
                _cx: &mut core::task::Context<'_>,
                packet_slot: &mut Option<PipelinePacket>,
            ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
                let packet = packet_slot.take().expect("poll_push without a packet");
                core::task::Poll::Ready({
                    if let PipelinePacket::DataFrame(f) = packet {
                        self.pts.push(f.timing.pts_ns);
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut sink = PtsSink::default();
        let n = 40u64;
        for i in 0..n {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    i420_grey(64, 64).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: (i + 1) * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
            // The map holds only the in-flight lookahead, never one slot per frame.
            assert!(
                enc.pts_by_frameno.len() < n as usize,
                "pts map stays bounded"
            );
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let mut got = sink.pts.clone();
        got.sort_unstable();
        let expected: Vec<u64> = (0..n).map(|i| (i + 1) * 33_000_000).collect();
        assert_eq!(got, expected, "each source pts is emitted exactly once");
        assert!(enc.pts_by_frameno.is_empty(), "pts map fully drains at EOS");
    }

    #[test]
    fn bitrate_target_applies_with_hysteresis() {
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        assert_eq!(
            enc.bitrate_bps, None,
            "default quantizer mode until a target arrives"
        );

        // First target always applies.
        enc.set_target_bitrate(1_000_000);
        assert_eq!(enc.bitrate_bps, Some(1_000_000));

        // A small change (< 20%) is damped to avoid a rebuild-per-estimate.
        enc.set_target_bitrate(1_050_000);
        assert_eq!(enc.bitrate_bps, Some(1_000_000), "5% change ignored");

        // A large change applies (and the rebuilt context is still usable).
        enc.set_target_bitrate(2_000_000);
        assert_eq!(enc.bitrate_bps, Some(2_000_000), "100% change applied");
        assert!(enc.ctx.is_some(), "rebuild left a live context");
    }

    #[tokio::test]
    async fn encodes_after_a_bitrate_change() {
        // A mid-stream bitrate retarget rebuilds the context; the encoder must
        // keep producing valid frames with monotonic timestamps afterward.
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..3u64 {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    i420_grey(64, 64).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        enc.set_target_bitrate(500_000);
        for i in 3..6u64 {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    i420_grey(64, 64).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert!(
            !sink.frames.is_empty(),
            "still produces frames after a bitrate change"
        );
        assert!(
            sink.frames.iter().all(|f| !f.is_empty()),
            "no empty packets after rebuild"
        );
    }

    #[test]
    fn no_frame_dropped_across_a_bitrate_change() {
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut emitted = 0usize;
        for i in 0..6u64 {
            emitted += enc
                .encode(&i420_grey(64, 64), i * 33_000_000)
                .unwrap()
                .len();
        }
        // The rebuild flushes the running context's lookahead (returned here),
        // so those buffered frames are not lost.
        emitted += enc.set_target_bitrate(2_000_000).len();
        for i in 6..12u64 {
            emitted += enc
                .encode(&i420_grey(64, 64), i * 33_000_000)
                .unwrap()
                .len();
        }
        emitted += enc.flush().unwrap().len();
        assert_eq!(
            emitted, 12,
            "every source frame is emitted across the rebuild"
        );
    }

    #[test]
    fn quantizer_property_round_trips_and_rejects_invalid() {
        let mut enc = Av1Enc::new();
        assert_eq!(
            enc.get_property("quantizer"),
            Some(PropValue::Uint(0)),
            "unset by default"
        );
        enc.set_property("quantizer", PropValue::Uint(60)).unwrap();
        assert_eq!(enc.get_property("quantizer"), Some(PropValue::Uint(60)));
        assert_eq!(enc.quantizer, Some(60), "onto the field the encoder reads");

        // rav1e's quantizer is 8-bit; anything above it is out of range.
        assert_eq!(
            enc.set_property("quantizer", PropValue::Uint(256)),
            Err(PropError::Value)
        );
        assert_eq!(
            enc.set_property("quantizer", PropValue::Str("high".into())),
            Err(PropError::Type)
        );
        assert_eq!(
            enc.get_property("quantizer"),
            Some(PropValue::Uint(60)),
            "a rejected set leaves the quantizer alone"
        );

        // 0 clears it, back to rav1e's default.
        enc.set_property("quantizer", PropValue::Uint(0)).unwrap();
        assert_eq!(enc.quantizer, None);
    }

    #[test]
    fn low_quantizer_encodes_more_bytes_than_high_quantizer() {
        // Constant quality is observable as size: the same frames at a low
        // quantizer (high quality) code to materially more bytes than at a high
        // one, which a bitrate-targeted or fixed-default encode would not show.
        fn encoded_bytes(quantizer: u8) -> usize {
            let mut enc = Av1Enc::new().with_speed(10).with_quantizer(quantizer);
            enc.configure_pipeline(&i420_caps(128, 128)).unwrap();
            let size = |packets: Vec<(Vec<u8>, u64)>| -> usize {
                packets.iter().map(|(data, _)| data.len()).sum()
            };
            let mut total = 0;
            for i in 0..6u64 {
                let frame = i420_noise(128, 128, i as u32 + 1);
                total += size(enc.encode(&frame, i * 33_000_000).unwrap());
            }
            total + size(enc.flush().unwrap())
        }
        let best = encoded_bytes(30);
        let worst = encoded_bytes(220);
        assert!(
            best > worst * 2,
            "quantizer 30 ({best} bytes) codes far more than quantizer 220 ({worst} bytes)"
        );
    }

    #[tokio::test]
    async fn mid_stream_quantizer_change_restarts_on_a_keyframe() {
        let mut enc = Av1Enc::new().with_speed(10).with_quantizer(60);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut sink = CaptureSink::default();
        async fn push(enc: &mut Av1Enc, sink: &mut CaptureSink, i: u64) {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    i420_noise(64, 64, i as u32 + 1).into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), sink)
                .await
                .unwrap();
        }
        for i in 0..4u64 {
            push(&mut enc, &mut sink, i).await;
        }
        enc.set_property("quantizer", PropValue::Uint(200)).unwrap();
        assert_eq!(enc.quantizer, Some(200));
        for i in 4..8u64 {
            push(&mut enc, &mut sink, i).await;
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        // rav1e emits one packet per input frame, so the four frames the old
        // context held (flushed by the rebuild, not dropped) are packets 0..4 and
        // the new context's first packet is 4.
        assert_eq!(sink.frames.len(), 8, "no frame lost across the rebuild");
        assert!(
            crate::av1parse::av1_keyframe(&sink.frames[4]),
            "the new-quantizer context starts on a keyframe"
        );
    }

    #[test]
    fn quantizer_and_bitrate_are_mutually_exclusive() {
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        enc.set_property("bitrate", PropValue::Uint(1_000_000))
            .unwrap();
        assert_eq!(enc.bitrate_bps, Some(1_000_000));

        // An explicit quantizer wins: rav1e runs rate control or a flat
        // quantizer, never both.
        enc.set_property("quantizer", PropValue::Uint(80)).unwrap();
        assert_eq!(enc.quantizer, Some(80));
        assert_eq!(enc.bitrate_bps, None, "the rate target is dropped");

        // A downstream BWE estimate does not override that.
        assert!(enc.set_target_bitrate(400_000).is_empty());
        assert_eq!(enc.bitrate_bps, None);
        assert_eq!(enc.quantizer, Some(80));

        // An explicit bitrate switches back to rate control.
        enc.set_property("bitrate", PropValue::Uint(2_000_000))
            .unwrap();
        assert_eq!(enc.bitrate_bps, Some(2_000_000));
        assert_eq!(enc.quantizer, None, "constant quality is dropped");
        assert!(enc.ctx.is_some(), "each switch left a live context");
    }
}
