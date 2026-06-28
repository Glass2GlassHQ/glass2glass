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
//! (`with_speed`, 0..=10); rate control uses the rav1e quantizer default.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    RawVideoFormat, Rate, VideoCodec,
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

/// Minimum percent change in target bitrate before the rav1e context is rebuilt.
/// rav1e cannot retarget at runtime, so each change costs a context rebuild (and
/// a keyframe); this damps a jittery BWE estimate.
const BITRATE_HYSTERESIS_PCT: u64 = 20;

/// Encodes raw planar-YUV video into an AV1 elementary stream.
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
    /// Target bitrate (bits/second) from downstream congestion control, or `None`
    /// for rav1e's default quantizer mode. A change rebuilds the rav1e context.
    bitrate_bps: Option<u32>,
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
            configured: false,
        }
    }

    /// Set the rav1e speed preset (0 slowest/best quality .. 10 fastest).
    pub fn with_speed(mut self, speed: u8) -> Self {
        self.speed = speed.min(10);
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
        let enc = EncoderConfig {
            width: self.width as usize,
            height: self.height as usize,
            bit_depth: depth,
            chroma_sampling: chroma_for(self.format).ok_or(G2gError::CapsMismatch)?,
            speed_settings: SpeedSettings::from_preset(self.speed),
            // 0 = rav1e's default quantizer mode; a downstream BWE target switches
            // to rate control (rav1e's `bitrate` is bits/second).
            bitrate: self.bitrate_bps.map_or(0, |b| b.min(i32::MAX as u32) as i32),
            ..Default::default()
        };
        let cfg = Config::new().with_encoder_config(enc);
        // rav1e packs 10/12-bit samples into `u16`; 8-bit uses `u8`.
        self.ctx = Some(if depth > 8 {
            RavCtx::U16(cfg.new_context::<u16>().map_err(|_| G2gError::CapsMismatch)?)
        } else {
            RavCtx::U8(cfg.new_context::<u8>().map_err(|_| G2gError::CapsMismatch)?)
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
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            packets,
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
    /// context (the next frame is then a keyframe). Hysteresis-gated: only act on
    /// a change of at least `BITRATE_HYSTERESIS` from the active target, so a
    /// jittery estimate near the frame rate does not thrash the encoder (each
    /// rebuild costs a keyframe). A bitrate drop is exactly when a fresh keyframe
    /// is wanted anyway. Rebuild failure leaves the current context running.
    /// Returns the packets flushed from the old context so the caller can emit
    /// them; empty when no rebuild happened.
    fn set_target_bitrate(&mut self, bps: u32) -> Vec<(Vec<u8>, u64)> {
        let bps = bps.max(1);
        let changed = match self.bitrate_bps {
            None => true,
            Some(cur) => {
                let (lo, hi) = (cur.min(bps), cur.max(bps));
                (hi - lo) as u64 * 100 >= cur as u64 * BITRATE_HYSTERESIS_PCT
            }
        };
        if !changed {
            return Vec::new();
        }
        self.bitrate_bps = Some(bps);
        // Not running yet: the next `build_context` (at configure) picks up the
        // target, nothing to flush.
        if self.ctx.is_none() {
            return Vec::new();
        }
        // Flush the running context's in-flight lookahead before the new-rate
        // context replaces it, so those frames are emitted instead of dropped.
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
                });
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, width, height, framerate } if chroma_for(*format).is_some() => {
                CapsSet::one(Caps::CompressedVideo {
                    codec: VideoCodec::Av1,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: framerate.clone(),
                })
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo { format, width, height, framerate } = absolute_caps else {
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let packets = self.encode(slice.as_slice(), frame.timing.pts_ns)?;
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
        };
        let sink = CapsSet::from_alternatives(Vec::from([
            any(RawVideoFormat::I420),
            any(RawVideoFormat::I422),
            any(RawVideoFormat::I444),
        ]));
        Vec::from([PadTemplate::sink(sink), PadTemplate::source(CapsSet::one(out))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1parse::Av1Parse;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, PushOutcome};

    fn i420_grey(w: usize, h: usize) -> Vec<u8> {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut v = alloc::vec![128u8; w * h]; // mid-grey luma
        v.extend(alloc::vec![128u8; cw * ch]); // U
        v.extend(alloc::vec![128u8; cw * ch]); // V
        v
    }

    fn i420_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
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
                FrameTiming { pts_ns: i * 33_000_000, ..FrameTiming::default() },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert!(!sink.frames.is_empty(), "the encoder produced AV1 frames");
        assert!(sink.frames.iter().all(|f| !f.is_empty()), "no empty packets");
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
            parse.process(PipelinePacket::DataFrame(f), &mut psink).await.unwrap();
        }
        let geometry = psink.caps.iter().find_map(|c| match c {
            Caps::CompressedVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => {
                Some((*w, *h))
            }
            _ => None,
        });
        assert_eq!(geometry, Some((64, 64)), "av1parse recovers the encoded 64x64 geometry");
    }

    #[tokio::test]
    async fn pts_map_is_bounded_and_round_trips() {
        #[derive(Default)]
        struct PtsSink {
            pts: Vec<u64>,
        }
        impl OutputSink for PtsSink {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
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
                MemoryDomain::System(SystemSlice::from_boxed(i420_grey(64, 64).into_boxed_slice())),
                FrameTiming { pts_ns: (i + 1) * 33_000_000, ..FrameTiming::default() },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
            // The map holds only the in-flight lookahead, never one slot per frame.
            assert!(enc.pts_by_frameno.len() < n as usize, "pts map stays bounded");
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
        assert_eq!(enc.bitrate_bps, None, "default quantizer mode until a target arrives");

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
                MemoryDomain::System(SystemSlice::from_boxed(i420_grey(64, 64).into_boxed_slice())),
                FrameTiming { pts_ns: i * 33_000_000, ..FrameTiming::default() },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        }
        enc.set_target_bitrate(500_000);
        for i in 3..6u64 {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(i420_grey(64, 64).into_boxed_slice())),
                FrameTiming { pts_ns: i * 33_000_000, ..FrameTiming::default() },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert!(!sink.frames.is_empty(), "still produces frames after a bitrate change");
        assert!(sink.frames.iter().all(|f| !f.is_empty()), "no empty packets after rebuild");
    }

    #[test]
    fn no_frame_dropped_across_a_bitrate_change() {
        let mut enc = Av1Enc::new().with_speed(10);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut emitted = 0usize;
        for i in 0..6u64 {
            emitted += enc.encode(&i420_grey(64, 64), i * 33_000_000).unwrap().len();
        }
        // The rebuild flushes the running context's lookahead (returned here),
        // so those buffered frames are not lost.
        emitted += enc.set_target_bitrate(2_000_000).len();
        for i in 6..12u64 {
            emitted += enc.encode(&i420_grey(64, 64), i * 33_000_000).unwrap().len();
        }
        emitted += enc.flush().unwrap().len();
        assert_eq!(emitted, 12, "every source frame is emitted across the rebuild");
    }
}
