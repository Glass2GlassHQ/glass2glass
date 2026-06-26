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
//! Scope (v1): 8-bit 4:2:0 (I420), geometry fixed at configure. The speed preset
//! is builder-configurable (`with_speed`, 0..=10); rate control uses the rav1e
//! quantizer default.

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
    FrameTypeOverride, SpeedSettings,
};

/// rav1e speed preset (0 slowest/best .. 10 fastest); 9 is a fast default for a
/// real-time-ish software encode.
const DEFAULT_SPEED: u8 = 9;

/// Minimum percent change in target bitrate before the rav1e context is rebuilt.
/// rav1e cannot retarget at runtime, so each change costs a context rebuild (and
/// a keyframe); this damps a jittery BWE estimate.
const BITRATE_HYSTERESIS_PCT: u64 = 20;

/// Encodes raw I420 video into an AV1 elementary stream.
pub struct Av1Enc {
    speed: u8,
    width: u32,
    height: u32,
    framerate: Rate,
    ctx: Option<Context<u8>>,
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

    fn input_template() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
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
        let enc = EncoderConfig {
            width: self.width as usize,
            height: self.height as usize,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            speed_settings: SpeedSettings::from_preset(self.speed),
            // 0 = rav1e's default quantizer mode; a downstream BWE target switches
            // to rate control (rav1e's `bitrate` is bits/second).
            bitrate: self.bitrate_bps.map_or(0, |b| b.min(i32::MAX as u32) as i32),
            ..Default::default()
        };
        let cfg = Config::new().with_encoder_config(enc);
        let ctx = cfg.new_context::<u8>().map_err(|_| G2gError::CapsMismatch)?;
        self.ctx = Some(ctx);
        self.pts_by_frameno.clear();
        self.next_frameno = 0;
        Ok(())
    }

    /// Encode one I420 access unit, returning the ready packets as `(data, pts)`.
    fn encode(&mut self, i420: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (y_size, c_size) = (w * h, cw * ch);
        if i420.len() < y_size + 2 * c_size {
            return Err(G2gError::CapsMismatch);
        }
        self.pts_by_frameno.insert(self.next_frameno, pts_ns);
        self.next_frameno += 1;
        // A pending keyframe request (downstream PLI) overrides this frame's type
        // to Key; consume the flag now. `FrameParameters` is not `Clone`, so it is
        // rebuilt per `send_frame` attempt below (the loop retries on EnoughData).
        let force_keyframe = core::mem::take(&mut self.force_keyframe);
        let frame_params = || {
            force_keyframe.then(|| FrameParameters {
                frame_type_override: FrameTypeOverride::Key,
                ..Default::default()
            })
        };
        let raw = {
            let ctx = self.ctx.as_mut().ok_or(G2gError::NotConfigured)?;
            let mut frame = ctx.new_frame();
            frame.planes[0].copy_from_raw_u8(&i420[..y_size], w, 1);
            frame.planes[1].copy_from_raw_u8(&i420[y_size..y_size + c_size], cw, 1);
            frame.planes[2].copy_from_raw_u8(&i420[y_size + c_size..y_size + 2 * c_size], cw, 1);
            let arc = Arc::new(frame);
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
        };
        Ok(self.map_pts(raw))
    }

    /// Flush the encoder at EOS and return the remaining packets.
    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let raw = {
            let ctx = self.ctx.as_mut().ok_or(G2gError::NotConfigured)?;
            let _ = ctx.send_frame(None);
            drain_ready(ctx)
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
        // A downstream bitrate estimate (WebRTC BWE) retargets the encoder.
        if let Some(bps) = feedback.bitrate_bps {
            self.set_target_bitrate(bps);
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
    fn set_target_bitrate(&mut self, bps: u32) {
        let bps = bps.max(1);
        let changed = match self.bitrate_bps {
            None => true,
            Some(cur) => {
                let (lo, hi) = (cur.min(bps), cur.max(bps));
                (hi - lo) as u64 * 100 >= cur as u64 * BITRATE_HYSTERESIS_PCT
            }
        };
        if changed {
            self.bitrate_bps = Some(bps);
            // Rebuild at the new rate if the encoder is already running; otherwise
            // the next `build_context` (at configure) picks up the target.
            if self.ctx.is_some() {
                let _ = self.build_context();
            }
        }
    }
}

/// Drain the packets rav1e has ready. `Encoded` means a frame was consumed
/// without emitting a packet (keep going); any other status means nothing more is
/// ready right now (`NeedMoreData`) or the stream is finished (`LimitReached`).
fn drain_ready(ctx: &mut Context<u8>) -> Vec<(Vec<u8>, u64)> {
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
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format: RawVideoFormat::I420, width, height, framerate } => {
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
        let Caps::RawVideo { format: RawVideoFormat::I420, width, height, framerate } =
            absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.build_context()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AV1 encoder",
            "Codec/Encoder/Video",
            "Encodes raw I420 video to AV1 via rav1e",
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
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(out)),
        ])
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
}
