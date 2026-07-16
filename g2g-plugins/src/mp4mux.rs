//! Fragmented-MP4 (ISO BMFF / CMAF) multiplexer element (M291): one H.264 or
//! H.265 elementary stream in (`Caps::CompressedVideo{H264|H265}`, Annex-B), an
//! ISO-BMFF byte stream out (`Caps::ByteStream{IsoBmff}`):
//!
//! ```text
//! ... ! x264enc ! mp4mux ! filesink location=out.mp4
//! ```
//!
//! The `mp4mux` / `qtmux` analog: wraps the pure [`Fmp4Muxer`] box writer and
//! forwards the muxed bytes downstream (to a `filesink`, `udpsink`, an HLS
//! segmenter, ...), the way gst muxing is a separate element feeding any sink.
//! `ftyp`+`moov` init segment once, then one `moof`+`mdat` fragment per access
//! unit, so a truncated recording stays valid to the last complete fragment.
//!
//! The muxer is built lazily on the first frame (its `moov` needs the in-band
//! parameter sets the first IDR carries), so a `CapsChanged` that refines the
//! geometry beforehand is reflected in the written tracks. CPU, `no_std`
//! baseline. Scope (v1): single video track; A/V multi-track interleave is
//! [`Mp4MuxN`](crate::mp4muxn).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, TagList, VideoCodec,
};

use crate::fmp4mux::Fmp4Muxer;

/// Muxes one H.264 / H.265 elementary stream into a fragmented-MP4 byte stream.
#[derive(Debug)]
pub struct Mp4Mux {
    /// Codec + geometry from the input caps, refined by `CapsChanged` until the
    /// first frame builds the muxer.
    codec: VideoCodec,
    width: u32,
    height: u32,
    tags: TagList,
    mux: Option<Fmp4Muxer>,
    configured: bool,
    emitted: u64,
    /// Target fragment duration in milliseconds (`0` = one fragment per access
    /// unit, the default). Batches access units into multi-sample CMAF / DASH
    /// fragments closed at the next keyframe once the target is reached.
    fragment_duration_ms: u64,
}

impl Default for Mp4Mux {
    fn default() -> Self {
        Self::new()
    }
}

impl Mp4Mux {
    pub fn new() -> Self {
        Self {
            codec: VideoCodec::H264,
            width: 0,
            height: 0,
            tags: TagList::new(),
            mux: None,
            configured: false,
            emitted: 0,
            fragment_duration_ms: 0,
        }
    }

    /// Attach stream metadata, written as a `moov/udta/meta/ilst` box in the init
    /// segment.
    pub fn with_tags(mut self, tags: TagList) -> Self {
        self.tags = tags;
        self
    }

    /// Batch access units into fragments of at least `ms` milliseconds (`0` keeps
    /// one fragment per AU); see [`fragment_duration_ms`](Self::fragment_duration_ms).
    pub fn with_fragment_duration_ms(mut self, ms: u64) -> Self {
        self.fragment_duration_ms = ms;
        self
    }

    /// Count of byte-stream frames forwarded (init segment + first fragment is one).
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }

    /// The compressed-video codecs `Fmp4Muxer` can carry: H.264 or H.265.
    fn accept_caps(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::CompressedVideo { codec, width, height, .. } = caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(codec, VideoCodec::H264 | VideoCodec::H265) {
            return Err(G2gError::CapsMismatch);
        }
        self.codec = *codec;
        if let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) {
            self.width = *w;
            self.height = *h;
        }
        // A built muxer rejects a post-moov codec swap; a geometry refinement is fine.
        if let Some(mux) = &mut self.mux {
            mux.update_caps(self.codec, self.width, self.height)?;
        }
        Ok(())
    }

    /// The elementary streams this muxer accepts on its sink pad.
    fn input_alternatives() -> Vec<Caps> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([video(VideoCodec::H264), video(VideoCodec::H265)])
    }
}

impl AsyncElement for Mp4Mux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in Self::input_alternatives() {
            if let Ok(c) = upstream_caps.intersect(&alt) {
                return Ok(c);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo { codec: VideoCodec::H264 | VideoCodec::H265, .. } => {
                CapsSet::one(Self::output_caps())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.accept_caps(absolute_caps)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "fragment-duration",
            PropKind::Uint,
            "target fragment duration, milliseconds (0 = one fragment per access unit)",
        )
        .with_default("0")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "fragment-duration" => {
                self.fragment_duration_ms = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "fragment-duration" => Some(PropValue::Uint(self.fragment_duration_ms)),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Build the box writer on the first AU (its moov needs the
                    // in-band parameter sets the first access unit carries).
                    let frag_ns = self.fragment_duration_ms.saturating_mul(1_000_000);
                    let mux = self.mux.get_or_insert_with(|| {
                        Fmp4Muxer::new(self.codec, self.width, self.height, self.tags.clone())
                            .with_fragment_duration_ns(frag_ns)
                    });
                    let bytes = mux.push_au(slice.as_slice(), frame.timing.pts_ns, frame.timing.duration_ns)?;
                    // In batched mode a buffering push yields no bytes yet; don't
                    // emit an empty frame.
                    if !bytes.is_empty() {
                        let out_frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                            FrameTiming { pts_ns: frame.timing.pts_ns, ..FrameTiming::default() },
                            self.emitted,
                        );
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    self.accept_caps(&c)?;
                }
                // Flush the final partial fragment (batched mode) before the runner
                // forwards EOS; a no-op in the default per-AU mode.
                PipelinePacket::Eos => {
                    if let Some(mux) = self.mux.as_mut() {
                        let tail = mux.flush();
                        if !tail.is_empty() {
                            let out_frame = Frame::new(
                                MemoryDomain::System(SystemSlice::from_boxed(tail.into_boxed_slice())),
                                FrameTiming::default(),
                                self.emitted,
                            );
                            self.emitted += 1;
                            out.push(PipelinePacket::DataFrame(out_frame)).await?;
                        }
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for Mp4Mux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Self::input_alternatives())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::PushOutcome;

    fn h264_caps(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    /// A 4-byte Annex-B start code prefix for a NAL of the given header byte +
    /// payload, so `split_annexb` / `parameter_sets` see a real AU.
    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
        frames: u64,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes.extend_from_slice(s.as_slice());
                        self.frames += 1;
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, ..FrameTiming::default() },
            0,
        ))
    }

    #[test]
    fn caps_codec_in_iso_bmff_out() {
        let m = Mp4Mux::new();
        assert!(m.intercept_caps(&h264_caps(320, 240)).is_ok());
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Nv12,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(m.intercept_caps(&raw).is_err());
        let CapsConstraint::DerivedOutput(f) = m.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(matches!(
            f(&h264_caps(320, 240)).alternatives(),
            [Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }]
        ));
    }

    #[tokio::test]
    async fn emits_iso_bmff_init_then_fragments() {
        // SPS (type 7), PPS (type 8), IDR (type 5) in the first AU so the moov's
        // avcC has its parameter sets; then a non-IDR AU.
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        let au0 = annexb(&[&sps, &pps, &idr]);
        let au1 = annexb(&[&[0x41u8, 0x9a, 0x00]]); // non-IDR slice

        let mut mux = Mp4Mux::new();
        mux.configure_pipeline(&h264_caps(320, 240)).unwrap();
        let mut sink = CaptureSink::default();
        mux.process(frame(au0, 0), &mut sink).await.unwrap();
        mux.process(frame(au1, 33_333_333), &mut sink).await.unwrap();

        assert_eq!(mux.emitted(), 2, "one out frame per AU");
        // The stream starts with `ftyp` and carries a `moov` (init segment) and
        // at least one `moof` fragment box.
        assert_eq!(&sink.bytes[4..8], b"ftyp", "ISO-BMFF starts with an ftyp box");
        let find = |needle: &[u8]| sink.bytes.windows(4).any(|w| w == needle);
        assert!(find(b"moov"), "init segment carries a moov");
        assert!(find(b"moof"), "fragments carry moof boxes");
        assert!(find(b"mdat"), "fragments carry mdat boxes");
    }

    /// Count `moof` fragment boxes and sum every `trun`'s sample count.
    fn moof_and_sample_count(bytes: &[u8]) -> (usize, u64) {
        let moofs = bytes.windows(4).filter(|w| *w == b"moof").count();
        let mut samples = 0u64;
        for (i, w) in bytes.windows(4).enumerate() {
            if w == b"trun" {
                // [..'trun'][version+flags:4][sample_count:4]...
                let off = i + 8;
                if off + 4 <= bytes.len() {
                    samples += u32::from_be_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]) as u64;
                }
            }
        }
        (moofs, samples)
    }

    #[tokio::test]
    async fn fragment_duration_batches_aus_into_keyframe_aligned_fragments() {
        // Six AUs at 30 fps: AU0 and AU5 are IDR (sync), the rest non-IDR.
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        let key = annexb(&[&sps, &pps, &idr]);
        let inter = || annexb(&[&[0x41u8, 0x9a, 0x00]]);
        let aus = [key.clone(), inter(), inter(), inter(), inter(), key.clone()];

        // Batched: a 10 ms target (each frame is ~33 ms) closes the fragment at the
        // next IDR, so AU0..AU4 form one fragment and AU5 the next (flushed at EOS).
        let mut mux = Mp4Mux::new().with_fragment_duration_ms(10);
        mux.configure_pipeline(&h264_caps(320, 240)).unwrap();
        let mut sink = CaptureSink::default();
        for (i, au) in aus.iter().enumerate() {
            mux.process(frame(au.clone(), i as u64 * 33_333_333), &mut sink).await.unwrap();
        }
        mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        let (moofs, samples) = moof_and_sample_count(&sink.bytes);
        assert_eq!(moofs, 2, "six AUs batch into two keyframe-aligned fragments");
        assert_eq!(samples, 6, "every access unit is preserved as a sample");

        // Default (per-AU): one fragment per access unit.
        let mut mux0 = Mp4Mux::new();
        mux0.configure_pipeline(&h264_caps(320, 240)).unwrap();
        let mut sink0 = CaptureSink::default();
        for (i, au) in aus.iter().enumerate() {
            mux0.process(frame(au.clone(), i as u64 * 33_333_333), &mut sink0).await.unwrap();
        }
        mux0.process(PipelinePacket::Eos, &mut sink0).await.unwrap();
        let (moofs0, samples0) = moof_and_sample_count(&sink0.bytes);
        assert_eq!(moofs0, 6, "per-AU mode emits one fragment per access unit");
        assert_eq!(samples0, 6);
    }
}
