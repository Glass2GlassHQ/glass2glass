//! VP8 / VP9 software encode element (VpxEnc, `vpx` feature): `RawVideo{I420}` in,
//! `CompressedVideo{Vp8|Vp9}` out, via libvpx through the `vpx-encode` crate.
//!
//! The GStreamer `vp8enc` / `vp9enc` analog. Unlike the pure-Rust [`crate::av1enc`]
//! (rav1e), this links the system `libvpx` and runs bindgen (clang) at build, so it
//! is gated behind the `vpx` feature and left out of pure-Rust / no_std builds.
//! Building needs libvpx installed (Windows: vcpkg `libvpx`; Debian/Ubuntu:
//! `libvpx-dev`; Fedora: `libvpx-devel`) plus clang/LLVM for bindgen.
//!
//! Scope (v1): 8-bit I420, geometry fixed at configure, single-pass
//! target-bitrate rate control. The codec (VP8/VP9) and bitrate are
//! builder-configurable.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, RawVideoFormat, Rate, VideoCodec,
};

use vpx_encode::{Config, Encoder, VideoCodecId};

/// Default target bitrate (kbps) when none is set.
const DEFAULT_BITRATE_KBPS: u32 = 1024;
/// Encode timestamps in milliseconds (timebase 1/1000): a frame's PTS in ns maps
/// to `pts_ns / 1_000_000` going in, and a packet's back to `pts * 1_000_000`.
const TIMEBASE_DEN: i32 = 1000;

/// Encodes raw I420 video into a VP8 or VP9 elementary stream via libvpx.
pub struct VpxEnc {
    codec: VideoCodec,
    bitrate_kbps: u32,
    width: u32,
    height: u32,
    framerate: Rate,
    enc: Option<Encoder>,
    emitted: u64,
    caps_sent: bool,
    configured: bool,
}

// SAFETY: `vpx_encode::Encoder` wraps a raw libvpx `vpx_codec_ctx_t` pointer and is
// thus `!Send` by default. The `multi-thread` runner requires `Send` so it can move
// a task between worker threads. We uphold that for `VpxEnc` by construction: the
// libvpx context carries no thread affinity (it is a heap allocation, no TLS or
// thread-bound handles), the runner drives a single element through `&mut self`
// (never concurrently), and the encoder is moved between threads but never shared,
// so there is no data race. Mirrors `MfDecode`'s contract.
unsafe impl Send for VpxEnc {}

impl core::fmt::Debug for VpxEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // vpx_encode's Encoder is not Debug; report the configuration instead.
        f.debug_struct("VpxEnc")
            .field("codec", &self.codec)
            .field("bitrate_kbps", &self.bitrate_kbps)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for VpxEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl VpxEnc {
    /// A VP9 encoder (the modern default); use [`with_codec`](Self::with_codec)
    /// for VP8.
    pub fn new() -> Self {
        Self {
            codec: VideoCodec::Vp9,
            bitrate_kbps: DEFAULT_BITRATE_KBPS,
            width: 0,
            height: 0,
            framerate: Rate::Any,
            enc: None,
            emitted: 0,
            caps_sent: false,
            configured: false,
        }
    }

    /// Select VP8 or VP9 (other codecs are rejected at `configure_pipeline`).
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Set the target bitrate in kbps (single-pass rate control).
    pub fn with_bitrate_kbps(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Count of encoded frames emitted.
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
            codec: self.codec,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    /// The libvpx codec id for the selected codec, or `CapsMismatch` for a codec
    /// this element does not encode.
    fn codec_id(&self) -> Result<VideoCodecId, G2gError> {
        match self.codec {
            VideoCodec::Vp8 => Ok(VideoCodecId::VP8),
            VideoCodec::Vp9 => Ok(VideoCodecId::VP9),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn build_encoder(&mut self) -> Result<(), G2gError> {
        let config = Config {
            width: self.width,
            height: self.height,
            timebase: [1, TIMEBASE_DEN],
            bitrate: self.bitrate_kbps,
            codec: self.codec_id()?,
        };
        let enc = Encoder::new(config).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.enc = Some(enc);
        Ok(())
    }

    /// Encode one I420 access unit, returning the ready packets as `(data, pts_ns)`.
    fn encode(&mut self, i420: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let expected = (self.width as usize) * (self.height as usize) * 3 / 2;
        if i420.len() < expected {
            return Err(G2gError::CapsMismatch);
        }
        let enc = self.enc.as_mut().ok_or(G2gError::NotConfigured)?;
        let pts = (pts_ns / 1_000_000) as i64;
        let packets =
            enc.encode(pts, i420).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let mut out = Vec::new();
        for frame in packets {
            out.push((frame.data.to_vec(), (frame.pts as u64) * 1_000_000));
        }
        Ok(out)
    }

    /// Flush the encoder at EOS and return the remaining packets, consuming the
    /// libvpx context (a fresh `configure_pipeline` would rebuild it).
    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let Some(enc) = self.enc.take() else {
            return Ok(Vec::new());
        };
        let mut finish = enc.finish().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let mut out = Vec::new();
        while let Some(frame) =
            finish.next().map_err(|_| G2gError::Hardware(HardwareError::Other))?
        {
            out.push((frame.data.to_vec(), (frame.pts as u64) * 1_000_000));
        }
        Ok(out)
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if !packets.is_empty() && !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps())).await?;
            self.caps_sent = true;
        }
        for (data, pts_ns) in packets {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for VpxEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo { format: RawVideoFormat::I420, width, height, framerate } => {
                CapsSet::one(Caps::CompressedVideo {
                    codec,
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
        self.codec_id()?; // reject a non-VP8/VP9 selection before building libvpx
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.build_encoder()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "VP8 / VP9 encoder",
            "Codec/Encoder/Video",
            "Encodes raw I420 video to VP8 or VP9 via libvpx",
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
                    // Flush libvpx; the runner's transform arm forwards EOS.
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

impl PadTemplates for VpxEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let video = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let source =
            CapsSet::from_alternatives(Vec::from([video(VideoCodec::Vp8), video(VideoCodec::Vp9)]));
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_template())), PadTemplate::source(source)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp9parse::Vp9Parse;
    use g2g_core::PushOutcome;

    fn i420_grey(w: usize, h: usize) -> Vec<u8> {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut v = alloc::vec![128u8; w * h];
        v.extend(alloc::vec![128u8; cw * ch]);
        v.extend(alloc::vec![128u8; cw * ch]);
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

    // Needs system libvpx to build/run (the `vpx` feature). Mirrors the av1enc
    // round-trip: encode I420, then recover the geometry with vp9parse.
    #[tokio::test]
    async fn encodes_i420_to_vp9_that_vp9parse_reads() {
        let mut enc = VpxEnc::new().with_codec(VideoCodec::Vp9);
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

        assert!(!sink.frames.is_empty(), "the encoder produced VP9 frames");
        assert!(sink.frames.iter().all(|f| !f.is_empty()), "no empty packets");

        let mut parse = Vp9Parse::new();
        parse
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                width: Dim::Range { min: 16, max: 65_535 },
                height: Dim::Range { min: 16, max: 65_535 },
                framerate: Rate::Any,
            })
            .unwrap();
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
        assert_eq!(geometry, Some((64, 64)), "vp9parse recovers the encoded geometry");
    }
}
