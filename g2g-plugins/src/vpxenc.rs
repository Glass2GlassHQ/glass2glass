//! VP8 / VP9 software encode element (VpxEnc, `vpx` feature): `RawVideo{I420}` in,
//! `CompressedVideo{Vp8|Vp9}` out, via libvpx with FFI that bindgen generates in
//! build.rs.
//!
//! The GStreamer `vp8enc` / `vp9enc` analog. Unlike the pure-Rust [`crate::av1enc`]
//! (rav1e), this links the system `libvpx` and runs bindgen (clang) at build, so it
//! is gated behind the `vpx` feature and left out of pure-Rust / no_std builds.
//! Building needs libvpx installed (Windows: vcpkg `libvpx`; Debian/Ubuntu:
//! `libvpx-dev`; Fedora: `libvpx-devel`) plus libclang for bindgen.
//!
//! Scope (v1): 8-bit I420, geometry fixed at configure, single-pass
//! target-bitrate rate control. The codec (VP8/VP9) and bitrate are
//! builder-configurable.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_int;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

#[allow(non_camel_case_types, dead_code, unreachable_pub)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/vpx_ffi.rs"));
}

/// Default target bitrate (kbps) when none is set.
const DEFAULT_BITRATE_KBPS: u32 = 1024;
/// Encode timestamps in milliseconds (timebase 1/1000): a frame's PTS in ns maps
/// to `pts_ns / 1_000_000` going in, and a packet's back to `pts * 1_000_000`.
const TIMEBASE_DEN: i32 = 1000;

/// Encodes raw I420 video into a VP8 or VP9 elementary stream via libvpx.
///
/// # Example
///
/// ```no_run
/// use g2g_core::VideoCodec;
/// use g2g_plugins::vpxenc::VpxEnc;
///
/// let encoder = VpxEnc::new()
///     .with_codec(VideoCodec::Vp8)
///     .with_bitrate_kbps(2_048);
/// ```
pub struct VpxEnc {
    codec: VideoCodec,
    bitrate_kbps: u32,
    width: u32,
    height: u32,
    framerate: Rate,
    enc: Option<VpxContext>,
    emitted: u64,
    caps_sent: bool,
    configured: bool,
}

impl core::fmt::Debug for VpxEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The libvpx context is not Debug, report the configuration instead.
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn build_encoder(&mut self) -> Result<(), G2gError> {
        self.enc = Some(VpxContext::new(
            self.codec,
            self.width,
            self.height,
            self.bitrate_kbps,
        )?);
        Ok(())
    }

    /// Encode one I420 access unit, returning the ready packets as `(data, pts_ns)`.
    fn encode(&mut self, i420: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let expected = crate::pixel::frame_byte_size(RawVideoFormat::I420, self.width, self.height);
        if i420.len() < expected {
            return Err(G2gError::CapsMismatch);
        }
        let enc = self.enc.as_mut().ok_or(G2gError::NotConfigured)?;
        enc.encode(i420, (pts_ns / 1_000_000) as i64)?;
        Ok(enc.drain_packets())
    }

    /// Flush the encoder at EOS and return the remaining packets, consuming the
    /// libvpx context (a fresh `configure_pipeline` would rebuild it).
    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let Some(mut enc) = self.enc.take() else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        loop {
            enc.encode_end_of_stream()?;
            let packets = enc.drain_packets();
            if packets.is_empty() {
                return Ok(out);
            }
            out.extend(packets);
        }
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
        // Both signals are honored by REBUILDING the encoder (M730): a fresh
        // libvpx context starts on a keyframe, and takes the new rate with it. The
        // rebuild is cheap for the software codecs; bitrate is hysteresis-gated
        // 20% like `Av1Enc` so BWE noise cannot thrash it.
        let mut rebuild = feedback.force_keyframe;
        if let Some(bps) = feedback.bitrate_bps {
            let target_kbps = (bps / 1000).max(1);
            let current = self.bitrate_kbps.max(1);
            if (target_kbps as i64 - current as i64).unsigned_abs() * 5 >= current as u64 {
                self.bitrate_kbps = target_kbps;
                rebuild = true;
            }
        }
        if rebuild && self.enc.is_some() {
            self.build_encoder()?;
        }
        Ok(())
    }
}

// An initialized libvpx encoder context, destroyed on drop. Boxed because libvpx
// does not document that an initialized context may move.
struct VpxContext {
    context: Box<ffi::vpx_codec_ctx_t>,
    width: u32,
    height: u32,
}

// SAFETY: libvpx keeps no thread-local or thread-bound state for a context, and
// every call goes through `&mut self`, so moving it between threads cannot race.
// The `multi-thread` runner needs `Send` to move the element between workers.
unsafe impl Send for VpxContext {}

// The libvpx encoder interface for `codec`, or `CapsMismatch` for a codec this
// element does not encode.
fn codec_interface(codec: VideoCodec) -> Result<*const ffi::vpx_codec_iface, G2gError> {
    // SAFETY: both functions take no arguments and return a pointer to a static
    // interface table inside libvpx.
    let interface = unsafe {
        match codec {
            VideoCodec::Vp8 => ffi::vpx_codec_vp8_cx(),
            VideoCodec::Vp9 => ffi::vpx_codec_vp9_cx(),
            _ => return Err(G2gError::CapsMismatch),
        }
    };
    if interface.is_null() {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(interface)
}

fn check_vpx(result: ffi::vpx_codec_err_t) -> Result<(), G2gError> {
    if result == ffi::VPX_CODEC_OK {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Other))
    }
}

// VP9 speed preset passed as VP8E_SET_CPUUSED, higher is faster.
const VP9_CPU_USED: c_int = 6;
const ENCODER_THREADS: u32 = 8;

impl VpxContext {
    fn new(
        codec: VideoCodec,
        width: u32,
        height: u32,
        bitrate_kbps: u32,
    ) -> Result<Self, G2gError> {
        let interface = codec_interface(codec)?;
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: `vpx_codec_enc_cfg_t` is a plain C struct of integers, rationals
        // and pointers, for which all-zero is a valid value.
        let mut config: ffi::vpx_codec_enc_cfg_t = unsafe { core::mem::zeroed() };
        // SAFETY: `interface` is a non-null libvpx encoder interface and `config`
        // is a live, writable config struct.
        check_vpx(unsafe { ffi::vpx_codec_enc_config_default(interface, &mut config, 0) })?;
        config.g_w = width;
        config.g_h = height;
        config.g_timebase.num = 1;
        config.g_timebase.den = TIMEBASE_DEN;
        config.rc_target_bitrate = bitrate_kbps;
        config.g_threads = ENCODER_THREADS;
        config.g_error_resilient = ffi::VPX_ERROR_RESILIENT_DEFAULT;

        // SAFETY: `vpx_codec_ctx_t` is a plain C struct, all-zero is the
        // uninitialized state `vpx_codec_enc_init_ver` expects.
        let mut context: Box<ffi::vpx_codec_ctx_t> = Box::new(unsafe { core::mem::zeroed() });
        // SAFETY: `context` and `config` are live, `interface` is non-null, and the
        // ABI version comes from the same headers the bindings were generated from.
        // libvpx copies `config` into the context, so it may go out of scope.
        check_vpx(unsafe {
            ffi::vpx_codec_enc_init_ver(
                &mut *context,
                interface,
                &config,
                0,
                ffi::VPX_ENCODER_ABI_VERSION as c_int,
            )
        })?;
        let mut encoder = Self {
            context,
            width,
            height,
        };
        if codec == VideoCodec::Vp9 {
            encoder.control(ffi::VP8E_SET_CPUUSED, VP9_CPU_USED)?;
            encoder.control(ffi::VP9E_SET_ROW_MT, 1)?;
        }
        Ok(encoder)
    }

    fn control(
        &mut self,
        control_id: ffi::vp8e_enc_control_id,
        value: c_int,
    ) -> Result<(), G2gError> {
        // SAFETY: the context is initialized, and both controls used here take a
        // single `int` argument.
        check_vpx(unsafe {
            ffi::vpx_codec_control_(&mut *self.context, control_id as c_int, value)
        })
    }

    // Queues one I420 frame. The caller has checked `i420` holds a full frame.
    fn encode(&mut self, i420: &[u8], pts: i64) -> Result<(), G2gError> {
        // SAFETY: `vpx_image_t` is a plain C struct, all-zero is valid before
        // `vpx_img_wrap` fills it.
        let mut image: ffi::vpx_image_t = unsafe { core::mem::zeroed() };
        // SAFETY: `i420` holds `width * height * 3 / 2` bytes and outlives the
        // `vpx_codec_encode` call below. libvpx only reads the wrapped pixels, so
        // the `*mut` cast never leads to a write.
        let wrapped = unsafe {
            ffi::vpx_img_wrap(
                &mut image,
                ffi::VPX_IMG_FMT_I420,
                self.width,
                self.height,
                1,
                i420.as_ptr().cast_mut(),
            )
        };
        if wrapped.is_null() {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: the context is initialized and `image` wraps live pixel data.
        check_vpx(unsafe {
            ffi::vpx_codec_encode(
                &mut *self.context,
                &image,
                pts,
                1,
                0,
                ffi::VPX_DL_REALTIME as core::ffi::c_ulong,
            )
        })
    }

    // A null image asks libvpx to emit the frames it is still holding.
    fn encode_end_of_stream(&mut self) -> Result<(), G2gError> {
        // SAFETY: the context is initialized, and a null image is the documented
        // flush request.
        check_vpx(unsafe {
            ffi::vpx_codec_encode(
                &mut *self.context,
                core::ptr::null(),
                -1,
                1,
                0,
                ffi::VPX_DL_REALTIME as core::ffi::c_ulong,
            )
        })
    }

    // Copies out every compressed frame packet as `(data, pts_ns)`.
    fn drain_packets(&mut self) -> Vec<(Vec<u8>, u64)> {
        let mut iterator: ffi::vpx_codec_iter_t = core::ptr::null();
        let mut out = Vec::new();
        loop {
            // SAFETY: the context is initialized, and `iterator` is null on the first
            // call and otherwise the value libvpx wrote back.
            let packet = unsafe { ffi::vpx_codec_get_cx_data(&mut *self.context, &mut iterator) };
            // SAFETY: a non-null packet stays valid until the next call on the context.
            let Some(packet) = (unsafe { packet.as_ref() }) else {
                return out;
            };
            if packet.kind != ffi::VPX_CODEC_CX_FRAME_PKT {
                continue;
            }
            // SAFETY: `kind` says the `frame` member of the union is the active one.
            let frame = unsafe { &packet.data.frame };
            // SAFETY: libvpx guarantees `buf` points at `sz` bytes for a frame packet.
            let data = unsafe { core::slice::from_raw_parts(frame.buf.cast::<u8>(), frame.sz) };
            out.push((data.to_vec(), (frame.pts as u64) * 1_000_000));
        }
    }
}

impl Drop for VpxContext {
    fn drop(&mut self) {
        // SAFETY: the context was initialized by `vpx_codec_enc_init_ver` and is
        // destroyed only here.
        unsafe { ffi::vpx_codec_destroy(&mut *self.context) };
    }
}

impl AsyncElement for VpxEnc {
    // M759: a re-encode to a compressed codec drops pixel-derived meta
    // (AnalyticsMeta); opaque side-data (BlobMeta) rides through.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    // Both signals are honored by an encoder rebuild (see `emit`), so they
    // stop here rather than relaying past the encoder.
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

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::CompressedVideo {
                codec,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::I420,
            width,
            height,
            framerate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        codec_interface(self.codec)?; // reject a non-VP8/VP9 selection before building libvpx
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

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("codec", PropKind::Str, "output codec: vp8 | vp9")
                .with_default("vp9"),
            PropertySpec::new("bitrate", PropKind::Uint, "target bitrate, bits/second")
                .with_default("1024000"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "codec" => {
                self.codec = match value.as_str().ok_or(PropError::Type)? {
                    "vp8" | "VP8" => VideoCodec::Vp8,
                    "vp9" | "VP9" => VideoCodec::Vp9,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            // g2g exposes bitrate in bits/second (the project convention); libvpx
            // rate control is kbps, so fold down here.
            "bitrate" => {
                let bps = value.as_uint().ok_or(PropError::Type)?;
                self.bitrate_kbps = ((bps / 1000) as u32).max(1);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "codec" => Some(PropValue::Str(
                match self.codec {
                    VideoCodec::Vp8 => "vp8",
                    _ => "vp9",
                }
                .into(),
            )),
            "bitrate" => Some(PropValue::Uint(self.bitrate_kbps as u64 * 1000)),
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let packets = self.encode(slice, frame.timing.pts_ns)?;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let source =
            CapsSet::from_alternatives(Vec::from([video(VideoCodec::Vp8), video(VideoCodec::Vp9)]));
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(source),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp9parse::Vp9Parse;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{FrameTiming, PushOutcome};

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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// VP8 frame-tag keyframe bit: byte 0 bit 0 is 0 for a key frame.
    fn vp8_is_keyframe(data: &[u8]) -> bool {
        data.first().is_some_and(|b| b & 0x1 == 0)
    }

    /// Sink that reports one `ForceKeyframe` after its second frame, then a low
    /// bitrate target (the WebRTC PLI + BWE shape).
    struct FeedbackSink {
        frames: alloc::vec::Vec<alloc::vec::Vec<u8>>,
        kf_sent: bool,
        bps: u32,
    }
    impl OutputSink for FeedbackSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            use core::task::Poll;
            if let PipelinePacket::DataFrame(f) = packet.take().expect("poll_push without a packet")
            {
                if let Some(sl) = f.domain.as_system_slice() {
                    self.frames.push(sl.to_vec());
                }
            }
            if self.frames.len() >= 2 && !core::mem::replace(&mut self.kf_sent, true) {
                return Poll::Ready(Ok(PushOutcome::Reconfigure(
                    g2g_core::Reconfigure::ForceKeyframe,
                )));
            }
            if self.bps > 0 {
                return Poll::Ready(Ok(PushOutcome::Bitrate(self.bps)));
            }
            Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// M730: a downstream PLI rebuilds the encoder, so the next emitted VP8
    /// frame is a key frame; a far-off bitrate target rebuilds at the new rate.
    #[tokio::test]
    async fn force_keyframe_and_retarget_rebuild_the_encoder() {
        let mut enc = VpxEnc::new()
            .with_codec(VideoCodec::Vp8)
            .with_bitrate_kbps(2_000);
        enc.configure_pipeline(&i420_caps(64, 64)).unwrap();
        let mut sink = FeedbackSink {
            frames: alloc::vec::Vec::new(),
            kf_sent: false,
            bps: 100_000,
        };
        for seq in 0..8u64 {
            // Vary the luma so inter frames are not degenerate.
            let mut buf = i420_grey(64, 64);
            buf[0] = seq as u8;
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                FrameTiming {
                    pts_ns: seq * 33_000_000,
                    ..FrameTiming::default()
                },
                seq,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        // Frame 0 is the natural keyframe; the PLI after frame 2 forces the
        // next emitted frame to be a keyframe again (encoder rebuilt).
        assert!(vp8_is_keyframe(&sink.frames[0]));
        let later_kf = sink.frames[2..].iter().any(|f| vp8_is_keyframe(f));
        assert!(later_kf, "the PLI produced a mid-stream keyframe");
        // The far-off bitrate target rebuilt at the new rate.
        assert_eq!(enc.bitrate_kbps, 100, "retargeted to 100 kb/s");
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

        assert!(!sink.frames.is_empty(), "the encoder produced VP9 frames");
        assert!(
            sink.frames.iter().all(|f| !f.is_empty()),
            "no empty packets"
        );

        let mut parse = Vp9Parse::new();
        parse
            .configure_pipeline(&Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                width: Dim::Range {
                    min: 16,
                    max: 65_535,
                },
                height: Dim::Range {
                    min: 16,
                    max: 65_535,
                },
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
            .unwrap();
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
            "vp9parse recovers the encoded geometry"
        );
    }
}
