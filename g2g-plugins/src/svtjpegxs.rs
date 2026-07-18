//! JPEG XS encode / decode via Intel SVT-JPEG-XS (M605): the ST 2110-22 compressed
//! essence. `SvtJpegXsEnc` takes planar YUV `RawVideo` and emits a
//! `CompressedVideo{JpegXs}` codestream (one independent codestream per frame);
//! `SvtJpegXsDec` is the inverse. Together they close the -22 loop: a plant can
//! encode to JPEG XS, ship it over `st2110jxsrtp` (RFC 9134), and decode on the far
//! side, at a fraction of -20's uncompressed bandwidth with sub-frame latency.
//!
//! Hand-rolled FFI to `libSvtJpegxs` (ISO/IEC 21122), no libavcodec: the `#[repr(C)]`
//! structs transcribe `SvtJpegxs*.h` and carry compile-time size assertions checked
//! against the installed headers (the technique `nvenc` / `nvdec` use). `build.rs`
//! links the library through pkg-config (`SvtJpegxs.pc`), so this whole module is
//! behind the `jpegxs` feature and Linux-only.
//!
//! Formats: planar 4:2:0 / 4:2:2 8-bit and 4:2:2 10-bit (`I420` / `I422` /
//! `I422p10`), 10-bit stored as 16-bit little-endian samples, matching the -20 /
//! -22 broadcast norm the `st2110` elements carry. The encoder runs in codestream
//! packetization mode (RFC 9134 `K=0`); the decoder discovers geometry / format from
//! the first codestream (`svt_jpeg_xs_decoder_init`) and refines its output caps via
//! `CapsChanged`, so a downstream sink sees the true format even though a compressed
//! caps carries no pixel layout.
//!
//! Threading: the SVT api structs hold a `private_ptr` to internal (threaded) state;
//! they are `!Send`. The runner drives an element through `&mut self` only, never
//! concurrently, so `unsafe impl Send` holds on the ownership-transfer grounds
//! documented on `FfmpegVideoDec` / `NvEnc`. The api struct is boxed so its address
//! stays stable across element moves.

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

// ================================================================
// FFI: struct layouts transcribed from SvtJpegxs*.h, size-asserted.
// ================================================================

/// Raw FFI to `libSvtJpegxs`. The `#[repr(C)]` structs mirror the C headers; each
/// carries a compile-time size assertion checked against the installed headers (LP64).
mod ffi {
    use super::c_void;

    pub(super) const API_VER_MAJOR: u64 = 0;
    pub(super) const API_VER_MINOR: u64 = 10;

    // SvtJxsErrorType_t values we act on.
    pub(super) const ERR_NONE: i32 = 0;
    pub(super) const ERR_END_OF_CODESTREAM: i32 = 0x8000_3005u32 as i32;

    // ColourFormat_t values.
    pub(super) const COLOUR_PLANAR_YUV420: i32 = 2;
    pub(super) const COLOUR_PLANAR_YUV422: i32 = 3;

    pub(super) const MAX_COMPONENTS: usize = 4;

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct ImageComponent {
        pub(super) width: u32,
        pub(super) height: u32,
        pub(super) byte_size: u32,
    }

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct ImageConfig {
        pub(super) width: u32,
        pub(super) height: u32,
        pub(super) bit_depth: u8,
        pub(super) format: i32, // ColourFormat_t
        pub(super) components_num: u8,
        pub(super) components: [ImageComponent; MAX_COMPONENTS],
    }
    const _: () = assert!(core::mem::size_of::<ImageConfig>() == 68);

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct ImageBuffer {
        pub(super) data_yuv: [*mut c_void; MAX_COMPONENTS],
        pub(super) stride: [u32; MAX_COMPONENTS],
        pub(super) alloc_size: [u32; MAX_COMPONENTS],
        pub(super) release_ctx_ptr: *mut c_void,
        pub(super) ready_to_release: u8,
    }
    const _: () = assert!(core::mem::size_of::<ImageBuffer>() == 80);

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct BitstreamBuffer {
        pub(super) buffer: *mut u8,
        pub(super) allocation_size: u32,
        pub(super) used_size: u32,
        pub(super) release_ctx_ptr: *mut c_void,
        pub(super) ready_to_release: u8,
        pub(super) last_packet_in_frame: u8,
    }
    const _: () = assert!(core::mem::size_of::<BitstreamBuffer>() == 32);

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct Frame {
        pub(super) image: ImageBuffer,
        pub(super) bitstream: BitstreamBuffer,
        pub(super) user_prv_ctx_ptr: *mut c_void,
    }
    const _: () = assert!(core::mem::size_of::<Frame>() == 120);

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct EncoderApi {
        pub(super) source_width: u32,
        pub(super) source_height: u32,
        pub(super) input_bit_depth: u8,
        pub(super) colour_format: i32, // ColourFormat_t
        pub(super) bpp_numerator: u32,
        pub(super) bpp_denominator: u32,
        pub(super) ndecomp_v: u32,
        pub(super) ndecomp_h: u32,
        pub(super) quantization: u32,
        pub(super) slice_height: u32,
        pub(super) use_cpu_flags: u64,
        pub(super) threads_num: u32,
        pub(super) cpu_profile: u8,
        pub(super) print_bands_info: u32,
        pub(super) coding_signs_handling: u32,
        pub(super) coding_significance: u32,
        pub(super) rate_control_mode: u32,
        pub(super) coding_vertical_prediction_mode: u32,
        pub(super) verbose: u32,
        pub(super) callback_send_data_available: *mut c_void,
        pub(super) callback_send_data_available_context: *mut c_void,
        pub(super) callback_get_data_available: *mut c_void,
        pub(super) callback_get_data_available_context: *mut c_void,
        pub(super) slice_packetization_mode: u8,
        pub(super) private_ptr: *mut c_void,
        pub(super) coding_raw_disable: u8,
        pub(super) cap_compat: u8,
        pub(super) padding: [u8; 62],
    }
    const _: () = assert!(core::mem::size_of::<EncoderApi>() == 192);

    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct DecoderApi {
        pub(super) use_cpu_flags: u64,
        pub(super) verbose: u32,
        pub(super) threads_num: u32,
        pub(super) packetization_mode: u8,
        pub(super) proxy_mode: i32, // proxy_mode_t
        pub(super) callback_send_data_available: *mut c_void,
        pub(super) callback_send_data_available_context: *mut c_void,
        pub(super) callback_get_data_available: *mut c_void,
        pub(super) callback_get_data_available_context: *mut c_void,
        pub(super) private_ptr: *mut c_void,
        pub(super) padding: [u8; 64],
    }
    const _: () = assert!(core::mem::size_of::<DecoderApi>() == 128);

    extern "C" {
        pub(super) fn svt_jpeg_xs_encoder_load_default_parameters(
            major: u64,
            minor: u64,
            enc_api: *mut EncoderApi,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_encoder_get_image_config(
            major: u64,
            minor: u64,
            enc_api: *mut EncoderApi,
            out_image_config: *mut ImageConfig,
            out_bytes_per_frame: *mut u32,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_encoder_init(
            major: u64,
            minor: u64,
            enc_api: *mut EncoderApi,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_encoder_close(enc_api: *mut EncoderApi);
        pub(super) fn svt_jpeg_xs_encoder_send_picture(
            enc_api: *mut EncoderApi,
            enc_input: *mut Frame,
            blocking_flag: u8,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_encoder_get_packet(
            enc_api: *mut EncoderApi,
            enc_output: *mut Frame,
            blocking_flag: u8,
        ) -> i32;

        pub(super) fn svt_jpeg_xs_decoder_init(
            major: u64,
            minor: u64,
            dec_api: *mut DecoderApi,
            bitstream_buf: *const u8,
            codestream_size: usize,
            out_image_config: *mut ImageConfig,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_decoder_close(dec_api: *mut DecoderApi);
        pub(super) fn svt_jpeg_xs_decoder_send_frame(
            dec_api: *mut DecoderApi,
            dec_input: *mut Frame,
            blocking_flag: u8,
        ) -> i32;
        pub(super) fn svt_jpeg_xs_decoder_get_frame(
            dec_api: *mut DecoderApi,
            dec_output: *mut Frame,
            blocking_flag: u8,
        ) -> i32;
    }
}

/// Map a planar `RawVideoFormat` to (SVT colour format, bit depth), or `None` if it
/// is not one of the JPEG XS input formats we support.
fn format_to_svt(f: RawVideoFormat) -> Option<(i32, u8)> {
    match f {
        RawVideoFormat::I420 => Some((ffi::COLOUR_PLANAR_YUV420, 8)),
        RawVideoFormat::I422 => Some((ffi::COLOUR_PLANAR_YUV422, 8)),
        RawVideoFormat::I422p10 => Some((ffi::COLOUR_PLANAR_YUV422, 10)),
        _ => None,
    }
}

/// The inverse: an SVT (colour format, bit depth) to a `RawVideoFormat`, or `None`.
fn svt_to_format(colour: i32, bit_depth: u8) -> Option<RawVideoFormat> {
    match (colour, bit_depth) {
        (ffi::COLOUR_PLANAR_YUV420, 8) => Some(RawVideoFormat::I420),
        (ffi::COLOUR_PLANAR_YUV422, 8) => Some(RawVideoFormat::I422),
        (ffi::COLOUR_PLANAR_YUV422, 10) => Some(RawVideoFormat::I422p10),
        _ => None,
    }
}

/// Extract (format, width, height) from an absolute `Caps::RawVideo` with a
/// JPEG-XS-mappable planar format.
fn raw_geometry(caps: &Caps) -> Result<(RawVideoFormat, u32, u32), G2gError> {
    match caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } if format_to_svt(*format).is_some() => Ok((*format, *w, *h)),
        _ => Err(G2gError::CapsMismatch),
    }
}

// ================================================================
// Encoder
// ================================================================

/// Encodes planar YUV `RawVideo` into a `CompressedVideo{JpegXs}` codestream via
/// SVT-JPEG-XS (one codestream per frame, RFC 9134 codestream packetization mode).
pub struct SvtJpegXsEnc {
    width: u32,
    height: u32,
    format: RawVideoFormat,
    /// Target bits per pixel as a fraction (`bpp_numerator` / `bpp_denominator`).
    bpp_num: u32,
    bpp_den: u32,
    /// Boxed so the C `private_ptr` inside it keeps a stable address across moves.
    enc: Option<Box<ffi::EncoderApi>>,
    /// Per-component byte sizes / plane widths from `get_image_config`.
    components: [(u32, u32); ffi::MAX_COMPONENTS], // (stride_samples, byte_size)
    components_num: usize,
    /// Output codestream scratch, sized to the encoder's max bytes per frame.
    out_buf: Vec<u8>,
    caps_sent: bool,
}

// SAFETY: `EncoderApi` holds a `private_ptr` to threaded SVT state and is `!Send`.
// The runner moves the element between worker tasks but drives it through `&mut self`
// only (never concurrently), so the handle is owned and moved, never aliased, the
// same contract `FfmpegVideoDec` / `NvEnc` uphold.
unsafe impl Send for SvtJpegXsEnc {}

impl core::fmt::Debug for SvtJpegXsEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SvtJpegXsEnc")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("bpp", &(self.bpp_num, self.bpp_den))
            .finish()
    }
}

impl Default for SvtJpegXsEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl SvtJpegXsEnc {
    /// A JPEG XS encoder targeting 6 bpp (a visually lossless mezzanine default).
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            format: RawVideoFormat::I422p10,
            bpp_num: 6,
            bpp_den: 1,
            enc: None,
            components: [(0, 0); ffi::MAX_COMPONENTS],
            components_num: 0,
            out_buf: Vec::new(),
            caps_sent: false,
        }
    }
}

impl Drop for SvtJpegXsEnc {
    fn drop(&mut self) {
        if let Some(enc) = self.enc.as_mut() {
            // SAFETY: `enc` was initialized by `svt_jpeg_xs_encoder_init`; close frees
            // its internal state exactly once (the element owns the only handle).
            unsafe { ffi::svt_jpeg_xs_encoder_close(&mut **enc) };
        }
    }
}

impl AsyncElement for SvtJpegXsEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let (_, w, h) = raw_geometry(upstream_caps)?;
        Ok(Caps::CompressedVideo {
            codec: VideoCodec::JpegXs,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, width, height) = raw_geometry(absolute_caps)?;
        let (colour, depth) = format_to_svt(format).ok_or(G2gError::CapsMismatch)?;
        self.format = format;
        self.width = width;
        self.height = height;

        // SAFETY: `api` is a correctly-sized, zeroed `EncoderApi`; the SVT calls read
        // / write only within it and the buffers we point at. Size is asserted above.
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); load_default_parameters then fills it.
        let zeroed: ffi::EncoderApi = unsafe { core::mem::zeroed() };
        let mut api: Box<ffi::EncoderApi> = Box::new(zeroed);
        // SAFETY: `api` is a correctly-sized `EncoderApi` (size asserted); the call
        // writes only within it.
        let rc = unsafe {
            ffi::svt_jpeg_xs_encoder_load_default_parameters(
                ffi::API_VER_MAJOR,
                ffi::API_VER_MINOR,
                &mut *api,
            )
        };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        api.source_width = width;
        api.source_height = height;
        api.input_bit_depth = depth;
        api.colour_format = colour;
        api.bpp_numerator = self.bpp_num;
        api.bpp_denominator = self.bpp_den.max(1);
        api.slice_packetization_mode = 0; // codestream mode (RFC 9134 K=0)
        api.verbose = 0; // VERBOSE_NONE

        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut cfg: ffi::ImageConfig = unsafe { core::mem::zeroed() };
        let mut bytes_per_frame: u32 = 0;
        // SAFETY: `api` is configured; `cfg` / `bytes_per_frame` are valid out params.
        let rc = unsafe {
            ffi::svt_jpeg_xs_encoder_get_image_config(
                ffi::API_VER_MAJOR,
                ffi::API_VER_MINOR,
                &mut *api,
                &mut cfg,
                &mut bytes_per_frame,
            )
        };
        if rc != ffi::ERR_NONE || bytes_per_frame == 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.components_num = usize::from(cfg.components_num).min(ffi::MAX_COMPONENTS);
        for i in 0..self.components_num {
            self.components[i] = (cfg.components[i].width, cfg.components[i].byte_size);
        }
        self.out_buf = vec![0u8; bytes_per_frame as usize];

        // SAFETY: `api` is fully configured; init allocates the encoder's state.
        let rc = unsafe {
            ffi::svt_jpeg_xs_encoder_init(ffi::API_VER_MAJOR, ffi::API_VER_MINOR, &mut *api)
        };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.enc = Some(api);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SVT JPEG XS encoder",
            "Codec/Encoder/Video",
            "Encodes planar YUV to a JPEG XS codestream (ISO/IEC 21122) via SVT-JPEG-XS",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "bpp",
            PropKind::Fraction,
            "Target bits per pixel (num/den); higher is higher quality / bitrate",
        )
        .with_default("6/1")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bpp" => {
                let (n, d) = value.as_fraction().ok_or(PropError::Type)?;
                if n <= 0 || d <= 0 {
                    return Err(PropError::Value);
                }
                self.bpp_num = n as u32;
                self.bpp_den = d as u32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bpp" => Some(PropValue::Fraction(
                self.bpp_num as i32,
                self.bpp_den as i32,
            )),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(Caps::CompressedVideo {
                            codec: VideoCodec::JpegXs,
                            width: Dim::Fixed(self.width),
                            height: Dim::Fixed(self.height),
                            framerate: Rate::Any,
                        }))
                        .await?;
                        self.caps_sent = true;
                    }
                    let codestream = self.encode_one(slice.as_slice())?;
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            codestream.into_boxed_slice(),
                        )),
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: frame.meta,
                    };
                    out.push(PipelinePacket::DataFrame(out_frame)).await
                }
                other => out.push(other).await,
            }
            .map(|_| ())
        })
    }
}

impl SvtJpegXsEnc {
    /// Encode one planar-YUV frame to a JPEG XS codestream (a fresh `Vec`). The input
    /// slice must hold the tightly-packed planes for the configured geometry.
    fn encode_one(&mut self, input: &[u8]) -> Result<Vec<u8>, G2gError> {
        let enc = self.enc.as_mut().ok_or(G2gError::NotConfigured)?;

        // Point the SVT image buffer at each plane in the contiguous input.
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut fr: ffi::Frame = unsafe { core::mem::zeroed() };
        let mut offset = 0usize;
        for i in 0..self.components_num {
            let (stride_samples, byte_size) = self.components[i];
            let end = offset
                .checked_add(byte_size as usize)
                .ok_or(G2gError::CapsMismatch)?;
            if end > input.len() {
                return Err(G2gError::CapsMismatch);
            }
            // SAFETY: the range [offset, end) is within `input` (checked above); the
            // pointer is only read by the encoder, which does not retain it past the
            // blocking `send_picture` call.
            fr.image.data_yuv[i] = unsafe { input.as_ptr().add(offset) as *mut c_void };
            fr.image.stride[i] = stride_samples;
            fr.image.alloc_size[i] = byte_size;
            offset = end;
        }
        fr.bitstream.buffer = self.out_buf.as_mut_ptr();
        fr.bitstream.allocation_size = self.out_buf.len() as u32;

        // SAFETY: `enc` is initialized; `fr` points at valid input planes and an
        // output buffer sized to the encoder's max bytes per frame. Blocking, so the
        // library consumes `fr` before returning.
        let rc = unsafe { ffi::svt_jpeg_xs_encoder_send_picture(&mut **enc, &mut fr, 1) };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut got: ffi::Frame = unsafe { core::mem::zeroed() };
        // SAFETY: blocking get returns this frame's packet (codestream mode: one
        // packet per picture); `got` is a valid out param.
        let rc = unsafe { ffi::svt_jpeg_xs_encoder_get_packet(&mut **enc, &mut got, 1) };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let len = got.bitstream.used_size as usize;
        let ptr = got.bitstream.buffer;
        if ptr.is_null() || len == 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: the library filled `len` valid bytes at `ptr` (its output buffer).
        let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
        Ok(bytes.to_vec())
    }
}

impl PadTemplates for SvtJpegXsEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let alts = [
            RawVideoFormat::I420,
            RawVideoFormat::I422,
            RawVideoFormat::I422p10,
        ]
        .map(|format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .to_vec();
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(alts)),
            PadTemplate::source(CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            })),
        ])
    }
}

// ================================================================
// Decoder
// ================================================================

/// Decodes a `CompressedVideo{JpegXs}` codestream into planar YUV `RawVideo` via
/// SVT-JPEG-XS. Geometry / format are discovered from the first codestream.
pub struct SvtJpegXsDec {
    /// Boxed for a stable C `private_ptr` address; `None` until the first frame.
    dec: Option<Box<ffi::DecoderApi>>,
    out_format: RawVideoFormat,
    width: u32,
    height: u32,
    /// (stride_samples, byte_size) per component, from decoder init.
    components: [(u32, u32); ffi::MAX_COMPONENTS],
    components_num: usize,
    /// Total output frame size (sum of component byte sizes).
    frame_size: usize,
    caps_sent: bool,
}

// SAFETY: same ownership-transfer contract as `SvtJpegXsEnc`: the `DecoderApi`'s
// `private_ptr` is driven through `&mut self` only, never shared across threads.
unsafe impl Send for SvtJpegXsDec {}

impl core::fmt::Debug for SvtJpegXsDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SvtJpegXsDec")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("out_format", &self.out_format)
            .finish()
    }
}

impl Default for SvtJpegXsDec {
    fn default() -> Self {
        Self::new()
    }
}

impl SvtJpegXsDec {
    /// A JPEG XS decoder. Output caps are discovered from the first codestream.
    pub fn new() -> Self {
        Self {
            dec: None,
            out_format: RawVideoFormat::I422p10,
            width: 0,
            height: 0,
            components: [(0, 0); ffi::MAX_COMPONENTS],
            components_num: 0,
            frame_size: 0,
            caps_sent: false,
        }
    }
}

impl Drop for SvtJpegXsDec {
    fn drop(&mut self) {
        if let Some(dec) = self.dec.as_mut() {
            // SAFETY: `dec` was initialized by `svt_jpeg_xs_decoder_init`; close frees
            // its state exactly once (the element owns the only handle).
            unsafe { ffi::svt_jpeg_xs_decoder_close(&mut **dec) };
        }
    }
}

impl AsyncElement for SvtJpegXsDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width,
                height,
                ..
            } => {
                // Output format is unknown until the first codestream; advertise the
                // broadcast-norm default and refine via CapsChanged at first frame.
                Ok(Caps::RawVideo {
                    format: RawVideoFormat::I422p10,
                    width: width.clone(),
                    height: height.clone(),
                    framerate: Rate::Any,
                })
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                ..
            } => Ok(ConfigureOutcome::Accepted),
            // A pass-through no-op configure with the downstream (raw) caps also lands
            // here after the runner cascades the source-side format.
            Caps::RawVideo { .. } => Ok(ConfigureOutcome::Accepted),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SVT JPEG XS decoder",
            "Codec/Decoder/Video",
            "Decodes a JPEG XS codestream (ISO/IEC 21122) to planar YUV via SVT-JPEG-XS",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let pixels = self.decode_one(slice.as_slice())?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(Caps::RawVideo {
                            format: self.out_format,
                            width: Dim::Fixed(self.width),
                            height: Dim::Fixed(self.height),
                            framerate: Rate::Any,
                        }))
                        .await?;
                        self.caps_sent = true;
                    }
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            pixels.into_boxed_slice(),
                        )),
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: frame.meta,
                    };
                    out.push(PipelinePacket::DataFrame(out_frame))
                        .await
                        .map(|_| ())
                }
                // A compressed CapsChanged upstream carries no pixel layout; swallow it
                // (our own output CapsChanged is emitted from the first decoded frame).
                PipelinePacket::CapsChanged(_) => Ok(()),
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

impl SvtJpegXsDec {
    /// Ensure the decoder is initialized from `codestream` (learning geometry /
    /// format), then decode one frame into a fresh contiguous planar buffer.
    fn decode_one(&mut self, codestream: &[u8]) -> Result<Vec<u8>, G2gError> {
        if self.dec.is_none() {
            self.init_from(codestream)?;
        }
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;

        let mut pixels = vec![0u8; self.frame_size];
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut fr: ffi::Frame = unsafe { core::mem::zeroed() };
        // Input bitstream (the library only reads it during the blocking send).
        fr.bitstream.buffer = codestream.as_ptr() as *mut u8;
        fr.bitstream.allocation_size = codestream.len() as u32;
        fr.bitstream.used_size = codestream.len() as u32;
        // Output planes into the contiguous buffer.
        let mut offset = 0usize;
        for i in 0..self.components_num {
            let (stride_samples, byte_size) = self.components[i];
            // SAFETY: offsets sum to `frame_size` == `pixels.len()`, so each plane
            // pointer stays within the allocation.
            fr.image.data_yuv[i] = unsafe { pixels.as_mut_ptr().add(offset) as *mut c_void };
            fr.image.stride[i] = stride_samples;
            fr.image.alloc_size[i] = byte_size;
            offset += byte_size as usize;
        }

        // SAFETY: `dec` is initialized; `fr` points at a valid input codestream and
        // output planes sized from the decoder's own config. Blocking calls.
        let rc = unsafe { ffi::svt_jpeg_xs_decoder_send_frame(&mut **dec, &mut fr, 1) };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut got: ffi::Frame = unsafe { core::mem::zeroed() };
        // SAFETY: blocking get returns the decoded frame into the planes we set on
        // `fr`; `got` is a valid out param. End-of-codestream is a non-fatal status.
        let rc = unsafe { ffi::svt_jpeg_xs_decoder_get_frame(&mut **dec, &mut got, 1) };
        if rc != ffi::ERR_NONE && rc != ffi::ERR_END_OF_CODESTREAM {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        Ok(pixels)
    }

    /// Initialize the decoder from the first codestream, learning geometry / format
    /// and the per-component plane layout.
    fn init_from(&mut self, codestream: &[u8]) -> Result<(), G2gError> {
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut api: Box<ffi::DecoderApi> = Box::new(unsafe { core::mem::zeroed() });
        api.packetization_mode = 0; // frame-based
        api.verbose = 0;
        // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
        let mut cfg: ffi::ImageConfig = unsafe { core::mem::zeroed() };
        // SAFETY: `api` is zeroed then set; `codestream` is a valid readable slice;
        // `cfg` is a valid out param. init parses the codestream header only.
        let rc = unsafe {
            ffi::svt_jpeg_xs_decoder_init(
                ffi::API_VER_MAJOR,
                ffi::API_VER_MINOR,
                &mut *api,
                codestream.as_ptr(),
                codestream.len(),
                &mut cfg,
            )
        };
        if rc != ffi::ERR_NONE {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.out_format = svt_to_format(cfg.format, cfg.bit_depth).ok_or(G2gError::CapsMismatch)?;
        self.width = cfg.width;
        self.height = cfg.height;
        self.components_num = usize::from(cfg.components_num).min(ffi::MAX_COMPONENTS);
        let mut total = 0usize;
        for i in 0..self.components_num {
            self.components[i] = (cfg.components[i].width, cfg.components[i].byte_size);
            total += cfg.components[i].byte_size as usize;
        }
        self.frame_size = total;
        self.dec = Some(api);
        Ok(())
    }
}

impl PadTemplates for SvtJpegXsDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let alts = [
            RawVideoFormat::I420,
            RawVideoFormat::I422,
            RawVideoFormat::I422p10,
        ]
        .map(|format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
        .to_vec();
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            })),
            PadTemplate::source(CapsSet::from_alternatives(alts)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::block_on;
    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct Capture {
        frames: Vec<Vec<u8>>,
        caps: Vec<Caps>,
    }
    impl OutputSink for Capture {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
                        }
                    }
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: Default::default(),
            sequence: 0,
            meta: Default::default(),
        })
    }

    /// A smooth planar I422p10 test frame (16-bit LE samples, low 10 bits): a
    /// horizontal ramp so JPEG XS compresses cleanly and the round trip is close.
    fn i422p10_ramp(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 4]; // Y (w*h) + Cb (w/2*h) + Cr (w/2*h), 2 bytes
        let mut write = |plane_off: usize, pw: usize, ph: usize, bias: u16| {
            for y in 0..ph {
                for x in 0..pw {
                    let v = (((x * 1023) / pw.max(1)) as u16).wrapping_add(bias) & 0x03FF;
                    let off = plane_off + (y * pw + x) * 2;
                    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        };
        let y_bytes = w * h * 2;
        let c_bytes = (w / 2) * h * 2;
        write(0, w, h, 0);
        write(y_bytes, w / 2, h, 128);
        write(y_bytes + c_bytes, w / 2, h, 256);
        buf
    }

    #[test]
    fn encode_then_decode_round_trips_i422p10() {
        let (w, h) = (128usize, 64usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I422p10,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(60 << 16),
        };
        let input = i422p10_ramp(w, h);

        // Encode at a high bpp so the round trip is near-lossless.
        let mut enc = SvtJpegXsEnc::new();
        enc.set_property("bpp", PropValue::Fraction(10, 1)).unwrap();
        enc.configure_pipeline(&caps).expect("encoder configures");
        let mut enc_out = Capture::default();
        block_on(enc.process(data_frame(input.clone()), &mut enc_out)).expect("encode");
        assert_eq!(enc_out.frames.len(), 1, "one codestream out");
        let codestream = enc_out.frames.remove(0);
        assert!(!codestream.is_empty(), "non-empty JPEG XS codestream");
        assert!(
            codestream.len() < input.len(),
            "codestream is smaller than raw"
        );
        assert_eq!(
            enc_out.caps[0],
            Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width: Dim::Fixed(w as u32),
                height: Dim::Fixed(h as u32),
                framerate: Rate::Any,
            }
        );

        // Decode; geometry / format discovered from the codestream.
        let mut dec = SvtJpegXsDec::new();
        let mut dec_out = Capture::default();
        block_on(dec.process(data_frame(codestream), &mut dec_out)).expect("decode");
        assert_eq!(dec_out.frames.len(), 1);
        let decoded = &dec_out.frames[0];
        assert_eq!(decoded.len(), input.len(), "same-size planar buffer back");
        assert_eq!(
            dec_out.caps[0],
            Caps::RawVideo {
                format: RawVideoFormat::I422p10,
                width: Dim::Fixed(w as u32),
                height: Dim::Fixed(h as u32),
                framerate: Rate::Any,
            }
        );

        // Near-lossless at 10 bpp: mean absolute 10-bit sample error is small.
        let mut sum_err = 0u64;
        let n = input.len() / 2;
        for i in 0..n {
            let a = u16::from_le_bytes([input[2 * i], input[2 * i + 1]]) & 0x03FF;
            let b = u16::from_le_bytes([decoded[2 * i], decoded[2 * i + 1]]) & 0x03FF;
            sum_err += u64::from(a.abs_diff(b));
        }
        let mean_err = sum_err as f64 / n as f64;
        assert!(
            mean_err < 8.0,
            "mean 10-bit error {mean_err} too high (round trip broken)"
        );
    }
}
