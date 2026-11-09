//! JPEG XS encode / decode via Intel SVT-JPEG-XS: the ST 2110-22 compressed
//! essence. `SvtJpegXsEnc` takes planar YUV `RawVideo` and emits a
//! `CompressedVideo{JpegXs}` codestream (one independent codestream per frame, or
//! one per field for interleaved video);
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
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, Segment, VideoCodec,
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
        pub(super) fn svt_jpeg_xs_decoder_get_single_frame_size(
            bitstream_buf: *const u8,
            bitstream_buf_size: usize,
            out_image_config: *mut ImageConfig,
            frame_size: *mut u32,
            fast_search: u32,
        ) -> i32;
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

fn raw_fields_per_frame(caps: &Caps) -> u32 {
    match caps {
        Caps::RawVideo {
            interlace: g2g_core::Interlace::Interleaved,
            ..
        } => 2,
        _ => 1,
    }
}

fn coded_fields_per_frame(caps: &Caps) -> u32 {
    match caps {
        Caps::CompressedVideo {
            alignment: g2g_core::BitstreamAlignment::FieldPair,
            ..
        } => 2,
        _ => 1,
    }
}

fn jpeg_xs_caps(width: u32, height: u32, fields_per_frame: u32) -> Caps {
    let alignment = if fields_per_frame == 2 {
        g2g_core::BitstreamAlignment::FieldPair
    } else {
        g2g_core::BitstreamAlignment::Unspecified
    };
    Caps::CompressedVideo {
        codec: VideoCodec::JpegXs,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment,
    }
}

fn interlace_of(fields_per_frame: u32) -> g2g_core::Interlace {
    if fields_per_frame == 2 {
        g2g_core::Interlace::Interleaved
    } else {
        g2g_core::Interlace::Any
    }
}

/// The codestream caps the encoder produces from raw `input`.
fn encoded_caps(input: &Caps) -> Result<Caps, G2gError> {
    let (_, w, h) = raw_geometry(input)?;
    Ok(jpeg_xs_caps(w, h, raw_fields_per_frame(input)))
}

/// The raw caps the decoder announces for a JPEG XS `input`.
fn decoded_caps(input: &Caps) -> Result<Caps, G2gError> {
    match input {
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
                interlace: interlace_of(coded_fields_per_frame(input)),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            })
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

#[derive(Debug, Clone, Copy)]
struct FieldLayout {
    format: RawVideoFormat,
    width: u32,
    frame_height: u32,
    fields_per_frame: u32,
}

impl FieldLayout {
    fn frame_bytes(&self) -> Result<usize, G2gError> {
        self.format
            .unpadded_frame_bytes(self.width, self.frame_height)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn check_codestream_planes(&self, config: &ffi::ImageConfig) -> Result<(), G2gError> {
        let plane_count = self.format.plane_count();
        if usize::from(config.components_num) != plane_count {
            return Err(G2gError::CapsMismatch);
        }
        let fields = u64::from(self.fields_per_frame);
        for (plane, component) in config.components[..plane_count].iter().enumerate() {
            let frame_rows = self
                .format
                .plane_rows(plane, self.frame_height)
                .ok_or(G2gError::CapsMismatch)?;
            let frame_plane_bytes = self
                .format
                .plane_bytes(plane, self.width, self.frame_height)
                .ok_or(G2gError::CapsMismatch)?;
            if u64::from(component.height) * fields != u64::from(frame_rows)
                || u64::from(component.byte_size) * fields != frame_plane_bytes
            {
                return Err(G2gError::CapsMismatch);
            }
        }
        Ok(())
    }

    fn point_at_field(
        &self,
        image: &mut ffi::ImageBuffer,
        frame_start: *mut u8,
        field: u32,
    ) -> Result<(), G2gError> {
        let sample_bytes = self.format.bytes_per_sample() as u64;
        for plane in 0..self.format.plane_count() {
            let offset = self
                .format
                .plane_offset(plane, self.width, self.frame_height)
                .ok_or(G2gError::CapsMismatch)?;
            let row_bytes = u64::from(
                self.format
                    .plane_stride(plane, self.width)
                    .ok_or(G2gError::CapsMismatch)?,
            );
            let plane_bytes = self
                .format
                .plane_bytes(plane, self.width, self.frame_height)
                .ok_or(G2gError::CapsMismatch)?;
            let skipped = u64::from(field) * row_bytes;
            let start = usize::try_from(offset + skipped).map_err(|_| G2gError::CapsMismatch)?;
            image.data_yuv[plane] = frame_start.wrapping_add(start) as *mut c_void;
            // svt strides count samples, and a doubled stride writes every other row
            image.stride[plane] =
                u32::try_from(row_bytes * u64::from(self.fields_per_frame) / sample_bytes)
                    .map_err(|_| G2gError::CapsMismatch)?;
            image.alloc_size[plane] =
                u32::try_from(plane_bytes - skipped).map_err(|_| G2gError::CapsMismatch)?;
        }
        Ok(())
    }
}

// ================================================================
// Encoder
// ================================================================

/// Encodes planar YUV `RawVideo` into a `CompressedVideo{JpegXs}` codestream via
/// SVT-JPEG-XS (one codestream per frame, RFC 9134 codestream packetization mode).
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::svtjpegxs::SvtJpegXsEnc;
///
/// let enc = SvtJpegXsEnc::new();
/// ```
pub struct SvtJpegXsEnc {
    layout: FieldLayout,
    /// Target bits per pixel as a fraction (`bpp_numerator` / `bpp_denominator`).
    bpp_num: u32,
    bpp_den: u32,
    /// Boxed so the C `private_ptr` inside it keeps a stable address across moves.
    enc: Option<Box<ffi::EncoderApi>>,
    /// Output codestream scratch, sized to the encoder's max bytes per codestream.
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
            .field("layout", &self.layout)
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
            layout: FieldLayout {
                format: RawVideoFormat::I422p10,
                width: 0,
                frame_height: 0,
                fields_per_frame: 1,
            },
            bpp_num: 6,
            bpp_den: 1,
            enc: None,
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

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        encoded_caps(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            encoded_caps(input)
                .map_or_else(|_| CapsSet::from_alternatives(Vec::new()), CapsSet::one)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, width, height) = raw_geometry(absolute_caps)?;
        let (colour, depth) = format_to_svt(format).ok_or(G2gError::CapsMismatch)?;
        let layout = FieldLayout {
            format,
            width,
            frame_height: height,
            fields_per_frame: raw_fields_per_frame(absolute_caps),
        };

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
        api.source_height = height / layout.fields_per_frame;
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
        layout.check_codestream_planes(&cfg)?;
        self.layout = layout;
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(jpeg_xs_caps(
                            self.layout.width,
                            self.layout.frame_height,
                            self.layout.fields_per_frame,
                        )))
                        .await?;
                        self.caps_sent = true;
                    }
                    let codestream = self.encode_one(slice)?;
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
    /// Encode one planar-YUV frame to JPEG XS (a fresh `Vec`): one codestream, or
    /// one per field back to back, top field first. The input slice must hold the
    /// tightly-packed planes for the configured geometry.
    fn encode_one(&mut self, input: &[u8]) -> Result<Vec<u8>, G2gError> {
        let enc = self.enc.as_mut().ok_or(G2gError::NotConfigured)?;
        if input.len() < self.layout.frame_bytes()? {
            return Err(G2gError::CapsMismatch);
        }
        let mut codestreams = Vec::new();
        for field in 0..self.layout.fields_per_frame {
            // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
            let mut fr: ffi::Frame = unsafe { core::mem::zeroed() };
            self.layout
                .point_at_field(&mut fr.image, input.as_ptr() as *mut u8, field)?;
            fr.bitstream.buffer = self.out_buf.as_mut_ptr();
            fr.bitstream.allocation_size = self.out_buf.len() as u32;

            // SAFETY: `enc` is initialized; `fr` points at planes inside `input` (its
            // length is checked against the frame size above, and the encoder only
            // reads them) and an output buffer sized to the encoder's max bytes per
            // codestream. Blocking, so the library consumes `fr` before returning.
            let rc = unsafe { ffi::svt_jpeg_xs_encoder_send_picture(&mut **enc, &mut fr, 1) };
            if rc != ffi::ERR_NONE {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
            let mut got: ffi::Frame = unsafe { core::mem::zeroed() };
            // SAFETY: blocking get returns this picture's packet (codestream mode: one
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
            codestreams.extend_from_slice(unsafe { core::slice::from_raw_parts(ptr, len) });
        }
        Ok(codestreams)
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .to_vec();
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(alts)),
            PadTemplate::source(CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
            })),
        ])
    }
}

// ================================================================
// Decoder
// ================================================================

/// Decodes a `CompressedVideo{JpegXs}` codestream into planar YUV `RawVideo` via
/// SVT-JPEG-XS. Geometry / format are discovered from the first codestream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::svtjpegxs::SvtJpegXsDec;
///
/// let dec = SvtJpegXsDec::new();
/// ```
pub struct SvtJpegXsDec {
    /// Boxed for a stable C `private_ptr` address; `None` until the first frame.
    dec: Option<Box<ffi::DecoderApi>>,
    layout: FieldLayout,
    caps_sent: bool,
    segment: Segment,
}

// SAFETY: same ownership-transfer contract as `SvtJpegXsEnc`: the `DecoderApi`'s
// `private_ptr` is driven through `&mut self` only, never shared across threads.
unsafe impl Send for SvtJpegXsDec {}

impl core::fmt::Debug for SvtJpegXsDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SvtJpegXsDec")
            .field("layout", &self.layout)
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
            layout: FieldLayout {
                format: RawVideoFormat::I422p10,
                width: 0,
                frame_height: 0,
                fields_per_frame: 1,
            },
            caps_sent: false,
            segment: Segment::new(),
        }
    }
}

impl Drop for SvtJpegXsDec {
    fn drop(&mut self) {
        self.close_decoder();
    }
}

impl AsyncElement for SvtJpegXsDec {
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
        decoded_caps(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            decoded_caps(input)
                .map_or_else(|_| CapsSet::from_alternatives(Vec::new()), CapsSet::one)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                ..
            } => {
                self.set_fields_per_frame(coded_fields_per_frame(absolute_caps));
                Ok(ConfigureOutcome::Accepted)
            }
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
                    if !self.segment.presents(frame.timing.pts_ns) {
                        g2g_core::g2g_log!(
                            g2g_core::log::Target::category(g2g_core::AsyncElement::log_category(
                                self
                            )),
                            "dropping frame pts {} outside the segment",
                            g2g_core::log::ClockTime(frame.timing.pts_ns)
                        );
                        return Ok(());
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let pixels = self.decode_one(slice)?;
                    if !self.caps_sent {
                        out.push(PipelinePacket::CapsChanged(Caps::RawVideo {
                            format: self.layout.format,
                            width: Dim::Fixed(self.layout.width),
                            height: Dim::Fixed(self.layout.frame_height),
                            framerate: Rate::Any,
                            interlace: interlace_of(self.layout.fields_per_frame),
                            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
                PipelinePacket::CapsChanged(caps) => {
                    if let Caps::CompressedVideo {
                        codec: VideoCodec::JpegXs,
                        ..
                    } = caps
                    {
                        self.set_fields_per_frame(coded_fields_per_frame(&caps));
                    }
                    Ok(())
                }
                PipelinePacket::Segment(segment) => {
                    self.segment = segment;
                    out.push(PipelinePacket::Segment(segment)).await.map(|_| ())
                }
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

impl SvtJpegXsDec {
    fn close_decoder(&mut self) {
        if let Some(mut dec) = self.dec.take() {
            // SAFETY: `dec` was initialized by `svt_jpeg_xs_decoder_init`; close frees
            // its state exactly once (the element owns the only handle).
            unsafe { ffi::svt_jpeg_xs_decoder_close(&mut *dec) };
        }
    }

    fn set_fields_per_frame(&mut self, fields_per_frame: u32) {
        if fields_per_frame == self.layout.fields_per_frame {
            return;
        }
        self.close_decoder();
        self.caps_sent = false;
        self.layout.fields_per_frame = fields_per_frame;
    }

    /// Ensure the decoder is initialized from `codestreams` (learning geometry /
    /// format), then decode one frame, or one field codestream into each field's
    /// rows, into a fresh contiguous planar buffer.
    fn decode_one(&mut self, codestreams: &[u8]) -> Result<Vec<u8>, G2gError> {
        if self.dec.is_none() {
            self.init_from(codestreams)?;
        }
        let dec = self.dec.as_mut().ok_or(G2gError::NotConfigured)?;

        let mut pixels = vec![0u8; self.layout.frame_bytes()?];
        let mut rest = codestreams;
        for field in 0..self.layout.fields_per_frame {
            let codestream_len = if field + 1 == self.layout.fields_per_frame {
                rest.len()
            } else {
                let mut len: u32 = 0;
                // SAFETY: `rest` is a valid readable slice and `len` a valid out param;
                // the image config out param is optional and passed as null.
                let rc = unsafe {
                    ffi::svt_jpeg_xs_decoder_get_single_frame_size(
                        rest.as_ptr(),
                        rest.len(),
                        core::ptr::null_mut(),
                        &mut len,
                        1,
                    )
                };
                if rc != ffi::ERR_NONE {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
                len as usize
            };
            if codestream_len == 0 || codestream_len > rest.len() {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            let (codestream, after) = rest.split_at(codestream_len);
            rest = after;

            // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
            let mut fr: ffi::Frame = unsafe { core::mem::zeroed() };
            // Input bitstream (the library only reads it during the blocking send).
            fr.bitstream.buffer = codestream.as_ptr() as *mut u8;
            fr.bitstream.allocation_size = codestream.len() as u32;
            fr.bitstream.used_size = codestream.len() as u32;
            self.layout
                .point_at_field(&mut fr.image, pixels.as_mut_ptr(), field)?;

            // SAFETY: `dec` is initialized; `fr` points at a valid input codestream and
            // output planes inside `pixels`, which is `frame_bytes` long. Blocking calls.
            let rc = unsafe { ffi::svt_jpeg_xs_decoder_send_frame(&mut **dec, &mut fr, 1) };
            if rc != ffi::ERR_NONE {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            // SAFETY: an all-zero bit pattern is valid for this `#[repr(C)]` POD struct (only integers / null pointers); the SVT call fills or reads it.
            let mut got: ffi::Frame = unsafe { core::mem::zeroed() };
            // SAFETY: blocking get returns the decoded picture into the planes we set on
            // `fr`; `got` is a valid out param. End-of-codestream is a non-fatal status.
            let rc = unsafe { ffi::svt_jpeg_xs_decoder_get_frame(&mut **dec, &mut got, 1) };
            if rc != ffi::ERR_NONE && rc != ffi::ERR_END_OF_CODESTREAM {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
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
        self.dec = Some(api);
        let layout = svt_to_format(cfg.format, cfg.bit_depth)
            .map(|format| FieldLayout {
                format,
                width: cfg.width,
                frame_height: cfg.height.saturating_mul(self.layout.fields_per_frame),
                fields_per_frame: self.layout.fields_per_frame,
            })
            .ok_or(G2gError::CapsMismatch)
            .and_then(|layout| layout.check_codestream_planes(&cfg).map(|()| layout));
        if layout.is_err() {
            self.close_decoder();
        }
        self.layout = layout?;
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .to_vec();
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::JpegXs,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
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

    fn mean_10bit_error(expected: &[u8], actual: &[u8]) -> f64 {
        let samples = expected.len() / 2;
        let sum: u64 = (0..samples)
            .map(|i| {
                let a = u16::from_le_bytes([expected[2 * i], expected[2 * i + 1]]) & 0x03FF;
                let b = u16::from_le_bytes([actual[2 * i], actual[2 * i + 1]]) & 0x03FF;
                u64::from(a.abs_diff(b))
            })
            .sum();
        sum as f64 / samples as f64
    }

    #[test]
    fn interleaved_frame_round_trips_as_two_field_codestreams() {
        let (w, field_h) = (128usize, 32usize);
        let h = 2 * field_h;
        let format = RawVideoFormat::I422p10;
        let top = i422p10_ramp(w, field_h);
        let bottom: Vec<u8> = top
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|sample| (0x03FF - u16::from_le_bytes(*sample)).to_le_bytes())
            .collect();
        let mut frame = vec![0u8; top.len() * 2];
        for plane in 0..format.plane_count() {
            let row = format.plane_stride(plane, w as u32).unwrap() as usize;
            let field_offset = format
                .plane_offset(plane, w as u32, field_h as u32)
                .unwrap() as usize;
            let frame_offset = format.plane_offset(plane, w as u32, h as u32).unwrap() as usize;
            for field_row in 0..format.plane_rows(plane, field_h as u32).unwrap() as usize {
                let source = field_offset + field_row * row;
                let target = frame_offset + 2 * field_row * row;
                frame[target..target + row].copy_from_slice(&top[source..source + row]);
                frame[target + row..target + 2 * row]
                    .copy_from_slice(&bottom[source..source + row]);
            }
        }
        let caps = Caps::RawVideo {
            format,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Interleaved,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };

        let mut enc = SvtJpegXsEnc::new();
        enc.set_property("bpp", PropValue::Fraction(10, 1)).unwrap();
        enc.configure_pipeline(&caps).expect("encoder configures");
        let mut enc_out = Capture::default();
        block_on(enc.process(data_frame(frame.clone()), &mut enc_out)).expect("encode");
        let coded_caps = jpeg_xs_caps(w as u32, h as u32, 2);
        assert_eq!(enc_out.caps, vec![coded_caps.clone()]);
        let codestreams = enc_out.frames.remove(0);

        let mut dec = SvtJpegXsDec::new();
        dec.configure_pipeline(&coded_caps)
            .expect("decoder configures");
        let mut dec_out = Capture::default();
        block_on(dec.process(data_frame(codestreams), &mut dec_out)).expect("decode");
        assert_eq!(
            dec_out.caps,
            vec![Caps::RawVideo {
                format,
                width: Dim::Fixed(w as u32),
                height: Dim::Fixed(h as u32),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Interleaved,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            }]
        );
        let mean_err = mean_10bit_error(&frame, &dec_out.frames[0]);
        assert!(mean_err < 8.0, "mean 10-bit error {mean_err} too high");
    }

    #[test]
    fn encode_then_decode_round_trips_i422p10() {
        let (w, h) = (128usize, 64usize);
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I422p10,
            width: Dim::Fixed(w as u32),
            height: Dim::Fixed(h as u32),
            framerate: Rate::Fixed(60 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
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
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }
        );

        // Near-lossless at 10 bpp: mean absolute 10-bit sample error is small.
        let mean_err = mean_10bit_error(&input, decoded);
        assert!(
            mean_err < 8.0,
            "mean 10-bit error {mean_err} too high (round trip broken)"
        );
    }
}
