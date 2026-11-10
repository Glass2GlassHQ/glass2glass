// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::ffi::c_void;
use core::ptr::NonNull;

use std::sync::Arc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::log::Target;
use g2g_core::{
    g2g_error, AudioFormat, Caps, CapsSet, Colorimetry, Dim, G2gError, HardwareError, Interlace,
    PropError, PropValue, Rate, RawVideoFormat,
};

use crate::st291::AncPacket;

pub const DECKLINK_LIBRARY: &str = "libDeckLinkAPI.so";

pub(crate) const NANOS_PER_SECOND: u64 = 1_000_000_000;

// slack between the first buffer and its display time, so the other sink on the device can catch up
pub const PREROLL_NS: u64 = 200_000_000;

// GStreamer's decklinkvideosink hands a frame over 30 ms before it is due
pub const SCHEDULE_LEAD_NS: u64 = 30_000_000;

pub(crate) fn log_target() -> Target<'static> {
    Target::category("decklink")
}

#[derive(Debug, PartialEq, Eq)]
pub struct DisplayMode {
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    pub framerate_numerator: u32,
    pub framerate_denominator: u32,
    pub interlaced: bool,
    framerate_q16: u32,
    fourcc: u32,
}

impl DisplayMode {
    pub fn frame_duration_ns(&self) -> u64 {
        NANOS_PER_SECOND * u64::from(self.framerate_denominator)
            / u64::from(self.framerate_numerator)
    }

    pub fn caps(&self, format: RawVideoFormat) -> Caps {
        let interlace = if self.interlaced {
            Interlace::Interleaved
        } else {
            Interlace::Progressive
        };
        Caps::RawVideo {
            format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
            interlace,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    // caps that declare no scan structure count as progressive
    pub fn matches(
        &self,
        width: u32,
        height: u32,
        framerate_q16: u32,
        interlace: Interlace,
    ) -> bool {
        self.width == width
            && self.height == height
            && self.framerate_q16 == framerate_q16
            && self.interlaced == (interlace == Interlace::Interleaved)
    }
}

const fn mode(
    name: &'static str,
    fourcc: &[u8; 4],
    width: u32,
    height: u32,
    framerate_numerator: u32,
    framerate_denominator: u32,
    interlaced: bool,
) -> DisplayMode {
    // rounds down as framerate_q16_from_fraction does, so it equals the rate in negotiated caps
    let framerate_q16 = ((framerate_numerator as u64) << 16) / framerate_denominator as u64;
    assert!(framerate_q16 <= u32::MAX as u64);
    DisplayMode {
        name,
        width,
        height,
        framerate_numerator,
        framerate_denominator,
        interlaced,
        framerate_q16: framerate_q16 as u32,
        fourcc: u32::from_be_bytes(*fourcc),
    }
}

// GStreamer's decklink mode names and order, without its widescreen SD modes, which differ only in pixel aspect ratio
pub const DISPLAY_MODES: &[DisplayMode] = &[
    mode("ntsc", b"ntsc", 720, 486, 30000, 1001, true),
    mode("ntsc2398", b"nt23", 720, 486, 24000, 1001, true),
    mode("pal", b"pal ", 720, 576, 25, 1, true),
    mode("ntsc-p", b"ntsp", 720, 486, 30000, 1001, false),
    mode("pal-p", b"palp", 720, 576, 25, 1, false),
    mode("1080p2398", b"23ps", 1920, 1080, 24000, 1001, false),
    mode("1080p24", b"24ps", 1920, 1080, 24, 1, false),
    mode("1080p25", b"Hp25", 1920, 1080, 25, 1, false),
    mode("1080p2997", b"Hp29", 1920, 1080, 30000, 1001, false),
    mode("1080p30", b"Hp30", 1920, 1080, 30, 1, false),
    mode("1080i50", b"Hi50", 1920, 1080, 25, 1, true),
    mode("1080i5994", b"Hi59", 1920, 1080, 30000, 1001, true),
    mode("1080i60", b"Hi60", 1920, 1080, 30, 1, true),
    mode("1080p50", b"Hp50", 1920, 1080, 50, 1, false),
    mode("1080p5994", b"Hp59", 1920, 1080, 60000, 1001, false),
    mode("1080p60", b"Hp60", 1920, 1080, 60, 1, false),
    mode("720p50", b"hp50", 1280, 720, 50, 1, false),
    mode("720p5994", b"hp59", 1280, 720, 60000, 1001, false),
    mode("720p60", b"hp60", 1280, 720, 60, 1, false),
    mode("1556p2398", b"2k23", 2048, 1556, 24000, 1001, false),
    mode("1556p24", b"2k24", 2048, 1556, 24, 1, false),
    mode("1556p25", b"2k25", 2048, 1556, 25, 1, false),
    mode("2kdcip2398", b"2d23", 2048, 1080, 24000, 1001, false),
    mode("2kdcip24", b"2d24", 2048, 1080, 24, 1, false),
    mode("2kdcip25", b"2d25", 2048, 1080, 25, 1, false),
    mode("2kdcip2997", b"2d29", 2048, 1080, 30000, 1001, false),
    mode("2kdcip30", b"2d30", 2048, 1080, 30, 1, false),
    mode("2kdcip50", b"2d50", 2048, 1080, 50, 1, false),
    mode("2kdcip5994", b"2d59", 2048, 1080, 60000, 1001, false),
    mode("2kdcip60", b"2d60", 2048, 1080, 60, 1, false),
    mode("2160p2398", b"4k23", 3840, 2160, 24000, 1001, false),
    mode("2160p24", b"4k24", 3840, 2160, 24, 1, false),
    mode("2160p25", b"4k25", 3840, 2160, 25, 1, false),
    mode("2160p2997", b"4k29", 3840, 2160, 30000, 1001, false),
    mode("2160p30", b"4k30", 3840, 2160, 30, 1, false),
    mode("2160p50", b"4k50", 3840, 2160, 50, 1, false),
    mode("2160p5994", b"4k59", 3840, 2160, 60000, 1001, false),
    mode("2160p60", b"4k60", 3840, 2160, 60, 1, false),
    mode("4kdcip2398", b"4d23", 4096, 2160, 24000, 1001, false),
    mode("4kdcip24", b"4d24", 4096, 2160, 24, 1, false),
    mode("4kdcip25", b"4d25", 4096, 2160, 25, 1, false),
    mode("4kdcip2997", b"4d29", 4096, 2160, 30000, 1001, false),
    mode("4kdcip30", b"4d30", 4096, 2160, 30, 1, false),
    mode("4kdcip50", b"4d50", 4096, 2160, 50, 1, false),
    mode("4kdcip5994", b"4d59", 4096, 2160, 60000, 1001, false),
    mode("4kdcip60", b"4d60", 4096, 2160, 60, 1, false),
    mode("8kp2398", b"8k23", 7680, 4320, 24000, 1001, false),
    mode("8kp24", b"8k24", 7680, 4320, 24, 1, false),
    mode("8kp25", b"8k25", 7680, 4320, 25, 1, false),
    mode("8kp2997", b"8k29", 7680, 4320, 30000, 1001, false),
    mode("8kp30", b"8k30", 7680, 4320, 30, 1, false),
    mode("8kp50", b"8k50", 7680, 4320, 50, 1, false),
    mode("8kp5994", b"8k59", 7680, 4320, 60000, 1001, false),
    mode("8kp60", b"8k60", 7680, 4320, 60, 1, false),
    mode("8kdcip2398", b"8d23", 8192, 4320, 24000, 1001, false),
    mode("8kdcip24", b"8d24", 8192, 4320, 24, 1, false),
    mode("8kdcip25", b"8d25", 8192, 4320, 25, 1, false),
    mode("8kdcip2997", b"8d29", 8192, 4320, 30000, 1001, false),
    mode("8kdcip30", b"8d30", 8192, 4320, 30, 1, false),
    mode("8kdcip50", b"8d50", 8192, 4320, 50, 1, false),
    mode("8kdcip5994", b"8d59", 8192, 4320, 60000, 1001, false),
    mode("8kdcip60", b"8d60", 8192, 4320, 60, 1, false),
    mode("640x480p60", b"vga6", 640, 480, 60, 1, false),
    mode("800x600p60", b"svg6", 800, 600, 60, 1, false),
    mode("1440x900p50", b"wxg5", 1440, 900, 50, 1, false),
    mode("1440x900p60", b"wxg6", 1440, 900, 60, 1, false),
    mode("1440x1080p50", b"sxg5", 1440, 1080, 50, 1, false),
    mode("1440x1080p60", b"sxg6", 1440, 1080, 60, 1, false),
    mode("1600x1200p50", b"uxg5", 1600, 1200, 50, 1, false),
    mode("1600x1200p60", b"uxg6", 1600, 1200, 60, 1, false),
    mode("1920x1200p50", b"wux5", 1920, 1200, 50, 1, false),
    mode("1920x1200p60", b"wux6", 1920, 1200, 60, 1, false),
    mode("1920x1440p50", b"1945", 1920, 1440, 50, 1, false),
    mode("1920x1440p60", b"1946", 1920, 1440, 60, 1, false),
    mode("2560x1440p50", b"wqh5", 2560, 1440, 50, 1, false),
    mode("2560x1440p60", b"wqh6", 2560, 1440, 60, 1, false),
    mode("2560x1600p50", b"wqx5", 2560, 1600, 50, 1, false),
    mode("2560x1600p60", b"wqx6", 2560, 1600, 60, 1, false),
];

pub(crate) const AUTO_MODE: &str = "auto";

// "auto" then every name in DISPLAY_MODES, in order
pub(crate) const MODE_VALUES: &str =
    "auto | ntsc | ntsc2398 | pal | ntsc-p | pal-p | 1080p2398 | 1080p24 | \
1080p25 | 1080p2997 | 1080p30 | 1080i50 | 1080i5994 | 1080i60 | 1080p50 | 1080p5994 | 1080p60 | \
720p50 | 720p5994 | 720p60 | 1556p2398 | 1556p24 | 1556p25 | 2kdcip2398 | 2kdcip24 | 2kdcip25 | \
2kdcip2997 | 2kdcip30 | 2kdcip50 | 2kdcip5994 | 2kdcip60 | 2160p2398 | 2160p24 | 2160p25 | \
2160p2997 | 2160p30 | 2160p50 | 2160p5994 | 2160p60 | 4kdcip2398 | 4kdcip24 | 4kdcip25 | \
4kdcip2997 | 4kdcip30 | 4kdcip50 | 4kdcip5994 | 4kdcip60 | 8kp2398 | 8kp24 | 8kp25 | 8kp2997 | \
8kp30 | 8kp50 | 8kp5994 | 8kp60 | 8kdcip2398 | 8kdcip24 | 8kdcip25 | 8kdcip2997 | 8kdcip30 | \
8kdcip50 | 8kdcip5994 | 8kdcip60 | 640x480p60 | 800x600p60 | 1440x900p50 | 1440x900p60 | \
1440x1080p50 | 1440x1080p60 | 1600x1200p50 | 1600x1200p60 | 1920x1200p50 | 1920x1200p60 | \
1920x1440p50 | 1920x1440p60 | 2560x1440p50 | 2560x1440p60 | 2560x1600p50 | 2560x1600p60";

fn mode_for_code(code: u32) -> Option<&'static DisplayMode> {
    DISPLAY_MODES.iter().find(|mode| mode.fourcc == code)
}

pub fn display_mode(name: &str) -> Option<&'static DisplayMode> {
    DISPLAY_MODES.iter().find(|mode| mode.name == name)
}

pub(crate) fn mode_from_property(
    value: &PropValue,
) -> Result<Option<&'static DisplayMode>, PropError> {
    match value.as_str().ok_or(PropError::Type)? {
        AUTO_MODE => Ok(None),
        name => display_mode(name).map(Some).ok_or(PropError::Value),
    }
}

pub(crate) fn mode_property_value(mode: Option<&'static DisplayMode>) -> PropValue {
    PropValue::Str(mode.map_or(AUTO_MODE, |mode| mode.name).into())
}

pub(crate) fn device_number_from_property(value: &PropValue) -> Result<u32, PropError> {
    let device_number = value.as_int().ok_or(PropError::Type)?;
    u32::try_from(device_number).map_err(|_| PropError::Value)
}

pub fn mode_for_caps(
    width: u32,
    height: u32,
    framerate_q16: u32,
    interlace: Interlace,
) -> Option<&'static DisplayMode> {
    DISPLAY_MODES
        .iter()
        .find(|mode| mode.matches(width, height, framerate_q16, interlace))
}

#[derive(Debug)]
pub(crate) struct PixelFormat {
    pub(crate) format: RawVideoFormat,
    code: &'static [u8; 4],
    // GStreamer's decklink video-format name
    pub(crate) name: &'static str,
}

// the raw formats the board plays and captures, with its pixel format codes
pub(crate) const PIXEL_FORMATS: [PixelFormat; 4] = [
    PixelFormat {
        format: RawVideoFormat::Uyvy,
        code: b"2vuy",
        name: "8bit-yuv",
    },
    PixelFormat {
        format: RawVideoFormat::Bgra8,
        code: b"BGRA",
        name: "8bit-bgra",
    },
    PixelFormat {
        format: RawVideoFormat::V210,
        code: b"v210",
        name: "10bit-yuv",
    },
    PixelFormat {
        format: RawVideoFormat::R210,
        code: b"r210",
        name: "10bit-rgb",
    },
];

fn format_for_pixel_code(code: u32) -> Option<RawVideoFormat> {
    PIXEL_FORMATS
        .iter()
        .find(|pixel| u32::from_be_bytes(*pixel.code) == code)
        .map(|pixel| pixel.format)
}

fn pixel_format_code(format: RawVideoFormat) -> Result<u32, G2gError> {
    PIXEL_FORMATS
        .iter()
        .find(|pixel| pixel.format == format)
        .map(|pixel| u32::from_be_bytes(*pixel.code))
        .ok_or(G2gError::CapsMismatch)
}

pub fn video_format_supported(format: RawVideoFormat) -> bool {
    PIXEL_FORMATS.iter().any(|pixel| pixel.format == format)
}

pub(crate) fn any_mode_caps() -> CapsSet {
    let caps = PIXEL_FORMATS
        .iter()
        .map(|pixel| Caps::RawVideo {
            format: pixel.format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        })
        .collect();
    CapsSet::from_alternatives(caps)
}

#[derive(Debug)]
pub struct ScheduledVideoFrame<'a> {
    pub pixels: &'a [u8],
    pub format: RawVideoFormat,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub display_time_ns: u64,
    pub duration_ns: u64,
    pub ancillary: &'a [AncPacket],
}

// one output device, with times in nanoseconds on the board's hardware clock
pub trait DeckLinkOutput: Send {
    fn enable_video(&mut self, mode: &'static DisplayMode, vanc: bool) -> Result<(), G2gError>;
    fn schedule_video(&mut self, frame: &ScheduledVideoFrame<'_>) -> Result<(), G2gError>;
    fn enable_audio(&mut self, format: AudioFormat, channels: u8) -> Result<(), G2gError>;
    fn begin_audio_preroll(&mut self) -> Result<(), G2gError>;
    // returns how many sample frames the board took
    fn schedule_audio(
        &mut self,
        samples: &[u8],
        sample_frames: u32,
        stream_time_ns: u64,
    ) -> Result<u32, G2gError>;
    fn start_playback(&mut self, start_time_ns: u64) -> Result<(), G2gError>;
    fn stop_playback(&mut self, stop_time_ns: u64) -> Result<(), G2gError>;
    fn hardware_time_ns(&mut self) -> Result<u64, G2gError>;
}

pub type DeckLinkOpener =
    Arc<dyn Fn(u32) -> Result<Box<dyn DeckLinkOutput>, G2gError> + Send + Sync>;

pub fn sdk_opener() -> DeckLinkOpener {
    Arc::new(SdkOutput::open)
}

#[repr(C)]
struct RawOutput {
    _opaque: [u8; 0],
}

// same layout as g2g_decklink_ancillary_packet in the C++ shim
#[repr(C)]
struct RawAncillaryPacket {
    user_data: *const u8,
    user_data_count: u32,
    line_number: u32,
    did: u8,
    sdid: u8,
}

// status codes of g2g_decklink_open, matching the enum in csrc/decklink_shim.cpp
const OPENED: i32 = 0;
const LIBRARY_MISSING: i32 = 1;
const NO_DEVICE: i32 = 2;
const NO_OUTPUT: i32 = 3;
const NO_ITERATOR: i32 = 4;
const NO_INPUT: i32 = 5;

fn opened_handle<Raw>(
    status: i32,
    handle: *mut Raw,
    device_number: u32,
) -> Result<NonNull<Raw>, G2gError> {
    match status {
        OPENED => NonNull::new(handle).ok_or(G2gError::Hardware(HardwareError::Other)),
        LIBRARY_MISSING => Err(G2gError::Hardware(HardwareError::LibraryNotFound(
            DECKLINK_LIBRARY,
        ))),
        NO_DEVICE => {
            g2g_error!(log_target(), "no device at device-number {device_number}");
            Err(G2gError::Hardware(HardwareError::Other))
        }
        NO_OUTPUT => {
            g2g_error!(log_target(), "device-number {device_number} has no output");
            Err(G2gError::Hardware(HardwareError::Other))
        }
        NO_INPUT => {
            g2g_error!(log_target(), "device-number {device_number} has no input");
            Err(G2gError::Hardware(HardwareError::Other))
        }
        NO_ITERATOR => {
            g2g_error!(
                log_target(),
                "{DECKLINK_LIBRARY} has no SDK 16.0 device iterator, the driver may be older than the SDK"
            );
            Err(G2gError::Hardware(HardwareError::Other))
        }
        hresult => Err(G2gError::Hardware(HardwareError::DeckLink(hresult))),
    }
}

extern "C" {
    fn g2g_decklink_open(device_number: u32, opened: *mut *mut RawOutput) -> i32;
    fn g2g_decklink_close(handle: *mut RawOutput);
    fn g2g_decklink_enable_video(handle: *mut RawOutput, display_mode: u32, vanc: bool) -> i32;
    fn g2g_decklink_schedule_video(
        handle: *mut RawOutput,
        pixels: *const u8,
        width: i32,
        height: i32,
        row_bytes: i32,
        pixel_format: u32,
        display_time: i64,
        duration: i64,
        time_scale: i64,
        packets: *const RawAncillaryPacket,
        packet_count: u32,
    ) -> i32;
    fn g2g_decklink_enable_audio(handle: *mut RawOutput, sample_type: u32, channels: u32) -> i32;
    fn g2g_decklink_begin_audio_preroll(handle: *mut RawOutput) -> i32;
    fn g2g_decklink_schedule_audio(
        handle: *mut RawOutput,
        samples: *const u8,
        sample_frames: u32,
        stream_time: i64,
        time_scale: i64,
        written: *mut u32,
    ) -> i32;
    fn g2g_decklink_start_playback(handle: *mut RawOutput, start_time: i64, time_scale: i64)
        -> i32;
    fn g2g_decklink_stop_playback(handle: *mut RawOutput, stop_time: i64, time_scale: i64) -> i32;
    fn g2g_decklink_hardware_time(
        handle: *mut RawOutput,
        time_scale: i64,
        hardware_time: *mut i64,
    ) -> i32;
}

const TIME_SCALE: i64 = NANOS_PER_SECOND as i64;

fn check(hresult: i32) -> Result<(), G2gError> {
    match hresult {
        0 => Ok(()),
        failure => Err(G2gError::Hardware(HardwareError::DeckLink(failure))),
    }
}

fn sdk_time(ns: u64) -> Result<i64, G2gError> {
    i64::try_from(ns).map_err(|_| G2gError::Hardware(HardwareError::Other))
}

fn sdk_dimension(value: u32) -> Result<i32, G2gError> {
    i32::try_from(value).map_err(|_| G2gError::CapsMismatch)
}

// ST 2038 line numbers from 0x7FE up carry no line
const FIRST_UNPOSITIONED_LINE: u16 = 0x7FE;
// the SDK picks the line of a packet on line zero
const SDK_AUTO_LINE: u32 = 0;

fn sdk_line(line_number: u16) -> u32 {
    if line_number >= FIRST_UNPOSITIONED_LINE {
        return SDK_AUTO_LINE;
    }
    u32::from(line_number)
}

#[derive(Debug)]
struct SdkOutput {
    handle: NonNull<RawOutput>,
}

// SAFETY: the SDK's output interface may be called from any thread, and its one owner makes one call at a time
unsafe impl Send for SdkOutput {}

impl SdkOutput {
    fn open(device_number: u32) -> Result<Box<dyn DeckLinkOutput>, G2gError> {
        let mut handle = core::ptr::null_mut();
        // SAFETY: `handle` is a valid place for the shim to write the opened handle to
        let status = unsafe { g2g_decklink_open(device_number, &mut handle) };
        let handle = opened_handle(status, handle, device_number)?;
        Ok(Box::new(SdkOutput { handle }))
    }

    fn raw(&self) -> *mut RawOutput {
        self.handle.as_ptr()
    }
}

impl Drop for SdkOutput {
    fn drop(&mut self) {
        // SAFETY: the handle came from g2g_decklink_open and is closed only here
        unsafe { g2g_decklink_close(self.raw()) }
    }
}

impl DeckLinkOutput for SdkOutput {
    fn enable_video(&mut self, mode: &'static DisplayMode, vanc: bool) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_enable_video(self.raw(), mode.fourcc, vanc) })
    }

    fn schedule_video(&mut self, frame: &ScheduledVideoFrame<'_>) -> Result<(), G2gError> {
        let pixel_format = pixel_format_code(frame.format)?;
        let row_bytes = sdk_dimension(frame.row_bytes)?;
        let height = sdk_dimension(frame.height)?;
        let frame_bytes = frame.row_bytes as usize * frame.height as usize;
        if frame.pixels.len() < frame_bytes {
            return Err(G2gError::CapsMismatch);
        }
        // the SDK takes 8-bit user data words and adds the parity bits itself
        let user_data: Vec<Vec<u8>> = frame
            .ancillary
            .iter()
            .map(AncPacket::user_data_bytes)
            .collect();
        let packets = frame
            .ancillary
            .iter()
            .zip(&user_data)
            .map(|(packet, bytes)| {
                Ok(RawAncillaryPacket {
                    user_data: bytes.as_ptr(),
                    user_data_count: u32::try_from(bytes.len())
                        .map_err(|_| G2gError::CapsMismatch)?,
                    line_number: sdk_line(packet.line_number),
                    did: packet.did,
                    sdid: packet.sdid,
                })
            })
            .collect::<Result<Vec<_>, G2gError>>()?;
        let packet_count = u32::try_from(packets.len()).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: the handle is open, the shim reads row_bytes * height bytes, which `pixels` holds, and each packet's user data, which `user_data` holds
        check(unsafe {
            g2g_decklink_schedule_video(
                self.raw(),
                frame.pixels.as_ptr(),
                sdk_dimension(frame.width)?,
                height,
                row_bytes,
                pixel_format,
                sdk_time(frame.display_time_ns)?,
                sdk_time(frame.duration_ns)?,
                TIME_SCALE,
                packets.as_ptr(),
                packet_count,
            )
        })
    }

    fn enable_audio(&mut self, format: AudioFormat, channels: u8) -> Result<(), G2gError> {
        // the SDK's sample type codes are the bit depths
        let sample_type = match format {
            AudioFormat::PcmS16Le => 16,
            AudioFormat::PcmS32Le => 32,
            _ => return Err(G2gError::CapsMismatch),
        };
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_enable_audio(self.raw(), sample_type, u32::from(channels)) })
    }

    fn begin_audio_preroll(&mut self) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_begin_audio_preroll(self.raw()) })
    }

    fn schedule_audio(
        &mut self,
        samples: &[u8],
        sample_frames: u32,
        stream_time_ns: u64,
    ) -> Result<u32, G2gError> {
        let mut written = 0;
        // SAFETY: the handle is open, and the caller sized `samples` to hold `sample_frames` frames of the enabled format
        check(unsafe {
            g2g_decklink_schedule_audio(
                self.raw(),
                samples.as_ptr(),
                sample_frames,
                sdk_time(stream_time_ns)?,
                TIME_SCALE,
                &mut written,
            )
        })?;
        Ok(written)
    }

    fn start_playback(&mut self, start_time_ns: u64) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe {
            g2g_decklink_start_playback(self.raw(), sdk_time(start_time_ns)?, TIME_SCALE)
        })
    }

    fn stop_playback(&mut self, stop_time_ns: u64) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe {
            g2g_decklink_stop_playback(self.raw(), sdk_time(stop_time_ns)?, TIME_SCALE)
        })
    }

    fn hardware_time_ns(&mut self) -> Result<u64, G2gError> {
        let mut time = 0;
        // SAFETY: the handle is open, and `time` is a valid place for the reading
        check(unsafe { g2g_decklink_hardware_time(self.raw(), TIME_SCALE, &mut time) })?;
        u64::try_from(time).map_err(|_| G2gError::Hardware(HardwareError::Other))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct VideoConnection {
    pub name: &'static str,
    sdk_bit: u32,
}

// GStreamer's decklink connection names, with the SDK's BMDVideoConnection bits
pub const VIDEO_CONNECTIONS: &[VideoConnection] = &[
    VideoConnection {
        name: "sdi",
        sdk_bit: 1 << 0,
    },
    VideoConnection {
        name: "hdmi",
        sdk_bit: 1 << 1,
    },
    VideoConnection {
        name: "optical-sdi",
        sdk_bit: 1 << 2,
    },
    VideoConnection {
        name: "component",
        sdk_bit: 1 << 3,
    },
    VideoConnection {
        name: "composite",
        sdk_bit: 1 << 4,
    },
    VideoConnection {
        name: "svideo",
        sdk_bit: 1 << 5,
    },
];

pub fn video_connection(name: &str) -> Option<&'static VideoConnection> {
    VIDEO_CONNECTIONS
        .iter()
        .find(|connection| connection.name == name)
}

// BMDDetectedVideoInputFormatFlags bits
const DETECTED_YCBCR422: u32 = 1 << 0;
const DETECTED_RGB444: u32 = 1 << 1;
const DETECTED_10_BIT: u32 = 1 << 4;
const DETECTED_8_BIT: u32 = 1 << 5;

// GStreamer captures 8-bit RGB as ARGB, the sink formats here have BGRA
const DETECTED_FORMATS: [(u32, RawVideoFormat); 4] = [
    (DETECTED_YCBCR422 | DETECTED_8_BIT, RawVideoFormat::Uyvy),
    (DETECTED_YCBCR422 | DETECTED_10_BIT, RawVideoFormat::V210),
    (DETECTED_RGB444 | DETECTED_8_BIT, RawVideoFormat::Bgra8),
    (DETECTED_RGB444 | DETECTED_10_BIT, RawVideoFormat::R210),
];

fn detected_format(flags: u32) -> Option<RawVideoFormat> {
    DETECTED_FORMATS
        .iter()
        .find(|(bits, _)| flags & bits == *bits)
        .map(|(_, format)| *format)
}

#[derive(Debug)]
pub struct CapturedVideoFrame {
    pub pixels: Vec<u8>,
    pub format: RawVideoFormat,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub hardware_time_ns: u64,
    pub no_signal: bool,
    pub ancillary: Vec<AncPacket>,
}

#[derive(Debug)]
pub enum InputEvent {
    Frame(CapturedVideoFrame),
    // None for an input mode or signal format the source cannot capture
    FormatChanged {
        mode: Option<&'static DisplayMode>,
        format: Option<RawVideoFormat>,
    },
    Failed(G2gError),
}

pub type InputEventSink = Box<dyn Fn(InputEvent) + Send + Sync>;

pub trait DeckLinkInput: Send {
    fn set_connection(&mut self, connection: &'static VideoConnection) -> Result<(), G2gError>;
    fn enable_video(
        &mut self,
        mode: &'static DisplayMode,
        format: RawVideoFormat,
        format_detection: bool,
    ) -> Result<(), G2gError>;
    fn start(&mut self, events: InputEventSink) -> Result<(), G2gError>;
    fn restart_video(
        &mut self,
        mode: &'static DisplayMode,
        format: RawVideoFormat,
    ) -> Result<(), G2gError>;
    fn hardware_time_ns(&mut self) -> Result<u64, G2gError>;
}

pub type DeckLinkInputOpener =
    Arc<dyn Fn(u32) -> Result<Box<dyn DeckLinkInput>, G2gError> + Send + Sync>;

pub fn sdk_input_opener() -> DeckLinkInputOpener {
    Arc::new(SdkInput::open)
}

#[repr(C)]
struct RawInput {
    _opaque: [u8; 0],
}

// same layout as g2g_decklink_input_frame in the C++ shim
#[repr(C)]
struct RawInputFrame {
    pixels: *const u8,
    row_bytes: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    hardware_time: i64,
    no_signal: bool,
    packets: *const RawAncillaryPacket,
    packet_count: u32,
}

type FrameArrived = unsafe extern "C" fn(context: *mut c_void, frame: *const RawInputFrame);
type FormatChanged =
    unsafe extern "C" fn(context: *mut c_void, display_mode: u32, detected_signal_flags: u32);
type CaptureFailed = unsafe extern "C" fn(context: *mut c_void, hresult: i32);

extern "C" {
    fn g2g_decklink_open_input(device_number: u32, opened: *mut *mut RawInput) -> i32;
    fn g2g_decklink_close_input(handle: *mut RawInput);
    fn g2g_decklink_set_input_connection(handle: *mut RawInput, connection: u32) -> i32;
    fn g2g_decklink_supports_format_detection(handle: *mut RawInput, supported: *mut bool) -> i32;
    fn g2g_decklink_enable_video_input(
        handle: *mut RawInput,
        display_mode: u32,
        pixel_format: u32,
        format_detection: bool,
    ) -> i32;
    fn g2g_decklink_start_input(
        handle: *mut RawInput,
        context: *mut c_void,
        time_scale: i64,
        frame_arrived: FrameArrived,
        format_changed: FormatChanged,
        capture_failed: CaptureFailed,
    ) -> i32;
    fn g2g_decklink_restart_video_input(
        handle: *mut RawInput,
        display_mode: u32,
        pixel_format: u32,
    ) -> i32;
    fn g2g_decklink_input_hardware_time(
        handle: *mut RawInput,
        time_scale: i64,
        hardware_time: *mut i64,
    ) -> i32;
}

fn captured_frame(raw: &RawInputFrame) -> Result<CapturedVideoFrame, G2gError> {
    let format = format_for_pixel_code(raw.pixel_format).ok_or(G2gError::CapsMismatch)?;
    let frame_bytes = (raw.row_bytes as usize)
        .checked_mul(raw.height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    // SAFETY: the shim's pixels hold row_bytes * height bytes for the length of the callback
    let pixels = unsafe { core::slice::from_raw_parts(raw.pixels, frame_bytes) }.to_vec();
    let packets: &[RawAncillaryPacket] = match raw.packet_count {
        0 => &[],
        // SAFETY: the shim's packets array holds packet_count entries for the length of the callback
        count => unsafe { core::slice::from_raw_parts(raw.packets, count as usize) },
    };
    let ancillary = packets
        .iter()
        .map(|packet| {
            // SAFETY: each packet's user data holds user_data_count bytes for the length of the callback
            let user_data = unsafe {
                core::slice::from_raw_parts(packet.user_data, packet.user_data_count as usize)
            };
            let mut ancillary = AncPacket::from_bytes(packet.did, packet.sdid, user_data);
            ancillary.line_number =
                u16::try_from(packet.line_number).map_err(|_| G2gError::CapsMismatch)?;
            Ok(ancillary)
        })
        .collect::<Result<Vec<_>, G2gError>>()?;
    Ok(CapturedVideoFrame {
        pixels,
        format,
        width: raw.width,
        height: raw.height,
        row_bytes: raw.row_bytes,
        hardware_time_ns: u64::try_from(raw.hardware_time)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?,
        no_signal: raw.no_signal,
        ancillary,
    })
}

unsafe extern "C" fn frame_arrived(context: *mut c_void, frame: *const RawInputFrame) {
    // SAFETY: the context is the event sink SdkInput::start boxed, which outlives the callback, and the frame is valid for this call
    let (events, frame) = unsafe { (&*(context as *const InputEventSink), &*frame) };
    events(match captured_frame(frame) {
        Ok(frame) => InputEvent::Frame(frame),
        Err(error) => InputEvent::Failed(error),
    });
}

unsafe extern "C" fn format_changed(
    context: *mut c_void,
    display_mode: u32,
    detected_signal_flags: u32,
) {
    // SAFETY: the context is the event sink SdkInput::start boxed, which outlives the callback
    let events = unsafe { &*(context as *const InputEventSink) };
    events(InputEvent::FormatChanged {
        mode: mode_for_code(display_mode),
        format: detected_format(detected_signal_flags),
    });
}

unsafe extern "C" fn capture_failed(context: *mut c_void, hresult: i32) {
    // SAFETY: the context is the event sink SdkInput::start boxed, which outlives the callback
    let events = unsafe { &*(context as *const InputEventSink) };
    events(InputEvent::Failed(G2gError::Hardware(
        HardwareError::DeckLink(hresult),
    )));
}

struct SdkInput {
    handle: NonNull<RawInput>,
    device_number: u32,
    // boxed again so the shim holds a thin pointer to it
    events: Option<Box<InputEventSink>>,
}

impl core::fmt::Debug for SdkInput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SdkInput")
            .field("device_number", &self.device_number)
            .finish()
    }
}

// SAFETY: the SDK's input interface may be called from any thread, and its one owner makes one call at a time
unsafe impl Send for SdkInput {}

impl SdkInput {
    fn open(device_number: u32) -> Result<Box<dyn DeckLinkInput>, G2gError> {
        let mut handle = core::ptr::null_mut();
        // SAFETY: `handle` is a valid place for the shim to write the opened handle to
        let status = unsafe { g2g_decklink_open_input(device_number, &mut handle) };
        let handle = opened_handle(status, handle, device_number)?;
        Ok(Box::new(SdkInput {
            handle,
            device_number,
            events: None,
        }))
    }

    fn raw(&self) -> *mut RawInput {
        self.handle.as_ptr()
    }
}

impl Drop for SdkInput {
    fn drop(&mut self) {
        // SAFETY: the handle came from g2g_decklink_open_input and is closed only here, which stops the callbacks before `events` drops
        unsafe { g2g_decklink_close_input(self.raw()) }
    }
}

impl DeckLinkInput for SdkInput {
    fn set_connection(&mut self, connection: &'static VideoConnection) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_set_input_connection(self.raw(), connection.sdk_bit) })
    }

    fn enable_video(
        &mut self,
        mode: &'static DisplayMode,
        format: RawVideoFormat,
        format_detection: bool,
    ) -> Result<(), G2gError> {
        if format_detection {
            let mut supported = false;
            // SAFETY: the handle is open, and `supported` is a valid place for the answer
            check(unsafe { g2g_decklink_supports_format_detection(self.raw(), &mut supported) })?;
            if !supported {
                g2g_error!(
                    log_target(),
                    "device-number {} cannot detect its input mode, set mode",
                    self.device_number
                );
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
        let pixel_format = pixel_format_code(format)?;
        // SAFETY: the handle is open for the life of `self`
        check(unsafe {
            g2g_decklink_enable_video_input(self.raw(), mode.fourcc, pixel_format, format_detection)
        })
    }

    fn start(&mut self, events: InputEventSink) -> Result<(), G2gError> {
        let events = Box::new(events);
        let context = &*events as *const InputEventSink as *mut c_void;
        // SAFETY: the handle is open, and `events` is kept in `self` until the input closes or the next start replaces its callback
        let started = check(unsafe {
            g2g_decklink_start_input(
                self.raw(),
                context,
                TIME_SCALE,
                frame_arrived,
                format_changed,
                capture_failed,
            )
        });
        self.events = Some(events);
        started
    }

    fn restart_video(
        &mut self,
        mode: &'static DisplayMode,
        format: RawVideoFormat,
    ) -> Result<(), G2gError> {
        let pixel_format = pixel_format_code(format)?;
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_restart_video_input(self.raw(), mode.fourcc, pixel_format) })
    }

    fn hardware_time_ns(&mut self) -> Result<u64, G2gError> {
        let mut time = 0;
        // SAFETY: the handle is open, and `time` is a valid place for the reading
        check(unsafe { g2g_decklink_input_hardware_time(self.raw(), TIME_SCALE, &mut time) })?;
        u64::try_from(time).map_err(|_| G2gError::Hardware(HardwareError::Other))
    }
}
