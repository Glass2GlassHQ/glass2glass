use core::ptr::NonNull;

use std::sync::Arc;

use alloc::boxed::Box;

use g2g_core::log::Target;
use g2g_core::{
    g2g_error, AudioFormat, Caps, Colorimetry, Dim, G2gError, HardwareError, Interlace, Rate,
    RawVideoFormat,
};

pub const DECKLINK_LIBRARY: &str = "libDeckLinkAPI.so";

pub(crate) const NANOS_PER_SECOND: u64 = 1_000_000_000;

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

pub fn display_mode(name: &str) -> Option<&'static DisplayMode> {
    DISPLAY_MODES.iter().find(|mode| mode.name == name)
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

// the board's pixel format code for each raw format the sink takes
pub(crate) const PIXEL_FORMATS: [(RawVideoFormat, &[u8; 4]); 4] = [
    (RawVideoFormat::Uyvy, b"2vuy"),
    (RawVideoFormat::Bgra8, b"BGRA"),
    (RawVideoFormat::V210, b"v210"),
    (RawVideoFormat::R210, b"r210"),
];

pub fn video_format_supported(format: RawVideoFormat) -> bool {
    PIXEL_FORMATS
        .iter()
        .any(|(supported, _)| *supported == format)
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
}

// one output device, with times in nanoseconds on the board's hardware clock
pub trait DeckLinkOutput: Send {
    fn enable_video(&mut self, mode: &'static DisplayMode) -> Result<(), G2gError>;
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

// status codes of g2g_decklink_open, matching the enum in csrc/decklink_shim.cpp
const OPENED: i32 = 0;
const LIBRARY_MISSING: i32 = 1;
const NO_DEVICE: i32 = 2;
const NO_OUTPUT: i32 = 3;
const NO_ITERATOR: i32 = 4;

extern "C" {
    fn g2g_decklink_open(device_number: u32, opened: *mut *mut RawOutput) -> i32;
    fn g2g_decklink_close(handle: *mut RawOutput);
    fn g2g_decklink_enable_video(handle: *mut RawOutput, display_mode: u32) -> i32;
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
        match status {
            OPENED => {
                let handle =
                    NonNull::new(handle).ok_or(G2gError::Hardware(HardwareError::Other))?;
                Ok(Box::new(SdkOutput { handle }))
            }
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
    fn enable_video(&mut self, mode: &'static DisplayMode) -> Result<(), G2gError> {
        // SAFETY: the handle is open for the life of `self`
        check(unsafe { g2g_decklink_enable_video(self.raw(), mode.fourcc) })
    }

    fn schedule_video(&mut self, frame: &ScheduledVideoFrame<'_>) -> Result<(), G2gError> {
        let pixel_format = PIXEL_FORMATS
            .iter()
            .find(|(format, _)| *format == frame.format)
            .map(|(_, code)| u32::from_be_bytes(**code))
            .ok_or(G2gError::CapsMismatch)?;
        let row_bytes = sdk_dimension(frame.row_bytes)?;
        let height = sdk_dimension(frame.height)?;
        let frame_bytes = frame.row_bytes as usize * frame.height as usize;
        if frame.pixels.len() < frame_bytes {
            return Err(G2gError::CapsMismatch);
        }
        // SAFETY: the handle is open, and the shim reads row_bytes * height bytes, which `pixels` holds
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
