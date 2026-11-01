// libasound is loaded when the first PCM or hint list opens, so a binary built
// with the ALSA elements still starts on a machine without it.

use alloc::ffi::CString;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CStr};

use g2g_core::{G2gError, HardwareError};

pub(crate) const LIBASOUND: &str = "libasound.so.2";

crate::shared_library::shared_library! {
    struct LibAsound = LIBASOUND;
    when missing return unreachable!("Pcm::open and pcm_hints load libasound before any other call");
    fn snd_pcm_open = "snd_pcm_open"(pcm: *mut *mut c_void, name: *const c_char, stream: c_int, mode: c_int) -> c_int;
    fn snd_pcm_close = "snd_pcm_close"(pcm: *mut c_void) -> c_int;
    fn snd_pcm_prepare = "snd_pcm_prepare"(pcm: *mut c_void) -> c_int;
    #[cfg(feature = "alsa-src")]
    fn snd_pcm_start = "snd_pcm_start"(pcm: *mut c_void) -> c_int;
    #[cfg(feature = "alsa-src")]
    fn snd_pcm_drop = "snd_pcm_drop"(pcm: *mut c_void) -> c_int;
    #[cfg(feature = "alsa-sink")]
    fn snd_pcm_drain = "snd_pcm_drain"(pcm: *mut c_void) -> c_int;
    fn snd_pcm_recover = "snd_pcm_recover"(pcm: *mut c_void, error: c_int, silent: c_int) -> c_int;
    fn snd_pcm_delay = "snd_pcm_delay"(pcm: *mut c_void, delay: *mut c_long) -> c_int;
    #[cfg(feature = "alsa-sink")]
    fn snd_pcm_writei = "snd_pcm_writei"(pcm: *mut c_void, buffer: *const c_void, frames: c_ulong) -> c_long;
    #[cfg(feature = "alsa-src")]
    fn snd_pcm_readi = "snd_pcm_readi"(pcm: *mut c_void, buffer: *mut c_void, frames: c_ulong) -> c_long;
    fn snd_pcm_get_chmap = "snd_pcm_get_chmap"(pcm: *mut c_void) -> *mut c_uint;
    fn snd_pcm_hw_params = "snd_pcm_hw_params"(pcm: *mut c_void, params: *mut c_void) -> c_int;
    fn snd_pcm_hw_params_malloc = "snd_pcm_hw_params_malloc"(params: *mut *mut c_void) -> c_int;
    fn snd_pcm_hw_params_free = "snd_pcm_hw_params_free"(params: *mut c_void) -> ();
    fn snd_pcm_hw_params_any = "snd_pcm_hw_params_any"(pcm: *mut c_void, params: *mut c_void) -> c_int;
    fn snd_pcm_hw_params_set_access = "snd_pcm_hw_params_set_access"(pcm: *mut c_void, params: *mut c_void, access: c_int) -> c_int;
    fn snd_pcm_hw_params_set_format = "snd_pcm_hw_params_set_format"(pcm: *mut c_void, params: *mut c_void, format: c_int) -> c_int;
    fn snd_pcm_hw_params_test_format = "snd_pcm_hw_params_test_format"(pcm: *mut c_void, params: *mut c_void, format: c_int) -> c_int;
    fn snd_pcm_hw_params_set_channels = "snd_pcm_hw_params_set_channels"(pcm: *mut c_void, params: *mut c_void, channels: c_uint) -> c_int;
    fn snd_pcm_hw_params_set_rate = "snd_pcm_hw_params_set_rate"(pcm: *mut c_void, params: *mut c_void, rate: c_uint, direction: c_int) -> c_int;
    fn snd_pcm_hw_params_set_period_time_near = "snd_pcm_hw_params_set_period_time_near"(pcm: *mut c_void, params: *mut c_void, microseconds: *mut c_uint, direction: *mut c_int) -> c_int;
    fn snd_pcm_hw_params_set_buffer_time_near = "snd_pcm_hw_params_set_buffer_time_near"(pcm: *mut c_void, params: *mut c_void, microseconds: *mut c_uint, direction: *mut c_int) -> c_int;
    #[cfg(feature = "alsa-src")]
    fn snd_pcm_hw_params_get_period_size = "snd_pcm_hw_params_get_period_size"(params: *const c_void, frames: *mut c_ulong, direction: *mut c_int) -> c_int;
    fn snd_pcm_hw_params_get_rate_min = "snd_pcm_hw_params_get_rate_min"(params: *const c_void, rate: *mut c_uint, direction: *mut c_int) -> c_int;
    fn snd_pcm_hw_params_get_rate_max = "snd_pcm_hw_params_get_rate_max"(params: *const c_void, rate: *mut c_uint, direction: *mut c_int) -> c_int;
    fn snd_pcm_hw_params_get_channels_min = "snd_pcm_hw_params_get_channels_min"(params: *const c_void, channels: *mut c_uint) -> c_int;
    fn snd_pcm_hw_params_get_channels_max = "snd_pcm_hw_params_get_channels_max"(params: *const c_void, channels: *mut c_uint) -> c_int;
    fn snd_device_name_hint = "snd_device_name_hint"(card: c_int, interface: *const c_char, hints: *mut *mut *mut c_void) -> c_int;
    fn snd_device_name_get_hint = "snd_device_name_get_hint"(hint: *const c_void, id: *const c_char) -> *mut c_char;
    fn snd_device_name_free_hint = "snd_device_name_free_hint"(hints: *mut *mut c_void) -> c_int;
}

// asound.h values
const SND_PCM_STREAM_PLAYBACK: c_int = 0;
const SND_PCM_STREAM_CAPTURE: c_int = 1;
const SND_PCM_NONBLOCK: c_int = 1;
const SND_PCM_ACCESS_RW_INTERLEAVED: c_int = 3;
// errno ENOSYS, negated as libasound reports errors
const LIBRARY_MISSING: i32 = -38;

pub(crate) fn require_library() -> Result<(), G2gError> {
    LibAsound::get()
        .map(|_| ())
        .ok_or(G2gError::Hardware(HardwareError::LibraryNotFound(
            LIBASOUND,
        )))
}

#[cfg(feature = "alsa-sink")]
pub(crate) fn library_installed() -> Result<(), String> {
    require_library().map_err(|_| alloc::format!("{LIBASOUND} is not installed"))
}

fn check(code: c_int) -> Result<(), i32> {
    if code < 0 {
        Err(code)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    Playback,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Format(c_int);

impl Format {
    pub(crate) const U8: Format = Format(1);
    pub(crate) const S16_LE: Format = Format(2);
    pub(crate) const S32_LE: Format = Format(10);
    pub(crate) const FLOAT_LE: Format = Format(14);
    pub(crate) const S24_3LE: Format = Format(32);
}

// asound.h `snd_pcm_chmap_position` values
pub(crate) mod channel_position {
    pub(crate) const MONO: u32 = 2;
    pub(crate) const FL: u32 = 3;
    pub(crate) const FR: u32 = 4;
    pub(crate) const RL: u32 = 5;
    pub(crate) const RR: u32 = 6;
    pub(crate) const FC: u32 = 7;
    pub(crate) const LFE: u32 = 8;
    pub(crate) const SL: u32 = 9;
    pub(crate) const SR: u32 = 10;
    pub(crate) const RC: u32 = 11;
    pub(crate) const FLC: u32 = 12;
    pub(crate) const FRC: u32 = 13;
}

#[derive(Debug)]
pub(crate) struct Pcm(*mut c_void);

// SAFETY: a snd_pcm_t has no thread affinity, and each Pcm is used from one thread at a time.
unsafe impl Send for Pcm {}

impl Pcm {
    pub(crate) fn open(name: &str, direction: Direction, nonblocking: bool) -> Result<Pcm, i32> {
        LibAsound::get().ok_or(LIBRARY_MISSING)?;
        let name = CString::new(name).map_err(|_| -22)?;
        let stream = match direction {
            Direction::Playback => SND_PCM_STREAM_PLAYBACK,
            Direction::Capture => SND_PCM_STREAM_CAPTURE,
        };
        let mode = if nonblocking { SND_PCM_NONBLOCK } else { 0 };
        let mut pcm = core::ptr::null_mut();
        // SAFETY: `pcm` and `name` are valid for the call.
        check(unsafe { snd_pcm_open(&mut pcm, name.as_ptr(), stream, mode) })?;
        Ok(Pcm(pcm))
    }

    pub(crate) fn hardware_parameters(&self) -> Result<HwParams<'_>, i32> {
        let mut params = core::ptr::null_mut();
        // SAFETY: `params` is valid for the call.
        check(unsafe { snd_pcm_hw_params_malloc(&mut params) })?;
        let hardware = HwParams { pcm: self, params };
        // SAFETY: both handles are open.
        check(unsafe { snd_pcm_hw_params_any(self.0, hardware.params) })?;
        Ok(hardware)
    }

    pub(crate) fn apply(&self, hardware: &HwParams<'_>) -> Result<(), i32> {
        // SAFETY: both handles are open.
        check(unsafe { snd_pcm_hw_params(self.0, hardware.params) })
    }

    pub(crate) fn prepare(&self) -> Result<(), i32> {
        // SAFETY: the handle is open.
        check(unsafe { snd_pcm_prepare(self.0) })
    }

    #[cfg(feature = "alsa-src")]
    pub(crate) fn start(&self) -> Result<(), i32> {
        // SAFETY: the handle is open.
        check(unsafe { snd_pcm_start(self.0) })
    }

    #[cfg(feature = "alsa-src")]
    pub(crate) fn drop_pending(&self) -> Result<(), i32> {
        // SAFETY: the handle is open.
        check(unsafe { snd_pcm_drop(self.0) })
    }

    #[cfg(feature = "alsa-sink")]
    pub(crate) fn drain(&self) -> Result<(), i32> {
        // SAFETY: the handle is open.
        check(unsafe { snd_pcm_drain(self.0) })
    }

    pub(crate) fn recover(&self, error: i32) -> Result<(), i32> {
        // SAFETY: the handle is open.
        check(unsafe { snd_pcm_recover(self.0, error, 1) })
    }

    pub(crate) fn delay(&self) -> Result<c_long, i32> {
        let mut delay: c_long = 0;
        // SAFETY: the handle is open and `delay` is valid for the call.
        check(unsafe { snd_pcm_delay(self.0, &mut delay) })?;
        Ok(delay)
    }

    #[cfg(feature = "alsa-sink")]
    pub(crate) fn write_interleaved(&self, bytes: &[u8], frames: usize) -> Result<usize, i32> {
        // SAFETY: `bytes` holds `frames` whole frames of the opened format.
        let written = unsafe { snd_pcm_writei(self.0, bytes.as_ptr().cast(), frames as c_ulong) };
        transferred(written)
    }

    #[cfg(feature = "alsa-src")]
    pub(crate) fn read_interleaved(&self, buffer: &mut [u8], frames: usize) -> Result<usize, i32> {
        // SAFETY: `buffer` has room for `frames` whole frames of the opened format.
        let read = unsafe { snd_pcm_readi(self.0, buffer.as_mut_ptr().cast(), frames as c_ulong) };
        transferred(read)
    }

    pub(crate) fn channel_map(&self) -> Option<Vec<u32>> {
        // SAFETY: the handle is open.
        let map = unsafe { snd_pcm_get_chmap(self.0) };
        if map.is_null() {
            return None;
        }
        // SAFETY: an snd_pcm_chmap_t is a channel count followed by that many positions.
        let positions = unsafe {
            let channels = *map as usize;
            core::slice::from_raw_parts(map.add(1), channels).to_vec()
        };
        // SAFETY: libasound allocates the map with malloc and leaves it to the caller.
        unsafe { libc::free(map.cast()) };
        Some(positions)
    }
}

fn transferred(result: c_long) -> Result<usize, i32> {
    if result < 0 {
        Err(result as i32)
    } else {
        Ok(result as usize)
    }
}

impl Drop for Pcm {
    fn drop(&mut self) {
        // SAFETY: the handle is open and closed only here.
        unsafe { snd_pcm_close(self.0) };
    }
}

#[derive(Debug)]
pub(crate) struct HwParams<'pcm> {
    pcm: &'pcm Pcm,
    params: *mut c_void,
}

impl HwParams<'_> {
    pub(crate) fn set_access_interleaved(&self) -> Result<(), i32> {
        // SAFETY: both handles are open.
        check(unsafe {
            snd_pcm_hw_params_set_access(self.pcm.0, self.params, SND_PCM_ACCESS_RW_INTERLEAVED)
        })
    }

    pub(crate) fn set_format(&self, format: Format) -> Result<(), i32> {
        // SAFETY: both handles are open.
        check(unsafe { snd_pcm_hw_params_set_format(self.pcm.0, self.params, format.0) })
    }

    pub(crate) fn supports_format(&self, format: Format) -> bool {
        // SAFETY: both handles are open.
        unsafe { snd_pcm_hw_params_test_format(self.pcm.0, self.params, format.0) == 0 }
    }

    pub(crate) fn set_channels(&self, channels: u32) -> Result<(), i32> {
        // SAFETY: both handles are open.
        check(unsafe { snd_pcm_hw_params_set_channels(self.pcm.0, self.params, channels) })
    }

    pub(crate) fn set_rate(&self, rate: u32) -> Result<(), i32> {
        // SAFETY: both handles are open.
        check(unsafe { snd_pcm_hw_params_set_rate(self.pcm.0, self.params, rate, 0) })
    }

    pub(crate) fn set_period_time_near(&self, microseconds: u32) -> Result<(), i32> {
        let (mut microseconds, mut direction) = (microseconds, 0);
        // SAFETY: both handles are open and the out-parameters are valid for the call.
        check(unsafe {
            snd_pcm_hw_params_set_period_time_near(
                self.pcm.0,
                self.params,
                &mut microseconds,
                &mut direction,
            )
        })
    }

    pub(crate) fn set_buffer_time_near(&self, microseconds: u32) -> Result<(), i32> {
        let (mut microseconds, mut direction) = (microseconds, 0);
        // SAFETY: both handles are open and the out-parameters are valid for the call.
        check(unsafe {
            snd_pcm_hw_params_set_buffer_time_near(
                self.pcm.0,
                self.params,
                &mut microseconds,
                &mut direction,
            )
        })
    }

    #[cfg(feature = "alsa-src")]
    pub(crate) fn period_size(&self) -> Result<c_ulong, i32> {
        let (mut frames, mut direction): (c_ulong, c_int) = (0, 0);
        // SAFETY: the parameters are allocated and the out-parameters are valid for the call.
        check(unsafe {
            snd_pcm_hw_params_get_period_size(self.params, &mut frames, &mut direction)
        })?;
        Ok(frames)
    }

    pub(crate) fn rate_range(&self) -> Result<(u32, u32), i32> {
        let (mut minimum, mut maximum, mut direction) = (0, 0, 0);
        // SAFETY: the parameters are allocated and the out-parameters are valid for the call.
        unsafe {
            check(snd_pcm_hw_params_get_rate_min(
                self.params,
                &mut minimum,
                &mut direction,
            ))?;
            check(snd_pcm_hw_params_get_rate_max(
                self.params,
                &mut maximum,
                &mut direction,
            ))?;
        }
        Ok((minimum, maximum))
    }

    pub(crate) fn channel_range(&self) -> Result<(u32, u32), i32> {
        let (mut minimum, mut maximum) = (0, 0);
        // SAFETY: the parameters are allocated and the out-parameters are valid for the call.
        unsafe {
            check(snd_pcm_hw_params_get_channels_min(
                self.params,
                &mut minimum,
            ))?;
            check(snd_pcm_hw_params_get_channels_max(
                self.params,
                &mut maximum,
            ))?;
        }
        Ok((minimum, maximum))
    }
}

impl Drop for HwParams<'_> {
    fn drop(&mut self) {
        // SAFETY: the parameters were allocated by snd_pcm_hw_params_malloc and are freed only here.
        unsafe { snd_pcm_hw_params_free(self.params) };
    }
}

#[derive(Debug)]
pub(crate) struct Hint {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) direction: Option<Direction>,
}

pub(crate) fn pcm_hints() -> Result<Vec<Hint>, i32> {
    LibAsound::get().ok_or(LIBRARY_MISSING)?;
    let mut hints: *mut *mut c_void = core::ptr::null_mut();
    // SAFETY: -1 asks for every card and `hints` is valid for the call.
    check(unsafe { snd_device_name_hint(-1, c"pcm".as_ptr(), &mut hints) })?;
    let mut listed = Vec::new();
    let mut entry = hints;
    // SAFETY: libasound ends the hint array with a null entry.
    while let Some(hint) = unsafe { entry.as_ref() }
        .copied()
        .filter(|hint| !hint.is_null())
    {
        let direction = hint_field(hint, c"IOID").and_then(|direction| match direction.as_str() {
            "Input" => Some(Direction::Capture),
            "Output" => Some(Direction::Playback),
            _ => None,
        });
        listed.push(Hint {
            name: hint_field(hint, c"NAME"),
            description: hint_field(hint, c"DESC"),
            direction,
        });
        // SAFETY: the array is null terminated and this entry was not null.
        entry = unsafe { entry.add(1) };
    }
    // SAFETY: `hints` came from snd_device_name_hint and is freed only here.
    unsafe { snd_device_name_free_hint(hints) };
    Ok(listed)
}

fn hint_field(hint: *mut c_void, id: &CStr) -> Option<String> {
    // SAFETY: `hint` is an entry of a live hint array.
    let value = unsafe { snd_device_name_get_hint(hint, id.as_ptr()) };
    if value.is_null() {
        return None;
    }
    // SAFETY: libasound returns a malloc'd, null-terminated string the caller frees.
    let text = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: as above.
    unsafe { libc::free(value.cast()) };
    Some(text)
}
