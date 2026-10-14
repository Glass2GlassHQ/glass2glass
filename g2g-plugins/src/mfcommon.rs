#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
use core::mem::ManuallyDrop;

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
use windows::Win32::Media::MediaFoundation::{
    IMFSample, IMFTransform, MFCreateMemoryBuffer, MFCreateSample, MFT_MESSAGE_COMMAND_DRAIN,
    MFT_OUTPUT_DATA_BUFFER, MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_E_NOTACCEPTING,
    MF_E_TRANSFORM_NEED_MORE_INPUT, MF_E_TRANSFORM_STREAM_CHANGE,
};
#[cfg(feature = "mf-video-src")]
use windows::Win32::Media::MediaFoundation::{MFShutdown, MFStartup, MFSTARTUP_FULL, MF_VERSION};
#[cfg(feature = "mf-video-src")]
use windows::Win32::System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED};

use g2g_core::{G2gError, HardwareError};

// MF sample times and durations count 100 ns units.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
const NANOS_PER_MF_TICK: u64 = 100;

// Drain-and-retry rounds before an MFT that keeps refusing input is an error.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
const MAX_INPUT_RETRIES: u32 = 64;

#[cfg(any(feature = "mf-decode", feature = "mf-video-src"))]
const LOW_WORD_MASK: u64 = 0xFFFF_FFFF;

pub(crate) fn mf_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode"))]
pub(crate) fn mf_ticks_to_ns(ticks: i64) -> u64 {
    (ticks.max(0) as u64).saturating_mul(NANOS_PER_MF_TICK)
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn ns_to_mf_ticks(ns: u64) -> i64 {
    (ns / NANOS_PER_MF_TICK) as i64
}

// MF packs a frame size as `(width << 32) | height` and a rate as `(num << 32) | den`.
#[cfg(any(feature = "mf-decode", feature = "mf-encode"))]
pub(crate) fn pack_u32_pair(high: u32, low: u32) -> u64 {
    ((high as u64) << 32) | (low as u64)
}

#[cfg(any(feature = "mf-decode", feature = "mf-video-src"))]
pub(crate) fn unpack_u32_pair(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & LOW_WORD_MASK) as u32)
}

// Runs `work` between CoInitializeEx + MFStartup and MFShutdown + CoUninitialize.
#[cfg(feature = "mf-video-src")]
pub(crate) fn with_media_foundation<R>(
    work: impl FnOnce() -> Result<R, G2gError>,
) -> Result<R, G2gError> {
    // SAFETY: COM + MF init on the calling thread, balanced before return.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let result = MFStartup(MF_VERSION, MFSTARTUP_FULL)
            .map_err(mf_err)
            .and_then(|()| work());
        let _ = MFShutdown();
        CoUninitialize();
        result
    }
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn output_provides_samples(transform: &IMFTransform) -> Result<bool, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    Ok(info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0)
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn make_input_sample(data: &[u8], pts_ns: u64) -> Result<IMFSample, G2gError> {
    let len = data.len() as u32;
    // SAFETY: buffer allocation, locked copy, and sample assembly on the
    // owning thread. `ptr` is valid for `len` bytes between Lock and Unlock.
    unsafe {
        let buffer = MFCreateMemoryBuffer(len).map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        buffer.Lock(&mut ptr, None, None).map_err(mf_err)?;
        core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        buffer.Unlock().map_err(mf_err)?;
        buffer.SetCurrentLength(len).map_err(mf_err)?;

        let sample = MFCreateSample().map_err(mf_err)?;
        sample.AddBuffer(&buffer).map_err(mf_err)?;
        sample
            .SetSampleTime(ns_to_mf_ticks(pts_ns))
            .map_err(mf_err)?;
        Ok(sample)
    }
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode"))]
pub(crate) fn sample_time_ns(sample: &IMFSample) -> u64 {
    // SAFETY: reading an attribute off a valid sample on the owning thread.
    unsafe { sample.GetSampleTime() }
        .map(mf_ticks_to_ns)
        .unwrap_or(0)
}

// Calls `read` on the sample's bytes (current length, not capacity) while the buffer is locked.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn read_sample_bytes<R>(
    sample: &IMFSample,
    read: impl FnOnce(&[u8]) -> R,
) -> Result<R, G2gError> {
    // SAFETY: contiguous-buffer access on the owning thread; `ptr` is valid
    // for `len` bytes between Lock and Unlock, where `read` runs.
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer().map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        let mut len: u32 = 0;
        buffer
            .Lock(&mut ptr, None, Some(&mut len))
            .map_err(mf_err)?;
        let result = read(core::slice::from_raw_parts(ptr, len as usize));
        buffer.Unlock().map_err(mf_err)?;
        Ok(result)
    }
}

// `Ok(false)` when the MFT must release outputs before it takes more input.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn process_input(
    transform: &IMFTransform,
    sample: &IMFSample,
) -> Result<bool, G2gError> {
    // SAFETY: COM call on the owning thread; `sample` is a valid IMFSample.
    match unsafe { transform.ProcessInput(0, sample, 0) } {
        Ok(()) => Ok(true),
        Err(e) if e.code() == MF_E_NOTACCEPTING => Ok(false),
        Err(e) => Err(mf_err(e)),
    }
}

// Feeds `sample`, draining between refusals, then drains what the MFT released.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn feed_sample<E>(
    element: &mut E,
    sample: &IMFSample,
    transform: impl Fn(&E) -> Result<&IMFTransform, G2gError>,
    mut drain: impl FnMut(&mut E) -> Result<(), G2gError>,
) -> Result<(), G2gError> {
    let mut refusals = 0u32;
    while !process_input(transform(element)?, sample)? {
        drain(element)?;
        refusals += 1;
        if refusals > MAX_INPUT_RETRIES {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
    }
    drain(element)
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn send_drain_command(transform: &IMFTransform) -> Result<(), G2gError> {
    // SAFETY: drain message on the owning thread.
    unsafe { transform.ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0) }.map_err(mf_err)
}

#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
#[derive(Debug)]
pub(crate) enum OutputPull {
    Sample(IMFSample),
    NeedInput,
    StreamChange,
}

// One `ProcessOutput`. Preallocates an `out_size` sample unless the MFT provides its own.
#[cfg(any(feature = "mf-decode", feature = "mf-encode", feature = "mf-aac"))]
pub(crate) fn process_output(
    transform: &IMFTransform,
    provides_samples: bool,
    out_size: u32,
) -> Result<OutputPull, G2gError> {
    // SAFETY: every call runs on the owning COM thread. The refs in the FFI
    // struct are reclaimed right after ProcessOutput regardless of its result.
    unsafe {
        let preallocated = if provides_samples {
            None
        } else {
            let buffer = MFCreateMemoryBuffer(out_size).map_err(mf_err)?;
            let sample = MFCreateSample().map_err(mf_err)?;
            sample.AddBuffer(&buffer).map_err(mf_err)?;
            Some(sample)
        };

        let mut out = [MFT_OUTPUT_DATA_BUFFER {
            dwStreamID: 0,
            pSample: ManuallyDrop::new(preallocated),
            dwStatus: 0,
            pEvents: ManuallyDrop::new(None),
        }];
        let mut status = 0u32;
        let r = transform.ProcessOutput(0, &mut out, &mut status);

        let out_sample = ManuallyDrop::into_inner(core::mem::replace(
            &mut out[0].pSample,
            ManuallyDrop::new(None),
        ));
        drop(ManuallyDrop::into_inner(core::mem::replace(
            &mut out[0].pEvents,
            ManuallyDrop::new(None),
        )));

        match r {
            Ok(()) => out_sample
                .map(OutputPull::Sample)
                .ok_or(G2gError::Hardware(HardwareError::Other)),
            Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => Ok(OutputPull::NeedInput),
            Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => Ok(OutputPull::StreamChange),
            Err(e) => Err(mf_err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "mf-decode")]
    #[test]
    fn u32_pair_round_trips() {
        assert_eq!(unpack_u32_pair(pack_u32_pair(1920, 1080)), (1920, 1080));
        assert_eq!(pack_u32_pair(1280, 720), (1280u64 << 32) | 720);
    }

    #[cfg(any(feature = "mf-decode", feature = "mf-encode"))]
    #[test]
    fn mf_ticks_are_100_ns() {
        assert_eq!(ns_to_mf_ticks(1_000_000_000), 10_000_000);
        assert_eq!(mf_ticks_to_ns(10_000_000), 1_000_000_000);
        assert_eq!(mf_ticks_to_ns(-5), 0);
    }
}
