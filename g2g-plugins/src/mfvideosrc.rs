//! Windows Media Foundation camera capture source. The video sibling of
//! [`WasapiSrc`](crate::wasapisrc::WasapiSrc): enumerates video capture devices
//! and drains frames from one via an `IMFSourceReader`, emitting `DataFrame`s in
//! system memory (NV12 or YUY2), so a live webcam feeds a g2g pipeline the same
//! way `V4l2Src` does on Linux.
//!
//! Pipeline shape: `MfVideoSrc -> VideoConvert -> sink` (NV12 is already a
//! downstream-friendly format; YUY2 needs the M89 unpack like `V4l2Src`).
//!
//! ## Threading
//!
//! Media Foundation is COM (MTA) and the synchronous Source Reader is driven
//! from one thread, so (like `WasapiSrc` / `MfDecode`) capture runs on a
//! dedicated worker spun up in `run`; captured frames cross to the async `run`
//! loop over a channel. A short COM/MF probe in `intercept_caps` reads the
//! device geometry up front so downstream solves against real caps.
//!
//! ## Status
//!
//! Authored against the `windows` 0.62 Media Foundation surface and the
//! `WasapiSrc` COM contract, but Windows-only and therefore not built or run on
//! the Linux dev host; it owes a first compile + a manual camera smoke test on
//! Windows (the `mf-decode` / `wasapi-src` elements share the same unverified-on-
//! Linux situation, see AGENTS.md).

use core::future::Future;
use core::pin::Pin;

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use tokio::sync::mpsc;

use windows::core::GUID;
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFMediaSource, IMFSourceReader, MFCreateAttributes,
    MFCreateSourceReaderFromMediaSource, MFCreateMediaType, MFEnumDeviceSources, MFShutdown,
    MFStartup, MFMediaType_Video, MFVideoFormat_NV12, MFVideoFormat_YUY2,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
    MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READER_FIRST_VIDEO_STREAM, MFSTARTUP_FULL, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError,
    LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat,
};

/// First video stream index for the Source Reader, as a `u32` (the constant is
/// the signed `MF_SOURCE_READER_FIRST_VIDEO_STREAM`).
fn first_video_stream() -> u32 {
    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32
}

/// The pixel format requested from the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfPixelFormat {
    /// Planar 4:2:0 (the decode/display-friendly default).
    Nv12,
    /// Packed 4:2:2 (the common cheap-webcam output; needs VideoConvert unpack).
    Yuy2,
}

impl MfPixelFormat {
    fn subtype(self) -> GUID {
        match self {
            MfPixelFormat::Nv12 => MFVideoFormat_NV12,
            MfPixelFormat::Yuy2 => MFVideoFormat_YUY2,
        }
    }

    fn raw_format(self) -> RawVideoFormat {
        match self {
            MfPixelFormat::Nv12 => RawVideoFormat::Nv12,
            MfPixelFormat::Yuy2 => RawVideoFormat::Yuyv,
        }
    }

    /// Bytes per frame for `width`x`height`.
    fn frame_bytes(self, width: u32, height: u32) -> usize {
        let px = (width as usize) * (height as usize);
        match self {
            MfPixelFormat::Nv12 => px * 3 / 2,
            MfPixelFormat::Yuy2 => px * 2,
        }
    }
}

/// Geometry probed from the device's chosen media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoConfig {
    format: MfPixelFormat,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
}

impl VideoConfig {
    fn fps(&self) -> u32 {
        self.fps_num.checked_div(self.fps_den).unwrap_or(0)
    }
}

#[derive(Debug)]
pub struct MfVideoSrc {
    device_index: u32,
    format: MfPixelFormat,
    /// 0 = run until error or downstream shutdown; else stop after N frames.
    frame_limit: u64,
    config: Option<VideoConfig>,
    configured: bool,
}

impl Default for MfVideoSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl MfVideoSrc {
    /// Capture NV12 from the first enumerated video device.
    pub fn new() -> Self {
        Self {
            device_index: 0,
            format: MfPixelFormat::Nv12,
            frame_limit: 0,
            config: None,
            configured: false,
        }
    }

    /// Select which enumerated capture device to open (0 = first).
    pub fn with_device_index(mut self, index: u32) -> Self {
        self.device_index = index;
        self
    }

    /// Request a pixel format (NV12 default, or YUY2 for cheaper webcams).
    pub fn with_format(mut self, format: MfPixelFormat) -> Self {
        self.format = format;
        self
    }

    /// Stop after `n` frames and emit EOS. Without it the source runs until an
    /// error or until downstream drops.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.config.is_none() {
            self.config = Some(probe_device(self.device_index, self.format)?);
        }
        let c = self.config.expect("just probed");
        Ok(Caps::RawVideo {
            format: c.format.raw_format(),
            width: Dim::Fixed(c.width),
            height: Dim::Fixed(c.height),
            framerate: Rate::Fixed(c.fps() << 16),
        })
    }
}

impl SourceLoop for MfVideoSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.probe())
    }

    /// Produces the geometry the device settles on during the probe, so a chain
    /// built on the camera takes the native arc-consistency path (like `V4l2Src`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.probe()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.config.is_none() {
            return Err(G2gError::NotConfigured);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Live source: one frame period of latency so the sink keeps a frame in hand.
    fn latency(&self) -> LatencyReport {
        let fps = self.config.map(|c| c.fps()).unwrap_or(30);
        let period_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let config = self.config.ok_or(G2gError::NotConfigured)?;
            let index = self.device_index;
            let limit = self.frame_limit;

            // Worker captures and streams frame payloads here; a ready signal
            // reports whether the device opened.
            let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<Vec<u8>>();
            let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), i32>>(1);
            let worker = thread::Builder::new()
                .name(alloc::string::String::from("g2g-mfvideosrc"))
                .spawn(move || capture_worker(index, config, limit, frame_tx, ready_tx))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            match ready_rx.recv_timeout(Duration::from_secs(5)) {
                Ok(Ok(())) => {}
                Ok(Err(code)) => {
                    let _ = worker.join();
                    return Err(G2gError::Hardware(HardwareError::MediaFoundation(code)));
                }
                Err(_) => {
                    let _ = worker.join();
                    return Err(G2gError::Hardware(HardwareError::MediaFoundation(0)));
                }
            }

            let fps = config.fps();
            let pts_step_ns = if fps > 0 { 1_000_000_000 / fps as u64 } else { 0 };
            let expected = config.format.frame_bytes(config.width, config.height);
            let mut seq = 0u64;
            let mut downstream_open = true;

            while let Some(bytes) = frame_rx.recv().await {
                // A short frame (driver hiccup) can't be consumed safely; skip it.
                if bytes.len() < expected {
                    continue;
                }
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let pts = seq * pts_step_ns;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: pts_step_ns,
                        capture_ns: pts,
                        arrival_ns,
                        keyframe: true, // raw frames are each independently presentable
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                seq += 1;
            }

            // Close the receiver so the worker's next send fails and it stops,
            // instead of capturing forever (frame_limit==0) and hanging join().
            drop(frame_rx);
            let _ = worker.join();
            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for MfVideoSrc {
    /// Produces raw video; a constructed instance fixes the geometry / rate
    /// during the probe. Both NV12 and YUY2 are advertised.
    fn pad_templates() -> Vec<PadTemplate> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(Vec::from([
            raw(RawVideoFormat::Nv12),
            raw(RawVideoFormat::Yuyv),
        ])))])
    }
}

// =================================================================
// COM/MF probe + capture worker
// =================================================================

fn mf_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

/// Open the device on a short-lived COM/MF thread, read the negotiated media
/// type, and map it to a [`VideoConfig`].
fn probe_device(index: u32, format: MfPixelFormat) -> Result<VideoConfig, G2gError> {
    let (tx, rx) = std_mpsc::sync_channel::<Result<VideoConfig, G2gError>>(1);
    thread::Builder::new()
        .name(alloc::string::String::from("g2g-mfvideosrc-probe"))
        .spawn(move || {
            // SAFETY: COM + MF init on this worker thread, balanced before exit.
            let result = unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                let r = MFStartup(MF_VERSION, MFSTARTUP_FULL)
                    .map_err(mf_err)
                    .and_then(|()| open_reader(index, format).map(|(_, cfg)| cfg));
                let _ = MFShutdown();
                CoUninitialize();
                r
            };
            let _ = tx.send(result);
        })
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    rx.recv_timeout(Duration::from_secs(5))
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?
}

/// Enumerate capture devices, activate device `index`, build a Source Reader,
/// pin the requested subtype, and read back the chosen geometry.
///
/// # Safety
/// Must run on a COM-initialised, MF-started thread.
unsafe fn open_reader(
    index: u32,
    format: MfPixelFormat,
) -> Result<(IMFSourceReader, VideoConfig), G2gError> {
    // SAFETY: MF object creation/queries on the owning thread.
    unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1).map_err(mf_err)?;
        let attrs = attrs.ok_or(G2gError::Hardware(HardwareError::MediaFoundation(0)))?;
        attrs
            .SetGUID(
                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
            )
            .map_err(mf_err)?;

        let mut devices: *mut Option<IMFActivate> = core::ptr::null_mut();
        let mut count: u32 = 0;
        MFEnumDeviceSources(&attrs, &mut devices, &mut count).map_err(mf_err)?;

        if index >= count {
            // Release whatever was returned before bailing.
            free_activates(devices, count);
            return Err(G2gError::Hardware(HardwareError::MediaFoundation(0)));
        }

        let activate = (*devices.add(index as usize))
            .clone()
            .ok_or(G2gError::Hardware(HardwareError::MediaFoundation(0)));
        let source: Result<IMFMediaSource, G2gError> =
            activate.and_then(|a| a.ActivateObject().map_err(mf_err));
        // Release the enumeration array regardless of the activation outcome.
        free_activates(devices, count);
        let source = source?;

        let reader = MFCreateSourceReaderFromMediaSource(&source, None).map_err(mf_err)?;

        // Pin the major type + requested subtype; the device picks the rest.
        let media_type = MFCreateMediaType().map_err(mf_err)?;
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf_err)?;
        media_type
            .SetGUID(&MF_MT_SUBTYPE, &format.subtype())
            .map_err(mf_err)?;
        let stream = first_video_stream();
        reader
            .SetCurrentMediaType(stream, None, &media_type)
            .map_err(mf_err)?;

        let current = reader.GetCurrentMediaType(stream).map_err(mf_err)?;
        // FRAME_SIZE packs (width << 32) | height; FRAME_RATE packs (num << 32) | den.
        let size = current.GetUINT64(&MF_MT_FRAME_SIZE).map_err(mf_err)?;
        let rate = current.GetUINT64(&MF_MT_FRAME_RATE).map_err(mf_err)?;
        let config = VideoConfig {
            format,
            width: (size >> 32) as u32,
            height: (size & 0xFFFF_FFFF) as u32,
            fps_num: (rate >> 32) as u32,
            fps_den: (rate & 0xFFFF_FFFF) as u32,
        };
        Ok((reader, config))
    }
}

/// Take + drop each `IMFActivate` in the enumeration array and free the array
/// memory allocated by `MFEnumDeviceSources`.
///
/// # Safety
/// `devices` must be the `count`-element array from `MFEnumDeviceSources`.
unsafe fn free_activates(devices: *mut Option<IMFActivate>, count: u32) {
    if devices.is_null() {
        return;
    }
    // SAFETY: each slot is a valid Option<IMFActivate>; taking drops its ref.
    unsafe {
        for i in 0..count as usize {
            let _ = (*devices.add(i)).take();
        }
        CoTaskMemFree(Some(devices.cast()));
    }
}

/// Capture worker: open the device, then pump frames to `frame_tx` until the
/// limit is reached, end-of-stream, or capture fails.
fn capture_worker(
    index: u32,
    config: VideoConfig,
    limit: u64,
    frame_tx: mpsc::UnboundedSender<Vec<u8>>,
    ready_tx: std_mpsc::SyncSender<Result<(), i32>>,
) {
    // SAFETY: COM + MF init on this worker thread, balanced below.
    let result = unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let r = MFStartup(MF_VERSION, MFSTARTUP_FULL)
            .map_err(mf_err)
            .and_then(|()| run_capture(index, config, limit, &frame_tx, &ready_tx));
        let _ = MFShutdown();
        CoUninitialize();
        r
    };
    if let Err(G2gError::Hardware(HardwareError::MediaFoundation(code))) = result {
        // If we never signalled ready, do so now so `run` stops waiting.
        let _ = ready_tx.try_send(Err(code));
    } else if result.is_err() {
        let _ = ready_tx.try_send(Err(0));
    }
}

/// # Safety
/// Must run on a COM-initialised, MF-started thread.
unsafe fn run_capture(
    index: u32,
    config: VideoConfig,
    limit: u64,
    frame_tx: &mpsc::UnboundedSender<Vec<u8>>,
    ready_tx: &std_mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), G2gError> {
    // SAFETY: reader setup + read loop on the owning thread.
    unsafe {
        let (reader, _cfg) = open_reader(index, config.format)?;
        let _ = ready_tx.try_send(Ok(()));

        let stream = first_video_stream();
        let mut emitted = 0u64;
        while limit == 0 || emitted < limit {
            let mut flags: u32 = 0;
            let mut timestamp: i64 = 0;
            let mut sample = None;
            reader
                .ReadSample(
                    stream,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
                .map_err(mf_err)?;

            if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                break;
            }
            let Some(sample) = sample else {
                // A null sample with no EOS flag is a gap (e.g. stream tick);
                // keep reading.
                continue;
            };

            // Flatten to one contiguous buffer, then copy it out under a lock.
            let buffer = sample.ConvertToContiguousBuffer().map_err(mf_err)?;
            let mut ptr: *mut u8 = core::ptr::null_mut();
            let mut len: u32 = 0;
            buffer
                .Lock(&mut ptr, None, Some(&mut len))
                .map_err(mf_err)?;
            let chunk = if ptr.is_null() {
                Vec::new()
            } else {
                core::slice::from_raw_parts(ptr, len as usize).to_vec()
            };
            buffer.Unlock().map_err(mf_err)?;

            if chunk.is_empty() {
                continue;
            }
            if frame_tx.send(chunk).is_err() {
                break; // consumer dropped
            }
            emitted += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = MfVideoSrc::new()
            .with_device_index(2)
            .with_format(MfPixelFormat::Yuy2)
            .with_frame_limit(10);
        assert_eq!(src.device_index, 2);
        assert_eq!(src.format, MfPixelFormat::Yuy2);
        assert_eq!(src.frame_limit, 10);
    }

    #[test]
    fn pixel_format_byte_sizes() {
        assert_eq!(MfPixelFormat::Nv12.frame_bytes(640, 480), 640 * 480 * 3 / 2);
        assert_eq!(MfPixelFormat::Yuy2.frame_bytes(640, 480), 640 * 480 * 2);
    }

    #[test]
    fn pixel_format_maps_to_raw() {
        assert_eq!(MfPixelFormat::Nv12.raw_format(), RawVideoFormat::Nv12);
        assert_eq!(MfPixelFormat::Yuy2.raw_format(), RawVideoFormat::Yuyv);
    }
}
