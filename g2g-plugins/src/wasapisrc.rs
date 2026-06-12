//! Windows WASAPI capture source (M32): the input mirror of [`WasapiSink`].
//! Captures interleaved PCM from the default audio capture endpoint (shared
//! mode) and emits `DataFrame`s, so a live microphone / line-in feeds a g2g
//! pipeline the same way `AudioTestSrc` feeds a synthetic tone.
//!
//! Caps come from the endpoint's mix format, probed during negotiation
//! (`intercept_caps`), so downstream solves against the device's real channel
//! count and rate. The format is reported as `PcmF32Le` (the usual shared-mode
//! mix format) or `PcmS16Le`.
//!
//! ## Threading
//!
//! WASAPI is COM and the capture client is driven from one thread, so capture
//! runs on a dedicated worker spun up in `run`. Captured buffers cross to the
//! async `run` loop over a channel; `run` stamps timing and pushes them. A
//! short COM-thread probe in `intercept_caps` reads the mix format up front.
//!
//! ## Scope
//!
//! Emits a fixed number of buffers then `Eos`, the bounded shape the test
//! sources use. A headless host (no capture endpoint) fails the probe loud,
//! so negotiation rejects the pipeline rather than hanging.

use core::future::Future;
use core::pin::Pin;

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::vec::Vec;

use tokio::sync::mpsc;

use windows::Win32::Media::Audio::{
    eCapture, eConsole, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_BUFFERFLAGS_SILENT, AUDCLNT_SHAREMODE_SHARED, WAVEFORMATEX,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Shared-mode endpoint buffer span (100-ns units), 200 ms.
const BUFFER_DURATION_HNS: i64 = 2_000_000;

/// The endpoint's PCM shape, probed from the mix format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AudioConfig {
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    block_align: usize,
}

#[derive(Debug)]
pub struct WasapiSrc {
    /// Number of captured buffers to emit before `Eos`.
    target_buffers: u64,
    config: Option<AudioConfig>,
    configured: bool,
}

impl WasapiSrc {
    /// Capture `target_buffers` buffers from the default endpoint, then end.
    pub fn new(target_buffers: u64) -> Self {
        Self {
            target_buffers,
            config: None,
            configured: false,
        }
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.config.is_none() {
            self.config = Some(probe_endpoint_format()?);
        }
        let c = self.config.expect("just probed");
        Ok(Caps::Audio {
            format: c.format,
            channels: c.channels,
            sample_rate: c.sample_rate,
        })
    }
}

impl SourceLoop for WasapiSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.probe())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.probe()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let config = self.config.ok_or(G2gError::NotConfigured)?;
            let target = self.target_buffers;

            // Worker captures from the endpoint and streams PCM chunks here;
            // a ready signal reports whether the endpoint opened.
            let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<Vec<u8>>();
            let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), ()>>(1);
            let worker = thread::Builder::new()
                .name(alloc::string::String::from("g2g-wasapisrc"))
                .spawn(move || capture_worker(config, target, chunk_tx, ready_tx))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            match ready_rx.recv_timeout(Duration::from_secs(5)) {
                Ok(Ok(())) => {}
                _ => {
                    let _ = worker.join();
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }

            let ns_per_frame = 1_000_000_000u64 / config.sample_rate as u64;
            let mut total_frames = 0u64;
            let mut sequence = 0u64;
            while let Some(bytes) = chunk_rx.recv().await {
                let frames = (bytes.len() / config.block_align) as u64;
                let pts_ns = total_frames * ns_per_frame;
                let duration_ns = frames * ns_per_frame;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns,
                        capture_ns: pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                    },
                    sequence,
                };
                sequence += 1;
                total_frames += frames;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            let _ = worker.join();
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

impl PadTemplates for WasapiSrc {
    /// PCM source pad. `Caps::Audio` has no open dims, so the template pins the
    /// common shapes per PCM format, as in `AudioTestSrc` / `WavSink`.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmF32Le),
            pcm(AudioFormat::PcmS16Le),
        ])))])
    }
}

// =================================================================
// COM-thread probe + capture worker
// =================================================================

/// Open the default capture endpoint, read its mix format, and map it to an
/// [`AudioConfig`]. Runs on a short-lived COM thread.
fn probe_endpoint_format() -> Result<AudioConfig, G2gError> {
    let (tx, rx) = std_mpsc::sync_channel::<Result<AudioConfig, G2gError>>(1);
    thread::Builder::new()
        .name(alloc::string::String::from("g2g-wasapisrc-probe"))
        .spawn(move || {
            // SAFETY: COM init + WASAPI queries on this worker thread, balanced
            // by CoUninitialize before it exits.
            let result = unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                let r = read_mix_format();
                CoUninitialize();
                r
            };
            let _ = tx.send(result);
        })
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    rx.recv_timeout(Duration::from_secs(5))
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?
}

/// Read the default capture endpoint's mix format into an [`AudioConfig`].
///
/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn read_mix_format() -> Result<AudioConfig, G2gError> {
    // SAFETY: WASAPI object creation/queries on the owning thread.
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eCapture, eConsole)
            .map_err(audio_err)?;
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(audio_err)?;
        let fmt_ptr = client.GetMixFormat().map_err(audio_err)?;
        let config = audio_config_from_format(&*fmt_ptr);
        CoTaskMemFree(Some(fmt_ptr.cast()));
        config
    }
}

/// Map a `WAVEFORMATEX` to an [`AudioConfig`]. 32-bit samples are reported as
/// float (the shared-mode mix format), 16-bit as signed PCM; other depths are
/// unsupported.
fn audio_config_from_format(fmt: &WAVEFORMATEX) -> Result<AudioConfig, G2gError> {
    let bits = fmt.wBitsPerSample;
    let tag = fmt.wFormatTag;
    let format = match (tag, bits) {
        (WAVE_FORMAT_PCM, 16) => AudioFormat::PcmS16Le,
        (WAVE_FORMAT_IEEE_FLOAT, 32) => AudioFormat::PcmF32Le,
        // EXTENSIBLE wraps the real subtype; the mix format is in practice
        // 32-bit float or 16-bit PCM, so map by bit depth.
        (WAVE_FORMAT_EXTENSIBLE, 32) => AudioFormat::PcmF32Le,
        (WAVE_FORMAT_EXTENSIBLE, 16) => AudioFormat::PcmS16Le,
        _ => return Err(G2gError::CapsMismatch),
    };
    Ok(AudioConfig {
        format,
        channels: fmt.nChannels as u8,
        sample_rate: fmt.nSamplesPerSec,
        block_align: fmt.nBlockAlign as usize,
    })
}

/// Capture worker: open + start the endpoint, then pump captured packets to
/// `chunk_tx` until `target` buffers are sent or capture fails.
fn capture_worker(
    config: AudioConfig,
    target: u64,
    chunk_tx: mpsc::UnboundedSender<Vec<u8>>,
    ready_tx: std_mpsc::SyncSender<Result<(), ()>>,
) {
    // SAFETY: COM init + capture on this worker thread, balanced below.
    let captured = unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let r = run_capture(config, target, &chunk_tx, &ready_tx);
        CoUninitialize();
        r
    };
    if captured.is_err() {
        // If we never signalled ready, do so now so `run` stops waiting.
        let _ = ready_tx.try_send(Err(()));
    }
}

/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn run_capture(
    config: AudioConfig,
    target: u64,
    chunk_tx: &mpsc::UnboundedSender<Vec<u8>>,
    ready_tx: &std_mpsc::SyncSender<Result<(), ()>>,
) -> Result<(), G2gError> {
    // SAFETY: WASAPI setup on the owning thread.
    let (client, capture) = unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eCapture, eConsole)
            .map_err(audio_err)?;
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(audio_err)?;
        let fmt_ptr = client.GetMixFormat().map_err(audio_err)?;
        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                BUFFER_DURATION_HNS,
                0,
                fmt_ptr,
                None,
            )
            .map_err(audio_err)?;
        CoTaskMemFree(Some(fmt_ptr.cast()));
        let capture: IAudioCaptureClient = client.GetService().map_err(audio_err)?;
        client.Start().map_err(audio_err)?;
        (client, capture)
    };

    let _ = ready_tx.try_send(Ok(()));

    // Bound the capture so a silent or stalled endpoint can't run forever.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut emitted = 0u64;
    while emitted < target {
        if Instant::now() >= deadline {
            break;
        }
        // SAFETY: packet query on the owning thread.
        let packet = unsafe { capture.GetNextPacketSize() }.map_err(audio_err)?;
        if packet == 0 {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let mut data_ptr: *mut u8 = core::ptr::null_mut();
        let mut frames = 0u32;
        let mut flags = 0u32;
        // SAFETY: GetBuffer hands back `frames` frames at `data_ptr`, valid
        // until ReleaseBuffer. We copy it out (or zero-fill on a SILENT
        // packet) before releasing.
        let chunk = unsafe {
            capture
                .GetBuffer(&mut data_ptr, &mut frames, &mut flags, None, None)
                .map_err(audio_err)?;
            let bytes = frames as usize * config.block_align;
            let chunk = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                alloc::vec![0u8; bytes]
            } else {
                core::slice::from_raw_parts(data_ptr, bytes).to_vec()
            };
            capture.ReleaseBuffer(frames).map_err(audio_err)?;
            chunk
        };

        if frames == 0 {
            continue;
        }
        if chunk_tx.send(chunk).is_err() {
            break; // consumer dropped
        }
        emitted += 1;
    }

    // SAFETY: stop on the owning thread.
    unsafe {
        let _ = client.Stop();
    }
    Ok(())
}

fn audio_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_mix_formats_to_audio_config() {
        let f32 = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_IEEE_FLOAT,
            nChannels: 2,
            nSamplesPerSec: 48_000,
            nAvgBytesPerSec: 48_000 * 8,
            nBlockAlign: 8,
            wBitsPerSample: 32,
            cbSize: 0,
        };
        let c = audio_config_from_format(&f32).unwrap();
        assert_eq!(c.format, AudioFormat::PcmF32Le);
        assert_eq!(c.channels, 2);
        assert_eq!(c.sample_rate, 48_000);
        assert_eq!(c.block_align, 8);

        let s16 = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM,
            nChannels: 1,
            nSamplesPerSec: 44_100,
            nAvgBytesPerSec: 44_100 * 2,
            nBlockAlign: 2,
            wBitsPerSample: 16,
            cbSize: 0,
        };
        let c = audio_config_from_format(&s16).unwrap();
        assert_eq!(c.format, AudioFormat::PcmS16Le);
        assert_eq!(c.block_align, 2);
    }

    #[test]
    fn rejects_unsupported_bit_depth() {
        let weird = WAVEFORMATEX {
            wFormatTag: WAVE_FORMAT_PCM,
            nChannels: 2,
            nSamplesPerSec: 48_000,
            nAvgBytesPerSec: 48_000 * 6,
            nBlockAlign: 6,
            wBitsPerSample: 24,
            cbSize: 0,
        };
        assert_eq!(audio_config_from_format(&weird), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn pad_template_is_pcm_source_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let source = WasapiSrc::pad_template(PadDirection::Source).expect("has source pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(source.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(WasapiSrc::pad_template(PadDirection::Sink).is_none());
    }
}
