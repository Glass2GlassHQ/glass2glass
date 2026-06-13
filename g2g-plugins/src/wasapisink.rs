//! Windows WASAPI render sink (M29): the audible output end of the audio path
//! `AudioTestSrc` / `WavSink` started in M25. Consumes interleaved PCM
//! (`PcmS16Le` or `PcmF32Le`) `DataFrame`s and plays them on the default
//! render endpoint via WASAPI shared mode, so an audio pipeline actually makes
//! sound instead of only writing a file.
//!
//! ## Threading
//!
//! WASAPI is COM and its render client is driven from one thread, so (like
//! `D3D11Sink`) all of it lives on a dedicated worker spun up at
//! `configure_pipeline`. The sink struct holds only `Send` handles (an mpsc
//! sender plus a shared counter); PCM bytes cross to the worker by value.
//!
//! ## Pacing
//!
//! The worker renders in real time: it tops up the shared-mode endpoint buffer
//! as the audio engine drains it. The source pushes faster than real time
//! (`AudioTestSrc` emits without sleeping), so the worker queues the pending
//! bytes and feeds them out at the device rate. On `Eos` it drains the queue
//! and waits for the endpoint to finish playing before stopping, so the tail
//! is not cut off.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioClient, IAudioRenderClient, IMMDeviceEnumerator, MMDeviceEnumerator,
    AUDCLNT_SHAREMODE_SHARED, WAVEFORMATEX,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, CLSCTX_INPROC_SERVER,
    COINIT_MULTITHREADED,
};

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audio::pcm_params;

/// Shared-mode endpoint buffer span (100-ns units), 200 ms. Large enough that
/// the worker's top-up cadence never starves the engine, small enough that the
/// drain-on-`Eos` wait is short.
const BUFFER_DURATION_HNS: i64 = 2_000_000;

/// Worker top-up cadence: roughly a quarter of the endpoint buffer.
const TOPUP_INTERVAL: Duration = Duration::from_millis(20);

/// Upper bound on the post-queue-drain wait for the endpoint to play out, so a
/// device that never reports empty padding can't wedge teardown.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Worker command. `Samples` carries one buffer of interleaved PCM bytes in the
/// negotiated format; `Shutdown` asks the worker to drain the queue, wait for
/// playout, and stop.
enum WorkerCmd {
    Samples(Vec<u8>),
    Shutdown,
}

pub struct WasapiSink {
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    caps: Option<Caps>,
    frames_rendered: Arc<AtomicU64>,
}

impl core::fmt::Debug for WasapiSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WasapiSink")
            .field("caps", &self.caps)
            .field(
                "frames_rendered",
                &self.frames_rendered.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for WasapiSink {
    fn default() -> Self {
        Self::new()
    }
}

impl WasapiSink {
    pub fn new() -> Self {
        Self {
            cmd_tx: None,
            worker: None,
            caps: None,
            frames_rendered: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Count of sample frames written to the endpoint. Useful in tests.
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered.load(Ordering::Relaxed)
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

impl Drop for WasapiSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Build a canonical `WAVEFORMATEX` for an interleaved PCM stream.
fn wave_format(tag: u16, bits: u16, channels: u16, rate: u32) -> WAVEFORMATEX {
    let block_align = channels * (bits / 8);
    WAVEFORMATEX {
        wFormatTag: tag,
        nChannels: channels,
        nSamplesPerSec: rate,
        nAvgBytesPerSec: rate * block_align as u32,
        nBlockAlign: block_align,
        wBitsPerSample: bits,
        cbSize: 0,
    }
}

impl AsyncElement for WasapiSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pcm_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only. `Caps::Audio` has no open dims, so the per-rate/channel
    /// acceptance rides the legacy intercept bridge, as in `WavSink`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pcm_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (tag, bits, channels, rate) = pcm_params(absolute_caps)?;

        if self.worker.is_some() {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = mpsc::channel::<WorkerCmd>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), ()>>(1);
        let rendered = Arc::clone(&self.frames_rendered);
        let format = wave_format(tag, bits, channels, rate);

        let join = thread::Builder::new()
            .name(alloc::string::String::from("g2g-wasapisink"))
            .spawn(move || {
                if let Err(e) = worker_main(format, rx, rendered, ready_tx) {
                    std::eprintln!("g2g-wasapisink worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // The worker reports whether the endpoint opened; a headless host (no
        // audio device) fails loud here rather than silently dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            _ => {
                let _ = tx.send(WorkerCmd::Shutdown);
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    tx.send(WorkerCmd::Samples(slice.as_slice().to_vec()))
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                // A mid-stream format change can't be honoured on an open
                // endpoint; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pcm_params(&c)?;
                    Ok(())
                }
                PipelinePacket::Flush => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
            }
        })
    }
}

impl PadTemplates for WasapiSink {
    /// Terminal PCM sink pad. `Caps::Audio` has no open dims, so the template
    /// pins the common shapes per PCM format, as in `WavSink`.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
    }
}

// =================================================================
// Worker thread: WASAPI shared-mode render
// =================================================================

fn worker_main(
    format: WAVEFORMATEX,
    rx: Receiver<WorkerCmd>,
    rendered: Arc<AtomicU64>,
    ready: SyncSender<Result<(), ()>>,
) -> Result<(), G2gError> {
    // SAFETY: COM is initialised MTA on this worker thread; every later WASAPI
    // call lands on the same thread. S_FALSE (already initialised) is fine.
    let render = match unsafe { open_endpoint(&format) } {
        Ok(state) => {
            let _ = ready.send(Ok(()));
            state
        }
        Err(e) => {
            let _ = ready.send(Err(()));
            // CoUninitialize is paired in open_endpoint's error paths via the
            // guard below only on success; on failure CoInitializeEx may or may
            // not have run, so balance it here best-effort.
            // SAFETY: balances the CoInitializeEx attempted in open_endpoint.
            unsafe { CoUninitialize() };
            return Err(e);
        }
    };

    let result = run_render_loop(&render, rx, &rendered);

    // SAFETY: stop the client and tear COM down on the owning thread.
    unsafe {
        let _ = render.client.Stop();
        CoUninitialize();
    }
    result
}

/// Live WASAPI render objects for the endpoint's lifetime.
struct RenderState {
    client: IAudioClient,
    render: IAudioRenderClient,
    /// Endpoint buffer capacity in sample frames.
    buffer_frames: u32,
    /// Bytes per sample frame (`nBlockAlign`).
    block_align: usize,
}

/// Open the default render endpoint in shared mode and start it. Initialises
/// COM on the calling (worker) thread.
///
/// # Safety
/// Must run on the worker thread that owns every subsequent WASAPI call.
unsafe fn open_endpoint(format: &WAVEFORMATEX) -> Result<RenderState, G2gError> {
    // SAFETY: COM/WASAPI object creation and configuration on the owning thread.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eConsole)
            .map_err(audio_err)?;
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None).map_err(audio_err)?;

        client
            .Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                0,
                BUFFER_DURATION_HNS,
                0,
                format,
                None,
            )
            .map_err(audio_err)?;

        let buffer_frames = client.GetBufferSize().map_err(audio_err)?;
        let render: IAudioRenderClient = client.GetService().map_err(audio_err)?;
        client.Start().map_err(audio_err)?;

        Ok(RenderState {
            client,
            render,
            buffer_frames,
            block_align: format.nBlockAlign as usize,
        })
    }
}

/// Pull PCM from the command channel into a byte queue and feed it to the
/// endpoint at the device rate, draining and playing out on shutdown.
fn run_render_loop(
    state: &RenderState,
    rx: Receiver<WorkerCmd>,
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    let mut queue: VecDeque<u8> = VecDeque::new();
    let mut closing = false;
    let mut drain_deadline: Option<Instant> = None;

    loop {
        if !closing {
            match rx.recv_timeout(TOPUP_INTERVAL) {
                Ok(WorkerCmd::Samples(bytes)) => queue.extend(bytes),
                Ok(WorkerCmd::Shutdown) => closing = true,
                Err(RecvTimeoutError::Disconnected) => closing = true,
                Err(RecvTimeoutError::Timeout) => {}
            }
            // Absorb whatever else is already queued so a burst from the source
            // lands in one pass.
            while let Ok(cmd) = rx.try_recv() {
                match cmd {
                    WorkerCmd::Samples(bytes) => queue.extend(bytes),
                    WorkerCmd::Shutdown => closing = true,
                }
            }
        }

        let wrote = top_up(state, &mut queue, rendered)?;

        if closing && queue.len() < state.block_align {
            // Queue exhausted: wait for the endpoint to play out the last
            // frames, then stop. Guarded so a stuck device can't wedge teardown.
            let deadline = *drain_deadline.get_or_insert_with(|| Instant::now() + DRAIN_TIMEOUT);
            // SAFETY: padding query on the owning thread.
            let padding = unsafe { state.client.GetCurrentPadding() }.map_err(audio_err)?;
            if padding == 0 || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        } else if wrote == 0 && closing {
            thread::sleep(Duration::from_millis(5));
        }
    }
    Ok(())
}

/// Write as many queued frames as the endpoint has free space for. Returns the
/// frame count written.
fn top_up(
    state: &RenderState,
    queue: &mut VecDeque<u8>,
    rendered: &AtomicU64,
) -> Result<u32, G2gError> {
    // SAFETY: padding query on the owning thread.
    let padding = unsafe { state.client.GetCurrentPadding() }.map_err(audio_err)?;
    let free_frames = state.buffer_frames.saturating_sub(padding);
    let have_frames = (queue.len() / state.block_align) as u32;
    let frames = free_frames.min(have_frames);
    if frames == 0 {
        return Ok(0);
    }

    let bytes = frames as usize * state.block_align;
    // SAFETY: GetBuffer hands back a writable region for exactly `frames`
    // frames; we fill all `bytes` of it, then release the same count. The
    // pointer is valid only between GetBuffer and ReleaseBuffer.
    unsafe {
        let ptr = state.render.GetBuffer(frames).map_err(audio_err)?;
        let dst = core::slice::from_raw_parts_mut(ptr, bytes);
        for (slot, byte) in dst.iter_mut().zip(queue.drain(..bytes)) {
            *slot = byte;
        }
        state.render.ReleaseBuffer(frames, 0).map_err(audio_err)?;
    }
    rendered.fetch_add(frames as u64, Ordering::Relaxed);
    Ok(frames)
}

fn audio_err(e: windows::core::Error) -> G2gError {
    // WASAPI errors are COM HRESULTs, the same carrier as the MF path.
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{WAVE_FORMAT_IEEE_FLOAT, WAVE_FORMAT_PCM};

    #[test]
    fn pcm_params_maps_formats_and_rejects_compressed() {
        let s16 = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(pcm_params(&s16), Ok((WAVE_FORMAT_PCM, 16, 2, 48_000)));
        let f32 = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        };
        assert_eq!(pcm_params(&f32), Ok((WAVE_FORMAT_IEEE_FLOAT, 32, 1, 44_100)));
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(pcm_params(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn wave_format_fields_are_canonical() {
        // stereo s16 at 48 kHz: block align 4, avg bytes 192000. WAVEFORMATEX
        // is packed, so copy each field to a local before asserting.
        let wfx = wave_format(WAVE_FORMAT_PCM, 16, 2, 48_000);
        assert_eq!({ wfx.nBlockAlign }, 4);
        assert_eq!({ wfx.nAvgBytesPerSec }, 192_000);
        assert_eq!({ wfx.cbSize }, 0);
        // mono f32 at 44.1 kHz: block align 4, avg bytes 176400.
        let wff = wave_format(WAVE_FORMAT_IEEE_FLOAT, 32, 1, 44_100);
        assert_eq!({ wff.nBlockAlign }, 4);
        assert_eq!({ wff.nAvgBytesPerSec }, 176_400);
    }

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = WasapiSink::new();
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(sink.intercept_caps(&pcm), Ok(pcm));
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(sink.intercept_caps(&opus), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = WasapiSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(WasapiSink::pad_template(PadDirection::Source).is_none());
    }
}
