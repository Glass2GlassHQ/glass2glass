//! Linux PulseAudio render sink. Plays interleaved PCM (`PcmS16Le` /
//! `PcmF32Le`) on the default PulseAudio (or PipeWire-pulse) server, the
//! higher-level sibling of [`AlsaSink`](crate::alsasink::AlsaSink) and the
//! Linux analog of the Windows-only [`WasapiSink`](crate::wasapisink::WasapiSink).
//!
//! ## Threading and pacing
//!
//! The libpulse "simple" API (`pa_simple_write`) is blocking, so all of it runs
//! on a dedicated worker spun up at `configure_pipeline`, exactly as in
//! `AlsaSink` / `WasapiSink`. The blocking write *is* the pacing: it returns
//! once the server-side buffer has room. On `Eos` the worker drains the buffer
//! (`pa_simple_drain`) so the tail is not cut off.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use libpulse_binding::sample::{Format as PaFormat, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// Negotiated PCM parameters as a PulseAudio `Spec`. Compressed audio
/// (AAC / Opus) is rejected structurally, as in `WasapiSink`.
fn pulse_spec(caps: &Caps) -> Result<Spec, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let format = match format {
        AudioFormat::PcmS16Le => PaFormat::S16le,
        AudioFormat::PcmF32Le => PaFormat::F32le,
        AudioFormat::Aac | AudioFormat::Opus => return Err(G2gError::CapsMismatch),
        _ => return Err(G2gError::CapsMismatch),
    };
    let spec = Spec {
        format,
        channels: *channels,
        rate: *sample_rate,
    };
    if !spec.is_valid() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(spec)
}

/// Worker command. `Samples` carries one buffer of interleaved PCM bytes in the
/// negotiated format; `Shutdown` asks the worker to drain and stop.
enum WorkerCmd {
    Samples(Vec<u8>),
    Shutdown,
}

pub struct PulseSink {
    app_name: String,
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    caps: Option<Caps>,
    bytes_written: Arc<AtomicU64>,
}

impl core::fmt::Debug for PulseSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseSink")
            .field("app_name", &self.app_name)
            .field("caps", &self.caps)
            .field("bytes_written", &self.bytes_written.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for PulseSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseSink {
    /// Render under the default application name `glass2glass`.
    pub fn new() -> Self {
        Self::with_app_name("glass2glass")
    }

    /// Render under a custom application name (shown in the PulseAudio mixer).
    pub fn with_app_name(name: impl Into<String>) -> Self {
        Self {
            app_name: name.into(),
            cmd_tx: None,
            worker: None,
            caps: None,
            bytes_written: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total PCM bytes written to the server. Useful in tests.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
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

impl Drop for PulseSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl AsyncElement for PulseSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pulse_spec(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only. `Caps::Audio` has no open dims, so the per-rate/channel
    /// acceptance rides the legacy intercept bridge, as in `WasapiSink`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pulse_spec(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let spec = pulse_spec(absolute_caps)?;

        if self.worker.is_some() {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = mpsc::channel::<WorkerCmd>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), i32>>(1);
        let written = Arc::clone(&self.bytes_written);
        let app_name = self.app_name.clone();

        let join = thread::Builder::new()
            .name(String::from("g2g-pulsesink"))
            .spawn(move || {
                worker_main(&app_name, spec, rx, written, ready_tx);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // The worker reports whether the server connection opened; a host with
        // no PulseAudio server fails loud here rather than dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(code)) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::PulseAudio(code)));
            }
            Err(_) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::PulseAudio(-1)));
            }
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PulseAudio audio sink",
            "Sink/Audio",
            "Plays interleaved PCM via PulseAudio",
            "g2g",
        )
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
                        .map_err(|_| G2gError::Hardware(HardwareError::PulseAudio(-1)))?;
                    Ok(())
                }
                // A mid-stream format change can't be honoured on an open
                // stream; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pulse_spec(&c)?;
                    Ok(())
                }
                PipelinePacket::Flush | PipelinePacket::Segment(_) => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for PulseSink {
    /// Terminal PCM sink pad. `Caps::Audio` has no open dims, so the template
    /// pins the common shapes per PCM format, as in `WasapiSink`.
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
// Worker thread: blocking libpulse simple write
// =================================================================

fn worker_main(
    app_name: &str,
    spec: Spec,
    rx: Receiver<WorkerCmd>,
    written: Arc<AtomicU64>,
    ready: SyncSender<Result<(), i32>>,
) {
    let simple = match Simple::new(
        None,             // default server
        app_name,         // application name
        Direction::Playback,
        None,             // default device
        "playback",       // stream description
        &spec,
        None,             // default channel map
        None,             // default buffering attributes
    ) {
        Ok(s) => {
            let _ = ready.send(Ok(()));
            s
        }
        Err(e) => {
            let _ = ready.send(Err(e.0));
            return;
        }
    };

    let mut closing = false;
    while !closing {
        match rx.recv() {
            Ok(WorkerCmd::Samples(bytes)) => {
                if simple.write(&bytes).is_err() {
                    break;
                }
                written.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            }
            Ok(WorkerCmd::Shutdown) | Err(_) => closing = true,
        }
    }
    // Play out whatever is still buffered, then stop.
    let _ = simple.drain();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_spec_maps_formats_and_rejects_compressed() {
        let s16 = pulse_spec(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        })
        .expect("s16 spec");
        assert_eq!(s16.format, PaFormat::S16le);
        assert_eq!((s16.channels, s16.rate), (2, 48_000));

        let f32 = pulse_spec(&Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        })
        .expect("f32 spec");
        assert_eq!(f32.format, PaFormat::F32le);

        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(pulse_spec(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = PulseSink::new();
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
        let sink = PulseSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PulseSink::pad_template(PadDirection::Source).is_none());
    }
}
