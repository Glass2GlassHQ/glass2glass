//! Linux PipeWire audio render sink. Plays interleaved PCM (`PcmS16Le` /
//! `PcmF32Le`) through the PipeWire graph, the modern Linux media layer and the
//! third Linux audio output alongside [`AlsaSink`](crate::alsasink::AlsaSink)
//! and [`PulseSink`](crate::pulsesink::PulseSink).
//!
//! ## Threading
//!
//! PipeWire is a callback-driven main loop pinned to one thread, so (like the
//! WASAPI / ALSA sinks) the whole loop runs on a dedicated worker spun up at
//! `configure_pipeline`. The element keeps only `Send` handles: a shared PCM
//! byte queue the realtime `process` callback drains, and a `pw::channel`
//! sender that asks the loop to quit on teardown.
//!
//! ## Pacing (leaky)
//!
//! Unlike ALSA's blocking `writei`, PipeWire's `process` callback pulls data on
//! its own clock and never blocks the producer, so this sink cannot backpressure
//! the graph. The shared queue is therefore bounded to ~1 s of audio and drops
//! the oldest bytes past that (the [`LinkPolicy::DropOldest`] analog for an
//! external clock). For a live source the queue stays near-empty; only a source
//! that runs faster than real time (e.g. unbounded `AudioTestSrc`) hits the cap.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::collections::VecDeque;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use pipewire as pw;
use pw::spa;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::pwaudio::{format_pod_bytes, frame_bytes, pw_params};

/// Control message to the loop thread. Only `Terminate` for now (quit the loop).
enum Ctrl {
    Terminate,
}

/// Shared PCM queue between the element and the realtime `process` callback.
type SharedQueue = Arc<Mutex<VecDeque<u8>>>;

pub struct PipeWireSink {
    ctrl_tx: Option<pw::channel::Sender<Ctrl>>,
    worker: Option<JoinHandle<()>>,
    queue: SharedQueue,
    high_water: usize,
    caps: Option<Caps>,
    bytes_queued: Arc<AtomicU64>,
}

impl core::fmt::Debug for PipeWireSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipeWireSink")
            .field("caps", &self.caps)
            .field("high_water", &self.high_water)
            .field("bytes_queued", &self.bytes_queued.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for PipeWireSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireSink {
    pub fn new() -> Self {
        Self {
            ctrl_tx: None,
            worker: None,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            high_water: 0,
            caps: None,
            bytes_queued: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total PCM bytes accepted from the pipeline (before any leaky drop).
    pub fn bytes_queued(&self) -> u64 {
        self.bytes_queued.load(Ordering::Relaxed)
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.ctrl_tx.take() {
            let _ = tx.send(Ctrl::Terminate);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
        if let Ok(mut q) = self.queue.lock() {
            q.clear();
        }
    }
}

impl Drop for PipeWireSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl AsyncElement for PipeWireSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pw_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only. `Caps::Audio` has no open dims, so per-rate/channel acceptance
    /// rides the legacy intercept bridge, as in the other audio sinks.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pw_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, rate) = pw_params(absolute_caps)?;

        if self.worker.is_some() {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let stride = frame_bytes(format, channels);
        // ~1 s of audio: bound the leaky queue.
        self.high_water = stride * rate as usize;

        let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), i32>>(1);
        let queue = Arc::clone(&self.queue);
        if let Ok(mut q) = queue.lock() {
            q.clear();
        }

        let join = thread::Builder::new()
            .name(String::from("g2g-pipewiresink"))
            .spawn(move || {
                worker_main(format, channels, rate, stride, queue, ctrl_rx, ready_tx);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(code)) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::PipeWire(code)));
            }
            Err(_) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::PipeWire(-1)));
            }
        }

        self.ctrl_tx = Some(ctrl_tx);
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
                    if self.worker.is_none() {
                        return Err(G2gError::NotConfigured);
                    }
                    let bytes = slice.as_slice();
                    let mut q = self
                        .queue
                        .lock()
                        .map_err(|_| G2gError::Hardware(HardwareError::PipeWire(-1)))?;
                    q.extend(bytes.iter().copied());
                    // Leaky bound: drop the oldest bytes past the high-water mark.
                    while q.len() > self.high_water {
                        q.pop_front();
                    }
                    drop(q);
                    self.bytes_queued
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    Ok(())
                }
                // A mid-stream format change can't be honoured on an open stream;
                // only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pw_params(&c)?;
                    Ok(())
                }
                PipelinePacket::Flush | PipelinePacket::Segment(_) => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
            }
        })
    }
}

impl PadTemplates for PipeWireSink {
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
// Worker thread: the PipeWire main loop
// =================================================================

fn worker_main(
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    queue: SharedQueue,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: std::sync::mpsc::SyncSender<Result<(), i32>>,
) {
    match build_and_run(format, channels, rate, stride, queue, ctrl_rx, &ready) {
        Ok(()) => {}
        Err(code) => {
            // If setup failed before `ready` was sent, report it; if it was
            // already sent (loop ran then exited), this send simply no-ops on a
            // closed channel.
            let _ = ready.send(Err(code));
        }
    }
}

fn build_and_run(
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    queue: SharedQueue,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| -1)?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| -1)?;
    let core = context.connect(None).map_err(|_| -1)?;

    let stream = pw::stream::Stream::new(
        &core,
        "g2g-pipewiresink",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::MEDIA_CATEGORY => "Playback",
        },
    )
    .map_err(|_| -1)?;

    let q = Arc::clone(&queue);
    let _listener = stream
        .add_local_listener_with_user_data(())
        .process(move |stream, ()| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let size = if let Some(slice) = data.data() {
                // Fill whole-frame-aligned capacity from the queue; pad the
                // remainder with silence so an underrun is quiet, not garbage.
                let cap = (slice.len() / stride) * stride;
                let mut filled = 0usize;
                if let Ok(mut q) = q.lock() {
                    let avail = q.len().min(cap);
                    for slot in slice.iter_mut().take(avail) {
                        *slot = q.pop_front().unwrap_or(0);
                    }
                    filled = avail;
                }
                for slot in slice.iter_mut().take(cap).skip(filled) {
                    *slot = 0;
                }
                cap
            } else {
                0
            };
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = stride as _;
            *chunk.size_mut() = size as _;
        })
        .register()
        .map_err(|_| -1)?;

    let values = format_pod_bytes(format, channels, rate);
    let mut params = [spa::pod::Pod::from_bytes(&values).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|_| -1)?;

    // Quit the loop when the element sends `Terminate` on teardown.
    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    // Setup succeeded; unblock configure_pipeline, then run the loop.
    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = PipeWireSink::new();
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(sink.intercept_caps(&pcm), Ok(pcm));
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(sink.intercept_caps(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = PipeWireSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PipeWireSink::pad_template(PadDirection::Source).is_none());
    }
}
