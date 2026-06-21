//! Linux PipeWire audio capture source. The capture sibling of
//! [`PipeWireSink`](crate::pipewiresink::PipeWireSink): connects an input stream
//! to the PipeWire graph and streams interleaved PCM (`PcmS16Le` / `PcmF32Le`)
//! `DataFrame`s downstream. The modern Linux microphone path (PipeWire replaces
//! v4l2 + PulseAudio + the screen-capture DBus dance); video / screen capture is
//! a follow-up on the same element.
//!
//! PipeWire is a callback-driven main loop pinned to one thread, so (like
//! `V4l2Src`) the loop runs on a dedicated worker thread that feeds the async
//! `run` loop over a channel. We request a fixed PCM format; the PipeWire
//! adapter converts the device to it, so the produced caps are deterministic
//! (no async `param_changed` round-trip needed for v1).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket,
};

use pipewire as pw;
use pw::spa;

use crate::pwaudio::{format_pod_bytes, frame_bytes, pw_params};

const DEFAULT_RATE: u32 = 48_000;
const DEFAULT_CHANNELS: u8 = 2;

/// Control message to the loop thread (quit on teardown).
enum Ctrl {
    Terminate,
}

#[derive(Debug)]
pub struct PipeWireSrc {
    format: AudioFormat,
    channels: u8,
    rate: u32,
    /// 0 = run until error or downstream shutdown; else stop after N frames
    /// (PipeWire buffers) and emit EOS. The bounded-capture / test path.
    frame_limit: u64,
    configured: bool,
}

impl Default for PipeWireSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireSrc {
    /// Capture S16LE stereo at 48 kHz by default.
    pub fn new() -> Self {
        Self {
            format: AudioFormat::PcmS16Le,
            channels: DEFAULT_CHANNELS,
            rate: DEFAULT_RATE,
            frame_limit: 0,
            configured: false,
        }
    }

    /// Request a PCM sample format (`PcmS16Le` or `PcmF32Le`).
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    /// Request a sample rate in Hz.
    pub fn with_rate(mut self, rate: u32) -> Self {
        self.rate = rate;
        self
    }

    /// Request a channel count.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    /// Stop after `n` captured buffers and emit EOS. Without this the source
    /// runs until an error or until downstream drops.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn caps(&self) -> Result<Caps, G2gError> {
        let caps = Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.rate,
        };
        // Reject a non-PCM request up front (the SPA mapping is PCM-only).
        pw_params(&caps)?;
        Ok(caps)
    }
}

impl SourceLoop for PipeWireSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps())
    }

    /// Produces the fixed PCM caps we ask the graph to convert to, so a chain
    /// built on the mic takes the native arc-consistency path (mirrors `V4l2Src`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(self.caps().map(|c| CapsConstraint::Produces(CapsSet::one(c))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Live source: one buffer period is device-driven, so report a small live
    /// latency hint rather than zero.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (spa_format, channels, rate) =
                pw_params(&self.caps()?).map_err(|_| G2gError::NotConfigured)?;
            let stride = frame_bytes(spa_format, channels);
            let limit = self.frame_limit;

            // Captured PCM buffers cross from the loop thread to here.
            let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
            // Control + a setup-result handshake (surface a connect failure).
            let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), i32>>(1);

            let handle = std::thread::Builder::new()
                .name(alloc::string::String::from("g2g-pipewiresrc"))
                .spawn(move || {
                    worker_main(spa_format, channels, rate, stride, audio_tx, ctrl_rx, ready_tx);
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // Block briefly for the stream to connect (sync ioctl-equivalent).
            match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(())) => {}
                Ok(Err(code)) => {
                    let _ = ctrl_tx.send(Ctrl::Terminate);
                    let _ = handle.join();
                    return Err(G2gError::Hardware(HardwareError::PipeWire(code)));
                }
                Err(_) => {
                    let _ = ctrl_tx.send(Ctrl::Terminate);
                    let _ = handle.join();
                    return Err(G2gError::Hardware(HardwareError::PipeWire(-1)));
                }
            }

            let frame_dur = if rate > 0 {
                1_000_000_000u64 / rate as u64
            } else {
                0
            };
            let mut seq = 0u64;
            let mut frames_total = 0u64; // sample frames, for PTS
            let mut downstream_open = true;

            while limit == 0 || seq < limit {
                let Some(bytes) = audio_rx.recv().await else {
                    break; // worker ended
                };
                if bytes.len() < stride {
                    continue;
                }
                let n_frames = (bytes.len() / stride) as u64;
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let pts = if rate > 0 {
                    frames_total * 1_000_000_000 / rate as u64
                } else {
                    0
                };
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: n_frames * frame_dur,
                        capture_ns: pts,
                        arrival_ns,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                frames_total += n_frames;
                seq += 1;
            }

            // Stop the loop and reap the worker.
            let _ = ctrl_tx.send(Ctrl::Terminate);
            let _ = handle.join();

            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for PipeWireSrc {
    /// Produces PCM; a constructed instance fixes the format / rate / channels.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
    }
}

// =================================================================
// Worker thread: the PipeWire capture main loop
// =================================================================

fn worker_main(
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: std::sync::mpsc::SyncSender<Result<(), i32>>,
) {
    if let Err(code) = build_and_run(format, channels, rate, stride, audio_tx, ctrl_rx, &ready) {
        let _ = ready.send(Err(code));
    }
}

fn build_and_run(
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| -1)?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| -1)?;
    let core = context.connect(None).map_err(|_| -1)?;

    let stream = pw::stream::Stream::new(
        &core,
        "g2g-pipewiresrc",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Music",
        },
    )
    .map_err(|_| -1)?;

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
            let offset = data.chunk().offset() as usize;
            let size = data.chunk().size() as usize;
            if size < stride {
                return;
            }
            if let Some(slice) = data.data() {
                let end = (offset + size).min(slice.len());
                if end > offset {
                    // Copy the valid region out and hand it to the async side.
                    let _ = audio_tx.send(slice[offset..end].to_vec());
                }
            }
        })
        .register()
        .map_err(|_| -1)?;

    let values = format_pod_bytes(format, channels, rate);
    let mut params = [spa::pod::Pod::from_bytes(&values).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|_| -1)?;

    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = PipeWireSrc::new()
            .with_format(AudioFormat::PcmF32Le)
            .with_rate(44_100)
            .with_channels(1)
            .with_frame_limit(5);
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!((src.channels, src.rate), (1, 44_100));
        assert_eq!(src.frame_limit, 5);
    }

    #[test]
    fn caps_reflect_request_and_reject_compressed() {
        let src = PipeWireSrc::new().with_rate(16_000).with_channels(1);
        assert_eq!(
            src.caps(),
            Ok(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 16_000,
            })
        );
        let bad = PipeWireSrc::new().with_format(AudioFormat::Opus);
        assert_eq!(bad.caps(), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn pad_template_is_pcm_source_only() {
        use g2g_core::{PadDirection, PadTemplates};
        assert!(PipeWireSrc::pad_template(PadDirection::Source).is_some());
        assert!(PipeWireSrc::pad_template(PadDirection::Sink).is_none());
    }
}
