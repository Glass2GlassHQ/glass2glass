//! M589: Linux audio-sink host smoke test (alsasink / pulsesink / pipewiresink).
//!
//! These render sinks were unit-tested (caps mapping, pad templates) but never
//! driven against a real audio device. This feeds each a short interleaved-PCM
//! sine tone on the host's default output, in every sample format and channel
//! layout the sinks open a device with, and asserts the full device path runs:
//! `configure_pipeline` opens the device, `process` writes PCM, `Eos` drains, and
//! the sink's byte/frame counter advances. A host with no reachable device (CI,
//! a headless box with no sound server) skips: `configure_pipeline` fails loud,
//! and the test treats a hardware error as "no device" rather than a failure.
//!
//! Each sink is behind its own cargo feature, so run with the ones built, e.g.
//! `cargo test -p g2g-plugins --features alsa-sink,pulse-sink,pipewire
//!  --test m589_audio_sink_smoke`. Validated on this Fedora / PipeWire host
//! (PulseAudio-on-PipeWire + pipewire-alsa); it plays a brief quiet tone.
#![cfg(any(feature = "alsa-sink", feature = "pulse-sink", feature = "pipewire"))]

use std::future::Future;
use std::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome,
};

const RATE: u32 = 48_000;
const FREQ: f32 = 440.0;
/// Total tone length. Short so an automated run is unobtrusive, long enough that
/// the device clock actually consumes several buffers.
const SECONDS: f32 = 0.3;
/// One pushed buffer's frame count (10 ms), so the sink sees several `DataFrame`s.
const CHUNK_FRAMES: usize = (RATE as usize) / 100;

/// Format / channel shapes exercised per sink: every sample format the sinks
/// open a device with, plus mono and the 5.1 / 7.1 layouts, so the format and
/// channel-map handling is covered rather than just the one common shape. A
/// host whose default device is stereo still exercises the > 2-channel open:
/// the sound server accepts the surround layout and downmixes it.
const SHAPES: &[(AudioFormat, u8)] = &[
    (AudioFormat::PcmS16Le, 2),
    (AudioFormat::PcmF32Le, 2),
    (AudioFormat::PcmS24Le, 2),
    (AudioFormat::PcmS32Le, 2),
    (AudioFormat::PcmU8, 2),
    (AudioFormat::PcmS16Le, 1),
    (AudioFormat::PcmS16Le, 6),
    (AudioFormat::PcmS24Le, 6),
    (AudioFormat::PcmS16Le, 8),
];

/// A terminal sink swallows nothing downstream; this stand-in satisfies the
/// `OutputSink` argument of `process`.
struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Encode one sample. The test's own little-endian encoder, so what reaches the
/// device is not produced by the same code the sinks lean on.
fn push_sample(out: &mut Vec<u8>, s: f32, format: AudioFormat) {
    match format {
        AudioFormat::PcmU8 => out.push(((s * 127.0) as i32 + 128) as u8),
        AudioFormat::PcmS16Le => out.extend_from_slice(&((s * 32767.0) as i16).to_le_bytes()),
        // 3-byte packed, the low 3 bytes of the 32-bit value.
        AudioFormat::PcmS24Le => {
            out.extend_from_slice(&((s * 8_388_607.0) as i32).to_le_bytes()[..3])
        }
        AudioFormat::PcmS32Le => {
            out.extend_from_slice(&((s * 2_147_483_647.0) as i32).to_le_bytes())
        }
        AudioFormat::PcmF32Le => out.extend_from_slice(&s.to_le_bytes()),
        _ => panic!("no encoder for {format:?}"),
    }
}

/// Interleaved sine tone, `frames` sample-frames across `channels`, in the given
/// PCM format, quiet (amplitude ~0.2).
fn tone(format: AudioFormat, channels: u8, frames: usize, phase0: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..frames {
        let t = (phase0 + i) as f32 / RATE as f32;
        let s = (2.0 * core::f32::consts::PI * FREQ * t).sin() * 0.2;
        for _ in 0..channels {
            push_sample(&mut out, s, format);
        }
    }
    out
}

fn pcm_frame(bytes: Vec<u8>, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: seq,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

/// Configure `sink`, feed the tone in 10 ms chunks, and drain on `Eos`. Returns
/// `Ok(false)` if the device is unreachable (a hardware error at configure),
/// `Ok(true)` if the full path ran, `Err` on an unexpected failure.
fn drive_tone<E: AsyncElement>(
    sink: &mut E,
    format: AudioFormat,
    channels: u8,
) -> Result<bool, G2gError> {
    let caps = Caps::Audio {
        format,
        channels,
        sample_rate: RATE,
    };
    match sink.configure_pipeline(&caps) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => return Ok(false),
        Err(e) => return Err(e),
    }
    let total = (SECONDS * RATE as f32) as usize;
    let mut done = 0usize;
    let mut seq = 0u64;
    let mut null = NullSink;
    while done < total {
        let n = CHUNK_FRAMES.min(total - done);
        let buf = tone(format, channels, n, done);
        block_on(sink.process(PipelinePacket::DataFrame(pcm_frame(buf, seq)), &mut null))?;
        done += n;
        seq += 1;
    }
    block_on(sink.process(PipelinePacket::Eos, &mut null))?;
    Ok(true)
}

#[cfg(feature = "alsa-sink")]
#[test]
fn alsasink_plays_a_tone() {
    use g2g_plugins::alsasink::AlsaSink;
    let mut ran = 0;
    for &(fmt, ch) in SHAPES {
        let mut sink = AlsaSink::new();
        match drive_tone(&mut sink, fmt, ch) {
            Ok(true) => {
                assert!(
                    sink.frames_rendered() > 0,
                    "alsasink rendered no {fmt:?} x{ch} frames"
                );
                eprintln!(
                    "m589 alsasink {fmt:?} x{ch}: rendered {} frames",
                    sink.frames_rendered()
                );
                ran += 1;
            }
            Ok(false) => eprintln!("skip m589 alsasink {fmt:?} x{ch}: no reachable ALSA device"),
            Err(e) => panic!("alsasink {fmt:?} x{ch} error: {e:?}"),
        }
    }
    if ran == 0 {
        eprintln!("skip m589 alsasink: no reachable ALSA device");
    }
}

#[cfg(feature = "pulse-sink")]
#[test]
fn pulsesink_plays_a_tone() {
    use g2g_plugins::pulsesink::PulseSink;
    let mut ran = 0;
    for &(fmt, ch) in SHAPES {
        let mut sink = PulseSink::new();
        match drive_tone(&mut sink, fmt, ch) {
            Ok(true) => {
                assert!(
                    sink.bytes_written() > 0,
                    "pulsesink wrote no {fmt:?} x{ch} bytes"
                );
                eprintln!(
                    "m589 pulsesink {fmt:?} x{ch}: wrote {} bytes",
                    sink.bytes_written()
                );
                ran += 1;
            }
            Ok(false) => eprintln!("skip m589 pulsesink {fmt:?} x{ch}: no reachable server"),
            Err(e) => panic!("pulsesink {fmt:?} x{ch} error: {e:?}"),
        }
    }
    if ran == 0 {
        eprintln!("skip m589 pulsesink: no reachable PulseAudio server");
    }
}

#[cfg(feature = "pipewire")]
#[test]
fn pipewiresink_plays_a_tone() {
    use g2g_plugins::pipewiresink::PipeWireSink;
    let mut ran = 0;
    for &(fmt, ch) in SHAPES {
        let mut sink = PipeWireSink::new();
        match drive_tone(&mut sink, fmt, ch) {
            Ok(true) => {
                assert!(
                    sink.bytes_queued() > 0,
                    "pipewiresink queued no {fmt:?} x{ch} bytes"
                );
                eprintln!(
                    "m589 pipewiresink {fmt:?} x{ch}: queued {} bytes",
                    sink.bytes_queued()
                );
                ran += 1;
            }
            Ok(false) => eprintln!("skip m589 pipewiresink {fmt:?} x{ch}: no reachable server"),
            Err(e) => panic!("pipewiresink {fmt:?} x{ch} error: {e:?}"),
        }
    }
    if ran == 0 {
        eprintln!("skip m589 pipewiresink: no reachable PipeWire server");
    }
}
