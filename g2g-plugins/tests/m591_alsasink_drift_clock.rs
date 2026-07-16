//! M591: AlsaSink DAC-disciplined clock (A/V sync, phase 1) host test.
//!
//! M590 built the `DriftClock` (least-squares fit of a playout rate into a
//! smooth `now_ns()`); this drives a real `AlsaSink` on the host's default
//! device and asserts the sink actually disciplines its provided clock from the
//! hardware. It plays a short tone, then checks that (1) the sink offers its
//! clock to election as an `AudioProvider` (so audio outranks a video sink's
//! plain `Provider`), (2) the worker fed the clock multiple `snd_pcm_delay`
//! observations, (3) the estimated playout rate is a sane ~1.0x (both timelines
//! are real time, so no wild slope), and (4) the clock is live (its `now_ns()`
//! advances across a real sleep).
//!
//! A host with no reachable ALSA device (CI, headless) skips: `configure_pipeline`
//! fails loud with a hardware error, treated as "no device" not a failure. Run:
//! `cargo test -p g2g-plugins --features alsa-sink --test m591_alsasink_drift_clock`.
//! Validated on this Fedora / PipeWire host (pipewire-alsa); plays a brief tone.
#![cfg(feature = "alsa-sink")]

use std::future::Future;
use std::pin::Pin;
use std::thread::sleep;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ClockPriority, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::alsasink::AlsaSink;

const RATE: u32 = 48_000;
const FREQ: f32 = 440.0;
/// Long enough that the worker writes many buffers and the clock gathers a real
/// window of observations, short enough to stay unobtrusive.
const SECONDS: f32 = 0.6;
/// One pushed buffer (10 ms), so the sink sees dozens of `DataFrame`s and the
/// worker disciplines the clock once per write.
const CHUNK_FRAMES: usize = (RATE as usize) / 100;

struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn tone(frames: usize, phase0: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..frames {
        let t = (phase0 + i) as f32 / RATE as f32;
        let s = (2.0 * core::f32::consts::PI * FREQ * t).sin() * 0.2;
        let sample = (s * i16::MAX as f32) as i16;
        for _ in 0..2 {
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    out
}

fn pcm_frame(bytes: Vec<u8>, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming { pts_ns: seq, ..Default::default() },
        sequence: seq,
        meta: Default::default(),
    }
}

#[test]
fn alsasink_disciplines_its_clock_from_the_device() {
    let mut sink = AlsaSink::new();

    // The offered clock: an AudioProvider so audio becomes the sync master.
    let cand = sink.provide_clock().expect("alsasink offers a clock by default");
    assert_eq!(cand.priority, ClockPriority::AudioProvider);
    let clock = sink.clock();

    let caps = Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: RATE };
    match sink.configure_pipeline(&caps) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            eprintln!("skip m590: no reachable ALSA device");
            return;
        }
        Err(e) => panic!("alsasink configure error: {e:?}"),
    }

    // Feed the tone in 10 ms chunks. `process` only queues to the worker; the
    // worker writes them (blocking-paced at the device rate) and disciplines the
    // clock once per write. `Eos` blocks until that whole backlog has drained,
    // so by the time it returns the clock has seen a full window of the real
    // playout.
    let total = (SECONDS * RATE as f32) as usize;
    let mut done = 0usize;
    let mut seq = 0u64;
    let mut null = NullSink;
    while done < total {
        let n = CHUNK_FRAMES.min(total - done);
        block_on(sink.process(PipelinePacket::DataFrame(pcm_frame(tone(n, done), seq)), &mut null))
            .expect("process");
        done += n;
        seq += 1;
    }
    block_on(sink.process(PipelinePacket::Eos, &mut null)).expect("eos");

    assert!(sink.frames_rendered() > 0, "alsasink rendered no frames");

    // The worker must have fed the clock a real window of observations, not
    // left it at the pass-through fallback.
    let obs = clock.observations();
    assert!(obs >= 2, "clock got only {obs} observations; discipline did not run");

    // Both the reference (monotonic) and the master (DAC playout) are real time,
    // so the estimated rate must be close to 1.0 (a small ppm drift, not garbage).
    let slope = clock.slope();
    assert!(
        (0.9..1.1).contains(&slope),
        "playout-rate estimate {slope} is implausible (expected ~1.0)",
    );

    // A disciplined clock is live: projecting the current reference through the
    // (now frozen) fit still advances, by roughly the elapsed real time (the
    // master tracks wall time at ~1.0x).
    let t0 = clock.now_ns();
    sleep(Duration::from_millis(50));
    let t1 = clock.now_ns();
    let advanced = t1.saturating_sub(t0);
    assert!(t1 > t0, "clock did not advance ({t0} -> {t1})");
    assert!(
        (20_000_000..120_000_000).contains(&advanced),
        "clock advanced {advanced} ns over a 50 ms sleep (expected ~50 ms)",
    );

    eprintln!(
        "m590 alsasink: {obs} observations, slope {slope:.6}, advanced {} ms over 50 ms sleep",
        advanced / 1_000_000
    );
}
