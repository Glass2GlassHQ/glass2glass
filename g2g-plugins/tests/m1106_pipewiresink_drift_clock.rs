//! M1106: PipeWireSink playout-disciplined clock host test.
//!
//! Drives a real `PipeWireSink` on the host's PipeWire graph and asserts the
//! sink disciplines its provided clock from the stream time. It plays a short
//! tone, then checks that (1) the sink offers its clock to election as an
//! `AudioProvider`, (2) the realtime callback fed the clock multiple
//! `pw_stream_get_time_n` observations, (3) the estimated rate is a sane ~1.0x
//! (both timelines are real time), and (4) the clock is live (its `now_ns()`
//! advances across a real sleep).
//!
//! A host with no reachable PipeWire daemon (CI, headless) skips:
//! `configure_pipeline` fails loud with a hardware error, treated as "no
//! device" not a failure. Run:
//! `cargo test -p g2g-plugins --features pipewire --test m1106_pipewiresink_drift_clock`.
#![cfg(feature = "pipewire")]

use std::thread::sleep;
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ClockPriority, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PushOutcome,
};
use g2g_plugins::pipewiresink::PipeWireSink;

const RATE: u32 = 48_000;
const FREQ: f32 = 440.0;
/// Long enough that the realtime callback runs for many graph cycles and the
/// clock gathers a real window of observations, short enough to stay
/// unobtrusive.
const SECONDS: f32 = 0.6;
const CHUNK_FRAMES: usize = (RATE as usize) / 100;

struct NullSink;
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
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
        timing: FrameTiming {
            pts_ns: seq,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

#[test]
fn pipewiresink_disciplines_its_clock_from_the_stream() {
    let mut sink = PipeWireSink::new();

    let cand = sink
        .provide_clock()
        .expect("pipewiresink offers a clock by default");
    assert_eq!(cand.priority, ClockPriority::AudioProvider);
    let clock = sink.clock();

    let caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: RATE,
    };
    match sink.configure_pipeline(&caps) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            eprintln!("skip m1106: no reachable PipeWire daemon");
            return;
        }
        Err(e) => panic!("pipewiresink configure error: {e:?}"),
    }

    // Feed the tone in 10 ms chunks. `process` only queues; the stream's
    // realtime callback drains the queue at the graph rate, disciplining the
    // clock once per cycle. `Eos` waits for the queue to drain, so by the time
    // it returns the clock has seen the whole playout window.
    let total = (SECONDS * RATE as f32) as usize;
    let mut done = 0usize;
    let mut seq = 0u64;
    let mut null = NullSink;
    while done < total {
        let n = CHUNK_FRAMES.min(total - done);
        block_on(sink.process(
            PipelinePacket::DataFrame(pcm_frame(tone(n, done), seq)),
            &mut null,
        ))
        .expect("process");
        done += n;
        seq += 1;
    }
    block_on(sink.process(PipelinePacket::Eos, &mut null)).expect("eos");

    let obs = clock.observations();
    assert!(
        obs >= 2,
        "clock got only {obs} observations; discipline did not run"
    );

    let slope = clock.slope();
    assert!(
        (0.9..1.1).contains(&slope),
        "playout-rate estimate {slope} is implausible (expected ~1.0)",
    );

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
        "m1106 pipewiresink: {obs} observations, slope {slope:.6}, advanced {} ms over 50 ms sleep",
        advanced / 1_000_000
    );
}
