//! M1148: the audio capture sources offer a disciplined clock, the input mirror
//! of the M1106 sink-side playout clock.
//!
//! Two things per element: the offered candidate sits at `Provider`, one rank
//! below the sinks' `AudioProvider` (a duplex pipeline keeps the playout clock
//! as master, a capture-only one elects this instead of the monotonic
//! fallback), and a real capture run disciplines the clock from the device's own
//! position (`snd_pcm_delay` for ALSA, `pw_stream_get_time_n` for PipeWire) so
//! its rate estimate is a sane ~1.0x of wall time.
//!
//! A host with no reachable capture device skips the device half: `run` fails
//! loud with a hardware error, read here as "no device" rather than a failure.
//! Run with the features built:
//! `cargo test -p g2g-plugins --features alsa-src,pipewire --test m1148_capture_drift_clock`.
#![cfg(any(feature = "alsa-src", feature = "pipewire"))]

use g2g_core::runtime::{block_on, SourceLoop};
use g2g_core::{ClockPriority, G2gError, OutputSink, PipelinePacket, PushOutcome};

/// Buffers to pull before checking the clock. At the ALSA default 10 ms period
/// that is ~300 ms of audio, and at a PipeWire 1024-frame quantum ~640 ms:
/// either way many device-position observations.
const BUFFERS: u64 = 30;

/// The rate estimate is a ratio of two real-time timelines, so it must sit near
/// 1.0; a card is off by tens to hundreds of ppm, never percent.
const PLAUSIBLE_RATE: core::ops::Range<f64> = 0.9..1.1;

struct Drain;
impl OutputSink for Drain {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Negotiate, configure and run `src` to its buffer limit. `false` when the
/// device / server is unreachable (the skip path).
fn capture<S: SourceLoop>(src: &mut S) -> bool {
    let caps = block_on(src.intercept_caps()).expect("capture source produces caps");
    match src.configure_pipeline(&caps) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => return false,
        Err(e) => panic!("configure error: {e:?}"),
    }
    let mut out = Drain;
    match block_on(src.run(&mut out)) {
        Ok(_) => true,
        Err(G2gError::Hardware(_)) => false,
        Err(e) => panic!("run error: {e:?}"),
    }
}

/// The clock is only worth electing once the device has actually disciplined it:
/// a real two-point rate estimate, and a slope that is a ratio of two real-time
/// timelines.
fn assert_disciplined(clock: &g2g_core::DriftClock, element: &str) {
    let observations = clock.observations();
    assert!(
        observations >= 2,
        "{element} clock got only {observations} observations; discipline did not run"
    );
    let slope = clock.slope();
    assert!(
        PLAUSIBLE_RATE.contains(&slope),
        "{element} capture-rate estimate {slope} is implausible (expected ~1.0)"
    );
    eprintln!("m1148 {element}: {observations} observations, slope {slope:.6}");
}

#[cfg(feature = "alsa-src")]
mod alsa {
    use super::*;
    use g2g_plugins::alsasrc::AlsaSrc;

    #[test]
    fn offers_a_provider_clock() {
        let cand = AlsaSrc::new()
            .provide_clock()
            .expect("alsasrc offers a clock by default");
        assert_eq!(cand.priority, ClockPriority::Provider);
    }

    #[test]
    fn a_real_capture_disciplines_the_clock() {
        let mut src = AlsaSrc::new().with_num_buffers(BUFFERS);
        let clock = src.clock();
        if !capture(&mut src) {
            eprintln!("skip m1148 alsasrc: no reachable ALSA capture device");
            return;
        }
        assert_disciplined(&clock, "alsasrc");
    }

    /// `provide-clock=false` keeps the per-period `snd_pcm_delay` probe out of
    /// the read loop entirely, so the clock stays untouched.
    #[test]
    fn provide_clock_off_leaves_the_clock_undisciplined() {
        let mut src = AlsaSrc::new().with_num_buffers(BUFFERS);
        src.set_property("provide-clock", g2g_core::PropValue::Bool(false))
            .unwrap();
        let clock = src.clock();
        if !capture(&mut src) {
            eprintln!("skip m1148 alsasrc provide-clock=false: no reachable ALSA capture device");
            return;
        }
        assert_eq!(clock.observations(), 0, "disabled clock was still fed");
    }
}

#[cfg(feature = "pipewire")]
mod pipewire {
    use super::*;
    use g2g_plugins::pipewiresrc::PipeWireSrc;

    #[test]
    fn offers_a_provider_clock() {
        let cand = PipeWireSrc::new()
            .provide_clock()
            .expect("pipewiresrc offers a clock by default");
        assert_eq!(cand.priority, ClockPriority::Provider);
    }

    #[test]
    fn a_real_capture_disciplines_the_clock() {
        let mut src = PipeWireSrc::new().with_frame_limit(BUFFERS);
        let clock = src.clock();
        if !capture(&mut src) {
            eprintln!("skip m1148 pipewiresrc: no reachable PipeWire daemon");
            return;
        }
        assert_disciplined(&clock, "pipewiresrc");
    }
}
