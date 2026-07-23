//! M307: on-device probe for the Android AAudio elements.
//!
//! The render path (`AAudioSink`) is validated hard: open the default output
//! stream, write synthetic PCM, assert frames reached the device. The capture
//! path (`AAudioSrc`) needs the `RECORD_AUDIO` runtime permission, which a bare
//! `/data/local/tmp` native binary (no APK / manifest) does not hold, so it is
//! attempted best-effort and reported, not asserted (an APK harness is the real
//! capture validation).
//!
//! Runs only on `aarch64-linux-android` (et al.) with the `aaudio` feature.
//! Build with cargo-ndk `--platform 26` (AAudio is API 26+), push, run as a bare
//! native binary. See `tools/android-aaudio-smoke.sh`.

#![cfg(all(target_os = "android", feature = "aaudio"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{AudioFormat, Caps, ConfigureOutcome, G2gError};
use g2g_plugins::aaudio::{AAudioSink, AAudioSrc};

#[derive(Default)]
struct Discard;
impl OutputSink for Discard {
    fn push<'a>(&'a mut self, _p: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

/// Counts the PCM data frames a source pushes.
#[derive(Default)]
struct CountFrames {
    frames: u64,
    bytes: usize,
}
impl OutputSink for CountFrames {
    fn push<'a>(&'a mut self, p: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        if let PipelinePacket::DataFrame(f) = &p {
            self.frames += 1;
            if let Some(s) = f.domain.as_system_slice() {
                self.bytes += s.len();
            }
        }
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

/// 10 ms of interleaved S16LE stereo sine at `rate`, a 440 Hz tone.
fn sine_buffer(rate: u32, channels: u8, seq: u64) -> Vec<u8> {
    let frames = (rate as u64 / 100).max(1); // 10 ms
    let mut buf = Vec::with_capacity(frames as usize * channels as usize * 2);
    for n in 0..frames {
        let t = (seq * frames + n) as f32 / rate as f32;
        // crude sine via the standard library (std is on for this feature).
        let v = (libm_sin(2.0 * core::f32::consts::PI * 440.0 * t) * 8000.0) as i16;
        for _ in 0..channels {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// `sin` without pulling libm: the test only needs an audible-ish wave, so a
/// 5-term Taylor series around the reduced angle is plenty.
fn libm_sin(x: f32) -> f32 {
    let tau = 2.0 * core::f32::consts::PI;
    let mut r = x % tau;
    if r > core::f32::consts::PI {
        r -= tau;
    } else if r < -core::f32::consts::PI {
        r += tau;
    }
    let (x2, mut term, mut sum) = (r * r, r, r);
    for k in 1..5 {
        term *= -x2 / (((2 * k) * (2 * k + 1)) as f32);
        sum += term;
    }
    sum
}

#[tokio::test]
async fn aaudio_sink_renders_pcm() {
    let (rate, channels) = (48_000u32, 2u8);
    let mut sink = AAudioSink::new();
    let caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels,
        sample_rate: rate,
    };
    let narrowed = sink.intercept_caps(&caps).expect("intercept caps");
    assert!(matches!(
        sink.configure_pipeline(&narrowed).expect("configure sink"),
        ConfigureOutcome::Accepted
    ));

    let mut nil = Discard;
    for seq in 0..50 {
        let pcm = sine_buffer(rate, channels, seq);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        };
        sink.process(PipelinePacket::DataFrame(frame), &mut nil)
            .await
            .expect("render buffer");
    }
    sink.process(PipelinePacket::Eos, &mut nil)
        .await
        .expect("eos");

    eprintln!(
        "=== M307 AAudioSink: {} PCM frames rendered ===",
        sink.rendered()
    );
    assert!(
        sink.rendered() > 0,
        "AAudioSink wrote no frames to the device"
    );
    eprintln!(">>> M307 AAudio render validated on device.");
}

#[tokio::test]
async fn aaudio_src_captures_pcm_best_effort() {
    // Capture needs RECORD_AUDIO; a bare native binary lacks it. Attempt and
    // report, do not fail the suite on a permission denial.
    let mut src = AAudioSrc::new(48_000, 2, 5);
    let caps = match SourceLoop::intercept_caps(&mut src).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!(">>> AAudioSrc open failed ({e:?}) - likely no RECORD_AUDIO; skipping capture probe");
            return;
        }
    };
    eprintln!("=== M307 AAudioSrc opened: {caps:?} ===");
    if src.configure_pipeline(&caps).is_err() {
        eprintln!(">>> AAudioSrc start failed; skipping capture probe");
        return;
    }
    let mut out = CountFrames::default();
    match src.run(&mut out).await {
        Ok(n) => {
            eprintln!(">>> captured {n} buffers ({} bytes)", out.bytes);
            assert_eq!(out.frames, n, "every captured buffer is a data frame");
        }
        Err(e) => eprintln!(">>> capture read failed ({e:?}); permission/headless limitation"),
    }
}
