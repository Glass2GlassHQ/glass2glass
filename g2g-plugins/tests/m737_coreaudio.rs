//! M737: Core Audio elements on the macOS CI runner. The runner has no audio
//! hardware, so these probe like the Android permission-gated tests: a failed
//! device open is reported and asserted as the graceful error path; with a
//! device present (a real Mac) they render / capture for real.
#![cfg(all(target_os = "macos", feature = "coreaudio"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{AsyncElement, AudioFormat, Caps, G2gError, OutputSink, PipelineClock, PushOutcome};
use g2g_plugins::coreaudio::{CoreAudioSink, CoreAudioSrc};
use g2g_plugins::registry::default_registry;

const RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
/// 20 ms of stereo S16 per buffer.
const FRAMES_PER_BUF: usize = (RATE as usize) / 50;

#[derive(Default)]
struct Collect {
    frames: usize,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames += 1;
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn pcm_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: RATE,
    }
}

/// One buffer of a 440 Hz sine, `index` buffers into the stream.
fn sine_frame(index: usize) -> Frame {
    let mut pcm = Vec::with_capacity(FRAMES_PER_BUF * CHANNELS as usize * 2);
    for n in 0..FRAMES_PER_BUF {
        let t = (index * FRAMES_PER_BUF + n) as f32 / RATE as f32;
        let s = (t * 440.0 * core::f32::consts::TAU).sin();
        let v = (s * 8000.0) as i16;
        for _ in 0..CHANNELS {
            pcm.extend_from_slice(&v.to_le_bytes());
        }
    }
    let dur = 1_000_000_000u64 * FRAMES_PER_BUF as u64 / RATE as u64;
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: index as u64 * dur,
            dts_ns: index as u64 * dur,
            duration_ns: dur,
            capture_ns: index as u64 * dur,
            ..FrameTiming::default()
        },
        sequence: index as u64,
        meta: Default::default(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn sink_renders_or_reports_no_device() {
    let mut sink = CoreAudioSink::new();
    let narrowed = sink.intercept_caps(&pcm_caps()).expect("intercept PCM");
    match sink.configure_pipeline(&narrowed) {
        Err(G2gError::Hardware(_)) => {
            // The probe result on a device-less runner: the open failed loud
            // and structured, not with a hang or a panic.
            eprintln!("skipping render: no audio output device");
            return;
        }
        other => {
            other.expect("configure");
        }
    }
    let mut out = Collect::default();
    for i in 0..10 {
        sink.process(PipelinePacket::DataFrame(sine_frame(i)), &mut out)
            .await
            .expect("render");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("drain");
    assert_eq!(
        sink.rendered(),
        (10 * FRAMES_PER_BUF) as u64,
        "every PCM frame handed to the device"
    );
    eprintln!("rendered {} PCM frames", sink.rendered());
}

#[tokio::test(flavor = "current_thread")]
async fn sink_runs_in_a_text_pipeline_or_reports_no_device() {
    if !CoreAudioSink::device_available() {
        eprintln!("skipping launch render: no audio output device");
        return;
    }
    let reg = default_registry();
    let line = "audiotestsrc num-buffers=10 ! coreaudiosink";
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    assert_eq!(stats.frames_consumed, 10, "every buffer rendered");
}

#[tokio::test(flavor = "current_thread")]
async fn src_captures_or_reports_no_device() {
    let mut src = CoreAudioSrc::new(RATE, CHANNELS, 3);
    let caps = src.intercept_caps().await.expect("caps");
    match src.configure_pipeline(&caps) {
        Err(G2gError::Hardware(_)) => {
            eprintln!("skipping capture: no audio input device");
            return;
        }
        other => {
            other.expect("configure");
        }
    }
    let mut out = Collect::default();
    match src.run(&mut out).await {
        Ok(n) => {
            assert_eq!(n, 3, "captured the requested buffers");
            assert_eq!(out.frames, 3, "every buffer reached the sink");
            eprintln!("captured {n} buffers");
        }
        Err(G2gError::Hardware(_)) => {
            // The queue opened but delivered nothing (a headless VM's phantom
            // input): the element surfaced it within its deadline, no hang.
            eprintln!("skipping capture: input device delivered no data");
        }
        Err(e) => panic!("capture failed unexpectedly: {e:?}"),
    }
}
