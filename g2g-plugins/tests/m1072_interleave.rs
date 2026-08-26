//! M1072 `interleave` / `deinterleave`: channel routing. N mono streams merge
//! into one N-channel stream and back, through the launch DSL (fan-in pads and
//! fan-out pads) and through direct `process` calls for the alignment and
//! rejection cases the DSL cannot stage.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::ffi::c_void;
use std::sync::Mutex;

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AudioFormat, Caps, G2gError, MultiInputElement, MultiOutputElement, OutputSink, PipelineClock,
    PushOutcome,
};
use g2g_plugins::appsink::set_appsink_callback;
use g2g_plugins::deinterleave::DeinterleaveN;
use g2g_plugins::interleave::Interleave;
use g2g_plugins::registry::default_registry;

const RATE: u32 = 48_000;
const S16_BYTES: usize = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One `appsink channel=` collector: the payload of every frame it receives.
#[derive(Default)]
struct Recorder {
    frames: Vec<Vec<u8>>,
}

extern "C" fn record(data: *const u8, len: usize, _pts_ns: u64, user: *mut c_void) {
    // SAFETY: `user` is the &Mutex<Recorder> registered for the channel, alive
    // for the whole run.
    let recorder = unsafe { &*(user as *const Mutex<Recorder>) };
    if data.is_null() && len == 0 {
        return; // the EOS marker carries no payload
    }
    // SAFETY: appsink passes `len` readable bytes for the call.
    let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
    recorder.lock().unwrap().frames.push(bytes);
}

/// Run `pipeline`, whose sinks are `appsink channel=<name>` for each name in
/// `channels`, and return what each recorded.
async fn run_recording(pipeline: &str, channels: &[&str]) -> Vec<Vec<Vec<u8>>> {
    let recorders: Vec<Box<Mutex<Recorder>>> = channels
        .iter()
        .map(|_| Box::new(Mutex::new(Recorder::default())))
        .collect();
    for (channel, recorder) in channels.iter().zip(&recorders) {
        let user = (&**recorder as *const Mutex<Recorder>) as *mut c_void;
        set_appsink_callback(channel, record, user);
    }
    let reg = default_registry();
    let graph = parse_launch(&reg, pipeline).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    recorders
        .iter()
        .map(|r| r.lock().unwrap().frames.clone())
        .collect()
}

/// The `channel`-th sample of every sample frame of an interleaved S16LE buffer.
fn channel_of(bytes: &[u8], channels: usize, channel: usize) -> Vec<i16> {
    bytes
        .chunks_exact(channels * S16_BYTES)
        .map(|frame| {
            let at = channel * S16_BYTES;
            i16::from_le_bytes([frame[at], frame[at + 1]])
        })
        .collect()
}

#[tokio::test]
async fn interleave_merges_two_mono_sources_into_stereo() {
    const BUFFERS: usize = 10;
    let merged = run_recording(
        "audiotestsrc num-buffers=10 channels=1 wave=sine ! i.sink_0 \
         audiotestsrc num-buffers=10 channels=1 wave=square ! i.sink_1 \
         interleave name=i ! appsink channel=m1072-merged",
        &["m1072-merged"],
    )
    .await;
    // The waveforms the pads carried, read from the same sources, so the
    // expected samples are the element's real input rather than a literal.
    let sine = run_recording(
        "audiotestsrc num-buffers=10 channels=1 wave=sine ! appsink channel=m1072-sine",
        &["m1072-sine"],
    )
    .await;
    let square = run_recording(
        "audiotestsrc num-buffers=10 channels=1 wave=square ! appsink channel=m1072-square",
        &["m1072-square"],
    )
    .await;

    let merged = &merged[0];
    assert_eq!(merged.len(), BUFFERS, "one stereo buffer per input pair");
    for (i, buffer) in merged.iter().enumerate() {
        assert_eq!(
            buffer.len(),
            sine[0][i].len() * 2,
            "a stereo buffer is twice a mono one"
        );
        assert_eq!(
            channel_of(buffer, 2, 0),
            channel_of(&sine[0][i], 1, 0),
            "the sine pad is the left channel"
        );
        assert_eq!(
            channel_of(buffer, 2, 1),
            channel_of(&square[0][i], 1, 0),
            "the square pad is the right channel"
        );
    }
}

#[tokio::test]
async fn deinterleave_splits_stereo_into_two_mono_streams() {
    const BUFFERS: usize = 10;
    let split = run_recording(
        "audiotestsrc num-buffers=10 channels=2 wave=saw ! deinterleave name=d \
         d.src_0 ! appsink channel=m1072-left \
         d.src_1 ! appsink channel=m1072-right",
        &["m1072-left", "m1072-right"],
    )
    .await;
    let stereo = run_recording(
        "audiotestsrc num-buffers=10 channels=2 wave=saw ! appsink channel=m1072-stereo",
        &["m1072-stereo"],
    )
    .await;

    assert_eq!(split[0].len(), BUFFERS, "every input buffer reached port 0");
    assert_eq!(split[1].len(), BUFFERS, "every input buffer reached port 1");
    for (i, source) in stereo[0].iter().enumerate() {
        assert_eq!(
            channel_of(&split[0][i], 1, 0),
            channel_of(source, 2, 0),
            "port 0 carries the even samples"
        );
        assert_eq!(
            channel_of(&split[1][i], 1, 0),
            channel_of(source, 2, 1),
            "port 1 carries the odd samples"
        );
    }
}

#[tokio::test]
async fn deinterleave_into_interleave_is_byte_exact() {
    let round_tripped = run_recording(
        "audiotestsrc num-buffers=5 channels=2 wave=triangle ! deinterleave name=d \
         d.src_0 ! i.sink_0 \
         d.src_1 ! i.sink_1 \
         interleave name=i ! appsink channel=m1072-roundtrip",
        &["m1072-roundtrip"],
    )
    .await;
    let source = run_recording(
        "audiotestsrc num-buffers=5 channels=2 wave=triangle ! appsink channel=m1072-source",
        &["m1072-source"],
    )
    .await;

    let flatten = |frames: &Vec<Vec<u8>>| frames.concat();
    assert_eq!(
        flatten(&round_tripped[0]),
        flatten(&source[0]),
        "splitting and re-merging returns the original samples"
    );
}

/// Collects what a fan-in pushes downstream.
#[derive(Default)]
struct Collect {
    frames: Vec<Vec<u8>>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(f)) = packet_slot.take() {
            self.frames
                .push(f.domain.as_system_slice().expect("system frame").to_vec());
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Collects what a fan-out pushes to each port.
struct PortTap {
    frames: Vec<Vec<Vec<u8>>>,
}

impl PortTap {
    fn new(ports: usize) -> Self {
        Self {
            frames: vec![Vec::new(); ports],
        }
    }
}

impl MultiOutputSink for PortTap {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(f)) = packet_slot.take() {
            self.frames[port].push(f.domain.as_system_slice().expect("system frame").to_vec());
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

fn pcm(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
    }
}

fn s16_frame(samples: &[i16]) -> PipelinePacket {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn samples_of(bytes: &[u8]) -> Vec<i16> {
    channel_of(bytes, 1, 0)
}

#[tokio::test]
async fn unequal_pad_lengths_emit_the_common_prefix_and_keep_the_remainder() {
    let mut element = Interleave::new(2);
    for pad in 0..2 {
        element
            .configure_pipeline(pad, &pcm(AudioFormat::PcmS16Le, 1, RATE))
            .expect("a mono pad configures");
    }
    let mut out = Collect::default();
    element
        .process(0, s16_frame(&[1, 2, 3]), &mut out)
        .await
        .unwrap();
    element
        .process(1, s16_frame(&[-1, -2]), &mut out)
        .await
        .unwrap();
    assert_eq!(out.frames.len(), 1, "nothing to emit until both pads have");
    assert_eq!(samples_of(&out.frames[0]), [1, -1, 2, -2]);

    // the third sample of pad 0 stayed queued and pairs up when pad 1 catches up.
    element
        .process(1, s16_frame(&[-3]), &mut out)
        .await
        .unwrap();
    assert_eq!(samples_of(&out.frames[1]), [3, -3]);
}

#[tokio::test]
async fn a_pad_at_eos_drains_the_longer_pad_with_silence() {
    let mut element = Interleave::new(2);
    let mut out = Collect::default();
    element
        .process(0, s16_frame(&[1, 2]), &mut out)
        .await
        .unwrap();
    element
        .process(1, PipelinePacket::Eos, &mut out)
        .await
        .unwrap();
    assert_eq!(
        samples_of(&out.frames[0]),
        [1, 0, 2, 0],
        "the ended pad's channel goes silent"
    );
}

#[tokio::test]
async fn a_pad_whose_rate_or_channel_count_disagrees_is_rejected() {
    let mut element = Interleave::new(2).with_rate(44_100);
    element
        .configure_pipeline(0, &pcm(AudioFormat::PcmS16Le, 1, 44_100))
        .expect("a mono pad at the declared rate configures");
    assert_eq!(
        element
            .configure_pipeline(1, &pcm(AudioFormat::PcmS16Le, 1, RATE))
            .unwrap_err(),
        G2gError::CapsMismatch,
        "the interleave never resamples, so a second rate is refused"
    );
    assert_eq!(
        element
            .configure_pipeline(1, &pcm(AudioFormat::PcmS16Le, 2, 44_100))
            .unwrap_err(),
        G2gError::CapsMismatch,
        "one channel per pad: a stereo pad has nowhere to go"
    );
    assert_eq!(
        element
            .configure_pipeline(1, &pcm(AudioFormat::PcmF32Le, 1, 44_100))
            .unwrap_err(),
        G2gError::CapsMismatch,
        "the interleave never converts, so a second format is refused"
    );
}

#[tokio::test]
async fn a_mid_stream_channel_count_change_is_rejected() {
    let mut element = DeinterleaveN::new(2);
    MultiOutputElement::configure_pipeline(&mut element, &pcm(AudioFormat::PcmS16Le, 2, RATE))
        .expect("a stereo input configures a two-port splitter");
    let mut out = PortTap::new(2);
    element
        .process(s16_frame(&[1, -1, 2, -2]), &mut out)
        .await
        .unwrap();
    assert_eq!(samples_of(&out.frames[0][0]), [1, 2]);
    assert_eq!(samples_of(&out.frames[1][0]), [-1, -2]);

    // A third channel would need a third pad, which the running graph cannot grow.
    let err = element
        .process(
            PipelinePacket::CapsChanged(pcm(AudioFormat::PcmS16Le, 3, RATE)),
            &mut out,
        )
        .await
        .unwrap_err();
    assert_eq!(err, G2gError::CapsMismatch);

    // The ports announced S16LE at negotiation, so a new format is refused too.
    let err = element
        .process(
            PipelinePacket::CapsChanged(pcm(AudioFormat::PcmF32Le, 2, RATE)),
            &mut out,
        )
        .await
        .unwrap_err();
    assert_eq!(err, G2gError::CapsMismatch);
}
