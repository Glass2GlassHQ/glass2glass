//! M1083 debug and test elements: `breakmydata` corrupts bytes on the way past,
//! `chopmydata` re-chunks a byte stream on a step boundary, `checksumsink`
//! digests every buffer, `errorignore` absorbs a failure from downstream, the
//! two media-typed fake sinks refuse a stream that was never decoded, and
//! `fpsdisplaysink` reports the rate its child achieves.
//!
//! `default_registry` and `fpsdisplaysink`'s clock are `std`-gated, so this file
//! is too: run with `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::task::{Context, Poll};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, FrameTiming,
    G2gError, HardwareError, Interlace, MemoryDomain, OutputSink, PipelinePacket, PropValue,
    PushOutcome, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::breakmydata::BreakMyData;
use g2g_plugins::checksumsink::ChecksumSink;
use g2g_plugins::chopmydata::ChopMyData;
use g2g_plugins::errorignore::ErrorIgnore;
use g2g_plugins::fakemediasink::{FakeAudioSink, FakeVideoSink};
use g2g_plugins::fpsdisplaysink::FpsDisplaySink;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    /// The payload of every data frame pushed.
    fn payloads(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => {
                    Some(f.domain.as_system_slice().expect("system frame").to_vec())
                }
                _ => None,
            })
            .collect()
    }
}

/// A downstream that fails every push, the stand-in for a branch that died.
#[derive(Default)]
struct FailingSink {
    pushes: u64,
    error: Option<G2gError>,
}

impl FailingSink {
    fn with_error(error: G2gError) -> Self {
        Self {
            pushes: 0,
            error: Some(error),
        }
    }
}

impl OutputSink for FailingSink {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take().expect("poll_push without a packet");
        self.pushes += 1;
        match &self.error {
            Some(error) => Poll::Ready(Err(error.clone())),
            None => Poll::Ready(Ok(PushOutcome::Accepted)),
        }
    }
}

fn byte_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

fn byte_stream_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

// ---------------------------------------------------------------------------
// breakmydata
// ---------------------------------------------------------------------------

/// Two buffers, with the skip landing inside the second one.
const CORRUPT_BUFFERS: u64 = 2;
const CORRUPT_BUFFER_BYTES: usize = 16;
const CORRUPT_SKIP: u64 = 24;
/// The value `set-to` writes over a broken byte.
const CORRUPT_SET_TO: i64 = 0xaa;
const CORRUPT_SEED: u64 = 7;

/// The input bytes, numbered so a corrupted one is visible.
fn intact_stream() -> Vec<u8> {
    (0..CORRUPT_BUFFERS as usize * CORRUPT_BUFFER_BYTES)
        .map(|i| i as u8)
        .collect()
}

/// Run `breakmydata` over [`intact_stream`] at the given probability, returning
/// what came out joined back together.
async fn run_corrupter(probability: f64, seed: u64) -> (Vec<u8>, u64) {
    let mut breaker = BreakMyData::new();
    breaker
        .set_property("probability", PropValue::Double(probability))
        .expect("`probability` property");
    breaker
        .set_property("set-to", PropValue::Int(CORRUPT_SET_TO))
        .expect("`set-to` property");
    breaker
        .set_property("skip", PropValue::Uint(CORRUPT_SKIP))
        .expect("`skip` property");
    breaker
        .set_property("seed", PropValue::Uint(seed))
        .expect("`seed` property");
    breaker
        .configure_pipeline(&byte_stream_caps())
        .expect("configure");

    let mut out = Collect::default();
    let stream = intact_stream();
    for sequence in 0..CORRUPT_BUFFERS {
        let start = sequence as usize * CORRUPT_BUFFER_BYTES;
        let bytes = stream[start..start + CORRUPT_BUFFER_BYTES].to_vec();
        breaker
            .process(byte_frame(bytes, sequence, sequence), &mut out)
            .await
            .expect("corrupting");
    }
    (out.payloads().concat(), breaker.corrupted())
}

#[tokio::test]
async fn breakmydata_leaves_the_skipped_bytes_alone_and_breaks_the_rest() {
    let (broken, corrupted) = run_corrupter(1.0, CORRUPT_SEED).await;
    let intact = intact_stream();
    let skip = CORRUPT_SKIP as usize;

    assert_eq!(
        &broken[..skip],
        &intact[..skip],
        "`skip` counts from the start of the stream, across buffers"
    );
    assert!(
        broken[skip..]
            .iter()
            .all(|b| i64::from(*b) == CORRUPT_SET_TO),
        "at probability 1 every byte past the skip takes `set-to`"
    );
    assert_eq!(
        corrupted,
        (intact.len() - skip) as u64,
        "every byte past the skip was overwritten, once"
    );
}

#[tokio::test]
async fn breakmydata_at_zero_probability_changes_nothing() {
    let (untouched, corrupted) = run_corrupter(0.0, CORRUPT_SEED).await;
    assert_eq!(untouched, intact_stream(), "no byte is touched");
    assert_eq!(corrupted, 0);
}

#[tokio::test]
async fn breakmydata_repeats_its_corruption_on_the_same_seed() {
    let (first, first_count) = run_corrupter(0.5, CORRUPT_SEED).await;
    let (second, second_count) = run_corrupter(0.5, CORRUPT_SEED).await;
    assert_eq!(first, second, "the same seed breaks the same bytes");
    assert_eq!(first_count, second_count);

    let intact = intact_stream();
    assert_ne!(first, intact, "half the bytes past the skip are broken");
    assert_eq!(
        &first[..CORRUPT_SKIP as usize],
        &intact[..CORRUPT_SKIP as usize],
        "the skip still holds at a partial probability"
    );
}

// ---------------------------------------------------------------------------
// chopmydata
// ---------------------------------------------------------------------------

const CHOP_MIN: i64 = 8;
const CHOP_MAX: i64 = 32;
const CHOP_STEP: i64 = 8;
/// A stream length that is not a whole number of `min-size` buffers, so the
/// remainder gst drops at EOS is visible.
const CHOP_INPUT_BUFFERS: u64 = 4;
const CHOP_INPUT_BUFFER_BYTES: usize = 101;

async fn run_chopper() -> Collect {
    let mut chopper = ChopMyData::new();
    for (name, value) in [
        ("min-size", CHOP_MIN),
        ("max-size", CHOP_MAX),
        ("step-size", CHOP_STEP),
    ] {
        chopper
            .set_property(name, PropValue::Int(value))
            .unwrap_or_else(|e| panic!("`{name}` property: {e:?}"));
    }
    chopper
        .configure_pipeline(&byte_stream_caps())
        .expect("configure");

    let mut out = Collect::default();
    for sequence in 0..CHOP_INPUT_BUFFERS {
        let start = sequence as usize * CHOP_INPUT_BUFFER_BYTES;
        let bytes = (start..start + CHOP_INPUT_BUFFER_BYTES)
            .map(|i| i as u8)
            .collect();
        chopper
            .process(byte_frame(bytes, sequence, sequence), &mut out)
            .await
            .expect("chopping");
    }
    chopper
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("tail flush");
    out
}

#[tokio::test]
async fn chopmydata_cuts_on_the_step_inside_the_bounds() {
    let chunks = run_chopper().await.payloads();
    assert!(chunks.len() > 1, "the stream is cut up");

    let total = CHOP_INPUT_BUFFERS as usize * CHOP_INPUT_BUFFER_BYTES;
    // The EOS tail goes out as whole `min-size` buffers, so the bytes that do
    // not fill one are dropped.
    let kept = total - total % CHOP_MIN as usize;
    let joined: Vec<u8> = chunks.concat();
    assert_eq!(joined.len(), kept, "the short tail is dropped, as gst does");
    let expected: Vec<u8> = (0..kept).map(|i| i as u8).collect();
    assert_eq!(joined, expected, "the bytes keep their order");

    for chunk in &chunks {
        let len = chunk.len() as i64;
        assert_eq!(len % CHOP_STEP, 0, "{len} is not a multiple of {CHOP_STEP}");
        assert!(
            (CHOP_MIN..=CHOP_MAX).contains(&len),
            "chunk of {len} bytes is outside [{CHOP_MIN}, {CHOP_MAX}]"
        );
    }
}

#[tokio::test]
async fn chopmydata_pins_every_chunk_when_the_bounds_meet() {
    let mut chopper = ChopMyData::new();
    for (name, value) in [
        ("min-size", CHOP_MIN),
        ("max-size", CHOP_MIN),
        ("step-size", CHOP_MIN),
    ] {
        chopper
            .set_property(name, PropValue::Int(value))
            .unwrap_or_else(|e| panic!("`{name}` property: {e:?}"));
    }
    chopper
        .configure_pipeline(&byte_stream_caps())
        .expect("configure");

    let mut out = Collect::default();
    let whole = CHOP_MIN as usize * 3;
    chopper
        .process(byte_frame(vec![0u8; whole], 0, 0), &mut out)
        .await
        .expect("chopping");
    let lengths: Vec<usize> = out.payloads().iter().map(Vec::len).collect();
    assert_eq!(
        lengths,
        vec![CHOP_MIN as usize; 3],
        "one size to draw means every chunk is that size"
    );
}

// ---------------------------------------------------------------------------
// checksumsink
// ---------------------------------------------------------------------------

/// The message the standard hash test vectors are stated for.
const DIGEST_INPUT: &[u8] = b"abc";
/// FIPS 180-4 / RFC 1321 digests of [`DIGEST_INPUT`], the published vectors.
const MD5_OF_ABC: &str = "900150983cd24fb0d6963f7d28e17f72";
const SHA1_OF_ABC: &str = "a9993e364706816aba3e25717850c26c9cd0d89d";
const SHA256_OF_ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const SHA512_OF_ABC: &str = concat!(
    "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a",
    "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
);
/// One and a half seconds, and the same instant in gst's `GST_TIME_FORMAT`.
const DIGEST_PTS_NS: u64 = 1_500_000_000;
const DIGEST_PTS_TEXT: &str = "0:00:01.500000000";

#[tokio::test]
async fn checksumsink_reports_each_buffer_under_the_named_hash() {
    for (hash, expected) in [
        ("md5", MD5_OF_ABC),
        ("sha1", SHA1_OF_ABC),
        ("sha256", SHA256_OF_ABC),
        ("sha512", SHA512_OF_ABC),
    ] {
        let (bus, handle) = Bus::new(4);
        let mut sink = ChecksumSink::new().with_bus(handle);
        sink.set_property("hash", PropValue::Str(hash.into()))
            .unwrap_or_else(|e| panic!("`hash={hash}`: {e:?}"));
        sink.configure_pipeline(&byte_stream_caps())
            .expect("configure");

        let mut out = Collect::default();
        sink.process(
            byte_frame(DIGEST_INPUT.to_vec(), DIGEST_PTS_NS, 0),
            &mut out,
        )
        .await
        .expect("digesting");

        assert_eq!(sink.digested(), 1);
        assert_eq!(
            sink.last_line(),
            format!("{DIGEST_PTS_TEXT} {expected}"),
            "{hash} of {:?} with its pts",
            core::str::from_utf8(DIGEST_INPUT).expect("ascii")
        );
        assert_eq!(
            bus.try_recv(),
            Some(BusMessage::Info(sink.last_line().into())),
            "the line the application collects is the line the sink kept"
        );
    }
}

#[tokio::test]
async fn checksumsink_defaults_to_sha1() {
    let mut sink = ChecksumSink::new();
    sink.configure_pipeline(&byte_stream_caps())
        .expect("configure");
    let mut out = Collect::default();
    sink.process(byte_frame(DIGEST_INPUT.to_vec(), 0, 0), &mut out)
        .await
        .expect("digesting");
    assert!(
        sink.last_line().ends_with(SHA1_OF_ABC),
        "gst's default hash is sha1, got {}",
        sink.last_line()
    );
}

// ---------------------------------------------------------------------------
// errorignore
// ---------------------------------------------------------------------------

/// Buffers pushed at a branch that fails every one of them.
const IGNORED_BUFFERS: u64 = 3;

/// The failure a dead branch downstream reports.
fn branch_failure() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

async fn configured_error_ignore(convert_to: &str) -> ErrorIgnore {
    let mut ignore = ErrorIgnore::new();
    ignore
        .set_property("convert-to", PropValue::Str(convert_to.into()))
        .expect("`convert-to` property");
    ignore
        .configure_pipeline(&byte_stream_caps())
        .expect("configure");
    ignore
}

#[tokio::test]
async fn errorignore_converts_every_failure_it_is_told_to() {
    let mut ignore = configured_error_ignore("ok").await;
    let mut out = FailingSink::with_error(branch_failure());

    for sequence in 0..IGNORED_BUFFERS {
        ignore
            .process(byte_frame(vec![0u8; 4], sequence, sequence), &mut out)
            .await
            .expect("the failure is converted to success");
    }
    assert_eq!(ignore.ignored(), IGNORED_BUFFERS);
    assert_eq!(
        out.pushes, IGNORED_BUFFERS,
        "every buffer is offered downstream, as gst's stateless element does"
    );
}

#[tokio::test]
async fn errorignore_reports_the_error_it_is_not_told_to_ignore() {
    let mut ignore = configured_error_ignore("ok").await;
    ignore
        .set_property("ignore-error", PropValue::Bool(false))
        .expect("`ignore-error` property");
    let mut out = FailingSink::with_error(branch_failure());

    assert_eq!(
        ignore
            .process(byte_frame(vec![0u8; 4], 0, 0), &mut out)
            .await
            .err(),
        Some(branch_failure()),
        "an error no property covers passes through"
    );
    assert_eq!(ignore.ignored(), 0);
}

/// `convert-to` names what the element reports instead. gst's default is
/// `not-linked`, which is g2g's "downstream is gone".
#[tokio::test]
async fn errorignore_convert_to_names_the_replacement() {
    let mut ignore = configured_error_ignore("not-linked").await;
    let mut out = FailingSink::with_error(branch_failure());
    assert_eq!(
        ignore
            .process(byte_frame(vec![0u8; 4], 0, 0), &mut out)
            .await
            .err(),
        Some(G2gError::Shutdown)
    );
}

/// A caps mismatch downstream is gst's `not-negotiated`, covered by its own
/// property rather than by `ignore-error`.
#[tokio::test]
async fn errorignore_separates_a_caps_mismatch_from_a_plain_error() {
    let mut ignore = configured_error_ignore("ok").await;
    ignore
        .set_property("ignore-error", PropValue::Bool(false))
        .expect("`ignore-error` property");
    let mut out = FailingSink::with_error(G2gError::CapsMismatch);
    ignore
        .process(byte_frame(vec![0u8; 4], 0, 0), &mut out)
        .await
        .expect("`ignore-notnegotiated` is on by default");

    ignore
        .set_property("ignore-notnegotiated", PropValue::Bool(false))
        .expect("`ignore-notnegotiated` property");
    assert_eq!(
        ignore
            .process(byte_frame(vec![0u8; 4], 1, 1), &mut out)
            .await
            .err(),
        Some(G2gError::CapsMismatch),
        "with the property off the mismatch passes through"
    );
}

// ---------------------------------------------------------------------------
// fakevideosink / fakeaudiosink
// ---------------------------------------------------------------------------

const FAKE_WIDTH: u32 = 320;
const FAKE_HEIGHT: u32 = 240;
/// 30 fps in the Q16 fixed point `Rate::Fixed` carries.
const FAKE_FPS_Q16: u32 = 30 << 16;
const FAKE_SAMPLE_RATE: u32 = 48_000;
const FAKE_CHANNELS: u8 = 2;
const FAKE_BUFFERS: u64 = 3;
const FAKE_BUFFER_BYTES: usize = 64;

fn raw_video_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(FAKE_WIDTH),
        height: Dim::Fixed(FAKE_HEIGHT),
        framerate: Rate::Fixed(FAKE_FPS_Q16),
        interlace: Interlace::Progressive,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn coded_video_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(FAKE_WIDTH),
        height: Dim::Fixed(FAKE_HEIGHT),
        framerate: Rate::Fixed(FAKE_FPS_Q16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn pcm_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: FAKE_CHANNELS,
        sample_rate: FAKE_SAMPLE_RATE,
    }
}

fn coded_audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: FAKE_CHANNELS,
        sample_rate: FAKE_SAMPLE_RATE,
    }
}

#[tokio::test]
async fn fakevideosink_takes_raw_video_and_refuses_the_rest() {
    let sink = FakeVideoSink::new();
    assert_eq!(
        sink.intercept_caps(&raw_video_caps()).ok(),
        Some(raw_video_caps())
    );
    assert_eq!(
        sink.intercept_caps(&coded_video_caps()).err(),
        Some(G2gError::CapsMismatch),
        "an undecoded stream is what this sink exists to refuse"
    );
    assert_eq!(
        sink.intercept_caps(&pcm_caps()).err(),
        Some(G2gError::CapsMismatch)
    );
}

#[tokio::test]
async fn fakeaudiosink_takes_pcm_and_refuses_the_rest() {
    let sink = FakeAudioSink::new();
    assert_eq!(sink.intercept_caps(&pcm_caps()).ok(), Some(pcm_caps()));
    assert_eq!(
        sink.intercept_caps(&coded_audio_caps()).err(),
        Some(G2gError::CapsMismatch),
        "`Caps::Audio` also carries the encoded formats"
    );
    assert_eq!(
        sink.intercept_caps(&raw_video_caps()).err(),
        Some(G2gError::CapsMismatch)
    );
}

#[tokio::test]
async fn fakevideosink_counts_what_it_swallowed() {
    let mut sink = FakeVideoSink::new();
    sink.set_property("silent", PropValue::Bool(false))
        .expect("`silent` property");
    sink.configure_pipeline(&raw_video_caps())
        .expect("configure");

    let mut out = Collect::default();
    for sequence in 0..FAKE_BUFFERS {
        sink.process(
            byte_frame(vec![0u8; FAKE_BUFFER_BYTES], sequence, sequence),
            &mut out,
        )
        .await
        .expect("swallowing");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");

    assert_eq!(sink.received(), FAKE_BUFFERS);
    assert!(sink.eos_seen());
    assert!(
        sink.last_message().contains(&FAKE_BUFFER_BYTES.to_string()),
        "`silent=false` records the buffer, got {:?}",
        sink.last_message()
    );
    assert!(
        out.packets.is_empty(),
        "a sink is terminal: nothing goes downstream"
    );
}

#[tokio::test]
async fn fakeaudiosink_stays_quiet_by_default() {
    let mut sink = FakeAudioSink::new();
    sink.configure_pipeline(&pcm_caps()).expect("configure");
    let mut out = Collect::default();
    sink.process(byte_frame(vec![0u8; FAKE_BUFFER_BYTES], 0, 0), &mut out)
        .await
        .expect("swallowing");
    assert_eq!(sink.received(), 1);
    assert_eq!(
        sink.last_message(),
        "",
        "gst's fake sinks default to silent"
    );
}

// ---------------------------------------------------------------------------
// fpsdisplaysink
// ---------------------------------------------------------------------------

/// A report interval short enough that every buffer past the first triggers one,
/// and the gap between buffers that makes it so.
const FPS_UPDATE_INTERVAL_MS: i64 = 1;
const FPS_FRAME_GAP_MS: u64 = 5;
const FPS_FRAMES: u64 = 3;

#[tokio::test]
async fn fpsdisplaysink_reports_what_its_child_rendered() {
    let (bus, handle) = Bus::new(16);
    let mut sink = FpsDisplaySink::new()
        .with_video_sink("fakesink")
        .with_bus(handle);
    sink.set_property(
        "fps-update-interval",
        PropValue::Int(FPS_UPDATE_INTERVAL_MS),
    )
    .expect("`fps-update-interval` property");
    sink.configure_pipeline(&raw_video_caps())
        .expect("the child accepts the caps");

    let mut out = Collect::default();
    for sequence in 0..FPS_FRAMES {
        sink.process(
            byte_frame(vec![0u8; FAKE_BUFFER_BYTES], sequence, sequence),
            &mut out,
        )
        .await
        .expect("presenting");
        tokio::time::sleep(Duration::from_millis(FPS_FRAME_GAP_MS)).await;
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");

    assert_eq!(sink.frames_rendered(), FPS_FRAMES);
    assert_eq!(sink.frames_dropped(), 0, "a fakesink child drops nothing");
    assert_eq!(
        sink.get_property("frames-rendered"),
        Some(PropValue::Uint(FPS_FRAMES)),
        "the count is readable as gst's read-only property"
    );

    let posted: Vec<String> = std::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::Info(text) => Some(text),
            _ => None,
        })
        .collect();
    assert!(
        posted
            .iter()
            .any(|line| line.starts_with(&format!("rendered: {FPS_FRAMES}, dropped: 0,"))),
        "a periodic report names the running counts: {posted:?}"
    );
    assert!(
        posted.iter().any(|line| line.starts_with("Max-fps:")),
        "eos reports the run's maximum, minimum and average: {posted:?}"
    );
    assert!(
        sink.last_message().starts_with("Max-fps:"),
        "the final report stays readable, got {:?}",
        sink.last_message()
    );
}

/// The child is a registered element name, and this element cannot be its own.
#[test]
fn fpsdisplaysink_refuses_itself_as_its_child() {
    let mut sink = FpsDisplaySink::new();
    assert!(sink
        .set_property("video-sink", PropValue::Str("fpsdisplaysink".into()))
        .is_err());
    assert_eq!(
        sink.get_property("video-sink"),
        Some(PropValue::Str("autovideosink".into())),
        "gst's default, resolved through the alias chain at build time"
    );
}
