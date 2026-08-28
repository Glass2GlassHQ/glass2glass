//! M1088 sources without one file behind them: `splitfilesrc` reads the parts of
//! a cut recording as one byte stream, `imagesequencesrc` stamps a file sequence
//! on a framerate grid, and `dataurisrc` plays the payload carried in a `data:`
//! URI.
//!
//! The reference is a checked-in container fixture cut into parts: the joined
//! stream has to type itself the way the whole file does and demux to the same
//! frame count, which is what makes the join byte-exact rather than merely
//! plausible.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, PipelineClock, PropValue};
use g2g_plugins::dataurisrc::DataUriSrc;
use g2g_plugins::multifilesrc::MultiFileSrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::splitfilesrc::SplitFileSrc;

/// An ffmpeg-authored MPEG-TS fixture: the whole file is the reference the parts
/// have to add back up to.
const TS_FIXTURE: &str = "aac_44100.ts";
/// The parts the fixture is cut into, at an offset that lands mid-packet so the
/// join cannot be right by accident.
const PART_COUNT: usize = 3;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g-m1088-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the working directory is created");
    dir
}

/// Cut `bytes` into `PART_COUNT` files named so their sort order is the file
/// order, and return the directory holding them.
fn write_parts(tag: &str, bytes: &[u8]) -> PathBuf {
    let dir = temp_dir(tag);
    let part_len = bytes.len().div_ceil(PART_COUNT);
    for (index, part) in bytes.chunks(part_len).enumerate() {
        std::fs::write(dir.join(format!("clip.ts.part{index:03}")), part)
            .expect("a part is written");
    }
    dir
}

/// The frames a launch line delivers to its sink.
async fn frames_consumed(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs")
        .frames_consumed
}

// ---------------------------------------------------------------------------
// splitfilesrc
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_parts_type_and_demux_like_the_whole_file() {
    let whole = std::fs::read(fixture(TS_FIXTURE)).expect("the fixture is checked in");
    let dir = write_parts("demux", &whole);

    let pattern = dir.join("clip.ts.part*").to_string_lossy().into_owned();
    let mut src = SplitFileSrc::new(&pattern);
    // No `bytestream-format`: the first part's header types the stream, since
    // `clip.ts.part000` has no usable extension.
    assert_eq!(
        futures_lite_block(src.intercept_caps()),
        Ok(Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs
        }),
        "the parts of a TS recording type as MPEG-TS"
    );

    // The fixture carries one AAC stream, which the demuxer selects by name.
    let from_parts = frames_consumed(&format!(
        "splitfilesrc location={pattern} ! tsdemux stream=aac ! fakesink"
    ))
    .await;
    let from_whole = frames_consumed(&format!(
        "filesrc location={} ! tsdemux stream=aac ! fakesink",
        fixture(TS_FIXTURE).display()
    ))
    .await;
    assert_eq!(
        from_parts, from_whole,
        "the joined parts demux to the same access units as the whole file"
    );
    assert!(from_whole > 0, "the fixture carries frames");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn the_joined_bytes_are_the_original_file() {
    let whole = std::fs::read(fixture(TS_FIXTURE)).expect("the fixture is checked in");
    let dir = write_parts("join", &whole);
    let out = dir.join("rejoined.ts");
    let line = format!(
        "splitfilesrc location={} blocksize=1024 ! filesink location={}",
        dir.join("clip.ts.part*").display(),
        out.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(
        std::fs::read(&out).expect("the sink wrote a file"),
        whole,
        "byte for byte the recording that was cut up"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// imagesequencesrc
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_image_sequence_is_stamped_on_its_framerate_grid() {
    const FPS: u32 = 10;
    const PERIOD_NS: u64 = 100_000_000;
    let dir = temp_dir("sequence");
    let pattern = dir.join("img%03d.png").to_string_lossy().into_owned();
    let files: Vec<Vec<u8>> = ["still_64x48_pal8.png", "still_64x48_gray16.png"]
        .iter()
        .map(|name| std::fs::read(fixture(name)).expect("the fixture is checked in"))
        .collect();
    for (index, bytes) in files.iter().enumerate() {
        // The names the `%03d` pattern expands to.
        std::fs::write(dir.join(format!("img{index:03}.png")), bytes).expect("an image is written");
    }

    let mut src = MultiFileSrc::new(&pattern).with_framerate(FPS, 1);
    // The `.png` pattern types the sequence, so no caps argument is needed.
    let caps = futures_lite_block(src.intercept_caps()).expect("the sequence types");
    let Caps::CompressedVideo {
        codec, framerate, ..
    } = &caps
    else {
        panic!("a still-image sequence, got {caps:?}");
    };
    assert_eq!(*codec, g2g_core::VideoCodec::Png);
    assert_eq!(*framerate, g2g_core::Rate::Fixed(FPS << 16));

    src.configure_pipeline(&caps).expect("its own caps");
    let mut sink = Collect::default();
    let count = src.run(&mut sink).await.expect("the sequence reads");
    assert_eq!(count, files.len() as u64);
    assert_eq!(sink.payloads, files, "each file is one buffer");
    assert_eq!(
        sink.timing,
        vec![(0, PERIOD_NS), (PERIOD_NS, PERIOD_NS)],
        "stamped on the stated grid"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn imagesequencesrc_is_a_registered_source_with_gsts_defaults() {
    let reg = default_registry();
    let src = reg
        .make_source("imagesequencesrc")
        .expect("`imagesequencesrc` is registered");
    assert_eq!(
        src.get_property("framerate"),
        Some(PropValue::Fraction(30, 1)),
        "gst's default rate"
    );
    assert_eq!(
        src.get_property("location"),
        Some(PropValue::Str("%05d".into())),
        "gst's default pattern"
    );
}

// ---------------------------------------------------------------------------
// dataurisrc
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_data_uri_carrying_a_still_decodes_through_decodebin() {
    let png = std::fs::read(fixture("still_64x48_pal8.png")).expect("the fixture is checked in");
    let uri = format!("data:image/png;base64,{}", base64(&png));
    let mut src = DataUriSrc::new(&uri);
    let caps = futures_lite_block(src.intercept_caps()).expect("the payload types");
    assert_eq!(
        caps,
        g2g_plugins::typefind::still_image_caps(g2g_core::VideoCodec::Png),
        "the payload's own header types it"
    );

    // Through the parser and decoder the auto-plug chain picks for that type.
    #[cfg(feature = "png")]
    {
        let line = format!("dataurisrc uri=\"{uri}\" ! decodebin ! fakesink");
        assert_eq!(frames_consumed(&line).await, 1, "the still decodes");
    }
}

#[tokio::test]
async fn a_data_uri_byte_stream_reaches_a_sink_whole() {
    let whole = std::fs::read(fixture(TS_FIXTURE)).expect("the fixture is checked in");
    let uri = format!("data:video/mp2t;base64,{}", base64(&whole));
    let dir = temp_dir("datauri");
    let out = dir.join("payload.ts");
    let line = format!(
        "dataurisrc uri=\"{uri}\" blocksize=4096 ! filesink location={}",
        out.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("the line parses: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(
        std::fs::read(&out).expect("the sink wrote a file"),
        whole,
        "the payload arrives byte for byte"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Standard-alphabet base64 (RFC 4648) with padding, so the test can build a
/// `data:` URI without a dependency.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const PAD: char = '=';
    let mut out = String::new();
    for group in bytes.chunks(3) {
        let mut packed = 0u32;
        for (index, byte) in group.iter().enumerate() {
            packed |= u32::from(*byte) << (16 - 8 * index);
        }
        let sextets = group.len() + 1;
        for index in 0..sextets {
            let sextet = (packed >> (18 - 6 * index)) & 0x3F;
            out.push(ALPHABET[sextet as usize] as char);
        }
        for _ in sextets..4 {
            out.push(PAD);
        }
    }
    out
}

/// Drive a `!Send`-free future to completion on the current thread: the source
/// caps futures are `Ready`, so one poll is enough.
fn futures_lite_block<F: core::future::Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = noop_waker();
    let mut cx = core::task::Context::from_waker(&waker);
    loop {
        if let core::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

fn noop_waker() -> core::task::Waker {
    use core::task::{RawWaker, RawWakerVTable, Waker};
    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|_| RawWaker::new(&(), &VTABLE), |_| {}, |_| {}, |_| {});
    // SAFETY: the vtable's clone returns the same no-op waker and every other
    // entry does nothing, so there is no state to keep valid.
    unsafe { Waker::from_raw(RawWaker::new(&(), &VTABLE)) }
}

#[derive(Default)]
struct Collect {
    payloads: Vec<Vec<u8>>,
    timing: Vec<(u64, u64)>,
}

impl g2g_core::OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<g2g_core::PipelinePacket>,
    ) -> core::task::Poll<Result<g2g_core::PushOutcome, g2g_core::G2gError>> {
        if let g2g_core::PipelinePacket::DataFrame(frame) =
            packet_slot.take().expect("poll_push without a packet")
        {
            self.payloads
                .push(frame.domain.as_system_slice().expect("system").to_vec());
            self.timing
                .push((frame.timing.pts_ns, frame.timing.duration_ns));
        }
        core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
    }
}
