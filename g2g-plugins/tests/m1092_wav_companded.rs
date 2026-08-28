//! M1092 `wavenc` writes the payloads `wavparse` reads: mu-law, A-law and IMA
//! ADPCM, not only PCM.
//!
//! ffmpeg's own files are the reference. Our `fmt ` chunk for a companded stream
//! must match theirs field for field, and a file we write must decode back to
//! the PCM ffmpeg decoded from its own, which is what the checked-in
//! `*_decoded.s16le` fixtures hold (they were made in `m1073_g711_adpcm.rs`).
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, OutputSink, PipelineClock,
    PipelinePacket, PropValue, PushOutcome,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::wavenc::WavEnc;

/// `RIFF` + size + `WAVE`, then 4-byte id + 4-byte size per chunk.
const RIFF_HEADER_LEN: usize = 12;
const CHUNK_HEADER_LEN: usize = 8;

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

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1092-{tag}-{}.bin", std::process::id()))
}

/// The body of the first chunk with this id.
fn chunk(file: &[u8], id: &[u8; 4]) -> Vec<u8> {
    let mut at = RIFF_HEADER_LEN;
    while at + CHUNK_HEADER_LEN <= file.len() {
        let size =
            u32::from_le_bytes([file[at + 4], file[at + 5], file[at + 6], file[at + 7]]) as usize;
        let body = at + CHUNK_HEADER_LEN;
        if &file[at..at + 4] == id {
            return file[body..(body + size).min(file.len())].to_vec();
        }
        at = body + size + size % 2;
    }
    panic!("no {} chunk", String::from_utf8_lossy(id));
}

/// Run a line whose `{}` is the output path, and return what it wrote.
async fn run_to_bytes(template: &str, tag: &str) -> Vec<u8> {
    let out = temp_path(tag);
    let _ = std::fs::remove_file(&out);
    let line = template.replace("{}", &out.display().to_string());
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote a file");
    std::fs::remove_file(&out).ok();
    bytes
}

/// `... ! wavparse ! <encoder> ! wavenc ! filesink`, i.e. the PCM fixture
/// re-filed in the coded format.
async fn recode(encoder: &str, tag: &str) -> Vec<u8> {
    run_to_bytes(
        &format!(
            "filesrc location={} ! wavparse ! {encoder} ! wavenc ! filesink location={{}}",
            fixture("sine_8k_mono_s16le.wav").display()
        ),
        tag,
    )
    .await
}

#[tokio::test]
async fn the_g711_fmt_chunks_match_ffmpegs() {
    for (encoder, reference) in [
        ("mulawenc", "sine_8k_mono_mulaw.wav"),
        ("alawenc", "sine_8k_mono_alaw.wav"),
    ] {
        let ours = recode(encoder, encoder).await;
        assert_eq!(
            chunk(&ours, b"fmt "),
            chunk(&read_fixture(reference), b"fmt "),
            "{encoder}: the `fmt ` chunk is ffmpeg's, field for field"
        );
    }
}

#[tokio::test]
async fn a_g711_file_we_wrote_decodes_to_ffmpegs_pcm() {
    for (encoder, decoder, reference) in [
        ("mulawenc", "mulawdec", "sine_8k_mono_mulaw_decoded.s16le"),
        ("alawenc", "alawdec", "sine_8k_mono_alaw_decoded.s16le"),
    ] {
        let ours = recode(encoder, &format!("{encoder}-roundtrip")).await;
        let written = temp_path(&format!("{encoder}-file"));
        std::fs::write(&written, &ours).expect("the coded file is written");
        let decoded = run_to_bytes(
            &format!(
                "filesrc location={} ! wavparse ! {decoder} ! filesink location={{}}",
                written.display()
            ),
            &format!("{encoder}-decoded"),
        )
        .await;
        std::fs::remove_file(&written).ok();
        assert_eq!(
            decoded,
            read_fixture(reference),
            "{encoder}: the file reads back as the PCM ffmpeg decodes from its own"
        );
    }
}

/// The profile that had no writer at all before this milestone.
#[tokio::test]
async fn the_wav_mulaw_encoding_profile_writes_a_file() {
    let bytes = run_to_bytes(
        &format!(
            "filesrc location={} ! wavparse ! encodebin profile=\"audio/x-wav:audio/x-mulaw\" ! filesink location={{}}",
            fixture("sine_8k_mono_s16le.wav").display()
        ),
        "profile",
    )
    .await;
    assert_eq!(
        chunk(&bytes, b"fmt "),
        chunk(&read_fixture("sine_8k_mono_mulaw.wav"), b"fmt "),
        "the profile wrote the same mu-law file the explicit chain does"
    );
}

// ---------------------------------------------------------------------------
// ADPCM: the block size lives in the header, not in the caps
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let PipelinePacket::DataFrame(frame) =
            packet_slot.take().expect("poll_push without a packet")
        {
            self.bytes
                .extend_from_slice(frame.domain.as_system_slice().expect("system"));
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn coded_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Driven directly, since the `adpcmenc` upstream of it is mono-only and a
/// stereo-placeholder source cannot negotiate into it (a separate gap).
#[tokio::test]
async fn the_adpcm_fmt_chunk_states_the_block_size() {
    /// The layout the fixture was encoded at: ffmpeg's own default.
    const BLOCK_ALIGN: u64 = 1024;
    const RATE: u32 = 8_000;
    let mut enc = WavEnc::new();
    let coded = Caps::Audio {
        format: AudioFormat::ImaAdpcm,
        channels: 1,
        sample_rate: RATE,
    };
    enc.set_property("blockalign", PropValue::Uint(BLOCK_ALIGN))
        .expect("the default block size");
    enc.configure_pipeline(&coded).expect("ADPCM negotiates");
    let mut sink = Collect::default();
    enc.process(coded_frame(vec![0u8; BLOCK_ALIGN as usize]), &mut sink)
        .await
        .expect("one block is written");

    let ours = chunk(&sink.bytes, b"fmt ");
    let reference = chunk(&read_fixture("sine_8k_mono_imaadpcm.wav"), b"fmt ");
    // Every field but `nAvgBytesPerSec`: ffmpeg writes the decoded PCM rate
    // there, this writes the coded one, and a reader uses neither to decode.
    const AVG_BYTES_PER_SEC: core::ops::Range<usize> = 8..12;
    assert_eq!(ours.len(), reference.len(), "the same `fmt ` layout");
    for (offset, (ours, theirs)) in ours.iter().zip(reference.iter()).enumerate() {
        if AVG_BYTES_PER_SEC.contains(&offset) {
            continue;
        }
        assert_eq!(ours, theirs, "`fmt ` byte {offset}");
    }
}

#[tokio::test]
async fn a_layout_that_never_arrives_fails_the_header() {
    let mut enc = WavEnc::new();
    // The sentinels a coded stream negotiates at, with no `CapsChanged` after.
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::Mulaw,
        channels: 0,
        sample_rate: 0,
    })
    .expect("the sentinels negotiate");
    let mut sink = Collect::default();
    assert_eq!(
        enc.process(coded_frame(vec![0u8; 8]), &mut sink)
            .await
            .err(),
        Some(G2gError::CapsMismatch),
        "no `fmt ` chunk can say unknown"
    );
}
