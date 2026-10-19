#![cfg(all(target_os = "linux", feature = "ffmpeg-long-tail"))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// ZMBV pal8 video with MS ADPCM audio in AVI, QuickTime RLE rgb24 with IMA4 in MOV.
// Each `.rgba` / `.rgb` / `.s16le` beside them is ffmpeg's decode of that stream.
const ZMBV_AVI: &str = "tagged_zmbv_adpcm_ms.avi";
const QTRLE_MOV: &str = "tagged_qtrle_ima4.mov";
// four GSM 6.10 blocks of seeded random bytes in WAV
const GSM_WAV: &str = "tagged_gsm_ms.wav";
// a fate-suite Screenpresso AVI cut inside its second frame, which libavcodec rejects
const SCREENPRESSO_CUT_AVI: &str = "tagged_screenpresso_cut.avi";
// every fixture carries 8 kHz mono audio
const AUDIO_CAPS: &str = "audio/x-raw,format=S16LE,rate=8000,channels=1";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1590-{tag}-{}.raw", std::process::id()))
}

// what `filesrc ! decodebin ! <tail> ! filesink` writes for `input`
async fn decode(input: &Path, tail: &str, tag: &str) -> Vec<u8> {
    let out = temp_path(tag);
    let line = format!(
        "filesrc location={} ! decodebin ! {tail} ! filesink location={}",
        input.display(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    bytes
}

fn expected(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).expect("the expected decode is committed")
}

#[tokio::test]
async fn a_palettized_avi_codec_decodes_to_rgba() {
    let decoded = decode(&fixture(ZMBV_AVI), "video/x-raw,format=RGBA", "zmbv").await;
    assert_eq!(decoded, expected(&format!("{ZMBV_AVI}.rgba")));
}

#[tokio::test]
async fn a_cut_frame_the_decoder_rejects_does_not_stop_the_stream() {
    let decoded = decode(
        &fixture(SCREENPRESSO_CUT_AVI),
        "video/x-raw,format=RGBA",
        "screenpresso",
    )
    .await;
    assert_eq!(decoded, expected(&format!("{SCREENPRESSO_CUT_AVI}.rgba")));
}

#[tokio::test]
async fn an_avi_adpcm_track_decodes_with_its_block_align() {
    let tail = format!("audioconvert ! {AUDIO_CAPS}");
    let decoded = decode(&fixture(ZMBV_AVI), &tail, "adpcm-ms").await;
    assert_eq!(decoded, expected(&format!("{ZMBV_AVI}.s16le")));
}

#[tokio::test]
async fn a_wav_codec_known_only_by_its_tag_decodes() {
    let tail = format!("audioconvert ! {AUDIO_CAPS}");
    let decoded = decode(&fixture(GSM_WAV), &tail, "gsm").await;
    assert_eq!(decoded, expected(&format!("{GSM_WAV}.s16le")));
}

#[tokio::test]
async fn a_quicktime_sample_entry_codec_decodes() {
    let decoded = decode(&fixture(QTRLE_MOV), "video/x-raw,format=RGB", "qtrle").await;
    assert_eq!(decoded, expected(&format!("{QTRLE_MOV}.rgb")));
}

#[tokio::test]
async fn a_quicktime_ima4_track_decodes_in_whole_chunks() {
    // ffmpeg decodes the last IMA4 block whole, the decoder clips it to the edit list
    const IMA4_BLOCK_SAMPLES: usize = 64;
    const SAMPLE_BYTES: usize = 2;
    let tail = format!("audioconvert ! {AUDIO_CAPS}");
    let decoded = decode(&fixture(QTRLE_MOV), &tail, "ima4").await;
    let reference = expected(&format!("{QTRLE_MOV}.s16le"));
    assert!(decoded.len() + IMA4_BLOCK_SAMPLES * SAMPLE_BYTES > reference.len());
    assert_eq!(decoded, reference[..decoded.len()]);
}

// the fixture with one sample entry fourcc swapped for one libavformat does not know
fn with_unknown_entry(fourcc: &[u8; 4], unknown: &[u8; 4], tag: &str) -> PathBuf {
    let mut bytes = std::fs::read(fixture(QTRLE_MOV)).expect("fixture");
    let at = bytes
        .windows(fourcc.len())
        .position(|window| window == fourcc)
        .expect("the sample entry");
    bytes[at..at + fourcc.len()].copy_from_slice(unknown);
    let path = temp_path(tag).with_extension("mov");
    std::fs::write(&path, bytes).expect("write");
    path
}

fn missing_decoders(input: &Path, consumer: &str) -> Vec<String> {
    let line = format!(
        "filesrc location={} ! decodebin ! {consumer} ! fakesink",
        input.display()
    );
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(input).ok();
    match parsed {
        Err(ParseError::NoStreamDecoder { codecs, .. }) => codecs,
        Err(other) => panic!("`{line}` fails naming no codec: {other}"),
        Ok(_) => panic!("`{line}` builds a graph"),
    }
}

#[test]
fn an_unknown_quicktime_tag_is_named_at_launch() {
    const UNKNOWN_VIDEO: &[u8; 4] = b"ZZZ9";
    const UNKNOWN_AUDIO: &[u8; 4] = b"ZZZ8";
    let video = with_unknown_entry(b"rle ", UNKNOWN_VIDEO, "unknown-video");
    let audio = with_unknown_entry(b"ima4", UNKNOWN_AUDIO, "unknown-audio");
    assert_eq!(missing_decoders(&video, "videoconvert"), ["ZZZ9"]);
    assert_eq!(missing_decoders(&audio, "audioconvert"), ["ZZZ8"]);
}
