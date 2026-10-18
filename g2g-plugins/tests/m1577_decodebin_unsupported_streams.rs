//! M1577: a bare `decodebin` fails at launch naming the codec it has no decoder
//! for, or the media kind the file lacks, instead of building a chain that fails
//! mid-run.

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_core::stream::StreamType;
use g2g_plugins::mpegts::{TsMuxer, STREAM_TYPE_AAC};
use g2g_plugins::registry::default_registry;

fn launch_error(tag: &str, ext: &str, bytes: &[u8], consumer: &str) -> ParseError {
    let path = std::env::temp_dir().join(format!("g2g-m1577-{tag}-{}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    let line = format!(
        "filesrc location={} ! decodebin ! {consumer} ! fakesink",
        path.display()
    );
    let parsed = parse_launch(&default_registry(), &line);
    std::fs::remove_file(&path).ok();
    match parsed {
        Err(error) => error,
        Ok(_) => panic!("`{line}` builds a graph"),
    }
}

fn missing_decoders(tag: &str, ext: &str, bytes: &[u8], consumer: &str) -> Vec<String> {
    match launch_error(tag, ext, bytes, consumer) {
        ParseError::NoStreamDecoder { codecs, .. } => codecs,
        other => panic!("{tag} fails naming no codec: {other}"),
    }
}

fn riff_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(body);
    out
}

const PCM_TAG: u16 = 0x0001;
const GSM_610_TAG: u16 = 0x0031;
const SAMPLE_RATE: u32 = 8_000;
const MONO: u16 = 1;
const PCM_BITS: u16 = 16;

fn wav(format_tag: u16, bits: u16) -> Vec<u8> {
    const DATA_LEN: usize = 64;
    let block_align = MONO * bits.max(8) / 8;
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&format_tag.to_le_bytes());
    fmt.extend_from_slice(&MONO.to_le_bytes());
    fmt.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    fmt.extend_from_slice(&(SAMPLE_RATE * u32::from(block_align)).to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    let body = [
        b"WAVE".as_slice(),
        &riff_chunk(b"fmt ", &fmt),
        &riff_chunk(b"data", &[0u8; DATA_LEN]),
    ]
    .concat();
    riff_chunk(b"RIFF", &body)
}

fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

// a HEIF still: its picture is an item in `meta`, and there is no `moov` at all
fn heif_still() -> Vec<u8> {
    const FULL_BOX_HEADER: [u8; 4] = [0; 4];
    let ftyp = mp4_box(b"ftyp", b"heic\0\0\0\0mif1heic");
    let meta = mp4_box(b"meta", &FULL_BOX_HEADER);
    [ftyp, meta, mp4_box(b"mdat", &[0u8; 16])].concat()
}

fn ts_with_stream(stream_type: u8, access_unit: &[u8]) -> Vec<u8> {
    const FRAME_TICKS: u64 = 3_000;
    let mut mux = TsMuxer::new(stream_type);
    // the probe sees a PES only once the next one starts
    (0..2)
        .flat_map(|frame| mux.push_au(access_unit, Some(frame * FRAME_TICKS), None, true))
        .collect()
}

const FLV_SOUND_FORMAT_NELLYMOSER: u8 = 6;

fn nellymoser_flv() -> Vec<u8> {
    const AUDIO_PRESENT: u8 = 0x04;
    const HEADER_LEN: u32 = 9;
    const TAG_AUDIO: u8 = 8;
    let body = [FLV_SOUND_FORMAT_NELLYMOSER << 4, 0, 0, 0];
    let mut out = b"FLV\x01".to_vec();
    out.push(AUDIO_PRESENT);
    out.extend_from_slice(&HEADER_LEN.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    out.push(TAG_AUDIO);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
    out.extend_from_slice(&[0; 7]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&0u32.to_be_bytes());
    out
}

#[test]
fn wav_names_its_format_tag() {
    let codecs = missing_decoders("gsm", "wav", &wav(GSM_610_TAG, 0), "audioconvert");
    assert_eq!(codecs, vec![format!("audio format tag {GSM_610_TAG:#06x}")]);
}

#[test]
fn heif_names_its_image_items() {
    let codecs = missing_decoders("heif", "heic", &heif_still(), "videoconvert");
    assert_eq!(codecs, vec!["HEIF image items".to_string()]);
}

#[test]
fn ts_names_a_stream_type_it_has_no_port_for() {
    const STREAM_TYPE_VVC: u8 = 0x33;
    let ts = ts_with_stream(STREAM_TYPE_VVC, &[0; 32]);
    let codecs = missing_decoders("vvc", "ts", &ts, "videoconvert");
    assert_eq!(codecs, vec!["vvc".to_string()]);
}

#[test]
fn ts_names_latm_labelled_as_adts() {
    const LOAS_SYNC: [u8; 2] = [0x56, 0xE0];
    let mut access_unit = LOAS_SYNC.to_vec();
    access_unit.extend_from_slice(&[0; 30]);
    let ts = ts_with_stream(STREAM_TYPE_AAC, &access_unit);
    let codecs = missing_decoders("latm", "ts", &ts, "audioconvert");
    assert_eq!(codecs, vec!["aac_latm".to_string()]);
}

#[test]
fn flv_names_its_sound_format() {
    let codecs = missing_decoders("nellymoser", "flv", &nellymoser_flv(), "audioconvert");
    assert_eq!(codecs, vec!["nellymoser".to_string()]);
}

#[test]
fn jpeg_ls_names_its_codec() {
    const JPEG_LS_HEADER: [u8; 8] = [0xFF, 0xD8, 0xFF, 0xF7, 0x00, 0x0B, 0x08, 0x00];
    let mut image = JPEG_LS_HEADER.to_vec();
    image.extend_from_slice(&[0; 32]);
    let codecs = missing_decoders("jls", "jls", &image, "videoconvert");
    assert_eq!(codecs, vec!["jpegls".to_string()]);
}

#[test]
fn video_consumer_on_an_audio_file_names_the_missing_kind() {
    match launch_error("pcm", "wav", &wav(PCM_TAG, PCM_BITS), "videoconvert") {
        ParseError::NoStreamOfKind { kind, .. } => assert_eq!(kind, StreamType::Video),
        other => panic!("fails without naming the missing kind: {other}"),
    }
}
