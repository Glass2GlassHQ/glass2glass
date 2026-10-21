#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;
#[path = "common/raw_convert.rs"]
mod raw_convert;

use g2g_core::{AudioFormat, Caps};
use g2g_plugins::mp4demuxn::{forwardable_streams, unsupported_sample_entries};
use mp4_demux::{demux_audio, demux_first_port, Demuxed};
use raw_convert::read_fixture;

// (fixture, the stream name selecting its track, the graph format, the suffix of ffmpeg's decode to it)
const PCM_FIXTURES: [(&str, &str, AudioFormat, &str); 10] = [
    ("mov_in24_be.mov", "pcm-s24le", AudioFormat::PcmS24Le, "s24le"),
    ("mov_in24_le.mov", "pcm-s24le", AudioFormat::PcmS24Le, "s24le"),
    ("mov_in32_le.mov", "pcm-s32le", AudioFormat::PcmS32Le, "s32le"),
    ("mov_fl32_be.mov", "pcm-f32le", AudioFormat::PcmF32Le, "f32le"),
    ("mov_fl32_le.mov", "pcm-f32le", AudioFormat::PcmF32Le, "f32le"),
    ("mov_lpcm_s16le.mov", "pcm-s16le", AudioFormat::PcmS16Le, "s16le"),
    ("mov_lpcm_s24be.mov", "pcm-s24le", AudioFormat::PcmS24Le, "s24le"),
    ("mov_lpcm_f32be.mov", "pcm-f32le", AudioFormat::PcmF32Le, "f32le"),
    ("mov_lpcm_u8.mov", "pcm-u8", AudioFormat::PcmU8, "u8"),
    ("mov_lpcm_s8.mov", "pcm-u8", AudioFormat::PcmU8, "u8"),
];

// the f64 tone in a version 1 `fl64` entry and a version 2 `lpcm` one, and how each is named
const REFUSED_FIXTURES: [(&str, &str); 2] = [
    ("mov_fl64_be.mov", "fl64 64-bit float big-endian PCM"),
    ("mov_lpcm_f64le.mov", "lpcm 64-bit float little-endian PCM"),
];

fn announced_format(demuxed: &Demuxed) -> Option<AudioFormat> {
    demuxed.caps.iter().find_map(|caps| match caps {
        Caps::Audio { format, .. } => Some(*format),
        _ => None,
    })
}

#[tokio::test]
async fn quicktime_pcm_entries_demux_to_ffmpegs_decode() {
    for (file, stream, format, decode) in PCM_FIXTURES {
        let mov = read_fixture(file);
        let expected = read_fixture(&format!("{file}.{decode}"));
        let single = demux_audio(&mov, stream).await;
        assert_eq!(announced_format(&single), Some(format), "{file}: qtdemux caps");
        assert!(single.frames.concat() == expected, "{file}: qtdemux samples");
        let multi = demux_first_port(&mov).await;
        assert_eq!(announced_format(&multi), Some(format), "{file}: Mp4DemuxN caps");
        assert!(multi.frames.concat() == expected, "{file}: Mp4DemuxN samples");
    }
}

#[test]
fn a_pcm_layout_with_no_graph_format_is_left_out_and_named() {
    for (file, name) in REFUSED_FIXTURES {
        let mov = read_fixture(file);
        assert!(forwardable_streams(&mov).is_empty(), "{file}: no track forwarded");
        assert_eq!(unsupported_sample_entries(&mov), [name], "{file}");
    }
}

const VORBIS_FIXTURE: &str = "mov_vorbis.mov";
// the identification, comment and setup header packet types, each followed by the codec name
const VORBIS_HEADER_TYPES: [u8; 3] = [0x01, 0x03, 0x05];
const VORBIS_MAGIC: &[u8] = b"vorbis";

fn assert_leads_with_vorbis_headers(demuxed: &Demuxed, what: &str) {
    assert!(demuxed.frames.len() > VORBIS_HEADER_TYPES.len(), "{what}: headers and audio");
    for (frame, kind) in demuxed.frames.iter().zip(VORBIS_HEADER_TYPES) {
        assert_eq!(frame.first(), Some(&kind), "{what}: header order");
        assert!(frame[1..].starts_with(VORBIS_MAGIC), "{what}: header magic");
    }
    let packets = demuxed.frames[VORBIS_HEADER_TYPES.len()..].concat();
    assert!(
        packets == read_fixture(&format!("{VORBIS_FIXTURE}.packets")),
        "{what}: the packets ffmpeg copies"
    );
}

#[tokio::test]
async fn a_vorbis_entry_leads_its_packets_with_its_three_headers() {
    let mov = read_fixture(VORBIS_FIXTURE);
    let single = demux_audio(&mov, "vorbis").await;
    assert_eq!(announced_format(&single), Some(AudioFormat::Vorbis));
    assert_leads_with_vorbis_headers(&single, "qtdemux");
    let multi = demux_first_port(&mov).await;
    assert_eq!(announced_format(&multi), Some(AudioFormat::Vorbis));
    assert_leads_with_vorbis_headers(&multi, "Mp4DemuxN");
}

#[cfg(feature = "vorbis")]
mod vorbis_decode {
    use super::*;
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_plugins::registry::default_registry;
    use raw_convert::{fixture, temp_path, ZeroClock};

    // the fixture script encodes 0.05 s of stereo tone at 8 kHz, which the edit list presents whole
    const CHANNELS: usize = 2;
    const SAMPLE_RATE: usize = 8_000;
    const TONE_SAMPLES: usize = SAMPLE_RATE / 20;
    const S16_BYTES: usize = 2;

    // each decoder rounds to int16 itself, and ffmpeg keeps the last packet's samples past the edit list's end
    fn assert_samples_agree(pcm: &[u8], reference: &[u8]) {
        let shared = pcm.len().min(reference.len());
        assert!(shared > 0, "both decodes produced audio");
        let samples = |bytes: &[u8]| {
            bytes[..shared]
                .as_chunks::<S16_BYTES>()
                .0
                .iter()
                .map(|sample| i32::from(i16::from_le_bytes(*sample)))
                .collect::<Vec<_>>()
        };
        let largest_difference = samples(pcm)
            .iter()
            .zip(samples(reference))
            .map(|(ours, theirs)| (ours - theirs).abs())
            .max()
            .unwrap_or_default();
        assert!(largest_difference <= 1, "off ffmpeg by {largest_difference}");
    }

    #[tokio::test]
    async fn a_vorbis_entry_decodes_to_ffmpegs_samples() {
        let out = temp_path("m1728-vorbis");
        let line = format!(
            "filesrc location={} ! qtdemux stream=vorbis ! vorbisdec ! audioconvert \
             ! audio/x-raw,format=S16LE,rate={SAMPLE_RATE},channels={CHANNELS} \
             ! filesink location={}",
            fixture(VORBIS_FIXTURE).display(),
            out.display()
        );
        let graph = parse_launch(&default_registry(), &line)
            .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
        let pcm = std::fs::read(&out).expect("the sink wrote its file");
        std::fs::remove_file(&out).ok();
        assert_eq!(pcm.len(), TONE_SAMPLES * CHANNELS * S16_BYTES, "the presented samples");
        assert_samples_agree(&pcm, &read_fixture(&format!("{VORBIS_FIXTURE}.s16le")));
    }
}
