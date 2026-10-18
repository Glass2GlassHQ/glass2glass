//! M1567: the MP4 demuxers read QuickTime PCM (`twos`, `sowt`, `raw `, `alaw`,
//! `ulaw`) into the graph's PCM layout one chunk per frame, and MP3 (`.mp3`, or
//! `mp4a` with an MPEG-1 / MPEG-2 audio esds) as its frames, so a bare
//! `decodebin` finds a decoder for them.
#![cfg(feature = "std")]

#[path = "common/audio_mp4.rs"]
mod audio_mp4;
#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mp4_demux.rs"]
mod mp4_demux;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{AudioFormat, Caps, PipelineClock};
use g2g_plugins::registry::default_registry;

use audio_mp4::{
    access_units, mp4a_entry, progressive_audio_mp4, quicktime_pcm_mov, sound_entry, CHANNELS,
    SAMPLE_RATE,
};
use mp4_demux::{demux_audio, demux_first_port, Demuxed};

// esds objectTypeIndication of MPEG-2 and MPEG-1 audio
const MPEG2_AUDIO: u8 = 0x69;
const MPEG1_AUDIO: u8 = 0x6B;

struct PcmCase {
    fourcc: &'static [u8; 4],
    sample_bits: u16,
    stream: &'static str,
    format: AudioFormat,
    // two frames as the file stores them, then as the graph carries them
    stored: Vec<u8>,
    graph: Vec<u8>,
}

impl PcmCase {
    fn frame_bytes(&self) -> usize {
        self.stored.len() / 2
    }

    // the stored frames as one chunk, then the first frame again as a shorter chunk
    fn file(&self) -> Vec<u8> {
        let chunks = [
            self.stored.clone(),
            self.stored[..self.frame_bytes()].to_vec(),
        ];
        quicktime_pcm_mov(
            &sound_entry(self.fourcc, self.sample_bits),
            &chunks,
            self.frame_bytes(),
        )
    }

    fn expected_frames(&self) -> Vec<Vec<u8>> {
        vec![
            self.graph.clone(),
            self.graph[..self.frame_bytes()].to_vec(),
        ]
    }
}

fn pcm_cases() -> Vec<PcmCase> {
    let stereo_16 = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    let swapped_16 = vec![0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07];
    let stereo_32: Vec<u8> = (1..=16).collect();
    let stereo_8 = vec![0x00, 0x7F, 0x80, 0xFF];
    let sign_flipped_8 = vec![0x80, 0xFF, 0x00, 0x7F];
    // QuickTime writes 16 as the depth of a G.711 entry
    const G711_ENTRY_BITS: u16 = 16;
    vec![
        PcmCase {
            fourcc: b"twos",
            sample_bits: 16,
            stream: "pcm-s16le",
            format: AudioFormat::PcmS16Le,
            stored: stereo_16.clone(),
            graph: swapped_16,
        },
        PcmCase {
            fourcc: b"sowt",
            sample_bits: 16,
            stream: "pcm-s16le",
            format: AudioFormat::PcmS16Le,
            stored: stereo_16.clone(),
            graph: stereo_16,
        },
        PcmCase {
            fourcc: b"sowt",
            sample_bits: 32,
            stream: "pcm-s32le",
            format: AudioFormat::PcmS32Le,
            stored: stereo_32.clone(),
            graph: stereo_32,
        },
        PcmCase {
            fourcc: b"sowt",
            sample_bits: 8,
            stream: "pcm-u8",
            format: AudioFormat::PcmU8,
            stored: stereo_8.clone(),
            graph: sign_flipped_8,
        },
        PcmCase {
            fourcc: b"raw ",
            sample_bits: 8,
            stream: "pcm-u8",
            format: AudioFormat::PcmU8,
            stored: stereo_8.clone(),
            graph: stereo_8.clone(),
        },
        PcmCase {
            fourcc: b"alaw",
            sample_bits: G711_ENTRY_BITS,
            stream: "alaw",
            format: AudioFormat::Alaw,
            stored: stereo_8.clone(),
            graph: stereo_8.clone(),
        },
        PcmCase {
            fourcc: b"ulaw",
            sample_bits: G711_ENTRY_BITS,
            stream: "mulaw",
            format: AudioFormat::Mulaw,
            stored: stereo_8.clone(),
            graph: stereo_8,
        },
    ]
}

fn assert_announces(demuxed: &Demuxed, format: AudioFormat, what: &str) {
    let expected = Caps::Audio {
        format,
        channels: CHANNELS as u8,
        sample_rate: SAMPLE_RATE,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    assert!(
        demuxed.caps.contains(&expected),
        "{what}: announces {expected:?}, got {:?}",
        demuxed.caps
    );
}

#[tokio::test]
async fn quicktime_pcm_leaves_both_demuxers_in_the_graph_layout() {
    for case in pcm_cases() {
        let what = core::str::from_utf8(case.fourcc).expect("ascii fourcc");
        let file = case.file();
        let single = demux_audio(&file, case.stream).await;
        assert_announces(&single, case.format, what);
        assert_eq!(
            single.frames,
            case.expected_frames(),
            "{what}: qtdemux frames"
        );
        let multi = demux_first_port(&file).await;
        assert_announces(&multi, case.format, what);
        assert_eq!(
            multi.frames,
            case.expected_frames(),
            "{what}: Mp4DemuxN frames"
        );
    }
}

#[tokio::test]
async fn mp3_tracks_leave_both_demuxers_as_their_frames() {
    let units = access_units();
    let entries = [
        ("mp4a MPEG-1", mp4a_entry(MPEG1_AUDIO, &[])),
        ("mp4a MPEG-2", mp4a_entry(MPEG2_AUDIO, &[])),
        (".mp3", sound_entry(b".mp3", 16)),
    ];
    for (what, entry) in entries {
        let file = progressive_audio_mp4(&entry, &units);
        let single = demux_audio(&file, "mp3").await;
        assert!(
            single.caps.iter().any(|caps| matches!(
                caps,
                Caps::Audio {
                    format: AudioFormat::Mp3,
                    ..
                }
            )),
            "{what}: MP3 caps announced, got {:?}",
            single.caps
        );
        assert_eq!(single.frames, units, "{what}: qtdemux frames unchanged");
        let multi = demux_first_port(&file).await;
        assert_eq!(multi.frames, units, "{what}: Mp4DemuxN frames unchanged");
    }
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn decodebin_plays_big_endian_quicktime_pcm() {
    let case = pcm_cases()
        .into_iter()
        .find(|case| case.fourcc == b"twos")
        .expect("the twos case");
    let dir = std::env::temp_dir();
    let source = dir.join(format!("g2g-m1567-{}.mov", std::process::id()));
    let decoded = dir.join(format!("g2g-m1567-{}.pcm", std::process::id()));
    std::fs::write(&source, case.file()).expect("write the source");
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert \
         ! audio/x-raw,format=S16LE,rate={SAMPLE_RATE},channels={CHANNELS} \
         ! filesink location={}",
        source.display(),
        decoded.display()
    );
    let graph = parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses: {e}"));
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    let output = std::fs::read(&decoded).expect("read the output");
    std::fs::remove_file(&source).ok();
    std::fs::remove_file(&decoded).ok();
    assert_eq!(output, case.expected_frames().concat());
}
