// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2621: `mpeghdec` decodes MPEG-H to the same PCM as Fraunhofer's own mpeghDecoder
//! program. Reads what tools/mpegh-fixtures.sh fetches and decodes, and skips without it.
#![cfg(feature = "mpegh")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mpegh_fixtures.rs"]
mod mpegh_fixtures;

use std::time::Duration;

use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_core::{AsyncElement, AudioFormat, Caps, ChannelLayout, PropValue};
use g2g_plugins::clock::WallClock;
use g2g_plugins::mpeghdec::MpeghDec;
use g2g_plugins::registry::default_registry;

use collect_sink::CollectSink;
use mpegh_fixtures::{
    data_frames, demux_mp4_audio, demux_ts_mpegh, first_caps, mpegh_fixture, MHA1_MP4, STEREO_MP4,
};

const CICP_STEREO: i64 = 2;
const CICP_5_1: i64 = 6;
const SURROUND_MP4: &str = "mpegh_cicp_06_baseline_32kbps.mp4";
// the elst of sample_mpegh_mha1.mp4: its encoder delay, then the samples it presents
const MHA1_EDIT_START: usize = 3072;
const MHA1_EDIT_LENGTH: usize = 56065;
const LINK_CAPACITY: usize = 4;
const RUN_TIMEOUT: Duration = Duration::from_secs(30);

struct Decoded {
    caps: Option<Caps>,
    samples: Vec<i32>,
}

async fn decode(demuxed: Vec<PipelinePacket>, target_layout: i64) -> Decoded {
    let mut decoder = MpeghDec::new();
    decoder
        .set_property("target-layout", PropValue::Int(target_layout))
        .expect("a CICP layout");
    decoder
        .configure_pipeline(&first_caps(&demuxed).expect("the demuxer announces caps"))
        .expect("MPEG-H caps");
    let mut sink = CollectSink::default();
    for packet in demuxed {
        decoder.process(packet, &mut sink).await.expect("decode");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain");
    let samples = data_frames(&sink.packets)
        .into_iter()
        .flat_map(|(bytes, _)| bytes)
        .collect::<Vec<u8>>()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&sample| i32::from_le_bytes(sample))
        .collect();
    Decoded {
        caps: first_caps(&sink.packets),
        samples,
    }
}

// the `data` chunk of a 24-bit PCM WAV, each sample sign-extended
fn wav_24_bit_samples(wav: &[u8]) -> Vec<i32> {
    let mut chunks = &wav[12..];
    while chunks.len() >= 8 {
        let id = &chunks[..4];
        let len = u32::from_le_bytes(chunks[4..8].try_into().unwrap()) as usize;
        let body = &chunks[8..8 + len];
        if id == b"data" {
            return body
                .as_chunks::<3>()
                .0
                .iter()
                .map(|&[low, middle, high]| i32::from_le_bytes([0, low, middle, high]) >> 8)
                .collect();
        }
        chunks = &chunks[8 + len + (len & 1)..];
    }
    panic!("the WAV has a data chunk");
}

// mpeghDecoder writes 24-bit WAVs, the top 24 bits of the decoder's samples
fn top_24_bits(samples: &[i32]) -> Vec<i32> {
    samples.iter().map(|sample| sample >> 8).collect()
}

fn assert_matches_reference(decoded: &Decoded, reference: &[i32]) {
    assert!(
        reference.iter().any(|&sample| sample != 0),
        "a silent reference"
    );
    assert_eq!(
        top_24_bits(&decoded.samples),
        reference,
        "the samples mpeghDecoder writes"
    );
}

#[tokio::test]
async fn stereo_decodes_to_the_reference_pcm() {
    let (Some(file), Some(reference)) = (
        mpegh_fixture(STEREO_MP4),
        mpegh_fixture("mpegh_cicp_02_baseline_32kbps_cicp_2.wav"),
    ) else {
        return;
    };
    let decoded = decode(demux_mp4_audio(file, "mpegh").await, CICP_STEREO).await;
    assert_eq!(
        decoded.caps,
        Some(Caps::Audio {
            format: AudioFormat::PcmS32Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::STEREO,
        })
    );
    assert_matches_reference(&decoded, &wav_24_bit_samples(&reference));
}

#[tokio::test]
async fn surround_decodes_to_the_reference_pcm_in_5_1_order() {
    let (Some(file), Some(reference)) = (
        mpegh_fixture(SURROUND_MP4),
        mpegh_fixture("mpegh_cicp_06_baseline_32kbps_cicp_6.wav"),
    ) else {
        return;
    };
    let decoded = decode(demux_mp4_audio(file, "mpegh").await, CICP_5_1).await;
    assert!(matches!(
        decoded.caps,
        Some(Caps::Audio {
            channels: 6,
            channel_layout: ChannelLayout::SURROUND_5_1,
            ..
        })
    ));
    assert_matches_reference(&decoded, &wav_24_bit_samples(&reference));
}

#[tokio::test]
async fn ts_cut_mid_access_unit_decodes_like_the_mp4() {
    let (Some(mp4), Some(ts)) = (
        mpegh_fixture(STEREO_MP4),
        mpegh_fixture("mpegh_cicp_02_baseline_32kbps_cont_2000.ts"),
    ) else {
        return;
    };
    let from_mp4 = decode(demux_mp4_audio(mp4, "mpegh").await, CICP_STEREO).await;
    let from_ts = decode(demux_ts_mpegh(&ts).await, CICP_STEREO).await;
    let shorter = from_mp4.samples.len().min(from_ts.samples.len());
    assert!(shorter > 0);
    assert_eq!(
        from_ts.samples[..shorter],
        from_mp4.samples[..shorter],
        "2000-byte PES packets reassemble into the MP4's access units"
    );
}

#[tokio::test]
async fn mha1_decodes_after_its_mhac_to_the_edited_reference_pcm() {
    let (Some(file), Some(reference)) = (
        mpegh_fixture(MHA1_MP4),
        mpegh_fixture("sample_mpegh_mha1_cicp_2.wav"),
    ) else {
        return;
    };
    let decoded = decode(demux_mp4_audio(file, "mpegh-raw").await, CICP_STEREO).await;
    assert!(matches!(
        decoded.caps,
        Some(Caps::Audio {
            format: AudioFormat::PcmS32Le,
            channels: 2,
            ..
        })
    ));
    // mpeghDecoder ignores the edit list, which the demuxer applies
    let edited = MHA1_EDIT_START * 2..(MHA1_EDIT_START + MHA1_EDIT_LENGTH) * 2;
    assert_matches_reference(&decoded, &wav_24_bit_samples(&reference)[edited]);
}

async fn run_launch_line(pipeline: &str) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mpegh")
        .join(STEREO_MP4);
    if !path.exists() {
        eprintln!(
            "{} absent, run tools/mpegh-fixtures.sh. skipping.",
            path.display()
        );
        return;
    }
    let line = format!(
        "filesrc location={} ! {pipeline} ! fakesink",
        path.display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line builds");
    tokio::time::timeout(
        RUN_TIMEOUT,
        RunBuilder::new(graph, &WallClock::new(), LINK_CAPACITY).run(),
    )
    .await
    .expect("the run ends")
    .expect("the pipeline runs");
}

#[tokio::test]
async fn launch_line_sets_the_decoder_properties() {
    run_launch_line(
        "qtdemux stream=mpegh ! mpeghdec target-layout=2 target-ref-level=-16 \
         drc-effect-type=noisy drc-cut-level=0.5 album-mode=true",
    )
    .await;
}

#[tokio::test]
async fn decodebin_plugs_the_mpegh_decoder() {
    run_launch_line("decodebin ! audioconvert").await;
}
