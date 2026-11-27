// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2620: MP4 `mhm1` / `mha1` and MPEG-TS stream_type 0x2D demux to MPEG-H audio. Reads
//! the streams tools/mpegh-fixtures.sh fetches and skips without them.
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mpegh_fixtures.rs"]
mod mpegh_fixtures;

use g2g_core::{AudioFormat, Caps, ChannelLayout};

use mpegh_fixtures::{
    data_frames, demux_mp4_audio, demux_ts_mpegh, first_caps, mpegh_fixture, MHA1_MP4, STEREO_MP4,
};

const STEREO_FRAGMENTED_MP4: &str = "mpegh_cicp_02_baseline_32kbps_fragmented.mp4";
// the reference channel layout of sample_mpegh_mha1.mp4's mhaC, CICP 19, is 7.1.4
const CICP_19_CHANNELS: u8 = 12;
// the stsz of mpegh_cicp_02_baseline_32kbps.mp4
const STEREO_SAMPLES: usize = 29;

fn stereo_mpegh(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

#[tokio::test]
async fn mhm1_samples_leave_as_mhas_in_both_mp4_layouts() {
    let (Some(progressive), Some(fragmented)) = (
        mpegh_fixture(STEREO_MP4),
        mpegh_fixture(STEREO_FRAGMENTED_MP4),
    ) else {
        return;
    };
    let progressive = demux_mp4_audio(progressive, "mpegh").await;
    let fragmented = demux_mp4_audio(fragmented, "mpegh").await;
    for packets in [&progressive, &fragmented] {
        assert_eq!(
            first_caps(packets),
            Some(stereo_mpegh(AudioFormat::MpeghMhas))
        );
    }
    let samples = data_frames(&progressive);
    assert_eq!(samples.len(), STEREO_SAMPLES);
    assert!(
        samples
            .windows(2)
            .all(|pair| pair[0].1.pts_ns < pair[1].1.pts_ns),
        "each sample keeps its own pts"
    );
    let sample_bytes = |frames: Vec<(Vec<u8>, g2g_core::frame::FrameTiming)>| {
        frames
            .into_iter()
            .map(|(bytes, _)| bytes)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        sample_bytes(data_frames(&fragmented)),
        sample_bytes(samples),
        "the fragmented file holds the same samples"
    );
}

#[tokio::test]
async fn mha1_samples_follow_their_mhac() {
    let Some(file) = mpegh_fixture(MHA1_MP4) else {
        return;
    };
    let packets = demux_mp4_audio(file, "mpegh-raw").await;
    assert_eq!(
        first_caps(&packets),
        Some(Caps::Audio {
            format: AudioFormat::MpeghRaw,
            channels: CICP_19_CHANNELS,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        })
    );
    let frames = data_frames(&packets);
    let (lead, _) = frames.first().expect("frames");
    assert_eq!(&lead[4..8], b"mhaC", "the mhaC box leads");
    assert!(frames.len() > 1, "access units follow the config");
}

#[tokio::test]
async fn every_ts_pes_packing_carries_the_same_mhas_bytes() {
    let packings = [
        "mpegh_cicp_02_baseline_32kbps_single.ts",
        "mpegh_cicp_02_baseline_32kbps_multi_5.ts",
        "mpegh_cicp_02_baseline_32kbps_cont_2000.ts",
    ];
    let Some(files) = packings
        .into_iter()
        .map(mpegh_fixture)
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let mut streams = Vec::new();
    for file in files {
        let packets = demux_ts_mpegh(&file).await;
        assert!(matches!(
            first_caps(&packets),
            Some(Caps::Audio {
                format: AudioFormat::MpeghMhas,
                ..
            })
        ));
        let bytes: Vec<u8> = data_frames(&packets)
            .into_iter()
            .flat_map(|(bytes, _)| bytes)
            .collect();
        assert!(!bytes.is_empty());
        streams.push(bytes);
    }
    assert!(
        streams.windows(2).all(|pair| pair[0] == pair[1]),
        "one access unit, five, or 2000 bytes per PES packet"
    );
}
