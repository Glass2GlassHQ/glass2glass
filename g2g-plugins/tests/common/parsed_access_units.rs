// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::{AsyncElement, VideoCodec};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::h265parse::H265Parse;
use g2g_plugins::typefind::elementary_video_caps;

use crate::collect_sink::{feed_in_chunks, CollectSink};

async fn parse<E: AsyncElement>(
    mut parser: E,
    codec: VideoCodec,
    stream: &[u8],
    sink: &mut CollectSink,
) {
    parser
        .configure_pipeline(&elementary_video_caps(codec))
        .expect("configure parser");
    feed_in_chunks(&mut parser, stream, sink).await;
}

// 29.97 fps
const FRAME_PERIOD_NS: u64 = 33_366_700;

pub(crate) async fn parsed_access_units(codec: VideoCodec, stream: &[u8]) -> Vec<Frame> {
    let mut sink = CollectSink::default();
    match codec {
        VideoCodec::H264 => parse(H264Parse::reframing(), codec, stream, &mut sink).await,
        VideoCodec::H265 => parse(H265Parse::reframing(), codec, stream, &mut sink).await,
        other => panic!("no Annex-B parser for {other:?}"),
    }
    let access_units: Vec<Frame> = sink
        .packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .zip(0u64..)
        .map(|(mut frame, index)| {
            let pts_ns = index * FRAME_PERIOD_NS;
            frame.timing.pts_ns = pts_ns;
            frame.timing.dts_ns = pts_ns;
            frame.timing.capture_ns = pts_ns;
            frame
        })
        .collect();
    assert!(
        access_units.len() > 1,
        "the parser splits the stream into access units"
    );
    access_units
}

#[allow(dead_code)] // not every test file that splits a stream wants the bytes alone
pub(crate) async fn parsed_access_unit_bytes(codec: VideoCodec, stream: &[u8]) -> Vec<Vec<u8>> {
    parsed_access_units(codec, stream)
        .await
        .into_iter()
        .map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("the parser emits system memory")
                .to_vec()
        })
        .collect()
}
