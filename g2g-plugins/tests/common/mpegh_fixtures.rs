// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps, PropValue};
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::tsdemux::{TsDemux, TsStream};

use crate::collect_sink::{feed_in_chunks, CollectSink};

pub(crate) const STEREO_MP4: &str = "mpegh_cicp_02_baseline_32kbps.mp4";
pub(crate) const MHA1_MP4: &str = "sample_mpegh_mha1.mp4";

// `None` with a note on stderr when tools/mpegh-fixtures.sh has not run
pub(crate) fn mpegh_fixture(name: &str) -> Option<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mpegh")
        .join(name);
    if !path.exists() {
        eprintln!(
            "{} absent, run tools/mpegh-fixtures.sh. skipping.",
            path.display()
        );
        return None;
    }
    Some(std::fs::read(&path).expect("read the MPEG-H fixture"))
}

pub(crate) async fn demux_mp4_audio(file: Vec<u8>, stream: &str) -> Vec<PipelinePacket> {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str(stream.into()))
        .expect("an MPEG-H stream name");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    let frame = Frame::from_system_bytes(file, FrameTiming::default(), 0);
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
}

pub(crate) async fn demux_ts_mpegh(file: &[u8]) -> Vec<PipelinePacket> {
    let mut demux = TsDemux::new().with_stream(TsStream::Mpegh);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    feed_in_chunks(&mut demux, file, &mut sink).await;
    sink.packets
}

pub(crate) fn data_frames(packets: &[PipelinePacket]) -> Vec<(Vec<u8>, FrameTiming)> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some((
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
                frame.timing,
            )),
            _ => None,
        })
        .collect()
}

pub(crate) fn first_caps(packets: &[PipelinePacket]) -> Option<Caps> {
    packets.iter().find_map(|packet| match packet {
        PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
        _ => None,
    })
}
