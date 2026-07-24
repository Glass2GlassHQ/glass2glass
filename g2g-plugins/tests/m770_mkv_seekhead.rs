//! M770: the Matroska muxers' seekable (two-pass) finalize mode. With
//! `seekable=true` the element buffers the file and emits it once at EOS with a
//! front `SeekHead` (the first element of the Segment data) whose Cues entry is
//! patched to the index appended after the last Cluster, so the file seeks from
//! byte 0: a reader parses the head, jumps straight to the `Cues`, and never
//! scans the Clusters. The default streaming output is unchanged.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::matroska::MatroskaDemuxer;
use g2g_plugins::mkvmux::MkvMux;
use g2g_plugins::mkvmuxn::MkvMuxN;

const ID_SEEK_HEAD: [u8; 4] = [0x11, 0x4D, 0x9B, 0x74];
const ID_CUES: [u8; 4] = [0x1C, 0x53, 0xBB, 0x6B];
const ID_CLUSTER: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

/// A minimal Annex-B H.264 IDR access unit (SPS + PPS + IDR).
fn h264_idr() -> Vec<u8> {
    let mut au = Vec::new();
    for nal in [
        vec![0x67, 0x42, 0x00, 0x1E],
        vec![0x68, 0xCE, 0x3C, 0x80],
        vec![0x65, 0x88, 0x84],
    ] {
        au.extend_from_slice(&[0, 0, 0, 1]);
        au.extend_from_slice(&nal);
    }
    au
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Common assertions on a seekable-finalized file.
fn assert_seekable_file(file: &[u8]) {
    // The SeekHead is the first element of the Segment data (before Info and
    // long before the first Cluster).
    let sh = find(file, &ID_SEEK_HEAD).expect("a SeekHead is written");
    let cluster = find(file, &ID_CLUSTER).expect("a Cluster is written");
    assert!(sh < cluster, "SeekHead sits at the front, not at EOS");

    // A reader that has seen only the pre-Cluster head already knows where the
    // Cues are: the demuxer's SeekHead parse resolves their absolute offset.
    let mut head_only = MatroskaDemuxer::new();
    head_only.push_data(&file[..cluster]);
    let cues_at = head_only
        .cue_index_offset()
        .expect("the head alone locates the Cues") as usize;
    assert_eq!(
        &file[cues_at..cues_at + 4],
        &ID_CUES,
        "the patched SeekPosition lands exactly on the Cues element"
    );
    assert!(cues_at > cluster, "the Cues trail the Clusters");

    // The whole file still demuxes, with a usable index.
    let mut demux = MatroskaDemuxer::new();
    demux.push_data(file);
    assert_eq!(demux.take_frames().len(), 3, "all frames recovered");
    assert!(!demux.cues().is_empty(), "Cues parsed");
    assert!(
        demux.cue_seek_offset(0).is_some(),
        "an indexed seek resolves"
    );
}

#[tokio::test]
async fn mkvmuxn_seekable_finalize_writes_front_seekhead() {
    let mut mux = MkvMuxN::new(1).with_seekable(true);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    let mut sink = CaptureSink::default();
    for i in 0..3u64 {
        mux.process(0, frame(h264_idr(), i * 33_000_000), &mut sink)
            .await
            .unwrap();
    }
    assert!(sink.frames.is_empty(), "two-pass: nothing emits before EOS");
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    assert_eq!(sink.frames.len(), 1, "the whole file emits once at EOS");
    assert_seekable_file(&sink.frames[0]);
}

#[tokio::test]
async fn mkvmux_seekable_finalize_writes_front_seekhead() {
    let mut mux = MkvMux::new().with_seekable(true);
    mux.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = CaptureSink::default();
    for i in 0..3u64 {
        mux.process(frame(h264_idr(), i * 33_000_000), &mut sink)
            .await
            .unwrap();
    }
    assert!(sink.frames.is_empty(), "two-pass: nothing emits before EOS");
    mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    assert_eq!(sink.frames.len(), 1, "the whole file emits once at EOS");
    assert_seekable_file(&sink.frames[0]);
}

#[tokio::test]
async fn default_streaming_output_has_no_front_seekhead() {
    let mut mux = MkvMux::new();
    mux.configure_pipeline(&h264_caps()).unwrap();
    let mut sink = CaptureSink::default();
    mux.process(frame(h264_idr(), 0), &mut sink).await.unwrap();
    mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    let stream: Vec<u8> = sink.frames.concat();
    assert!(
        find(&stream, &ID_SEEK_HEAD).is_none(),
        "the streaming path is unchanged (no SeekHead)"
    );
}
