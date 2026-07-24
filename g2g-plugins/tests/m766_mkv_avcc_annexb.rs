//! M766: the Matroska demuxers convert an AVCC / HVCC length-prefixed H.264 /
//! H.265 track (the container-native framing, declared by the `avcC` / `hvcC`
//! `CodecPrivate`) to the Annex-B framing the g2g pipeline assumes, prepending
//! the config record's parameter sets on keyframes. Round-trips our own A/V
//! muxer (which writes avcC + AVCC samples) through `MkvDemux` and `MkvDemuxN`
//! and asserts the demuxed frames are byte-exactly the original Annex-B access
//! units, so a bare `decodebin` on an `.mkv` can NAL-split them.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, MultiOutputElement,
    MultiOutputSink, OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemux, MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

fn video_caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
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
                    self.bytes.extend_from_slice(s);
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Frames per port from a `MultiOutputElement`.
#[derive(Default)]
struct PortCapture {
    frames: Vec<(usize, Vec<u8>)>,
}
impl MultiOutputSink for PortCapture {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((port, s.to_vec()));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        1
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

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// Mux the given Annex-B access units as one video track (our A/V muxer writes
/// an avcC / hvcC `CodecPrivate` + length-prefixed samples), returning the
/// Matroska byte stream.
async fn mux_track(codec: VideoCodec, aus: &[Vec<u8>]) -> Vec<u8> {
    let mut mux = MkvMuxN::new(1);
    mux.configure_pipeline(0, &video_caps(codec)).unwrap();
    let mut sink = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(0, frame(au.clone(), i as u64 * 33_000_000), &mut sink)
            .await
            .unwrap();
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

/// Demux `mkv` through the single-output `MkvDemux` and return the frames.
async fn demux_frames(mkv: Vec<u8>) -> Vec<Vec<u8>> {
    let mut demux = MkvDemux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("byte-stream input accepted");
    let mut sink = CaptureSink::default();
    demux
        .process(frame(mkv, 0), &mut sink)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.frames
}

#[tokio::test]
async fn h264_avcc_track_demuxes_as_annexb() {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];
    let p: &[u8] = &[0x41, 0x9a, 0x11, 0x22];

    let aus = vec![annexb(&[sps, pps, idr]), annexb(&[p])];
    let mkv = mux_track(VideoCodec::H264, &aus).await;
    let frames = demux_frames(mkv).await;

    assert_eq!(frames.len(), 2, "both access units recovered");
    assert_eq!(
        frames[0],
        annexb(&[sps, pps, idr]),
        "keyframe is Annex-B with the parameter sets ahead of the IDR"
    );
    assert_eq!(frames[1], annexb(&[p]), "delta frame is Annex-B, no sets");
}

#[tokio::test]
async fn h265_hvcc_track_demuxes_as_annexb() {
    // Minimal HEVC NALs: type in bits 6..1 of byte 0 (VPS 32, SPS 33, PPS 34,
    // IDR_W_RADL 19, TRAIL_R 1). The SPS body is long enough for the hvcC
    // record's 12-byte PTL copy.
    let vps: &[u8] = &[0x40, 0x01, 0x0c, 0x01, 0xff, 0xff];
    let sps: &[u8] = &[
        0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x03, 0x00, 0x5d,
    ];
    let pps: &[u8] = &[0x44, 0x01, 0xc1, 0x72, 0xb4];
    let idr: &[u8] = &[0x26, 0x01, 0xaf, 0x08];
    let trail: &[u8] = &[0x02, 0x01, 0xd0, 0x09];

    let aus = vec![annexb(&[vps, sps, pps, idr]), annexb(&[trail])];
    let mkv = mux_track(VideoCodec::H265, &aus).await;
    let frames = demux_frames(mkv).await;

    assert_eq!(frames.len(), 2, "both access units recovered");
    assert_eq!(
        frames[0],
        annexb(&[vps, sps, pps, idr]),
        "keyframe is Annex-B with VPS/SPS/PPS ahead of the IDR"
    );
    assert_eq!(frames[1], annexb(&[trail]), "delta frame is Annex-B");
}

#[tokio::test]
async fn mkvdemuxn_port_forwards_annexb() {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0x88];
    let pps: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00];

    let aus = vec![annexb(&[sps, pps, idr])];
    let mkv = mux_track(VideoCodec::H264, &aus).await;

    let mut demux = MkvDemuxN::new(vec![MkvStream::H264]);
    let mut sink = PortCapture::default();
    demux
        .process(frame(mkv, 0), &mut sink)
        .await
        .expect("demux");

    assert_eq!(sink.frames.len(), 1);
    assert_eq!(sink.frames[0].0, 0, "routed to the H.264 port");
    assert_eq!(
        sink.frames[0].1,
        annexb(&[sps, pps, idr]),
        "the fan-out port forwards Annex-B too"
    );
}
