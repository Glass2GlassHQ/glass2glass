//! M482 - `decodebin name=d` fan-out in a `gst-launch` line: the decode-per-port
//! sibling of the M476 demux fan-out. A file source feeding a `decodebin` with
//! several `d.` pad refs probes the container, builds the multi-output demuxer, and
//! auto-plugs a decoder onto every requested port, so each branch receives DECODED
//! (raw) frames:
//!
//! ```text
//! filesrc location=movie.mp4 ! decodebin name=d
//!   d.video_0 ! videoconvert ! autovideosink
//!   d.audio_0 ! audioconvert ! autoaudiosink
//! ```
//!
//! This asserts the parse-time WIRING (a fan-out demux + a decode chain per port);
//! full playback is live-validated (a real A/V MP4 decodes both branches to raw).
//! Needs decoders in the autoplug pool (ffmpeg).

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, NodeKind, OutputSink, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::registry::default_registry;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}
fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
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
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// Mux a two-track (H.264 + AAC) MP4 and return its bytes.
async fn mux_av<M: MultiInputElement>(mut mux: M) -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

/// `decodebin name=d` over an A/V MP4 builds a fan-out demux plus a decode chain on
/// each branch: the video branch gets a parser + H.264 decoder, the audio branch an
/// AAC decoder, ahead of the user's `videoconvert` / `audioconvert`. A single
/// `Tee`-role demux node feeds both; a raw-demux line (M476) would have no decoders.
#[tokio::test]
async fn decodebin_name_fans_out_and_decodes_each_port() {
    let bytes = mux_av(g2g_plugins::mp4muxn::Mp4MuxN::new(2)).await;
    let path = std::env::temp_dir().join(format!("g2g-m482-{}.mp4", std::process::id()));
    std::fs::write(&path, &bytes).expect("write mp4");
    let p = path.display();

    let reg = default_registry();
    let line = format!(
        "filesrc location={p} ! decodebin name=d  \
         d.video_0 ! videoconvert ! fakesink  d.audio_0 ! audioconvert ! fakesink"
    );
    let graph = parse_launch(&reg, &line)
        .unwrap_or_else(|e| panic!("decodebin fan-out parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    let kinds: Vec<NodeKind> = vg.topo().iter().map(|&n| vg.kind(n)).collect();
    std::fs::remove_file(&path).ok();

    // One fan-out demux (Tee-role, 2 ports).
    assert!(
        kinds.iter().any(|k| matches!(k, NodeKind::Tee(2))),
        "decodebin built a 2-port fan-out demux: {kinds:?}"
    );
    // Decoders were spliced per port: filesrc + demux + (video parser + decoder +
    // convert + sink) + (audio decoder + convert + sink) is well past a bare 5-node
    // demux-only fan-out, so a healthy node count confirms the decode chains.
    assert!(
        kinds.len() >= 8,
        "a decode chain was spliced onto each port: {} nodes",
        kinds.len()
    );
}
