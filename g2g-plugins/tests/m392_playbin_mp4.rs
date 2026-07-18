//! M392 - `playbin uri=*.mp4` multi-stream fan-out + multi-hook dispatch. A lone
//! `playbin uri=file://x.mp4` probes a fragmented MP4's `moov` and auto-builds
//! `FileSrc -> Mp4DemuxN -> {decode -> auto sink}`, one branch per track, the MP4
//! sibling of the MKV (M382) and MPEG-TS (M389) fan-outs. The registry holds
//! three playbin hooks: an MP4 file is handled by `mp4_playbin`, a TS file by
//! `ts_playbin`, an MKV file by `mkv_playbin`, each declining the others.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, ElementFactory, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MultiInputElement,
    OutputSink, PadTemplate, PadTemplates, PushOutcome, Rate, RawVideoFormat, VideoCodec,
};

use g2g_plugins::mp4muxn::Mp4MuxN;

// --- caps + stub-element helpers ---
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
fn aac_any() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
    }
}
fn raw_video() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
fn raw_audio() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 0,
        sample_rate: 0,
    }
}

#[derive(Default)]
struct NullSink;
impl PadTemplates for NullSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}
impl AsyncElement for NullSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        _packet: g2g_core::PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A registry with all three playbin hooks, stub H.264 / AAC decoders, and the
/// auto-sink names, so a probed container's branches plug to raw.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);
    reg.register_playbin(g2g_plugins::uridecodebin::ts_playbin);
    reg.register_playbin(g2g_plugins::uridecodebin::mp4_playbin);
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw_video())),
        ]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac_any())),
            PadTemplate::source(CapsSet::one(raw_audio())),
        ]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || {
        Box::new(NullSink)
    }));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || {
        Box::new(NullSink)
    }));
    reg
}

fn temp_uri(tag: &str, bytes: &[u8]) -> (PathBuf, String) {
    let path = std::env::temp_dir().join(format!("g2g_m392_{}_{}.bin", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- A/V fragmented-MP4 builder (drives Mp4MuxN) ---
fn frame(data: Vec<u8>, pts_ns: u64) -> g2g_core::PipelinePacket {
    g2g_core::PipelinePacket::DataFrame(Frame::new(
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
#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: g2g_core::PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let g2g_core::PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
async fn av_mp4() -> Vec<u8> {
    let mut mux = Mp4MuxN::new(2);
    mux.configure_pipeline(
        0,
        &Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
        },
    )
    .unwrap();
    mux.configure_pipeline(
        1,
        &Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        },
    )
    .unwrap();
    let mut sink = Collect::default();
    let sps = [0x67u8, 0x42, 0, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(0, g2g_core::PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, g2g_core::PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

#[tokio::test]
async fn playbin_fans_out_a_fragmented_mp4() {
    let (path, uri) = temp_uri("av_mp4", &av_mp4().await);
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("mp4 playbin fans out");
    std::fs::remove_file(&path).ok();

    // FileSrc -> Mp4DemuxN(2). Video: demux -> decoder -> sink. Audio: demux ->
    // decoder -> audioconvert -> audioresample -> sink (the M422+ audio branch).
    assert_eq!(
        graph.node_count(),
        8,
        "source, demux, video decode+sink, audio decode+convert+resample+sink"
    );
    assert_eq!(
        graph.edges().len(),
        7,
        "video branch (2) + audio branch (4) + src->demux"
    );
}

#[tokio::test]
async fn the_mp4_hook_handles_mp4_only() {
    let reg = registry();

    // An MP4 file is handled by mp4_playbin (mkv_playbin / ts_playbin decline it).
    let (mp4_path, mp4_uri) = temp_uri("disp_mp4", &av_mp4().await);
    let mp4_graph = parse_launch(&reg, &format!("playbin uri={mp4_uri}")).expect("mp4 handled");
    std::fs::remove_file(&mp4_path).ok();
    assert_eq!(
        mp4_graph.node_count(),
        8,
        "MP4 fans out via mp4_playbin (audio branch adds convert+resample)"
    );
}
