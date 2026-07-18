//! M412 - `playbin uri=*.mp4` routes an embedded subtitle track through a
//! `TextOverlayN` onto the video. When a probed MP4 carries both a video track and
//! a `tx3g` timed-text track, `mp4_playbin` builds
//! `FileSrc -> Mp4DemuxN -> { video: decode -> videoconvert(RGBA8) -> overlay ;
//! text: -> overlay } -> videoconvert(NV12) -> autovideosink`, so the subtitles
//! render on screen with no sidecar file. An MP4 with no text track keeps the
//! plain per-stream A/V fan-out (M392), proving the overlay is additive.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{parse_launch, ElementFactory, GraphNode, LaunchFactory, Registry};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, OutputSink, PadTemplate,
    PadTemplates, Rate, RawVideoFormat, VideoCodec,
};

// --- caps + stub-element helpers (as in m392) ---
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
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

/// A registry with the MP4 playbin hook, a stub H.264 decoder, and the auto sinks.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_playbin(g2g_plugins::uridecodebin::mp4_playbin);
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264_any())),
            PadTemplate::source(CapsSet::one(raw_video())),
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
    let path = std::env::temp_dir().join(format!("g2g_m412_{}_{}.mp4", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- progressive MP4 fixture builder (mdat first, so chunk offsets are constant) ---
fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}
fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = be32(payload.len() as u32 + 8).to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}
fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut p = vec![0u8; 4]; // version 0 + flags 0
    p.extend_from_slice(payload);
    mp4_box(kind, &p)
}
fn tkhd(track_id: u32, w: u32, h: u32) -> Vec<u8> {
    let mut c = vec![0u8; 80];
    c[8..12].copy_from_slice(&be32(track_id));
    c[72..76].copy_from_slice(&be32(w << 16));
    c[76..80].copy_from_slice(&be32(h << 16));
    full_box(b"tkhd", &c)
}
fn mdhd(timescale: u32) -> Vec<u8> {
    let mut c = vec![0u8; 16];
    c[8..12].copy_from_slice(&be32(timescale));
    full_box(b"mdhd", &c)
}
fn hdlr(handler: &[u8; 4]) -> Vec<u8> {
    let mut c = vec![0u8; 20];
    c[4..8].copy_from_slice(handler);
    full_box(b"hdlr", &c)
}
fn stsd(entry: &[u8]) -> Vec<u8> {
    let mut p = be32(1).to_vec();
    p.extend_from_slice(entry);
    full_box(b"stsd", &p)
}
fn stsz(sizes: &[u32]) -> Vec<u8> {
    let mut b = vec![0u8; 8]; // default_size 0, then count
    b[4..8].copy_from_slice(&be32(sizes.len() as u32));
    for s in sizes {
        b.extend_from_slice(&be32(*s));
    }
    full_box(b"stsz", &b)
}
fn stts(count: u32, delta: u32) -> Vec<u8> {
    let mut b = be32(1).to_vec();
    b.extend_from_slice(&be32(count));
    b.extend_from_slice(&be32(delta));
    full_box(b"stts", &b)
}
fn stsc(spc: u32) -> Vec<u8> {
    let mut b = be32(1).to_vec();
    b.extend_from_slice(&be32(1)); // first_chunk
    b.extend_from_slice(&be32(spc)); // samples_per_chunk
    b.extend_from_slice(&be32(1)); // sample_desc_index
    full_box(b"stsc", &b)
}
fn stco(offset: u32) -> Vec<u8> {
    let mut b = be32(1).to_vec();
    b.extend_from_slice(&be32(offset));
    full_box(b"stco", &b)
}
fn avc1() -> Vec<u8> {
    let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
    let pps: &[u8] = &[0x68, 0xce];
    let mut avcc = vec![0u8; 5];
    avcc.push(0xE1); // 1 SPS
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps);
    avcc.push(1); // 1 PPS
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps);
    let mut p = vec![0u8; 78];
    p.extend_from_slice(&mp4_box(b"avcC", &avcc));
    mp4_box(b"avc1", &p)
}
fn trak(tkhd: &[u8], mdhd: &[u8], hdlr: &[u8], stbl: &[u8]) -> Vec<u8> {
    let minf = mp4_box(b"minf", &mp4_box(b"stbl", stbl));
    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, &minf].concat());
    mp4_box(b"trak", &[tkhd, &mdia].concat())
}

/// A progressive MP4 with a video track and, when `with_text`, a `tx3g` subtitle
/// track. mdat is written first so the absolute chunk offsets are constant (8 and
/// 8 + video-sample-len), needing no placeholder rebuild.
fn mp4_with_optional_text(with_text: bool) -> Vec<u8> {
    // One AVCC video sample (4-byte length + 4-byte IDR NAL) and one tx3g cue.
    let vsample: &[u8] = &[0, 0, 0, 4, 0x65, 0x88, 0x84, 0x00];
    let cue = {
        let text = b"Hello";
        let mut s = (text.len() as u16).to_be_bytes().to_vec();
        s.extend_from_slice(text);
        s
    };
    let off_v = 8u32; // after the mdat box header
    let off_t = off_v + vsample.len() as u32;

    let v_stbl = [
        stsd(&avc1()),
        stsz(&[vsample.len() as u32]),
        stts(1, 3000),
        stsc(1),
        stco(off_v),
    ]
    .concat();
    let video_trak = trak(&tkhd(1, 320, 240), &mdhd(90_000), &hdlr(b"vide"), &v_stbl);

    let mut traks = video_trak;
    let mut mdat_body = vsample.to_vec();
    if with_text {
        let tx3g = mp4_box(b"tx3g", &[0u8; 8]);
        let t_stbl = [
            stsd(&tx3g),
            stsz(&[cue.len() as u32]),
            stts(1, 1000),
            stsc(1),
            stco(off_t),
        ]
        .concat();
        traks.extend_from_slice(&trak(&tkhd(2, 0, 0), &mdhd(1000), &hdlr(b"text"), &t_stbl));
        mdat_body.extend_from_slice(&cue);
    }
    let moov = mp4_box(b"moov", &traks);

    let mut file = mp4_box(b"mdat", &mdat_body);
    file.extend_from_slice(&moov);
    file
}

/// Incoming-edge count per node id, to spot the fan-in (the overlay muxer).
fn inbound_counts(graph: &Graph<GraphNode>) -> std::collections::HashMap<u32, usize> {
    let mut counts = std::collections::HashMap::new();
    for e in graph.edges() {
        *counts.entry(e.dst.node.0).or_insert(0) += 1;
    }
    counts
}

#[test]
fn subtitle_track_routes_through_a_text_overlay() {
    let (path, uri) = temp_uri("vid_text", &mp4_with_optional_text(true));
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("mp4+subs playbin builds");
    std::fs::remove_file(&path).ok();

    // FileSrc, Mp4DemuxN, h264 stub, videoconvert(RGBA8), TextOverlayN,
    // videoconvert(NV12), autovideosink.
    assert_eq!(
        graph.node_count(),
        7,
        "source + demux + decode + 2 converts + overlay + sink"
    );
    assert_eq!(
        graph.edges().len(),
        7,
        "video decode path + text join + sink path"
    );

    // Exactly one node is a fan-in (the overlay), fed by the video convert and the
    // demux's text port; everything else has a single input.
    let counts = inbound_counts(&graph);
    let fan_ins: Vec<_> = counts.values().filter(|&&n| n >= 2).collect();
    assert_eq!(fan_ins.len(), 1, "one fan-in node: the subtitle overlay");
    assert_eq!(
        *fan_ins[0], 2,
        "the overlay joins the video and the text streams"
    );
}

#[test]
fn an_mp4_without_subtitles_keeps_the_plain_fanout() {
    let (path, uri) = temp_uri("vid_only", &mp4_with_optional_text(false));
    let reg = registry();
    let graph =
        parse_launch(&reg, &format!("playbin uri={uri}")).expect("video-only playbin builds");
    std::fs::remove_file(&path).ok();

    // FileSrc -> Mp4DemuxN -> h264 stub -> autovideosink: no overlay inserted.
    assert_eq!(
        graph.node_count(),
        4,
        "source + demux + decode + sink, no overlay"
    );
    let counts = inbound_counts(&graph);
    assert!(
        counts.values().all(|&n| n <= 1),
        "no fan-in node without a subtitle track"
    );
}
