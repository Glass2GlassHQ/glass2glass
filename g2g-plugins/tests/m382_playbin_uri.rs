//! M382 - `playbin uri=X` auto-fan-out. A lone `playbin uri=file://x.mkv` in a
//! text pipeline probes the Matroska container, then auto-builds
//! `FileSrc -> MkvDemuxN -> {decode -> auto sink}` with one branch per forwardable
//! stream, the multi-stream counterpart of the `playbin` macro (M196). The core
//! `parse_launch` calls a registry-installed `PlaybinHook`; the Matroska probe
//! lives in `g2g-plugins` (cross-crate by design).
//!
//! Structural (the graph is asserted, not run, like m379): the hook only fires for
//! a lone `playbin`, only for a probed Matroska file, and a registry without the
//! hook (or a non-Matroska file) falls back to the single-stream `playbin`.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::PipelinePacket;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, ElementFactory, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, RawVideoFormat, Rate, VideoCodec,
};

// --- caps helpers (the stub decoders' templates) ---
fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn aac_any() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_audio() -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 0, sample_rate: 0 }
}

/// An accept-anything sink, registered under the auto-sink names so the hook's
/// `make_element("autovideosink"/"autoaudiosink")` resolves. Tolerates the demux's
/// per-port retyping `CapsChanged`.
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
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A registry with the `file://` handler, stub H.264 / AAC decoders (so the
/// per-port auto-plug reaches raw), and the auto-sink names. The playbin hook is
/// added separately so a no-hook control registry can omit it.
fn base_registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_uri(g2g_plugins::uridecodebin::file_handler());
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([PadTemplate::sink(CapsSet::one(aac_any())), PadTemplate::source(CapsSet::one(raw_audio()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || Box::new(NullSink)));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || Box::new(NullSink)));
    reg
}

/// Write `bytes` to a unique temp file and return its `file://` URI.
fn temp_uri(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, String) {
    let path =
        std::env::temp_dir().join(format!("g2g_m382_{}_{}.bin", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write temp fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- A/V mux fixture: a real two-track (H.264 + AAC) Matroska file ---
fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(g2g_core::frame::Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        g2g_core::frame::FrameTiming { pts_ns, dts_ns: pts_ns, ..g2g_core::frame::FrameTiming::default() },
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
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(g2g_core::PushOutcome::Accepted)
        })
    }
}
async fn mux_av() -> Vec<u8> {
    use g2g_core::MultiInputElement;
    use g2g_plugins::mkvmuxn::MkvMuxN;
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_any()).unwrap();
    mux.configure_pipeline(1, &Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink).await.unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn playbin_uri_auto_fans_out_a_matroska_file() {
    let file = mux_av().await;
    let (path, uri) = temp_uri("av", &file);

    let mut reg = base_registry();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);

    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("playbin fans out");
    std::fs::remove_file(&path).ok();

    // FileSrc -> MkvDemuxN(2). Video port: demux -> decoder -> auto video sink.
    // Audio port: demux -> decoder -> audioconvert -> audioresample -> auto audio
    // sink (the M422+ audio branch, so the sink sees a fixed PCM format while the
    // converters absorb the stream's real channels / rate).
    // Nodes: source + demux + video(decoder+sink) + audio(decoder+convert+resample+sink) = 8.
    // Edges: src->demux(1) + video branch(2) + audio branch(4) = 7.
    assert_eq!(graph.node_count(), 8, "source, demux, video decode+sink, audio decode+convert+resample+sink");
    assert_eq!(graph.edges().len(), 7, "video branch (2 edges) + audio branch (4 edges) + src->demux");
}

#[tokio::test]
async fn playbin_uri_without_hook_stays_single_stream() {
    let file = mux_av().await;
    let (path, uri) = temp_uri("nohook", &file);

    // No register_playbin: parse_launch falls through to the single-stream
    // playbin expansion (source -> one decoder -> one auto sink).
    let reg = base_registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("single-stream playbin");
    std::fs::remove_file(&path).ok();

    assert_eq!(graph.node_count(), 3, "source, one decoder, one sink: single-stream playbin");
}

#[tokio::test]
async fn playbin_uri_non_matroska_falls_through() {
    // A file the Matroska probe cannot parse: the hook declines (Ok(None)) and the
    // single-stream playbin path takes over, so it is still a 3-node graph.
    let (path, uri) = temp_uri("notmkv", b"this is not a matroska container");

    let mut reg = base_registry();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);

    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("falls back to playbin");
    std::fs::remove_file(&path).ok();

    assert_eq!(graph.node_count(), 3, "non-Matroska declines: single-stream playbin");
}

/// A `playbin` that is *not* alone (here, with a trailing element) does not take
/// the auto-fan-out path; the lone-element guard leaves it to the normal builder.
#[test]
fn non_lone_playbin_is_not_hooked() {
    // `playbin ! fakesink` is two elements, so `lone_playbin_uri` returns None
    // and the hook never fires. With no `uri=` the single-stream expansion errors
    // (MissingUri) rather than the hook running, proving the guard held.
    let mut reg = base_registry();
    reg.register_launch(LaunchFactory::of::<NullSink>("fakesink", || Box::new(NullSink)));
    reg.register_playbin(|_, _| panic!("hook must not fire for a non-lone playbin"));

    let err = parse_launch(&reg, "playbin ! fakesink").unwrap_err();
    // Two-element line, playbin with no uri -> the normal builder's MissingUri.
    let _ = err; // any error is fine; the point is the hook panic did not fire.
}
