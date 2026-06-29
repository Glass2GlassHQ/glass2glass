//! M395 - HLS rendition discovery + `playbin uri=hls://...` fan-out core. The
//! network probe (`hls_playbin`) is validated live, not in CI; this exercises the
//! network-free assembly (`build_hls_ts_fanout`) that turns a master variant's
//! discovered streams into `HlsSrc -> TsDemuxN -> {decode -> auto sink}`, plus the
//! `variant_streams` rendition discovery that feeds it.
#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{ElementFactory, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, Rate, StreamType, VideoCodec,
};

use g2g_plugins::hls::{parse, Playlist};
use g2g_plugins::hlssrc::{variant_streams, HlsStreamInfo};
use g2g_plugins::uridecodebin::{build_hls_fmp4_fanout, build_hls_ts_fanout};

fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn aac_any() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: g2g_core::RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_audio() -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 0, sample_rate: 0 }
}

#[derive(Default)]
struct NullSink;
impl PadTemplates for NullSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}
impl AsyncElement for NullSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;
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

fn registry() -> Registry {
    let mut reg = Registry::new();
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

fn muxed(stream_type: StreamType, caps: Caps, video: bool) -> HlsStreamInfo {
    HlsStreamInfo { stream_type, caps, video, uri: None, name: String::new(), language: None }
}

#[test]
fn discovers_renditions_and_fans_out_a_muxed_ts_variant() {
    // Rendition discovery: a master with a muxed A/V variant yields a video + an
    // audio stream, both carried in the variant's own TS segments.
    let master_text = "#EXTM3U\n\
        #EXT-X-STREAM-INF:BANDWIDTH=2400000,RESOLUTION=1280x720,CODECS=\"avc1.4d401e,mp4a.40.2\"\n\
        720p.m3u8\n";
    let Playlist::Master(master) = parse(master_text).unwrap() else {
        panic!("expected master");
    };
    let streams = variant_streams(&master, &master.variants[0]);
    assert_eq!(streams.len(), 2);

    // The fan-out assembles HlsSrc -> TsDemuxN(2) -> two decode branches.
    let reg = registry();
    let graph = build_hls_ts_fanout(&reg, "https://example.com/master.m3u8", &streams)
        .expect("fan-out builds")
        .expect("two muxed streams fan out");
    assert_eq!(graph.node_count(), 6, "source, demux, two decoders, two auto sinks");
    assert_eq!(graph.edges().len(), 5, "one decode branch per muxed stream");
}

#[test]
fn declines_a_single_stream_variant() {
    // Only one routable muxed stream: not a fan-out, decline to the single-stream
    // handler (Ok(None)).
    let reg = registry();
    let streams = vec![muxed(StreamType::Video, h264_any(), true)];
    let graph = build_hls_ts_fanout(&reg, "https://example.com/v.m3u8", &streams).unwrap();
    assert!(graph.is_none(), "a lone video stream does not fan out");
}

#[test]
fn ignores_separate_audio_renditions_for_the_ts_demuxer() {
    // A separate-rendition audio (its own playlist URI) is not muxed in the
    // variant's TS, so it cannot route through the one TsDemuxN: with only the
    // muxed video left, the builder declines.
    let reg = registry();
    let streams = vec![
        muxed(StreamType::Video, h264_any(), true),
        HlsStreamInfo {
            stream_type: StreamType::Audio,
            caps: aac_any(),
            video: false,
            uri: Some("audio/en.m3u8".into()),
            name: "en".into(),
            language: Some("en".into()),
        },
    ];
    let graph = build_hls_ts_fanout(&reg, "https://example.com/v.m3u8", &streams).unwrap();
    assert!(graph.is_none(), "separate-rendition audio is not fanned through TsDemuxN");
}

/// An fMP4 / CMAF HLS variant fans out via `Mp4DemuxN`, its tracks discovered from
/// the `#EXT-X-MAP` init segment (here a two-track A/V file's ftyp+moov, built by
/// `Mp4MuxN`). The network-free assembly (`build_hls_fmp4_fanout`) is what the
/// hook calls after fetching the init.
#[test]
fn fmp4_variant_fans_out_via_mp4demuxn() {
    use core::future::Future;
    use g2g_core::frame::{Frame, FrameTiming};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::runtime::block_on;
    use g2g_core::{MultiInputElement, PipelinePacket, PushOutcome};
    use g2g_plugins::mp4muxn::Mp4MuxN;

    #[derive(Default)]
    struct ByteCapture {
        bytes: Vec<u8>,
    }
    impl OutputSink for ByteCapture {
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
            FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
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
    fn adts(payload: &[u8]) -> Vec<u8> {
        let len = payload.len() + 7;
        let mut au = vec![
            0xFF, 0xF1, (1 << 6) | (3 << 2), ((2 & 3) << 6) | ((len >> 11) & 3) as u8,
            ((len >> 3) & 0xFF) as u8, (((len & 7) << 5) as u8) | 0x1F, 0xFC,
        ];
        au.extend_from_slice(payload);
        au
    }

    // Mux a two-track (H.264 + AAC) fragmented MP4; its ftyp+moov is the init.
    let init = block_on(async {
        let mut mux = Mp4MuxN::new(2);
        mux.configure_pipeline(0, &h264_any()).unwrap();
        mux.configure_pipeline(1, &Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }).unwrap();
        let mut sink = ByteCapture::default();
        let sps = [0x67u8, 0x42, 0, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
        mux.process(1, frame(adts(&[0xA1, 0xA2]), 0), &mut sink).await.unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
        sink.bytes
    });

    let reg = registry();
    let graph = build_hls_fmp4_fanout(&reg, "https://example.com/master.m3u8", &init)
        .expect("fmp4 fan-out builds")
        .expect("two tracks fan out");
    assert_eq!(graph.node_count(), 6, "source, demux, two decoders, two auto sinks");
    assert_eq!(graph.edges().len(), 5);
}
