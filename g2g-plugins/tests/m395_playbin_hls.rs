//! M395 - HLS rendition discovery + `playbin uri=hls://...` fan-out core. The
//! network probe (`hls_playbin`) is validated live, not in CI; this exercises the
//! network-free assembly (`build_hls_ts_fanout`) that turns the streams of a
//! variant's first TS segment into `HlsSrc -> TsDemuxN -> {decode -> auto sink}`,
//! plus the `variant_streams` rendition discovery the separate-audio graph reads.
//! The M1169 checks at the bottom cover the open-ported siblings a lone
//! `fallbacksrc uri=hls://...` builds from the same probe.
#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::runtime::{attach_seek_controller, ElementFactory, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink,
    PadTemplate, PadTemplates, Rate, StreamType, VideoCodec,
};

use g2g_plugins::hls::{parse, Playlist};
use g2g_plugins::hlssrc::variant_streams;
use g2g_plugins::uridecodebin::{
    build_hls_fmp4_fanout, build_hls_separate_fanout, build_hls_ts_fanout, hls_fmp4_uri_fanout,
    hls_ts_uri_fanout,
};

const AV_SEGMENT: &str = "av_h264_aac44100.ts";
const VIDEO_ONLY_SEGMENT: &str = "h264_64x48_25fps_gop10.ts";
const TWO_AAC_SEGMENT: &str = "multitrack_h264_aac_hls_segment.ts";

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}
fn aac_any() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}
fn raw_video() -> Caps {
    Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}
fn raw_audio() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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

fn registry() -> Registry {
    let mut reg = Registry::new();
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

fn ts_fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

#[test]
fn fans_out_the_streams_of_a_muxed_ts_segment() {
    // The fan-out assembles HlsSrc -> TsDemuxN(2) -> two decode branches. The audio
    // branch adds audioconvert + audioresample (the M422+ chain) so the sink sees a
    // fixed PCM format: video(decoder+sink) + audio(decoder+convert+resample+sink).
    let reg = registry();
    let graph = build_hls_ts_fanout(
        &reg,
        &g2g_core::runtime::PlaybinSinks::AUTO,
        "https://example.com/master.m3u8",
        &ts_fixture(AV_SEGMENT),
    )
    .expect("fan-out builds")
    .expect("two muxed streams fan out");
    assert_eq!(
        graph.node_count(),
        9,
        "source, demux, video decode+deinterlace+sink, audio decode+convert+resample+sink"
    );
    assert_eq!(
        graph.edges().len(),
        8,
        "video branch (3, with the M935 auto deinterlace) + audio branch (4) + src->demux"
    );
}

#[test]
fn declines_a_single_stream_segment() {
    // Only one forwardable muxed stream: not a fan-out, decline to the
    // single-stream handler (Ok(None)).
    let reg = registry();
    let graph = build_hls_ts_fanout(
        &reg,
        &g2g_core::runtime::PlaybinSinks::AUTO,
        "https://example.com/v.m3u8",
        &ts_fixture(VIDEO_ONLY_SEGMENT),
    )
    .unwrap();
    assert!(graph.is_none(), "a lone video stream does not fan out");
}

/// A variant with a separate audio rendition fans out as two independent sources
/// merged into one graph: the video variant (own TS) and the audio rendition
/// playlist, each `HlsSrc -> TsDemuxN -> decode -> sink`.
#[test]
fn separate_audio_rendition_builds_two_source_chains() {
    let master_text = "#EXTM3U\n\
        #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aud\",NAME=\"en\",LANGUAGE=\"en\",DEFAULT=YES,URI=\"a/en.m3u8\"\n\
        #EXT-X-STREAM-INF:BANDWIDTH=2400000,CODECS=\"avc1.4d401e,mp4a.40.2\",AUDIO=\"aud\"\n\
        v.m3u8\n";
    let Playlist::Master(master) = parse(master_text).unwrap() else {
        panic!("expected master");
    };
    let streams = variant_streams(&master, &master.variants[0]);
    // One muxed video + one separate-rendition audio (its own playlist URI).
    assert_eq!(streams.len(), 2);
    assert!(streams[0].video && streams[0].uri.is_none());
    assert_eq!(streams[1].uri.as_deref(), Some("a/en.m3u8"));

    let reg = registry();
    let graph = build_hls_separate_fanout(
        &reg,
        &g2g_core::runtime::PlaybinSinks::AUTO,
        "https://example.com/master.m3u8",
        &streams,
        "https://example.com/a/en.m3u8",
    )
    .expect("separate fan-out builds")
    .expect("video + audio chains");
    // Two chains: video = HlsSrc -> TsDemuxN -> decoder -> deinterlace -> sink
    // (5 nodes, 4 edges, with the M935 auto deinterlace); audio = HlsSrc ->
    // TsDemuxN -> decoder -> audioconvert -> audioresample -> sink (6 nodes,
    // 5 edges, the M422+ audio branch). Total 11 nodes, 9 edges.
    assert_eq!(
        graph.node_count(),
        11,
        "video chain (5, with deinterlace) + audio chain (6, with convert+resample)"
    );
    assert_eq!(graph.edges().len(), 9);
}

/// The `#EXT-X-MAP` init segment of a two-track (H.264 + AAC) fragmented MP4:
/// the `ftyp` + `moov` `Mp4MuxN` writes ahead of the first fragment, which is
/// what an fMP4 HLS variant's tracks are discovered from.
fn fmp4_init() -> Vec<u8> {
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
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
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
    fn adts(payload: &[u8]) -> Vec<u8> {
        let len = payload.len() + 7;
        let mut au = vec![
            0xFF,
            0xF1,
            (1 << 6) | (3 << 2),
            ((2 & 3) << 6) | ((len >> 11) & 3) as u8,
            ((len >> 3) & 0xFF) as u8,
            (((len & 7) << 5) as u8) | 0x1F,
            0xFC,
        ];
        au.extend_from_slice(payload);
        au
    }

    block_on(async {
        let mut mux = Mp4MuxN::new(2);
        mux.configure_pipeline(0, &h264_any()).unwrap();
        mux.configure_pipeline(
            1,
            &Caps::Audio {
                format: AudioFormat::Aac,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            },
        )
        .unwrap();
        let mut sink = ByteCapture::default();
        let sps = [0x67u8, 0x42, 0, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
            .await
            .unwrap();
        mux.process(1, frame(adts(&[0xA1, 0xA2]), 0), &mut sink)
            .await
            .unwrap();
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        sink.bytes
    })
}

/// An fMP4 / CMAF HLS variant fans out via `Mp4DemuxN`, its tracks discovered
/// from the `#EXT-X-MAP` init segment. The network-free assembly
/// (`build_hls_fmp4_fanout`) is what the hook calls after fetching the init.
#[test]
fn fmp4_variant_fans_out_via_mp4demuxn() {
    let reg = registry();
    let graph = build_hls_fmp4_fanout(
        &reg,
        &g2g_core::runtime::PlaybinSinks::AUTO,
        "https://example.com/master.m3u8",
        &fmp4_init(),
    )
    .expect("fmp4 fan-out builds")
    .expect("two tracks fan out");
    // Audio track goes through the M422+ branch (decoder+convert+resample+sink),
    // the video one through the M935 auto deinterlace: source + demux +
    // video(decoder+deinterlace+sink) + audio(decoder+convert+resample+sink).
    assert_eq!(
        graph.node_count(),
        9,
        "source, demux, video decode+deinterlace+sink, audio decode+convert+resample+sink"
    );
    assert_eq!(graph.edges().len(), 8);
}

// qtdemux holds the whole stream, so the seek goes to it and not to the HlsSrc feeding it
#[test]
fn qtdemux_behind_hlssrc_takes_the_seek() {
    let line = "hlssrc location=http://127.0.0.1:1/master.m3u8 ! qtdemux ! fakesink";
    let mut graph =
        g2g_core::runtime::parse_launch(&g2g_plugins::registry::default_registry(), line).unwrap();
    attach_seek_controller(&mut graph).expect("exactly one element takes the seek");
}

#[test]
fn a_ts_variant_seeks_through_its_source() {
    let mut graph = build_hls_ts_fanout(
        &registry(),
        &g2g_core::runtime::PlaybinSinks::AUTO,
        "https://example.com/master.m3u8",
        &ts_fixture(AV_SEGMENT),
    )
    .unwrap()
    .unwrap();
    attach_seek_controller(&mut graph).expect("the HlsSrc takes the seek");
}

// --- M1169: the open-ported siblings a lone `fallbacksrc uri=hls://...` uses ---
//
// These stop at the demuxer instead of closing on sinks, because the caller puts
// a `fallbackswitch` between each port and its sink. Same probe, same packaging
// split, so the checks below assert the ports rather than a whole graph.

/// A muxed MPEG-TS segment's streams become one fan-out port each, tagged with
/// the kind the caller picks that port's switch and sink by.
#[test]
fn a_muxed_ts_segment_reports_a_port_per_stream() {
    let fanout = hls_ts_uri_fanout("https://example.com/master.m3u8", &ts_fixture(AV_SEGMENT))
        .expect("a muxed A/V segment fans out");
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio],
        "one port per muxed stream, in demux port order"
    );
    assert!(matches!(
        fanout.ports[0].caps,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        }
    ));
    assert!(matches!(
        fanout.ports[1].caps,
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        }
    ));
}

/// CODECS names AAC once for both tracks, the segment's PMT lists each one.
#[test]
fn two_aac_tracks_get_a_port_each() {
    let fanout = hls_ts_uri_fanout(
        "https://example.com/master.m3u8",
        &ts_fixture(TWO_AAC_SEGMENT),
    )
    .expect("a muxed segment fans out");
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio, StreamType::Audio]
    );
}

/// Bytes with no PMT carry no forwardable stream, so the hook declines.
#[test]
fn a_segment_without_a_pmt_gets_no_fanout() {
    assert!(hls_ts_uri_fanout("https://example.com/v.m3u8", &[]).is_none());
}

/// The fMP4 sibling: the ports come from the `#EXT-X-MAP` init segment's `moov`,
/// the same tracks `build_hls_fmp4_fanout` decodes.
#[test]
fn an_fmp4_variant_reports_a_port_per_track() {
    let fanout = hls_fmp4_uri_fanout("https://example.com/master.m3u8", &fmp4_init())
        .expect("a two-track init fans out");
    assert_eq!(
        fanout
            .ports
            .iter()
            .map(|p| p.stream_type)
            .collect::<Vec<_>>(),
        vec![StreamType::Video, StreamType::Audio]
    );

    // An init segment with no forwardable track: decline.
    assert!(hls_fmp4_uri_fanout("https://example.com/master.m3u8", &[]).is_none());
}
