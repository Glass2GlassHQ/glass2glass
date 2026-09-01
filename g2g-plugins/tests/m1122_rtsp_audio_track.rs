//! M1122: `RtspSrcN` plays an RTSP stream's audio track alongside its video.
//!
//! The offline tests cover the pad layout and the graph the `playbin
//! uri=rtsp://...` hook assembles. The live test needs an A/V RTSP feed:
//!
//! ```sh
//! G2G_RTSP_AV_TEST_URL=rtsp://localhost:8554/avpattern cargo test -p g2g-plugins \
//!     --features "rtsp ffmpeg" --test m1122_rtsp_audio_track -- --ignored --nocapture
//! ```

#![cfg(feature = "rtsp")]

use g2g_plugins::rtspsrcn::{RtspSrcN, AUDIO_PORT, VIDEO_PORT};

use g2g_core::MultiOutputSource;

/// The feed the live test plays when `G2G_RTSP_AV_TEST_URL` is unset.
const DEFAULT_FEED_URL: &str = "rtsp://localhost:8554/avpattern";

#[test]
fn a_video_only_element_has_one_pad() {
    let src = RtspSrcN::new("rtsp://example/stream");
    assert_eq!(src.output_count(), 1);
    assert!(
        src.output_caps(VIDEO_PORT).is_ok(),
        "the video pad is always present"
    );
}

#[test]
fn linking_two_pads_adds_the_audio_output() {
    let src = RtspSrcN::new("rtsp://example/stream").with_outputs(2);
    assert_eq!(src.output_count(), 2);
    assert!(matches!(
        src.output_caps(AUDIO_PORT),
        Ok(g2g_core::Caps::Audio {
            format: g2g_core::AudioFormat::Aac,
            ..
        })
    ));
}

/// A launch line spells the fan-out with two output pads and resolves
/// `rtspsrcn` from the default registry.
#[test]
fn launch_line_builds_the_av_fanout() {
    let reg = g2g_plugins::registry::default_registry();
    let line = format!(
        "rtspsrcn name=s location={DEFAULT_FEED_URL} \
         s. ! fakesink  s. ! fakesink"
    );
    g2g_core::runtime::parse_launch(&reg, &line).expect("rtspsrcn resolves with two linked pads");
}

/// The `playbin uri=rtsp://...` fan-out: one session, a decode branch per track.
/// Network-free (the DESCRIBE result is supplied), so it runs in CI.
#[cfg(all(feature = "ffmpeg", feature = "std"))]
mod fanout {
    use super::*;

    use g2g_core::{AudioFormat, Caps, Dim, Rate, VideoCodec};
    use g2g_plugins::rtspsrcn::RtspTracks;
    use g2g_plugins::uridecodebin::build_rtsp_av_fanout;

    fn tracks(audio: Option<Caps>) -> RtspTracks {
        RtspTracks {
            video: Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            audio,
        }
    }

    #[test]
    fn an_av_stream_gets_a_decode_branch_per_track() {
        let reg = g2g_plugins::registry::default_registry();
        let graph = build_rtsp_av_fanout(
            &reg,
            DEFAULT_FEED_URL,
            &tracks(Some(Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
            })),
        )
        .expect("the fan-out assembles")
        .expect("an audio track builds the two-pad graph");
        // One RtspSrcN, a video branch (parse, decode, deinterlace, sink) and an
        // audio branch (parse, decode, audioconvert, audioresample, sink).
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edges().len(), 9);
    }

    /// The hook's own path: DESCRIBE a live feed and assemble the same graph the
    /// network-free builder does.
    #[test]
    #[ignore = "requires a live RTSP A/V feed (G2G_RTSP_AV_TEST_URL, default rtsp://localhost:8554/avpattern)"]
    fn the_playbin_hook_describes_a_live_feed_and_fans_it_out() {
        let url =
            std::env::var("G2G_RTSP_AV_TEST_URL").unwrap_or_else(|_| DEFAULT_FEED_URL.to_string());
        let reg = g2g_plugins::registry::default_registry();
        let graph = g2g_plugins::uridecodebin::rtsp_playbin(&reg, &url)
            .expect("the hook assembles")
            .unwrap_or_else(|| {
                panic!("the hook declined {url}: no A/V feed there (is mediamtx running?)")
            });
        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edges().len(), 9);
    }

    #[test]
    fn a_video_only_stream_declines_to_the_single_pad_source() {
        let reg = g2g_plugins::registry::default_registry();
        assert!(
            build_rtsp_av_fanout(&reg, DEFAULT_FEED_URL, &tracks(None))
                .expect("the fan-out assembles")
                .is_none(),
            "without audio the hook must decline so rtsp_handler plugs RtspSrc"
        );
    }
}

/// The live decode of both tracks, which needs the ffmpeg decoders.
#[cfg(all(feature = "ffmpeg", feature = "std"))]
mod live {
    use super::*;

    /// User-Agent the DESCRIBE probe sends.
    const PROBE_USER_AGENT: &str = "g2g-m1122-test";

    use core::future::Future;
    use core::pin::Pin;
    use std::sync::Arc;
    use std::time::Duration;

    use g2g_core::graph::Graph;
    use g2g_core::runtime::{run_graph, GraphNodeRef};
    use g2g_core::{
        AsyncElement, AudioFormat, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink,
        PipelineClock, PipelinePacket,
    };
    use g2g_plugins::rtspsrcn::probe_tracks;

    /// Access units each pad emits before its EOS. At the feed's frame rate the
    /// video pad is the slower of the two, so this also bounds the run length.
    const FRAMES_PER_PAD: u64 = 30;

    /// In-flight packets per link.
    const LINK_CAPACITY: usize = 4;

    /// How far apart the two decoded timelines may start. Audio is emitted from
    /// the first packet while video waits for the first key frame, so on a
    /// mid-GOP tune-in the video timeline legitimately starts one GOP later.
    const MAX_AV_START_SKEW_NS: u64 = 2_000_000_000;

    /// Wall-clock budget for the whole live run: connect, play, and decode
    /// `FRAMES_PER_PAD` frames on each pad.
    const RUN_BUDGET: Duration = Duration::from_secs(30);

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    /// Records what one decoded branch produced: frame count, the first frame's
    /// PTS, and the caps the decoder last declared.
    #[derive(Debug, Default)]
    struct Decoded {
        frames: u64,
        first_pts_ns: Option<u64>,
        caps: Option<Caps>,
    }

    #[derive(Debug)]
    struct ProbeSink {
        seen: Arc<std::sync::Mutex<Decoded>>,
    }

    impl AsyncElement for ProbeSink {
        type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

        fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
            Ok(caps.clone())
        }
        fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
            CapsConstraint::AcceptsAny
        }
        fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            self.seen.lock().unwrap().caps = Some(caps.clone());
            Ok(ConfigureOutcome::Accepted)
        }
        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                let mut seen = self.seen.lock().unwrap();
                match packet {
                    PipelinePacket::DataFrame(f) => {
                        seen.first_pts_ns.get_or_insert(f.timing.pts_ns);
                        seen.frames += 1;
                    }
                    PipelinePacket::CapsChanged(caps) => seen.caps = Some(caps),
                    _ => {}
                }
                Ok(())
            })
        }
    }

    #[tokio::test]
    #[ignore = "requires a live RTSP A/V feed (G2G_RTSP_AV_TEST_URL, default rtsp://localhost:8554/avpattern)"]
    async fn both_tracks_arrive_and_decode_on_one_timeline() {
        use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
        use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};

        let url =
            std::env::var("G2G_RTSP_AV_TEST_URL").unwrap_or_else(|_| DEFAULT_FEED_URL.to_string());

        // The expected geometry / sample rate / channel count come from the
        // feed's own SDP, so the assertions below track the server rather than
        // a hardcoded copy of it.
        let tracks = probe_tracks(&url, PROBE_USER_AGENT)
            .await
            .unwrap_or_else(|| panic!("no RTSP feed at {url} (is mediamtx running?)"));
        let audio_caps = tracks
            .audio
            .clone()
            .unwrap_or_else(|| panic!("{url} offers no audio track; this test needs an A/V feed"));
        let Caps::Audio {
            format: audio_format,
            channels: sdp_channels,
            sample_rate: sdp_sample_rate,
        } = audio_caps
        else {
            panic!("audio track caps are not Caps::Audio: {audio_caps:?}");
        };
        assert_eq!(audio_format, AudioFormat::Aac, "this feed should carry AAC");
        eprintln!(
            "m1122: {url}\n  video {:?}\n  audio {audio_caps:?}",
            tracks.video
        );

        let video_seen = Arc::new(std::sync::Mutex::new(Decoded::default()));
        let audio_seen = Arc::new(std::sync::Mutex::new(Decoded::default()));

        let mut src = RtspSrcN::new(url.clone())
            .with_tracks(&tracks)
            .with_frame_limit(FRAMES_PER_PAD);
        assert_eq!(src.output_count(), 2, "the SDP's audio must add a pad");
        let mut video_dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        let mut audio_dec = FfmpegAudioDec::new();
        let mut video_sink = ProbeSink {
            seen: video_seen.clone(),
        };
        let mut audio_sink = ProbeSink {
            seen: audio_seen.clone(),
        };

        let mut graph: Graph<GraphNodeRef> = Graph::new();
        let node = graph.add_fanout_src(GraphNodeRef::fanout_source_ref(&mut src), 2);
        let vdec = graph.add_transform(GraphNodeRef::element_ref(&mut video_dec));
        let vsnk = graph.add_sink(GraphNodeRef::element_ref(&mut video_sink));
        let adec = graph.add_transform(GraphNodeRef::element_ref(&mut audio_dec));
        let asnk = graph.add_sink(GraphNodeRef::element_ref(&mut audio_sink));
        graph.link(node.output(VIDEO_PORT as u8), vdec).unwrap();
        graph.link(vdec, vsnk).unwrap();
        graph.link(node.output(AUDIO_PORT as u8), adec).unwrap();
        graph.link(adec, asnk).unwrap();

        let stats = tokio::time::timeout(RUN_BUDGET, run_graph(graph, &ZeroClock, LINK_CAPACITY))
            .await
            .expect("the live A/V run must finish within the budget")
            .expect("the live A/V run must succeed");
        eprintln!("m1122: {stats:?}");

        let video = video_seen.lock().unwrap();
        let audio = audio_seen.lock().unwrap();
        eprintln!("m1122: video {video:?}\n  audio {audio:?}");

        assert!(video.frames > 0, "no video frames decoded");
        assert!(audio.frames > 0, "no audio frames decoded");

        // Video decodes to raw frames at the SDP's geometry.
        let Caps::CompressedVideo {
            width: sdp_width,
            height: sdp_height,
            ..
        } = tracks.video
        else {
            panic!(
                "video track caps are not compressed video: {:?}",
                tracks.video
            );
        };
        match video.caps.as_ref().expect("video sink saw caps") {
            Caps::RawVideo { width, height, .. } => {
                assert_eq!(*width, sdp_width, "decoded width must match the SDP");
                assert_eq!(*height, sdp_height, "decoded height must match the SDP");
            }
            other => panic!("video branch did not decode to raw video: {other:?}"),
        }

        // Audio decodes to PCM at the SDP's sample rate and channel count.
        match audio.caps.as_ref().expect("audio sink saw caps") {
            Caps::Audio {
                format,
                channels,
                sample_rate,
            } => {
                assert!(
                    matches!(
                        format,
                        AudioFormat::PcmS16Le | AudioFormat::PcmF32Le | AudioFormat::PcmS32Le
                    ),
                    "audio branch did not decode to PCM: {format:?}"
                );
                assert_eq!(
                    *sample_rate, sdp_sample_rate,
                    "decoded sample rate must match the SDP"
                );
                assert_eq!(
                    *channels, sdp_channels,
                    "decoded channel count must match the SDP"
                );
            }
            other => panic!("audio branch did not decode to PCM: {other:?}"),
        }

        // Both tracks ride one timeline: the video pad's first PTS is late by at
        // most the key-frame wait, never a separate origin per stream.
        let video_start = video.first_pts_ns.expect("video PTS");
        let audio_start = audio.first_pts_ns.expect("audio PTS");
        let skew = video_start.abs_diff(audio_start);
        assert!(
            skew <= MAX_AV_START_SKEW_NS,
            "A/V timelines start {skew} ns apart (video {video_start}, audio {audio_start}), \
             more than the {MAX_AV_START_SKEW_NS} ns key-frame allowance"
        );
    }
}
