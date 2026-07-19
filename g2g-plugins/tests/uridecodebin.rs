//! uridecodebin front-door tests (M92): URI parsing + scheme dispatch + decode
//! chain assembly via `Registry::build_uridecodebin`. Assembly is checked
//! structurally (which elements get spliced), which needs no real media; the
//! ffmpeg-gated case confirms a real H.264 decoder is auto-plugged for a raw
//! target.

#![cfg(all(feature = "std", feature = "udp-ingress"))]

use g2g_core::runtime::{is_raw_video, Registry, UriError};
use g2g_core::{Caps, VideoCodec};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::uridecodebin;

fn is_h264(c: &Caps) -> bool {
    matches!(
        c,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        }
    )
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_uri(uridecodebin::udp_handler())
        .register_uri(uridecodebin::file_handler());
    reg
}

#[test]
fn rejects_malformed_uri() {
    let reg = registry();
    let err = reg
        .build_uridecodebin("not-a-uri", FakeSink::new(), &is_raw_video, 4)
        .expect_err("no scheme separator");
    assert!(matches!(err, UriError::Malformed));
}

#[test]
fn rejects_unknown_scheme() {
    let reg = registry();
    let err = reg
        .build_uridecodebin("ftp://host/file", FakeSink::new(), &is_raw_video, 4)
        .expect_err("ftp is not registered");
    assert!(matches!(err, UriError::UnknownScheme));
}

#[test]
fn udp_uri_with_no_decoder_cannot_reach_raw() {
    // The udp handler builds a UdpSrc (H.264), but no decoder is registered, so
    // the chain to a raw target cannot be assembled.
    let reg = registry();
    let err = reg
        .build_uridecodebin("udp://127.0.0.1:5004", FakeSink::new(), &is_raw_video, 4)
        .expect_err("no decoder registered");
    assert!(
        matches!(err, UriError::Decode(_)),
        "decode-chain failure, got {err:?}"
    );
}

#[test]
fn udp_uri_to_h264_target_is_a_direct_source_sink_graph() {
    // Target = H.264 (the source's own type): the chain is empty, so the graph
    // is just source -> sink. Proves URI parse -> source construction -> link.
    let reg = registry();
    let graph = reg
        .build_uridecodebin("udp://0.0.0.0:5004", FakeSink::new(), &is_h264, 4)
        .expect("empty chain links source straight to sink");
    assert_eq!(
        graph.finish().expect("graph validates").node_count(),
        2,
        "source + sink, no decoder"
    );
}

#[test]
fn bad_udp_authority_is_malformed() {
    let reg = registry();
    let err = reg
        .build_uridecodebin("udp://not:a:port", FakeSink::new(), &is_h264, 4)
        .expect_err("unparseable host:port");
    assert!(matches!(err, UriError::Malformed));
}

// The real payoff: with an H.264 decoder registered, a udp:// URI auto-plugs
// source -> decoder -> sink down to a raw target. Needs the ffmpeg decoder.
#[cfg(feature = "ffmpeg")]
#[test]
fn udp_uri_autoplugs_a_decoder_to_reach_raw() {
    use g2g_core::runtime::ElementFactory;
    use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};

    let mut reg = registry();
    reg.register(ElementFactory::of::<FfmpegH264Dec>(
        "ffmpegh264dec",
        |_out| Box::new(FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12)),
    ));

    let graph = reg
        .build_uridecodebin("udp://0.0.0.0:5004", FakeSink::new(), &is_raw_video, 4)
        .expect("decoder bridges H.264 to raw");
    assert_eq!(
        graph.finish().expect("graph validates").node_count(),
        3,
        "source -> ffmpegh264dec -> sink auto-plugged"
    );
}
