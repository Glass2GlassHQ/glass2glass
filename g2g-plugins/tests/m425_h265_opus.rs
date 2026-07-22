//! M425: H.265 access-unit re-framing auto-insert + Opus decode auto-plug.
//!
//! Two follow-ups to the M421 (H.264 re-framer) / M422 (AAC decode) playback work:
//!
//! - The auto-plug parser provider now prepends an access-unit-re-framing
//!   `h265parse` before an H.265 decoder, exactly as it does for H.264, so a
//!   TS / HLS HEVC stream (one PES per buffer, not access-unit-aligned) is fed one
//!   coded picture per packet instead of mis-framed slices.
//! - An Opus audio track now auto-plugs `opusdec` in the playbin audio branch. The
//!   demuxer must surface the track's concrete channel count (libopus is created
//!   per channel count, unlike AAC where libavcodec discovers it), which
//!   `mkvdemux::forwardable_streams` now does; with the channels-0 placeholder no
//!   decode chain plugs.
//!
//! Each test carries its own imports so no import is unused under a feature combo
//! that compiles out the test (the file is otherwise `std`-only scaffolding).

#![cfg(feature = "std")]

/// The auto-plug parser provider prepends a re-framing `h265parse` to the chain
/// that decodes H.265, the HEVC sibling of the M421 H.264 behaviour, so the
/// decoder is fed one access unit per packet. `decodebin` applies the prepend
/// (`autoplug_names` reports the bare decoder), so the spliced chain is one node
/// longer than the decoder alone.
#[cfg(feature = "ffmpeg")]
#[test]
fn decodebin_inserts_h265_reframer_before_the_decoder() {
    use g2g_core::runtime::{is_raw_video, GraphNode, GraphNodeRef};
    use g2g_core::{Caps, Dim, Graph, Rate, VideoCodec};
    use g2g_plugins::fakesink::FakeSink;
    use g2g_plugins::h265parse::H265Parse;
    use g2g_plugins::registry::default_registry;

    let h265 = Caps::CompressedVideo {
        codec: VideoCodec::H265,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let reg = default_registry();
    let decoder_only = reg
        .autoplug_names(&h265, &is_raw_video, 6)
        .expect("an H.265 decode chain plugs");

    // Splice the decode chain into a graph; decodebin prepends the re-framer.
    let mut g: Graph<GraphNode> = Graph::new();
    let head = g.add_transform(GraphNodeRef::element(H265Parse::new()));
    let tail = g.add_sink(GraphNodeRef::element(FakeSink::new()));
    let spliced = reg
        .decodebin(&mut g, head, tail, &h265, &is_raw_video, 6)
        .expect("H.265 -> raw chain");

    assert_eq!(
        spliced.len(),
        decoder_only.len() + 1,
        "decodebin splices the re-framing h265parse ahead of the decoder ({decoder_only:?})"
    );
}

/// An Opus track with a concrete channel count auto-plugs `opusdec` to raw PCM.
#[cfg(feature = "opus")]
#[test]
fn opus_with_concrete_channels_auto_plugs_opusdec() {
    use g2g_core::runtime::is_raw_audio;
    use g2g_core::{AudioFormat, Caps};
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();
    let opus = Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 0,
    };
    let names = reg
        .autoplug_names(&opus, &is_raw_audio, 6)
        .expect("a concrete-channel Opus stream plugs a decode chain");
    assert!(
        names.contains(&"opusdec"),
        "opusdec auto-plugs for Opus -> PCM: {names:?}"
    );
}

/// libopus is created per channel count, but a demuxer only knows the real
/// count once it parses `OpusHead`, so `OpusDec` accepts the channels-0
/// placeholder at configure (deferring the decoder to the `CapsChanged` that
/// carries the real count) instead of rejecting it. A concrete count configures
/// the decoder up front (the direct `OpusParse` path).
#[cfg(feature = "opus")]
#[test]
fn opusdec_defers_placeholder_channels_at_configure() {
    use g2g_core::element::AsyncElement;
    use g2g_core::{AudioFormat, Caps};
    use g2g_plugins::opusdec::OpusDec;

    let opus = |ch| Caps::Audio {
        format: AudioFormat::Opus,
        channels: ch,
        sample_rate: 0,
    };
    // The channels-0 placeholder is accepted (decoder deferred), so the
    // OggDemux / decodebin path negotiates before the count is known.
    assert!(OpusDec::new().configure_pipeline(&opus(0)).is_ok());
    // A concrete mono / stereo count configures the decoder immediately.
    assert!(OpusDec::new().configure_pipeline(&opus(1)).is_ok());
    assert!(OpusDec::new().configure_pipeline(&opus(2)).is_ok());
}
