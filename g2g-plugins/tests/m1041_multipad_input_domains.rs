//! M1041: the remaining multi-pad elements that touch frame bytes declare the
//! domains they can read, finishing the sweep `m1039_multipad_input_domains`
//! started. A fan-in left on the `ALL` default lets the allocation cascade
//! leave a GPU producer on the device, and the element then fails the frame at
//! run time instead of the graph downloading it.

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::MultiInputElement;

const SYSTEM_ONLY: DomainSet = DomainSet::only(MemoryDomainKind::System);

#[test]
fn caption_and_subtitle_overlays_take_system_frames() {
    assert_eq!(
        g2g_plugins::ccinsert::CcInsert::new().input_domains(),
        SYSTEM_ONLY
    );
    assert_eq!(
        g2g_plugins::subpictureoverlay::SubPictureOverlay::new().input_domains(),
        SYSTEM_ONLY
    );
    assert_eq!(
        g2g_plugins::textoverlay::TextOverlayN::new().input_domains(),
        SYSTEM_ONLY
    );
}

/// The interleave is a pure router: it orders whole frames by PTS and never
/// looks inside one, so it stays domain-transparent and a GPU producer feeding
/// it is not forced to download.
#[test]
fn interleave_mux_stays_domain_transparent() {
    let mux = g2g_plugins::mux::InterleaveMux::new(
        2,
        g2g_core::Caps::Audio {
            format: g2g_core::AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
    );
    assert_eq!(mux.input_domains(), DomainSet::ALL);
}

#[cfg(feature = "webrtc")]
#[test]
fn webrtc_session_sink_takes_system_frames() {
    let sink = g2g_plugins::webrtcsession::WebRtcSessionSink::new("http://127.0.0.1:8080/whip");
    assert_eq!(sink.input_domains(), SYSTEM_ONLY);
}

#[cfg(feature = "webrtc-livekit")]
#[test]
fn livekit_sink_takes_system_frames() {
    let sink = g2g_plugins::livekitsink::LiveKitSink::new("ws://127.0.0.1:7880", "room", "g2g");
    assert_eq!(sink.input_domains(), SYSTEM_ONLY);
}
