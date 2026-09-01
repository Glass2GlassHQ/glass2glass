//! M1043: the last sources on the old frame-limit builder move to gst
//! `basesrc`'s `num-buffers`, so -1 runs forever, n emits exactly n, and 0 emits
//! nothing and goes straight to EOS. The pipewire pair already had the property
//! and contradicted it in the builder (`with_frame_limit(0)` meant unlimited);
//! the rest had the builder and no property at all.
//!
//! The run tests need no daemon, no SFU and no device: a zero limit ends the
//! source before it opens anything, which is the point. They are wrapped in a
//! timeout, since the old reading of 0 blocks forever.
//!
//! ```sh
//! cargo test -p g2g-plugins --features pipewire --test m1043_num_buffers_sweep
//! cargo test -p g2g-plugins --features webrtc,webrtc-livekit \
//!     --test m1043_num_buffers_sweep
//! cargo test -p g2g-plugins --features local-ipc,local-dmabuf,libcamera \
//!     --test m1043_num_buffers_sweep
//! ```

#![cfg(any(
    feature = "webrtc",
    feature = "webrtc-livekit",
    all(
        target_os = "linux",
        any(
            feature = "pipewire",
            feature = "libcamera",
            feature = "local-ipc",
            feature = "local-dmabuf"
        )
    )
))]

mod numbuffers_common;

#[allow(unused_imports)]
use numbuffers_common::{assert_builder_matches_num_buffers, assert_num_buffers_round_trips};

/// How long a zero-limit `run` may take before we call it hung. Generous: the
/// correct path does no IO at all.
#[allow(dead_code)]
const ZERO_LIMIT_DEADLINE: core::time::Duration = core::time::Duration::from_secs(5);

/// Caps for the elements whose `configure_pipeline` only marks readiness: they
/// ignore the argument, so any well-formed video caps stand in for the solved
/// ones.
#[allow(dead_code)]
fn raw_video_caps() -> g2g_core::Caps {
    g2g_core::Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: g2g_core::Dim::Fixed(640),
        height: g2g_core::Dim::Fixed(480),
        framerate: g2g_core::Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Progressive,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// ---- pipewire: the property was already right, the builder was not ----

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[test]
fn pipewiresrc_builder_agrees_with_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::pipewiresrc::PipeWireSrc;
    assert_num_buffers_round_trips!(PipeWireSrc::new());
    assert_builder_matches_num_buffers!(|n| PipeWireSrc::new().with_frame_limit(n));
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[test]
fn pipewirevideosrc_builder_agrees_with_num_buffers() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;
    assert_num_buffers_round_trips!(PipeWireVideoSrc::new());
    assert_builder_matches_num_buffers!(|n| PipeWireVideoSrc::new().with_frame_limit(n));
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[tokio::test]
async fn pipewiresrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos, Collect};
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{AudioFormat, Caps};
    use g2g_plugins::pipewiresrc::PipeWireSrc;

    let mut src = PipeWireSrc::new().with_frame_limit(0);
    src.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: 48_000,
    })
    .expect("pipewiresrc caps");

    let mut out = Collect::default();
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not open a capture stream")
        .expect("run");
    assert_only_eos(&out, emitted);
}

#[cfg(all(target_os = "linux", feature = "pipewire"))]
#[tokio::test]
async fn pipewirevideosrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos, Collect};
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;

    let mut src = PipeWireVideoSrc::new().with_frame_limit(0);
    src.configure_pipeline(&raw_video_caps())
        .expect("pipewirevideosrc caps");

    let mut out = Collect::default();
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not open a capture stream")
        .expect("run");
    assert_only_eos(&out, emitted);
}

// ---- webrtc ----

#[cfg(feature = "webrtc")]
#[test]
fn whepsrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::webrtcwhepsrc::WebRtcWhepSrc;
    assert_num_buffers_round_trips!(WebRtcWhepSrc::new("https://sfu.invalid/whep"));
    assert_builder_matches_num_buffers!(
        |n| WebRtcWhepSrc::new("https://sfu.invalid/whep").with_frame_limit(n)
    );
}

#[cfg(feature = "webrtc")]
#[tokio::test]
async fn whepsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos, Collect};
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::webrtcwhepsrc::WebRtcWhepSrc;

    let mut src = WebRtcWhepSrc::new("https://sfu.invalid/whep").with_frame_limit(0);
    src.configure_pipeline(&raw_video_caps())
        .expect("whepsrc caps");

    let mut out = Collect::default();
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not POST to the SFU")
        .expect("run");
    assert_only_eos(&out, emitted);
}

// ---- livekit: room subscriber + duplex participant ----

#[cfg(feature = "webrtc-livekit")]
#[test]
fn livekitsrc_num_buffers_round_trips() {
    use g2g_core::MultiOutputSource;
    use g2g_plugins::livekitsrc::LiveKitSrc;
    assert_num_buffers_round_trips!(LiveKitSrc::new("ws://127.0.0.1:7880", "room", "g2g"));
    assert_builder_matches_num_buffers!(
        |n| LiveKitSrc::new("ws://127.0.0.1:7880", "room", "g2g").with_frame_limit(n)
    );
}

#[cfg(feature = "webrtc-livekit")]
#[tokio::test]
async fn livekitsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos_on_every_pad, CollectPorts};
    use g2g_core::MultiOutputSource;
    use g2g_plugins::livekitsrc::LiveKitSrc;

    let mut src = LiveKitSrc::new("ws://127.0.0.1:7880", "room", "g2g").with_frame_limit(0);
    let mut out = CollectPorts::new(src.output_count());
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not join the room")
        .expect("run");
    assert_only_eos_on_every_pad(&out, emitted);
}

/// The duplex participant counts RECEIVED access units, so a zero limit is over
/// before it joins: both recv pads get their EOS and nothing is published.
#[cfg(feature = "webrtc-livekit")]
#[tokio::test]
async fn livekitduplex_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos_on_every_pad, CollectPorts, NoInbound};
    use g2g_core::MultiDuplexSession;
    use g2g_plugins::livekitduplex::LiveKitDuplex;

    let mut session = LiveKitDuplex::new("ws://127.0.0.1:7880", "room", "g2g").with_frame_limit(0);
    let mut inbound = NoInbound;
    let mut out = CollectPorts::new(session.output_count());
    let received = tokio::time::timeout(ZERO_LIMIT_DEADLINE, session.run(&mut inbound, &mut out))
        .await
        .expect("a zero limit must not join the room")
        .expect("run");
    assert_only_eos_on_every_pad(&out, received);
}

// ---- local zero-copy IPC ----

#[cfg(all(target_os = "linux", feature = "local-ipc"))]
#[test]
fn localcudasrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::localcuda::LocalCudaSrc;
    assert_num_buffers_round_trips!(LocalCudaSrc::new("/tmp/g2g-cuda-m1043.sock"));
    assert_builder_matches_num_buffers!(
        |n| LocalCudaSrc::new("/tmp/g2g-cuda-m1043.sock").with_frame_limit(n)
    );
}

#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
#[test]
fn dmabufsrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::localdmabuf::DmaBufSrc;
    assert_num_buffers_round_trips!(DmaBufSrc::new("/tmp/g2g-dmabuf-m1043.sock"));
    assert_builder_matches_num_buffers!(
        |n| DmaBufSrc::new("/tmp/g2g-dmabuf-m1043.sock").with_frame_limit(n)
    );
}

/// No sender ever connects here: a zero limit must not wait for one. (The CUDA
/// sibling needs a device to configure, so it stops at the property halves.)
#[cfg(all(target_os = "linux", feature = "local-dmabuf"))]
#[tokio::test]
async fn dmabufsrc_num_buffers_zero_emits_only_eos() {
    use crate::numbuffers_common::{assert_only_eos, Collect};
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::localdmabuf::DmaBufSrc;

    let mut src = DmaBufSrc::new("/tmp/g2g-dmabuf-m1043-zero.sock").with_frame_limit(0);
    src.configure_pipeline(&raw_video_caps())
        .expect("dmabufsrc caps");

    let mut out = Collect::default();
    let emitted = tokio::time::timeout(ZERO_LIMIT_DEADLINE, src.run(&mut out))
        .await
        .expect("a zero limit must not wait for a sender")
        .expect("run");
    assert_only_eos(&out, emitted);
}

// ---- libcamera ----

#[cfg(all(target_os = "linux", feature = "libcamera"))]
#[test]
fn libcamerasrc_num_buffers_round_trips() {
    use g2g_core::runtime::SourceLoop;
    use g2g_plugins::libcamerasrc::LibCameraSrc;
    assert_num_buffers_round_trips!(LibCameraSrc::new());
    assert_builder_matches_num_buffers!(|n| LibCameraSrc::new().with_frame_limit(n));
}
