//! RTSP source element: [`RtspSrcN`] with the video pad alone.

use std::string::String;

use g2g_core::ElementMetadata;

use crate::oneport::OnePortSource;
use crate::rtspsrcn::{RtspSrcN, RTSPSRC_PROPS};

/// # Example
///
/// ```no_run
/// use g2g_plugins::rtspsrc::RtspSrc;
///
/// let element = RtspSrc::new("rtsp://192.168.1.10:554/stream1")
///     .with_credentials("admin", "secret")
///     .with_expected_dims(1920, 1080);
/// ```
pub type RtspSrc = OnePortSource<RtspSrcN>;

impl RtspSrc {
    pub fn new<S: Into<String>>(url: S) -> Self {
        OnePortSource::wrap(
            RtspSrcN::new(url),
            RTSPSRC_PROPS,
            ElementMetadata::new(
                "RTSP source",
                "Source/Network",
                "Receives an RTSP / RTP H.264 or H.265 stream via retina",
                "g2g",
            ),
        )
    }

    pub fn with_frame_limit(self, n: u64) -> Self {
        self.map(|src| src.with_frame_limit(n))
    }

    pub fn with_user_agent<S: Into<String>>(self, ua: S) -> Self {
        self.map(|src| src.with_user_agent(ua))
    }

    pub fn with_credentials<U: Into<String>, P: Into<String>>(self, user: U, pass: P) -> Self {
        self.map(|src| src.with_credentials(user, pass))
    }

    pub fn with_expected_dims(self, width: u32, height: u32) -> Self {
        self.map(|src| src.with_expected_dims(width, height))
    }

    pub fn with_reconnect(self, max_attempts: u32) -> Self {
        self.map(|src| src.with_reconnect(max_attempts))
    }

    pub fn with_reconnect_backoff(self, initial_ms: u64, max_ms: u64) -> Self {
        self.map(|src| src.with_reconnect_backoff(initial_ms, max_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{Caps, CapsConstraint, Dim, VideoCodec};

    #[tokio::test]
    async fn with_expected_dims_skips_probe_and_returns_fixed_geometry() {
        // Fast path: caller-supplied geometry skips the network probe, so
        // intercept_caps resolves without any I/O. (Useful for tests in
        // sandbox environments where the network is unreachable.)
        let mut src = RtspSrc::new("rtsp://example/stream").with_expected_dims(1920, 1080);
        let caps = src.intercept_caps().await.expect("caps");
        match caps {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                ..
            } => {
                assert_eq!(codec, VideoCodec::H264);
                assert_eq!(width, Dim::Fixed(1920));
                assert_eq!(height, Dim::Fixed(1080));
            }
            other => panic!("expected compressed video caps, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn intercept_caps_memoizes_expected_dims_across_repeat_calls() {
        // The runner queries `caps_constraint` (which awaits `intercept_caps`)
        // on every re-fixate retry. With `with_expected_dims` no probe runs,
        // so two consecutive queries must return the same fixed caps without
        // I/O.
        let mut src = RtspSrc::new("rtsp://example/stream").with_expected_dims(640, 480);
        let a = src.intercept_caps().await.expect("first");
        let b = src.intercept_caps().await.expect("second");
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn caps_constraint_is_produces_from_expected_dims() {
        let mut src = RtspSrc::new("rtsp://example/stream").with_expected_dims(1920, 1080);
        match src.caps_constraint().await.expect("constraint") {
            CapsConstraint::Produces(set) => {
                let alts = set.alternatives();
                assert_eq!(alts.len(), 1);
                assert!(matches!(
                    alts[0],
                    Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        width: Dim::Fixed(1920),
                        height: Dim::Fixed(1080),
                        ..
                    }
                ));
            }
            other => panic!("expected Produces, got {other:?}"),
        };
    }
}
