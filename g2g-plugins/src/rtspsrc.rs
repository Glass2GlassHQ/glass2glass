//! RTSP source element wrapping the `retina` crate.
//!
//! M5: H.264 access units emitted as `Frame` with `MemoryDomain::System`
//! (encoded bitstream, not decoded pixels). RTP timestamps are converted
//! to ns using the stream's RTP clock rate; the first frame's RTP PTS
//! becomes the pipeline-time origin so emitted timestamps start at 0.
//!
//! M7: retina is now configured with `FrameFormat::SIMPLE`, so each access
//! unit arrives Annex-B–framed with SPS/PPS prepended to every key frame.
//! That makes the bitstream directly consumable by [`crate::h264parse`].
//! Dimensions and (when available) framerate are read from retina's
//! depacketized `VideoParameters` and emitted as a `CapsChanged` packet
//! before the first `DataFrame`; mid-stream parameter changes (signaled by
//! `VideoFrame::has_new_parameters`) trigger a fresh `CapsChanged`.
//!
//! M18 item 5: `intercept_caps` is async (`SourceLoop::CapsFuture`), so
//! the source can probe the server for real SDP dims/fps during
//! negotiation. The default path connects, DESCRIBEs, SETUPs, extracts
//! the H.264 [`VideoParameters`], drops the session, and caches the
//! discovered caps for `run` to skip re-probing. [`RtspSrc::with_expected_dims`]
//! short-circuits the probe entirely, returning the caller-supplied
//! geometry without I/O (useful for tests and offline negotiation).
//!
//! Remaining limitations:
//! - The probe and `run`'s session open are distinct connections; the
//!   server pays for two DESCRIBEs at startup. A future optimization is
//!   to stash the post-SETUP session and consume it in `run`.
//! - Each frame still copies retina's `Vec<u8>` into a `Box<[u8]>`. A
//!   `Bytes`-aware `SystemSlice` variant (zero-copy) is deferred.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use std::string::{String, ToString};
use std::sync::Arc;

use alloc::boxed::Box;

use futures_util::StreamExt;
use retina::client::{Described, PlayOptions, Session, SessionGroup, SessionOptions, SetupOptions};
use retina::codec::{CodecItem, FrameFormat, ParametersRef, VideoParameters};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, Rate, VideoCodec,
};

/// Reconnect policy for [`RtspSrc`]. Off by default: a session failure
/// surfaces as `G2gError::Hardware` and the source ends. When enabled,
/// the source retries up to `max_attempts` times with exponential backoff
/// between `initial_backoff_ms` and `max_backoff_ms`. A `PipelinePacket::Flush`
/// is emitted to downstream between sessions to signal the discontinuity
/// (the decoder flushes its state, the sink resets `last_sequence`).
#[derive(Debug, Clone, Copy)]
struct ReconnectPolicy {
    max_attempts: u32,
    initial_backoff_ms: u64,
    max_backoff_ms: u64,
}

impl ReconnectPolicy {
    const DISABLED: Self = Self {
        max_attempts: 0,
        initial_backoff_ms: 0,
        max_backoff_ms: 0,
    };
}

/// Post-SETUP retina session stashed by `intercept_caps`, consumed by
/// `run`'s first session attempt so the server isn't asked for two
/// DESCRIBE / SETUP round-trips at startup. Reconnect attempts after
/// the first network failure still rebuild from scratch — by definition
/// the stashed session is gone once the connection drops.
struct StashedSession {
    session: Session<Described>,
    video_idx: usize,
    caps: Caps,
}

impl core::fmt::Debug for StashedSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StashedSession")
            .field("video_idx", &self.video_idx)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct RtspSrc {
    url: String,
    user_agent: String,
    target_frames: Option<u64>,
    reconnect: ReconnectPolicy,
    /// Caller-supplied expected geometry. When set, `intercept_caps`
    /// returns these as fixed dims without performing a network probe
    /// (useful for tests and known cameras). The SDP's actual dims still
    /// arrive as a mid-stream `CapsChanged`; if they differ, downstream
    /// rebuilds.
    expected_dims: Option<(u32, u32)>,
    /// Caps discovered by the async `intercept_caps` probe. Cached so
    /// the runner can call `caps_constraint` more than once during
    /// re-fixate retries without a second DESCRIBE.
    discovered_caps: Option<Caps>,
    /// Post-SETUP retina session captured by the probe. `run` takes it
    /// on the first session attempt and skips straight to PLAY, saving
    /// a redundant DESCRIBE + SETUP round-trip at startup.
    stashed_session: Option<StashedSession>,
    configured: bool,
}

impl RtspSrc {
    pub fn new<S: Into<String>>(url: S) -> Self {
        Self {
            url: url.into(),
            user_agent: "glass2glass/0.1".to_string(),
            target_frames: None,
            reconnect: ReconnectPolicy::DISABLED,
            expected_dims: None,
            discovered_caps: None,
            stashed_session: None,
            configured: false,
        }
    }

    /// Stop emitting after this many `DataFrame` packets (excludes `Eos`).
    /// Without a limit, the source runs until the server disconnects.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.target_frames = Some(n);
        self
    }

    pub fn with_user_agent<S: Into<String>>(mut self, ua: S) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Declare the stream's expected pixel geometry. Opt-in fix for
    /// workaround #1: the RTSP handshake (and thus the real SDP dims)
    /// only completes inside `run`, so by default `intercept_caps`
    /// advertises a wide placeholder Range and the real geometry lands
    /// later via `CapsChanged`. When you already know the camera
    /// resolution, set it here so the chain negotiates fixed dims at
    /// startup (a downstream sink sizes its surface once, instead of
    /// building at the placeholder min and rebuilding on the first
    /// `CapsChanged`). If the actual SDP dims differ, the mid-stream
    /// `CapsChanged` still corrects them.
    pub fn with_expected_dims(mut self, width: u32, height: u32) -> Self {
        self.expected_dims = Some((width, height));
        self
    }

    /// Enable automatic reconnect on session failure. `max_attempts == 0`
    /// (the default via [`Self::new`]) disables reconnect entirely;
    /// pass any positive count to opt in. Backoff starts at 250 ms and
    /// doubles per attempt, capped at 5 s. Use [`Self::with_reconnect_backoff`]
    /// to override.
    ///
    /// Reconnect triggers on network/protocol errors only — a server-side
    /// graceful end-of-stream (eg a VOD finishing) is treated as final
    /// and the source emits EOS without retrying.
    pub fn with_reconnect(mut self, max_attempts: u32) -> Self {
        self.reconnect.max_attempts = max_attempts;
        if self.reconnect.initial_backoff_ms == 0 {
            self.reconnect.initial_backoff_ms = 250;
        }
        if self.reconnect.max_backoff_ms == 0 {
            self.reconnect.max_backoff_ms = 5_000;
        }
        self
    }

    /// Override the exponential-backoff bounds used by [`Self::with_reconnect`].
    /// `initial_backoff_ms` is the wait before the first retry; each
    /// subsequent retry doubles up to `max_backoff_ms`.
    pub fn with_reconnect_backoff(mut self, initial_ms: u64, max_ms: u64) -> Self {
        self.reconnect.initial_backoff_ms = initial_ms;
        self.reconnect.max_backoff_ms = max_ms;
        self
    }
}

impl SourceLoop for RtspSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            // Fast path: caller-supplied geometry skips the probe.
            // Framerate stays as a fixable Range (Any would be rejected
            // by `Caps::fixate`); the real value lands as a mid-stream
            // `CapsChanged` if the SDP carries one.
            if let Some((w, h)) = self.expected_dims {
                return Ok(Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    framerate: Rate::Range {
                        min_q16: 1 << 16,
                        max_q16: 240 << 16,
                    },
                });
            }
            // Memoized probe: a re-fixate retry skips the second DESCRIBE.
            if let Some(c) = &self.discovered_caps {
                return Ok(c.clone());
            }
            let stashed =
                probe_session_with_reconnect(&self.url, &self.user_agent, &self.reconnect).await?;
            let caps = stashed.caps.clone();
            self.discovered_caps = Some(caps.clone());
            self.stashed_session = Some(stashed);
            Ok(caps)
        })
    }

    /// Produces the caps discovered by the negotiation-time probe (or the
    /// `with_expected_dims` hint), so the chain takes the native arc-consistency
    /// path. Mid-stream SDP/SPS refinements still arrive as `CapsChanged` from
    /// `run`.
    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        let caps = self.intercept_caps().await?;
        Ok(CapsConstraint::Produces(CapsSet::one(caps)))
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP source",
            "Source/Network",
            "Receives an RTSP / RTP H.264 stream via retina",
            "g2g",
        )
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let emitted = run_rtsp(&mut *self, out).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// Outer reconnect orchestrator. Calls [`run_session`] for each connect
/// attempt; on network/protocol failure, waits according to the
/// reconnect policy and tries again until either the frame limit is hit,
/// the server gracefully closes, or `max_attempts` is exhausted.
///
/// State threaded across sessions:
/// - `total_emitted`: cumulative `DataFrame` count, also the next frame's
///   `sequence`. Continues monotonically across reconnects.
/// - `pts_base_ns`: PTS offset applied to the per-session relative PTS so
///   downstream sees monotonic timestamps even across discontinuities.
///   Bumped by `1 s` (a deliberate gap to mark the reconnect boundary)
///   after each successful session before the next one runs.
async fn run_rtsp(
    src: &mut RtspSrc,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let limit = src.target_frames.unwrap_or(u64::MAX);
    let mut total_emitted: u64 = 0;
    let mut pts_base_ns: u64 = 0;
    let mut attempt: u32 = 0;
    let mut backoff_ms = src.reconnect.initial_backoff_ms.max(1);
    let mut last_session_max_pts: u64 = 0;

    loop {
        let emitted_before = total_emitted;
        let outcome =
            run_session(src, out, &mut total_emitted, pts_base_ns, limit, &mut last_session_max_pts)
                .await;
        match outcome {
            SessionOutcome::LimitReached | SessionOutcome::GracefulEnd => {
                return Ok(total_emitted);
            }
            SessionOutcome::DownstreamError(e) => {
                // A downstream `out.push` failure (eg sink panicked) is
                // never something a reconnect can fix.
                return Err(e);
            }
            SessionOutcome::NetworkError(e) => {
                if src.reconnect.max_attempts == 0 {
                    return Err(e);
                }
                attempt += 1;
                if attempt > src.reconnect.max_attempts {
                    return Err(e);
                }
                std::eprintln!(
                    "rtsp: session ended ({:?}); reconnect {}/{} after {}ms",
                    e,
                    attempt,
                    src.reconnect.max_attempts,
                    backoff_ms,
                );
                // Tell downstream we have a discontinuity coming. The
                // ffmpeg decoder flushes its codec state on Flush so
                // the next session's IDR primes cleanly; sinks reset
                // their `last_sequence` so the new (continuing) sequence
                // counter isn't rejected as out-of-order.
                let _ = out.push(PipelinePacket::Flush).await;
                // Push PTS forward past the gap before the next session, but
                // only if this session emitted frames; otherwise the base is
                // unchanged so a flapping empty reconnect doesn't inflate PTS
                // by a second each attempt.
                if total_emitted > emitted_before {
                    pts_base_ns = last_session_max_pts.saturating_add(1_000_000_000);
                }

                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = backoff_ms
                    .saturating_mul(2)
                    .min(src.reconnect.max_backoff_ms.max(backoff_ms));
            }
        }
    }
}

/// Result of a single connect+play+drain session.
#[derive(Debug)]
enum SessionOutcome {
    /// `target_frames` reached: we emitted enough, stop cleanly.
    LimitReached,
    /// retina's demuxer returned `None` — the server closed the stream
    /// without error (typical for VOD reaching end-of-file).
    GracefulEnd,
    /// retina or the network errored. Eligible for reconnect.
    #[allow(dead_code)]
    NetworkError(G2gError),
    /// A downstream `out.push` failed. Not retryable.
    DownstreamError(G2gError),
}

/// Run one RTSP session: DESCRIBE, SETUP, PLAY, drain until error /
/// graceful end / frame limit. Updates `total_emitted` in place and
/// records the highest PTS we emitted for the reconnect-gap computation.
async fn run_session(
    src: &mut RtspSrc,
    out: &mut dyn OutputSink,
    total_emitted: &mut u64,
    pts_base_ns: u64,
    limit: u64,
    last_session_max_pts: &mut u64,
) -> SessionOutcome {
    // Fast path: the probe in `intercept_caps` already left a
    // post-SETUP session and the parsed caps. Take them and skip
    // straight to PLAY. Reconnect attempts after a network failure
    // can't reuse this; `stashed_session` is `take`n once and stays
    // `None` for any later session.
    let (session, video_idx, current_caps) = match src.stashed_session.take() {
        Some(stashed) => (stashed.session, stashed.video_idx, Some(stashed.caps)),
        None => match connect_describe_setup(&src.url, &src.user_agent).await {
            Ok((s, idx)) => {
                let caps = caps_from_video_params(video_params_for(s.streams(), idx));
                (s, idx, caps)
            }
            Err(e) => return SessionOutcome::NetworkError(e),
        },
    };
    let clock_rate = u64::from(session.streams()[video_idx].clock_rate_hz());
    let mut current_caps = current_caps;

    let played = match session.play(PlayOptions::default()).await {
        Ok(p) => p,
        Err(_) => return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other)),
    };

    let mut demuxed = match played.demuxed() {
        Ok(d) => d,
        Err(_) => return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other)),
    };

    if let Some(caps) = &current_caps {
        if let Err(e) = out.push(PipelinePacket::CapsChanged(caps.clone())).await {
            return SessionOutcome::DownstreamError(e);
        }
    }

    // retina (FrameFormat::SIMPLE) only prepends SPS/PPS on keyframes
    // (`is_random_access_point`). When we tune into a live stream mid-GOP
    // the first arriving access units are typically P-frames; emitting them
    // would feed slices to the decoder before any parameter set, producing
    // "non-existing PPS N referenced" and stalling until the next IDR. Drop
    // everything up to and including the wait for the first keyframe so the
    // decoder's first input is always SPS + PPS + IDR.
    let mut seen_keyframe = false;
    let mut origin_rtp: Option<i64> = None;
    let mut session_max_pts: u64 = pts_base_ns;

    while *total_emitted < limit {
        let item = match demuxed.next().await {
            Some(Ok(item)) => item,
            Some(Err(_)) => {
                *last_session_max_pts = session_max_pts;
                return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other));
            }
            None => {
                *last_session_max_pts = session_max_pts;
                return SessionOutcome::GracefulEnd;
            }
        };

        let CodecItem::VideoFrame(vf) = item else {
            continue;
        };

        if !seen_keyframe {
            if !vf.is_random_access_point() {
                continue;
            }
            seen_keyframe = true;
        }

        if vf.has_new_parameters() {
            let refreshed =
                caps_from_video_params(video_params_for(demuxed.streams(), video_idx));
            if refreshed != current_caps {
                if let Some(caps) = &refreshed {
                    if let Err(e) = out.push(PipelinePacket::CapsChanged(caps.clone())).await {
                        return SessionOutcome::DownstreamError(e);
                    }
                }
                current_caps = refreshed;
            }
        }

        let rtp_pts = vf.timestamp().timestamp();
        let origin = *origin_rtp.get_or_insert(rtp_pts);
        let rel_rtp = rtp_pts.saturating_sub(origin).max(0) as u64;
        let rel_pts_ns = rel_rtp.saturating_mul(1_000_000_000) / clock_rate.max(1);
        let pts_ns = pts_base_ns.saturating_add(rel_pts_ns);
        session_max_pts = session_max_pts.max(pts_ns);

        let bytes = vf.into_data().into_boxed_slice();

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: 0,
                capture_ns: pts_ns,
                arrival_ns: g2g_core::metrics::monotonic_ns(),
            },
            sequence: *total_emitted,
            meta: Default::default(),
        };

        if let Err(e) = out.push(PipelinePacket::DataFrame(frame)).await {
            *last_session_max_pts = session_max_pts;
            return SessionOutcome::DownstreamError(e);
        }
        *total_emitted += 1;
    }

    *last_session_max_pts = session_max_pts;
    SessionOutcome::LimitReached
}

/// Reconnect-aware wrapper around `probe_caps`: applies the source's
/// reconnect policy so a transient connect failure during negotiation
/// retries with the same backoff `run` uses for mid-session drops.
/// With `ReconnectPolicy::DISABLED` (the default) this is a single
/// probe attempt.
///
/// On success the returned [`StashedSession`] holds the post-SETUP
/// retina session so the caller can stash it and let `run` skip the
/// duplicate DESCRIBE + SETUP at startup.
async fn probe_session_with_reconnect(
    url: &str,
    user_agent: &str,
    policy: &ReconnectPolicy,
) -> Result<StashedSession, G2gError> {
    let mut attempt: u32 = 0;
    let mut backoff_ms = policy.initial_backoff_ms.max(1);
    let max_attempts = policy.max_attempts;
    loop {
        match probe_session(url, user_agent).await {
            Ok(stashed) => return Ok(stashed),
            // `CapsMismatch` is a structural problem (bad URL, no H.264
            // stream): retrying won't help, surface immediately.
            Err(G2gError::CapsMismatch) => return Err(G2gError::CapsMismatch),
            Err(e) => {
                if attempt >= max_attempts {
                    return Err(e);
                }
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                attempt += 1;
                backoff_ms = backoff_ms.saturating_mul(2).min(policy.max_backoff_ms.max(backoff_ms));
            }
        }
    }
}

/// Async probe used by `intercept_caps` to discover real SDP geometry
/// before negotiation. Performs DESCRIBE + SETUP, extracts H.264
/// `VideoParameters`, and returns the post-SETUP session alongside the
/// parsed caps so the source can stash both and let `run` skip the
/// duplicate connect.
///
/// Failure modes are flattened to `G2gError::Hardware` / `CapsMismatch`:
/// the source has no caps to advertise, so the runner fails negotiation
/// cleanly. A caller that wants to advertise fixed caps without I/O
/// should use [`RtspSrc::with_expected_dims`] instead.
async fn probe_session(url: &str, user_agent: &str) -> Result<StashedSession, G2gError> {
    let (session, video_idx) = connect_describe_setup(url, user_agent).await?;
    let caps = caps_from_video_params(video_params_for(session.streams(), video_idx))
        .ok_or(G2gError::CapsMismatch)?;
    Ok(StashedSession { session, video_idx, caps })
}

/// Shared DESCRIBE + SETUP step. Used both by the probe path (caching
/// the result in `RtspSrc::stashed_session`) and the reconnect path in
/// `run_session` (after the stash is exhausted). Returns the post-SETUP
/// session and the negotiated video stream index.
async fn connect_describe_setup(
    url: &str,
    user_agent: &str,
) -> Result<(Session<Described>, usize), G2gError> {
    let url = url::Url::parse(url).map_err(|_| G2gError::CapsMismatch)?;
    let session_group = Arc::new(SessionGroup::default());
    let opts = SessionOptions::default()
        .session_group(session_group)
        .user_agent(user_agent.to_string());
    let mut session = Session::describe(url, opts)
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    let video_idx = session
        .streams()
        .iter()
        .position(|s| s.media() == "video" && matches!(s.encoding_name(), "h264" | "h265"))
        .ok_or(G2gError::CapsMismatch)?;
    session
        .setup(
            video_idx,
            SetupOptions::default().frame_format(FrameFormat::SIMPLE),
        )
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    Ok((session, video_idx))
}

fn video_params_for(streams: &[retina::client::Stream], idx: usize) -> Option<&VideoParameters> {
    match streams[idx].parameters() {
        Some(ParametersRef::Video(v)) => Some(v),
        _ => None,
    }
}

fn caps_from_video_params(params: Option<&VideoParameters>) -> Option<Caps> {
    let p = params?;
    let (w, h) = p.pixel_dimensions();
    Some(Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: rate_from_frame_rate(p.frame_rate()),
    })
}

/// Convert retina's `(numerator, denominator)` representation of frame
/// duration into our Q16-fps `Rate`. retina returns frame duration in
/// seconds (eg `(1, 15)` → 1/15 s per frame → 15 fps), so the Q16 fps
/// value is `(denominator << 16) / numerator`. We compute in `u64` to
/// keep the shift from overflowing for large denominators (eg NTSC's
/// `30000/1001` representation).
fn rate_from_frame_rate(frame_rate: Option<(u32, u32)>) -> Rate {
    match frame_rate {
        Some((num, denom)) if num > 0 => {
            let q16 = (u64::from(denom) << 16) / u64::from(num);
            if q16 <= u64::from(u32::MAX) {
                Rate::Fixed(q16 as u32)
            } else {
                Rate::Any
            }
        }
        _ => Rate::Any,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_unset_yields_rate_any() {
        assert_eq!(rate_from_frame_rate(None), Rate::Any);
    }

    #[tokio::test]
    async fn with_expected_dims_skips_probe_and_returns_fixed_geometry() {
        // Fast path: caller-supplied geometry skips the network probe, so
        // intercept_caps resolves without any I/O. (Useful for tests in
        // sandbox environments where the network is unreachable.)
        let mut src = RtspSrc::new("rtsp://example/stream").with_expected_dims(1920, 1080);
        let caps = src.intercept_caps().await.expect("caps");
        match caps {
            Caps::CompressedVideo { codec, width, height, .. } => {
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

    #[test]
    fn frame_rate_zero_numerator_yields_rate_any() {
        // Numerator of 0 would divide-by-zero; treat as unknown.
        assert_eq!(rate_from_frame_rate(Some((0, 30))), Rate::Any);
    }

    #[test]
    fn frame_rate_15fps_round_trips_to_q16() {
        // 15 fps in retina form: 1/15 s per frame.
        assert_eq!(rate_from_frame_rate(Some((1, 15))), Rate::Fixed(15 << 16));
    }

    #[test]
    fn frame_rate_ntsc_29_97_uses_full_q16_precision() {
        // 29.97 fps as the canonical (1001, 30000) duration form.
        // Expected: 30000/1001 ≈ 29.9700; in Q16 that's ~1_963_098.
        match rate_from_frame_rate(Some((1001, 30000))) {
            Rate::Fixed(q16) => {
                // Allow ±1 LSB for integer-division rounding.
                let expected = ((30000u64 << 16) / 1001) as u32;
                assert_eq!(q16, expected);
                // Sanity: q16 / 2^16 ≈ 29.97
                let int_part = q16 >> 16;
                assert_eq!(int_part, 29);
            }
            other => panic!("expected Rate::Fixed, got {other:?}"),
        }
    }
}

