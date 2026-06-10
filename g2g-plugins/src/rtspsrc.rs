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
//! Remaining limitations:
//! - `intercept_caps()` still returns `Dim::Any` / `Rate::Any` because the
//!   network handshake happens inside `run`. A pre-play caps probe would
//!   require extending `SourceLoop`.
//! - Each frame still copies retina's `Vec<u8>` into a `Box<[u8]>`. A
//!   `Bytes`-aware `SystemSlice` variant (zero-copy) is deferred.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use std::string::{String, ToString};
use std::sync::Arc;

use alloc::boxed::Box;

use futures_util::StreamExt;
use retina::client::{PlayOptions, Session, SessionGroup, SessionOptions, SetupOptions};
use retina::codec::{CodecItem, FrameFormat, ParametersRef, VideoParameters};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink,
    PipelinePacket, Rate, VideoFormat,
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

#[derive(Debug)]
pub struct RtspSrc {
    url: String,
    user_agent: String,
    target_frames: Option<u64>,
    reconnect: ReconnectPolicy,
    configured: bool,
}

impl RtspSrc {
    pub fn new<S: Into<String>>(url: S) -> Self {
        Self {
            url: url.into(),
            user_agent: "glass2glass/0.1".to_string(),
            target_frames: None,
            reconnect: ReconnectPolicy::DISABLED,
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

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        // The real dims come from the SDP, which only lands after DESCRIBE
        // (inside `run`). To survive Phase 2's `Caps::fixate()` we advertise
        // a wide Range rather than `Any` (`Any` is unfixable and aborts
        // negotiation). The fixated placeholder is overwritten by the first
        // `CapsChanged` we push once the session connects; downstream
        // elements that care about geometry (e.g. allocators) only act on
        // the cascaded refinement, not on this placeholder.
        Ok(Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Range { min: 16, max: 8192 },
            height: Dim::Range { min: 16, max: 8192 },
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: 240 << 16,
            },
        })
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let emitted = run_rtsp(self, out).await?;
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
    src: &RtspSrc,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let limit = src.target_frames.unwrap_or(u64::MAX);
    let mut total_emitted: u64 = 0;
    let mut pts_base_ns: u64 = 0;
    let mut attempt: u32 = 0;
    let mut backoff_ms = src.reconnect.initial_backoff_ms.max(1);
    let mut last_session_max_pts: u64 = 0;

    loop {
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
                // Push PTS forward past the gap before the next session.
                pts_base_ns = last_session_max_pts.saturating_add(1_000_000_000);

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
    src: &RtspSrc,
    out: &mut dyn OutputSink,
    total_emitted: &mut u64,
    pts_base_ns: u64,
    limit: u64,
    last_session_max_pts: &mut u64,
) -> SessionOutcome {
    let url = match url::Url::parse(&src.url) {
        Ok(u) => u,
        // A bad URL is fatal across reconnects too — emit it as a
        // network error and let the policy decide (the test for
        // `127.0.0.1:1/no-such-server` flows through here).
        Err(_) => return SessionOutcome::NetworkError(G2gError::CapsMismatch),
    };

    let session_group = Arc::new(SessionGroup::default());
    let opts = SessionOptions::default()
        .session_group(session_group)
        .user_agent(src.user_agent.clone());

    let mut session = match Session::describe(url, opts).await {
        Ok(s) => s,
        Err(_) => return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other)),
    };

    let video_idx = match session
        .streams()
        .iter()
        .position(|s| s.media() == "video" && matches!(s.encoding_name(), "h264" | "h265"))
    {
        Some(i) => i,
        None => return SessionOutcome::NetworkError(G2gError::CapsMismatch),
    };

    let clock_rate = u64::from(session.streams()[video_idx].clock_rate_hz());

    if session
        .setup(
            video_idx,
            SetupOptions::default().frame_format(FrameFormat::SIMPLE),
        )
        .await
        .is_err()
    {
        return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other));
    }

    // After SETUP, retina has parsed the SPS/PPS from the SDP (when present)
    // and exposes it as a `VideoParameters`. We use it to emit a fixed-cap
    // `CapsChanged` before the first frame so downstream elements can size
    // hardware allocations without waiting for the first bitstream parse.
    let mut current_caps =
        caps_from_video_params(video_params_for(session.streams(), video_idx));

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

        let frame_caps = current_caps.clone().unwrap_or_else(any_h264_caps);
        let bytes = vf.into_data().into_boxed_slice();

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            caps: frame_caps,
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: 0,
                capture_ns: pts_ns,
            },
            sequence: *total_emitted,
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

fn video_params_for(streams: &[retina::client::Stream], idx: usize) -> Option<&VideoParameters> {
    match streams[idx].parameters() {
        Some(ParametersRef::Video(v)) => Some(v),
        _ => None,
    }
}

fn caps_from_video_params(params: Option<&VideoParameters>) -> Option<Caps> {
    let p = params?;
    let (w, h) = p.pixel_dimensions();
    Some(Caps::Video {
        format: VideoFormat::H264,
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

fn any_h264_caps() -> Caps {
    Caps::Video {
        format: VideoFormat::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}
