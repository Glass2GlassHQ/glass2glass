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

#[derive(Debug)]
pub struct RtspSrc {
    url: String,
    user_agent: String,
    target_frames: Option<u64>,
    configured: bool,
}

impl RtspSrc {
    pub fn new<S: Into<String>>(url: S) -> Self {
        Self {
            url: url.into(),
            user_agent: "glass2glass/0.1".to_string(),
            target_frames: None,
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
}

impl SourceLoop for RtspSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        // SPS parsing is M6; until then advertise H.264 with any geometry.
        Ok(Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
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

async fn run_rtsp(
    src: &RtspSrc,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let url = url::Url::parse(&src.url).map_err(|_| G2gError::CapsMismatch)?;

    let session_group = Arc::new(SessionGroup::default());
    let opts = SessionOptions::default()
        .session_group(session_group)
        .user_agent(src.user_agent.clone());

    let mut session = Session::describe(url, opts)
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let video_idx = session
        .streams()
        .iter()
        .position(|s| {
            s.media() == "video" && matches!(s.encoding_name(), "h264" | "h265")
        })
        .ok_or(G2gError::CapsMismatch)?;

    let clock_rate = u64::from(session.streams()[video_idx].clock_rate_hz());

    session
        .setup(
            video_idx,
            SetupOptions::default().frame_format(FrameFormat::SIMPLE),
        )
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    // After SETUP, retina has parsed the SPS/PPS from the SDP (when present)
    // and exposes it as a `VideoParameters`. We use it to emit a fixed-cap
    // `CapsChanged` before the first frame so downstream elements can size
    // hardware allocations without waiting for the first bitstream parse.
    let mut current_caps =
        caps_from_video_params(video_params_for(session.streams(), video_idx));

    let played = session
        .play(PlayOptions::default())
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let mut demuxed = played
        .demuxed()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    if let Some(caps) = &current_caps {
        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
    }

    let limit = src.target_frames.unwrap_or(u64::MAX);
    let mut emitted: u64 = 0;
    let mut origin_rtp: Option<i64> = None;

    while emitted < limit {
        let item = match demuxed.next().await {
            Some(Ok(item)) => item,
            Some(Err(_)) => return Err(G2gError::Hardware(HardwareError::Other)),
            None => break,
        };

        let CodecItem::VideoFrame(vf) = item else {
            continue;
        };

        if vf.has_new_parameters() {
            // Re-read parameters via the demuxer (which proxies to the
            // underlying playing session) and emit CapsChanged if they
            // actually represent a different geometry/framerate from what
            // we last advertised.
            let refreshed =
                caps_from_video_params(video_params_for(demuxed.streams(), video_idx));
            if refreshed != current_caps {
                if let Some(caps) = &refreshed {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                }
                current_caps = refreshed;
            }
        }

        let rtp_pts = vf.timestamp().timestamp();
        let origin = *origin_rtp.get_or_insert(rtp_pts);
        let rel_rtp = rtp_pts.saturating_sub(origin).max(0) as u64;
        let pts_ns = rel_rtp.saturating_mul(1_000_000_000) / clock_rate.max(1);

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
            sequence: emitted,
        };

        out.push(PipelinePacket::DataFrame(frame)).await?;
        emitted += 1;
    }

    Ok(emitted)
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
