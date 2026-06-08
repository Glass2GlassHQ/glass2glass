//! RTSP source element wrapping the `retina` crate.
//!
//! M5: H.264 access units emitted as `Frame` with `MemoryDomain::System`
//! (encoded bitstream, not decoded pixels). RTP timestamps are converted
//! to ns using the stream's RTP clock rate; the first frame's RTP PTS
//! becomes the pipeline-time origin so emitted timestamps start at 0.
//!
//! Limitations to be addressed in later milestones:
//! - `intercept_caps()` returns `Dim::Any` / `Rate::Any` because we don't
//!   parse the SPS until the bitstream arrives (M6).
//! - Each frame copies retina's `Bytes` into a fresh `Box<[u8]>`. A
//!   `Bytes`-aware `SystemSlice` variant (zero-copy) is deferred.
//! - DESCRIBE / SETUP / PLAY happen lazily inside `run`, so caps
//!   negotiation gets placeholder caps. A pre-play phase that exposes
//!   real caps before `run` would be cleaner but needs a trait extension.

use core::future::Future;
use core::pin::Pin;
use std::string::{String, ToString};
use std::sync::Arc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use bytes::Buf;
use futures_util::StreamExt;
use retina::client::{PlayOptions, Session, SessionGroup, SessionOptions, SetupOptions};
use retina::codec::CodecItem;

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
        .setup(video_idx, SetupOptions::default())
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let played = session
        .play(PlayOptions::default())
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let mut demuxed = played
        .demuxed()
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let caps = src.intercept_caps()?;
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

        let rtp_pts = vf.timestamp().timestamp();
        let origin = *origin_rtp.get_or_insert(rtp_pts);
        let rel_rtp = rtp_pts.saturating_sub(origin).max(0) as u64;
        let pts_ns = rel_rtp.saturating_mul(1_000_000_000) / clock_rate.max(1);

        let mut data = vf.data();
        let len = data.remaining();
        let mut bytes: Vec<u8> = alloc::vec![0u8; len];
        data.copy_to_slice(&mut bytes);

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            caps: caps.clone(),
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
