//! Multi-track RTSP playback (`RtspSrcN`, `rtsp` feature): one retina session,
//! one output pad per subscribed track. The fan-out sibling of
//! [`RtspSrc`](crate::rtspsrc::RtspSrc), which plays the video track alone.
//!
//! Output 0 is the first H.264 / H.265 video stream in the SDP, emitted
//! Annex-B framed exactly as `RtspSrc` emits it. Output 1, when the SDP offers
//! a supported audio stream, is that audio: AAC arrives ADTS-framed (retina's
//! [`FrameFormat::SIMPLE`] wraps it, matching what `Mp4DemuxN` synthesizes from
//! an `AudioSpecificConfig`), G.711 as raw companded samples. Without audio the
//! element has a single output and behaves like `RtspSrc`.
//!
//! Timestamps: the two streams run on different RTP clocks (90 kHz video,
//! 48 kHz AAC), so each frame's PTS comes from retina's
//! [`Timestamp::elapsed`](retina::Timestamp::elapsed), which is normal play
//! time in that stream's own clock units against the `RTP-Info` origin the
//! server declares for both. Converting per stream and rebasing both on the
//! first frame of the session puts video and audio on one timeline that starts
//! at zero with the stream's own A/V offset preserved. A server that omits
//! `rtptime` falls back to each stream's first packet at NPT 0
//! ([`InitialTimestampPolicy::Permissive`]), which starts the tracks together
//! but cannot preserve the offset.
//!
//! The pad count is fixed before the run, so a graph builder that needs the
//! real track list calls [`blocking_probe_tracks`] (a DESCRIBE) first; that is
//! what the `playbin uri=rtsp://...` hook in [`uridecodebin`](crate::uridecodebin)
//! does. A launch line instead declares the count by linking pads
//! (`rtspsrcn name=s location=... s. ! ... s. ! ...`).

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use std::string::{String, ToString};
use std::vec::Vec;

use futures_util::StreamExt;
use retina::client::{Described, InitialTimestampPolicy, PlayOptions, Session, SetupOptions};
use retina::codec::{AudioParameters, CodecItem, FrameFormat, ParametersRef};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::g711::{G711_CLOCK_RATE_HZ, G711_DEFAULT_CHANNELS};
use crate::rtspsrc::{
    caps_from_video_params, connect_describe, video_params_for, video_stream_index, LowerTransport,
    ReconnectPolicy, ReconnectState,
};

/// User-Agent sent on DESCRIBE / SETUP unless `user-agent` overrides it. Also
/// what [`blocking_probe_tracks`] identifies as, so a server sees one client.
pub const DEFAULT_USER_AGENT: &str = "glass2glass/0.1";

/// The video track's output pad.
pub const VIDEO_PORT: usize = 0;
/// The audio track's output pad, present only when the SDP offered audio.
pub const AUDIO_PORT: usize = 1;

/// Widest geometry the placeholder video caps accept, so `fixate` has a value
/// to pick and the real dimensions can arrive later as a `CapsChanged`.
const MIN_PLACEHOLDER_DIM: u32 = 2;
const MAX_PLACEHOLDER_DIM: u32 = 8192;
const MAX_PLACEHOLDER_FPS: u32 = 240;

/// Gap inserted in the emitted timeline at a reconnect boundary, so downstream
/// sees the discontinuity as a jump rather than as overlapping timestamps.
const RECONNECT_PTS_GAP_NS: u64 = 1_000_000_000;

const NICK_AAC: &str = "aac";
const NICK_ALAW: &str = "alaw";
const NICK_MULAW: &str = "mulaw";
const AUDIO_FORMAT_NICKS: &str = "aac | alaw | mulaw";

/// The tracks a DESCRIBE found, as negotiation caps.
#[derive(Debug, Clone, PartialEq)]
pub struct RtspTracks {
    /// The first H.264 / H.265 stream, with the SDP's geometry.
    pub video: Caps,
    /// The first supported audio stream as the SDP declares it (real channels
    /// and sample rate), or `None` when the SDP offers none. The pad negotiates
    /// with the decoder-facing form of this, see [`RtspSrcN::with_tracks`].
    pub audio: Option<Caps>,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::rtspsrcn::RtspSrcN;
///
/// let src = RtspSrcN::new("rtsp://192.168.1.10:554/stream1").with_outputs(2);
/// ```
#[derive(Debug)]
pub struct RtspSrcN {
    url: String,
    user_agent: String,
    creds: Option<retina::client::Credentials>,
    transports: Vec<LowerTransport>,
    reconnect: ReconnectPolicy,
    /// Access units each output emits before its `Eos`. `u64::MAX` runs until
    /// the server disconnects.
    frame_limit: u64,
    /// Caller-supplied geometry, taking precedence over [`Self::probed_video`]
    /// the way `RtspSrc::with_expected_dims` takes precedence over its probe.
    expected_dims: Option<(u32, u32)>,
    /// Video caps a DESCRIBE resolved, so the chain negotiates the real
    /// geometry instead of the placeholder range.
    probed_video: Option<Caps>,
    /// Negotiation caps of the audio pad, or `None` for a video-only element
    /// (one output).
    audio: Option<Caps>,
}

impl RtspSrcN {
    pub fn new<S: Into<String>>(url: S) -> Self {
        Self {
            url: url.into(),
            user_agent: DEFAULT_USER_AGENT.to_string(),
            creds: None,
            transports: Vec::from([LowerTransport::Tcp]),
            reconnect: ReconnectPolicy::DISABLED,
            frame_limit: u64::MAX,
            expected_dims: None,
            probed_video: None,
            audio: None,
        }
    }

    /// Take the pad count from a launch line's linked outputs: 2 or more adds
    /// the audio pad (AAC unless `audio-format` says otherwise), 1 leaves the
    /// element video-only.
    pub fn with_outputs(mut self, outputs: usize) -> Self {
        self.audio = (outputs >= 2).then(|| audio_caps(AudioFormat::Aac, 0, 0));
        self
    }

    /// Build from a DESCRIBE's [`RtspTracks`]: the real video geometry and, when
    /// the SDP carried audio, an audio pad negotiating at
    /// [`negotiation_audio_caps`] of that stream.
    pub fn with_tracks(mut self, tracks: &RtspTracks) -> Self {
        self.probed_video = Some(tracks.video.clone());
        self.audio = tracks.audio.as_ref().map(negotiation_audio_caps);
        self
    }

    /// Stop each output after this many access units. Without a limit the
    /// element runs until the server disconnects.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    pub fn with_user_agent<S: Into<String>>(mut self, ua: S) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// RTSP DESCRIBE / SETUP credentials, as on
    /// [`RtspSrc::with_credentials`](crate::rtspsrc::RtspSrc::with_credentials).
    pub fn with_credentials<U: Into<String>, P: Into<String>>(mut self, user: U, pass: P) -> Self {
        self.creds = Some(retina::client::Credentials {
            username: user.into(),
            password: pass.into(),
        });
        self
    }

    /// Declare the video track's geometry, skipping the placeholder range so a
    /// downstream sink sizes its surface once.
    pub fn with_expected_dims(mut self, width: u32, height: u32) -> Self {
        self.expected_dims = Some((width, height));
        self
    }

    /// Retry a failed session up to `max_attempts` times, as
    /// [`RtspSrc::with_reconnect`](crate::rtspsrc::RtspSrc::with_reconnect) does.
    /// Both pads see a `Flush` between sessions and the emitted timeline jumps a
    /// second forward, so the two stay aligned across the discontinuity.
    pub fn with_reconnect(mut self, max_attempts: u32) -> Self {
        self.reconnect.max_attempts = max_attempts;
        self.reconnect.fill_backoff_defaults();
        self
    }

    /// Override the exponential-backoff bounds used by [`Self::with_reconnect`].
    pub fn with_reconnect_backoff(mut self, initial_ms: u64, max_ms: u64) -> Self {
        self.reconnect.initial_backoff_ms = initial_ms;
        self.reconnect.max_backoff_ms = max_ms;
        self
    }

    /// The video pad's negotiation caps: caller-supplied geometry first, then a
    /// DESCRIBE's, then a range wide enough for the SPS to refine.
    fn video_caps(&self) -> Caps {
        if let Some((w, h)) = self.expected_dims.filter(|(w, h)| *w > 0 && *h > 0) {
            return Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate: placeholder_framerate(),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            };
        }
        self.probed_video.clone().unwrap_or(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Range {
                min: MIN_PLACEHOLDER_DIM,
                max: MAX_PLACEHOLDER_DIM,
            },
            height: Dim::Range {
                min: MIN_PLACEHOLDER_DIM,
                max: MAX_PLACEHOLDER_DIM,
            },
            framerate: placeholder_framerate(),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
    }

    /// The audio encoding this element subscribes, or `None` when video-only.
    fn audio_format(&self) -> Option<AudioFormat> {
        match self.audio {
            Some(Caps::Audio { format, .. }) => Some(format),
            _ => None,
        }
    }
}

fn placeholder_framerate() -> Rate {
    Rate::Range {
        min_q16: 1 << 16,
        max_q16: MAX_PLACEHOLDER_FPS << 16,
    }
}

/// Compressed-audio caps in the "unknown until parsed" form the demuxers use:
/// AAC negotiates at `0/0` (its caps intersect by strict equality, so a
/// concrete rate would miss a decoder's wildcard sink) and takes its real
/// channels / rate from a runtime `CapsChanged`; G.711 carries the nominal
/// values it is statically bound to.
fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Nominal negotiation caps for an audio format named without an SDP to read.
/// RTP payload types 0 (PCMU) and 8 (PCMA) are statically bound to 8 kHz mono.
fn default_audio_caps(format: AudioFormat) -> Caps {
    match format {
        AudioFormat::Aac => audio_caps(format, 0, 0),
        _ => audio_caps(format, G711_DEFAULT_CHANNELS, G711_CLOCK_RATE_HZ),
    }
}

/// The caps an audio pad negotiates with, from what the SDP declared. AAC hands
/// the decode chain the `0/0` form its sink template carries: a compressed
/// `sample_rate` is matched for equality, so advertising the SDP's real rate
/// finds no decoder at all. G.711 keeps the SDP's rate, which its decoder
/// declares an alternative for.
pub fn negotiation_audio_caps(declared: &Caps) -> Caps {
    match declared {
        Caps::Audio {
            format: AudioFormat::Aac,
            ..
        } => default_audio_caps(AudioFormat::Aac),
        other => other.clone(),
    }
}

/// The audio format an SDP encoding name maps to, or `None` for one g2g does
/// not decode off RTP.
fn audio_format_for(encoding_name: &str) -> Option<AudioFormat> {
    match encoding_name {
        "mpeg4-generic" => Some(AudioFormat::Aac),
        "pcma" => Some(AudioFormat::Alaw),
        "pcmu" => Some(AudioFormat::Mulaw),
        _ => None,
    }
}

fn audio_format_for_nick(nick: &str) -> Option<AudioFormat> {
    match nick {
        NICK_AAC => Some(AudioFormat::Aac),
        NICK_ALAW => Some(AudioFormat::Alaw),
        NICK_MULAW => Some(AudioFormat::Mulaw),
        _ => None,
    }
}

fn audio_format_nick(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Alaw => NICK_ALAW,
        AudioFormat::Mulaw => NICK_MULAW,
        _ => NICK_AAC,
    }
}

/// The first audio stream whose encoding g2g decodes, and the format it maps to.
fn audio_stream_index(
    streams: &[retina::client::Stream],
    want: Option<AudioFormat>,
) -> Option<(usize, AudioFormat)> {
    streams.iter().enumerate().find_map(|(i, s)| {
        if s.media() != "audio" {
            return None;
        }
        let format = audio_format_for(s.encoding_name())?;
        match want {
            Some(w) if w != format => None,
            _ => Some((i, format)),
        }
    })
}

fn audio_params_for(streams: &[retina::client::Stream], idx: usize) -> Option<&AudioParameters> {
    match streams[idx].parameters() {
        Some(ParametersRef::Audio(a)) => Some(a),
        _ => None,
    }
}

/// What the SDP declares the audio track to be: its channel count and sample
/// rate, as [`RtspTracks`] reports them. [`negotiation_audio_caps`] turns this
/// into what the pad advertises.
fn declared_audio_caps(format: AudioFormat, params: Option<&AudioParameters>) -> Caps {
    match params {
        Some(p) => audio_caps(
            format,
            u8::try_from(p.channels().get()).unwrap_or(u8::MAX),
            p.clock_rate(),
        ),
        None => default_audio_caps(format),
    }
}

/// [`probe_tracks`] on a throwaway current-thread runtime, so a synchronous
/// graph builder (the `playbin uri=rtsp://...` hook) can size the fan-out before
/// the graph exists. `None` when the server is unreachable, offers no
/// H.264 / H.265 video, or the caller is already inside a runtime (blocking one
/// of its threads would deadlock it): an async caller awaits [`probe_tracks`].
pub fn blocking_probe_tracks(url: &str, user_agent: &str) -> Option<RtspTracks> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return None;
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;
    rt.block_on(probe_tracks(url, user_agent))
}

/// DESCRIBE `url` and report the tracks g2g can play from it: the first
/// H.264 / H.265 video stream and the first supported audio stream, with the
/// geometry and channel layout the SDP declares. `None` when the server is
/// unreachable or carries no video.
pub async fn probe_tracks(url: &str, user_agent: &str) -> Option<RtspTracks> {
    let session = connect_describe(url, user_agent, None).await.ok()?;
    let streams = session.streams();
    let video_idx = video_stream_index(streams)?;
    let video = caps_from_video_params(video_params_for(streams, video_idx))?;
    let audio = audio_stream_index(streams, None)
        .map(|(idx, format)| declared_audio_caps(format, audio_params_for(streams, idx)));
    Some(RtspTracks { video, audio })
}

static RTSPSRCN_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "RTSP URL (rtsp://host:port/path)",
    ),
    PropertySpec::new(
        "user-agent",
        PropKind::Str,
        "User-Agent sent on DESCRIBE/SETUP",
    )
    .with_default(DEFAULT_USER_AGENT),
    PropertySpec::new("user-id", PropKind::Str, "RTSP auth username"),
    PropertySpec::new("user-pw", PropKind::Str, "RTSP auth password"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units each pad emits then EOS (-1 = until the server disconnects)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "width",
        PropKind::Uint,
        "expected frame width; with height, negotiates fixed geometry (0 = placeholder range)",
    )
    .with_default("0"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "expected frame height; with width, negotiates fixed geometry (0 = placeholder range)",
    )
    .with_default("0"),
    PropertySpec::new(
        "audio-format",
        PropKind::Str,
        "audio encoding to subscribe on the second pad",
    )
    .with_default(NICK_AAC)
    .with_enum_values(AUDIO_FORMAT_NICKS),
    PropertySpec::new(
        "reconnect",
        PropKind::Uint,
        "max reconnect attempts after a session failure (0 = no reconnect)",
    )
    .with_default("0"),
    PropertySpec::new(
        "reconnect-backoff",
        PropKind::Uint,
        "wait before the first reconnect attempt, milliseconds (doubles per retry)",
    )
    .with_default("250"),
    PropertySpec::new(
        "reconnect-backoff-max",
        PropKind::Uint,
        "cap on the doubling reconnect backoff, milliseconds",
    )
    .with_default("5000"),
    PropertySpec::new(
        "protocols",
        PropKind::Flags,
        "lower transports to request at SETUP, tried in the order written",
    )
    .with_enum_values("udp | tcp")
    .with_default("tcp"),
];

impl MultiOutputSource for RtspSrcN {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        1 + usize::from(self.audio.is_some())
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(self.video_caps()),
            AUDIO_PORT => self.audio.clone().ok_or(G2gError::CapsMismatch),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RTSPSRCN_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.url = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "user-agent" => {
                self.user_agent = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            // user-id / user-pw build the one Credentials together; either may be
            // set first, so each preserves the other's current value.
            "user-id" => {
                let username = value.as_str().ok_or(PropError::Type)?.to_string();
                let password = self
                    .creds
                    .as_ref()
                    .map(|c| c.password.clone())
                    .unwrap_or_default();
                self.creds = Some(retina::client::Credentials { username, password });
                Ok(())
            }
            "user-pw" => {
                let password = value.as_str().ok_or(PropError::Type)?.to_string();
                let username = self
                    .creds
                    .as_ref()
                    .map(|c| c.username.clone())
                    .unwrap_or_default();
                self.creds = Some(retina::client::Credentials { username, password });
                Ok(())
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            // width / height fill the expected-dims pair one leg at a time; the
            // pair only takes effect once both are nonzero (see video_caps).
            "width" => {
                let w = value.as_uint().ok_or(PropError::Type)? as u32;
                let h = self.expected_dims.map(|(_, h)| h).unwrap_or(0);
                self.expected_dims = Some((w, h));
                Ok(())
            }
            "height" => {
                let h = value.as_uint().ok_or(PropError::Type)? as u32;
                let w = self.expected_dims.map(|(w, _)| w).unwrap_or(0);
                self.expected_dims = Some((w, h));
                Ok(())
            }
            "audio-format" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                let format = audio_format_for_nick(nick).ok_or(PropError::Value)?;
                self.audio = Some(default_audio_caps(format));
                Ok(())
            }
            "reconnect" => {
                let n = value.as_uint().ok_or(PropError::Type)?;
                self.reconnect.max_attempts = n.min(u32::MAX as u64) as u32;
                self.reconnect.fill_backoff_defaults();
                Ok(())
            }
            "reconnect-backoff" => {
                self.reconnect.initial_backoff_ms = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "reconnect-backoff-max" => {
                self.reconnect.max_backoff_ms = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "protocols" => {
                let mut order = Vec::new();
                for nick in value.as_flags().ok_or(PropError::Type)? {
                    let t = LowerTransport::from_nick(nick.as_str()).ok_or(PropError::Value)?;
                    if !order.contains(&t) {
                        order.push(t);
                    }
                }
                if order.is_empty() {
                    return Err(PropError::Value);
                }
                self.transports = order;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.url.clone())),
            "user-agent" => Some(PropValue::Str(self.user_agent.clone())),
            "user-id" => Some(PropValue::Str(
                self.creds
                    .as_ref()
                    .map(|c| c.username.clone())
                    .unwrap_or_default(),
            )),
            "user-pw" => Some(PropValue::Str(
                self.creds
                    .as_ref()
                    .map(|c| c.password.clone())
                    .unwrap_or_default(),
            )),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "width" => Some(PropValue::Uint(
                self.expected_dims.map(|(w, _)| w).unwrap_or(0) as u64,
            )),
            "height" => Some(PropValue::Uint(
                self.expected_dims.map(|(_, h)| h).unwrap_or(0) as u64,
            )),
            "audio-format" => Some(PropValue::Str(
                audio_format_nick(self.audio_format()?).to_string(),
            )),
            "reconnect" => Some(PropValue::Uint(self.reconnect.max_attempts as u64)),
            "reconnect-backoff" => Some(PropValue::Uint(self.reconnect.initial_backoff_ms)),
            "reconnect-backoff-max" => Some(PropValue::Uint(self.reconnect.max_backoff_ms)),
            "protocols" => Some(PropValue::Flags(
                self.transports
                    .iter()
                    .map(|t| t.nick().to_string())
                    .collect(),
            )),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if crate::numbuffers::finished_at_zero_limit_multi(
                self.frame_limit,
                self.output_count(),
                out,
            )
            .await?
            {
                return Ok(0);
            }
            run_rtsp_n(self, out).await
        })
    }
}

/// Per-output emission state, threaded across sessions so a reconnect continues
/// each pad's sequence counter rather than restarting it.
#[derive(Debug)]
struct PortState {
    emitted: u64,
    eos_sent: bool,
}

impl PortState {
    fn new() -> Self {
        Self {
            emitted: 0,
            eos_sent: false,
        }
    }
}

/// Outer reconnect orchestrator, the fan-out twin of `rtspsrc`'s: run a
/// session, and on a network failure wait per the policy and run another with
/// the timeline pushed past the gap. Every output gets exactly one `Eos` before
/// this returns `Ok`.
async fn run_rtsp_n(src: &mut RtspSrcN, out: &mut dyn MultiOutputSink) -> Result<u64, G2gError> {
    let ports = src.output_count();
    let mut state: Vec<PortState> = (0..ports).map(|_| PortState::new()).collect();
    let mut pts_base_ns: u64 = 0;
    let mut retry = ReconnectState::new(&src.reconnect);

    loop {
        let mut session_max_pts = pts_base_ns;
        let outcome = run_session(src, out, &mut state, pts_base_ns, &mut session_max_pts).await;
        match outcome {
            SessionOutcome::LimitReached | SessionOutcome::GracefulEnd => break,
            SessionOutcome::DownstreamError(e) => return Err(e),
            SessionOutcome::NetworkError(e) => {
                let Some(backoff_ms) = retry.next_delay_ms(&src.reconnect) else {
                    return Err(e);
                };
                std::eprintln!(
                    "rtspsrcn: session ended ({:?}); reconnect {}/{} after {}ms",
                    e,
                    retry.attempt,
                    src.reconnect.max_attempts,
                    backoff_ms,
                );
                for (port, port_state) in state.iter().enumerate() {
                    if !port_state.eos_sent {
                        let _ = out.push_to(port, PipelinePacket::Flush).await;
                    }
                }
                pts_base_ns = session_max_pts.saturating_add(RECONNECT_PTS_GAP_NS);
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
        }
    }

    let mut total = 0;
    for (port, port_state) in state.iter_mut().enumerate() {
        total += port_state.emitted;
        if !port_state.eos_sent {
            out.push_to(port, PipelinePacket::Eos).await?;
            port_state.eos_sent = true;
        }
    }
    Ok(total)
}

/// Result of a single connect + play + drain session, as in `rtspsrc`.
#[derive(Debug)]
enum SessionOutcome {
    /// Every output reached `num-buffers`.
    LimitReached,
    /// The server closed the stream without error.
    GracefulEnd,
    /// retina or the network errored. Eligible for reconnect.
    NetworkError(G2gError),
    /// A downstream push failed. Not retryable.
    DownstreamError(G2gError),
}

/// DESCRIBE, SETUP both tracks, PLAY, and drain until every output hits its
/// limit, the server ends the stream, or the session errors.
async fn run_session(
    src: &mut RtspSrcN,
    out: &mut dyn MultiOutputSink,
    state: &mut [PortState],
    pts_base_ns: u64,
    session_max_pts: &mut u64,
) -> SessionOutcome {
    let want_audio = src.audio_format();
    let (session, video_idx, audio_idx) = match connect_describe_setup(
        &src.url,
        &src.user_agent,
        src.creds.as_ref(),
        &src.transports,
        want_audio,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return SessionOutcome::NetworkError(e),
    };

    let video_clock_rate = u64::from(session.streams()[video_idx].clock_rate_hz());
    let audio_clock_rate = audio_idx
        .map(|i| u64::from(session.streams()[i].clock_rate_hz()))
        .unwrap_or(1);
    let mut video_caps = caps_from_video_params(video_params_for(session.streams(), video_idx));

    // Both streams are set up, so the server's `RTP-Info` origin is what puts
    // them on one timeline; permissive keeps a server that omits `rtptime`
    // playable (each stream then starts at its own first packet).
    let play_options = PlayOptions::default().initial_timestamp(InitialTimestampPolicy::Permissive);
    let played = match session.play(play_options).await {
        Ok(p) => p,
        Err(_) => return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other)),
    };
    let mut demuxed = match played.demuxed() {
        Ok(d) => d,
        Err(_) => return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other)),
    };

    if let Some(caps) = &video_caps {
        if let Err(e) = out
            .push_to(VIDEO_PORT, PipelinePacket::CapsChanged(caps.clone()))
            .await
        {
            return SessionOutcome::DownstreamError(e);
        }
    }
    if let Some(caps) = &src.audio {
        if let Err(e) = out
            .push_to(AUDIO_PORT, PipelinePacket::CapsChanged(caps.clone()))
            .await
        {
            return SessionOutcome::DownstreamError(e);
        }
    }

    // As in `rtspsrc`: retina prepends SPS/PPS only to key frames, so a mid-GOP
    // tune-in must drop access units until the first one or the decoder sees
    // slices with no parameter set.
    let mut seen_keyframe = false;
    let mut origin_ns: Option<u64> = None;
    let limit = src.frame_limit;

    while state.iter().any(|p| p.emitted < limit) {
        let item = match demuxed.next().await {
            Some(Ok(item)) => item,
            Some(Err(_)) => {
                return SessionOutcome::NetworkError(G2gError::Hardware(HardwareError::Other))
            }
            None => return SessionOutcome::GracefulEnd,
        };

        let (port, timestamp, duration_ns, bytes, keyframe) = match item {
            CodecItem::VideoFrame(vf) if vf.stream_id() == video_idx => {
                if !seen_keyframe {
                    if !vf.is_random_access_point() {
                        continue;
                    }
                    seen_keyframe = true;
                }
                if vf.has_new_parameters() {
                    let refreshed =
                        caps_from_video_params(video_params_for(demuxed.streams(), video_idx));
                    if refreshed != video_caps {
                        if let Some(caps) = &refreshed {
                            if let Err(e) = out
                                .push_to(VIDEO_PORT, PipelinePacket::CapsChanged(caps.clone()))
                                .await
                            {
                                return SessionOutcome::DownstreamError(e);
                            }
                        }
                        video_caps = refreshed;
                    }
                }
                let timestamp = vf.timestamp();
                let bytes = vf.into_data().into_boxed_slice();
                // IDR NAL => independently decodable. H.265 reports false (the
                // helper is H.264-specific), the safe default.
                let keyframe = crate::h264util::h264_au_is_keyframe(&bytes);
                (VIDEO_PORT, timestamp, 0, bytes, keyframe)
            }
            CodecItem::AudioFrame(af) if Some(af.stream_id()) == audio_idx => {
                let duration_ns = u64::from(af.frame_length().get()).saturating_mul(1_000_000_000)
                    / audio_clock_rate.max(1);
                // Every AAC / G.711 frame decodes on its own.
                (
                    AUDIO_PORT,
                    af.timestamp(),
                    duration_ns,
                    Box::from(af.data()),
                    true,
                )
            }
            _ => continue,
        };

        if state[port].emitted >= limit {
            continue;
        }

        let clock_rate = match port {
            VIDEO_PORT => video_clock_rate,
            _ => audio_clock_rate,
        };
        let raw_ns =
            (timestamp.elapsed().max(0) as u64).saturating_mul(1_000_000_000) / clock_rate.max(1);
        let origin = *origin_ns.get_or_insert(raw_ns);
        let pts_ns = pts_base_ns.saturating_add(raw_ns.saturating_sub(origin));
        *session_max_pts = (*session_max_pts).max(pts_ns);

        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                capture_ns: pts_ns,
                arrival_ns: g2g_core::metrics::monotonic_ns(),
                keyframe,
            },
            sequence: state[port].emitted,
            meta: Default::default(),
        };
        if let Err(e) = out.push_to(port, PipelinePacket::DataFrame(frame)).await {
            return SessionOutcome::DownstreamError(e);
        }
        state[port].emitted += 1;

        // A pad that reached its limit ends now, so its branch drains while the
        // others keep receiving.
        if state[port].emitted >= limit && !state[port].eos_sent {
            if let Err(e) = out.push_to(port, PipelinePacket::Eos).await {
                return SessionOutcome::DownstreamError(e);
            }
            state[port].eos_sent = true;
        }
    }

    SessionOutcome::LimitReached
}

/// DESCRIBE and SETUP the video stream plus, when asked for, the audio one.
/// Both tracks take the same lower transport, so a server that refuses UDP
/// moves the whole session to interleaved TCP rather than splitting it.
async fn connect_describe_setup(
    url: &str,
    user_agent: &str,
    creds: Option<&retina::client::Credentials>,
    transports: &[LowerTransport],
    want_audio: Option<AudioFormat>,
) -> Result<(Session<Described>, usize, Option<usize>), G2gError> {
    let mut session = connect_describe(url, user_agent, creds).await?;
    let video_idx = video_stream_index(session.streams()).ok_or(G2gError::CapsMismatch)?;
    let audio_idx = match want_audio {
        Some(want) => Some(
            audio_stream_index(session.streams(), Some(want))
                .ok_or(G2gError::CapsMismatch)?
                .0,
        ),
        None => None,
    };

    let mut setup_err = None;
    for transport in transports {
        let options = || {
            SetupOptions::default()
                .frame_format(FrameFormat::SIMPLE)
                .transport(transport.retina())
        };
        match session.setup(video_idx, options()).await {
            Ok(()) => {}
            Err(e) => {
                setup_err = Some(e);
                continue;
            }
        }
        match audio_idx {
            None => return Ok((session, video_idx, None)),
            Some(idx) => match session.setup(idx, options()).await {
                Ok(()) => return Ok((session, video_idx, Some(idx))),
                Err(e) => setup_err = Some(e),
            },
        }
    }
    Err(match setup_err {
        Some(_) => G2gError::Hardware(HardwareError::Other),
        // An empty transport list cannot happen: `protocols` rejects an empty set.
        None => G2gError::CapsMismatch,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_only_has_one_output_and_audio_adds_a_second() {
        let src = RtspSrcN::new("rtsp://example/stream");
        assert_eq!(src.output_count(), 1);
        assert!(src.output_caps(AUDIO_PORT).is_err());
        let src = src.with_outputs(2);
        assert_eq!(src.output_count(), 2);
        assert_eq!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );
    }

    #[test]
    fn placeholder_video_caps_fixate() {
        let src = RtspSrcN::new("rtsp://example/stream");
        src.output_caps(VIDEO_PORT)
            .expect("video caps")
            .fixate()
            .expect("the placeholder must survive fixate");
    }

    #[test]
    fn expected_dims_win_over_the_placeholder() {
        let src = RtspSrcN::new("rtsp://example/stream").with_expected_dims(1920, 1080);
        assert!(matches!(
            src.output_caps(VIDEO_PORT),
            Ok(Caps::CompressedVideo {
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                ..
            })
        ));
    }

    #[test]
    fn probed_tracks_set_both_pads() {
        let tracks = RtspTracks {
            video: Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            audio: Some(Caps::Audio {
                format: AudioFormat::Aac,
                channels: 1,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }),
        };
        let src = RtspSrcN::new("rtsp://example/stream").with_tracks(&tracks);
        assert_eq!(src.output_count(), 2);
        assert_eq!(src.output_caps(VIDEO_PORT), Ok(tracks.video.clone()));
        // The SDP's 48 kHz mono AAC negotiates at 0/0: a compressed sample rate
        // is matched for equality, so the real rate would find no decoder.
        assert_eq!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Aac,
                channels: 0,
                sample_rate: 0,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );
    }

    #[test]
    fn g711_keeps_the_sdp_rate_its_decoder_declares() {
        let declared = Caps::Audio {
            format: AudioFormat::Alaw,
            channels: G711_DEFAULT_CHANNELS,
            sample_rate: G711_CLOCK_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(negotiation_audio_caps(&declared), declared);
    }

    #[test]
    fn audio_format_property_round_trips_and_selects_g711_nominal_caps() {
        let mut src = RtspSrcN::new("rtsp://example/stream").with_outputs(2);
        src.set_property("audio-format", PropValue::Str(NICK_ALAW.to_string()))
            .expect("alaw is a declared value");
        assert_eq!(
            src.get_property("audio-format"),
            Some(PropValue::Str(NICK_ALAW.to_string()))
        );
        assert_eq!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Alaw,
                channels: G711_DEFAULT_CHANNELS,
                sample_rate: G711_CLOCK_RATE_HZ,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );
        assert_eq!(
            src.set_property("audio-format", PropValue::Str("vorbis".to_string())),
            Err(PropError::Value)
        );
    }

    #[test]
    fn sdp_encoding_names_map_to_audio_formats() {
        assert_eq!(audio_format_for("mpeg4-generic"), Some(AudioFormat::Aac));
        assert_eq!(audio_format_for("pcma"), Some(AudioFormat::Alaw));
        assert_eq!(audio_format_for("pcmu"), Some(AudioFormat::Mulaw));
        // g2g has no RTP depayload/decode path for these, so the pad is not offered.
        assert_eq!(audio_format_for("g722"), None);
        assert_eq!(audio_format_for("opus"), None);
    }

    #[test]
    fn num_buffers_and_location_round_trip() {
        let mut src = RtspSrcN::new("rtsp://example/stream");
        src.set_property("location", PropValue::Str("rtsp://other/s".to_string()))
            .expect("location");
        src.set_property("num-buffers", PropValue::Int(12))
            .expect("num-buffers");
        assert_eq!(
            src.get_property("location"),
            Some(PropValue::Str("rtsp://other/s".to_string()))
        );
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(12)));
    }
}
