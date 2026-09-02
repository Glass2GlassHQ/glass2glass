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
    VideoCodec, NTP_TO_UNIX_EPOCH_SECS,
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

/// SDP `a=rtpmap` encoding name of an uncompressed ONVIF metadata track
/// (Streaming spec 5.2.2.4).
const ONVIF_METADATA_ENCODING: &str = "vnd.onvif.metadata";
/// The GZIP variant's encoding name as retina spells it. The Streaming spec
/// writes it [`ONVIF_METADATA_SPEC_GZIP_ENCODING`] instead.
const ONVIF_METADATA_GZIP_ENCODING: &str = "vnd.onvif.metadata.gzip";
/// The GZIP variant's encoding name as the Streaming spec writes it. retina
/// 0.4.19 builds no depacketizer for this spelling, and a set-up stream it
/// cannot depacketize fails the whole session at `demuxed()`, so a track
/// advertised this way is skipped rather than played.
const ONVIF_METADATA_SPEC_GZIP_ENCODING: &str = "vnd.onvif.metadata+gzip";
/// EXI-coded metadata tracks. g2g has no EXI decoder, so they are skipped.
const ONVIF_METADATA_EXI_ENCODINGS: &[&str] =
    &["vnd.onvif.metadata.exi.onvif", "vnd.onvif.metadata.exi.ext"];

/// Largest document one gzip-compressed metadata packet may inflate to. A
/// camera's scene description is kilobytes; the bound stops a crafted few-KiB
/// payload from inflating to gigabytes.
const MAX_INFLATED_METADATA_LEN: usize = 4 * 1024 * 1024;

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
    /// Whether the SDP offers an ONVIF analytics metadata track this element
    /// can play. Reported, not subscribed: the metadata pad is opt-in through
    /// [`RtspSrcN::with_onvif_metadata`] / the `onvif-metadata` property.
    pub onvif_metadata: bool,
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
    /// Whether the ONVIF analytics metadata track gets its own output pad.
    onvif_metadata: bool,
    /// Output pads a launch line asked for by linking them, so setting
    /// `onvif-metadata` after construction can trade the audio pad for the
    /// metadata one instead of growing the element past the linked count.
    requested_outputs: Option<usize>,
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
            onvif_metadata: false,
            requested_outputs: None,
        }
    }

    /// Take the pad count from a launch line's linked outputs: 2 or more adds
    /// the audio pad (AAC unless `audio-format` says otherwise), 1 leaves the
    /// element video-only.
    pub fn with_outputs(mut self, outputs: usize) -> Self {
        self.requested_outputs = Some(outputs);
        self.fill_audio_pad();
        self
    }

    /// Subscribe the SDP's ONVIF analytics metadata track on a pad of its own,
    /// after video and audio. Off by default, so a graph that only wants
    /// pictures negotiates exactly as before. Each frame on it is one complete
    /// `tt:MetadataStream` document, ready for `onvifmetadataparse`.
    pub fn with_onvif_metadata(mut self, on: bool) -> Self {
        self.onvif_metadata = on;
        self.fill_audio_pad();
        self
    }

    /// Give the audio pad whatever room the linked-pad count leaves once video
    /// and the optional metadata pad have taken theirs. Only a launch line fixes
    /// that count; a builder caller names the pads it wants directly.
    fn fill_audio_pad(&mut self) {
        let Some(outputs) = self.requested_outputs else {
            return;
        };
        let without_audio = 1 + usize::from(self.onvif_metadata);
        self.audio = (outputs > without_audio).then(|| audio_caps(AudioFormat::Aac, 0, 0));
    }

    /// The metadata track's output pad, after video and audio.
    fn metadata_port(&self) -> Option<usize> {
        self.onvif_metadata
            .then(|| 1 + usize::from(self.audio.is_some()))
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

/// The first ONVIF analytics metadata track g2g can play, and whether its
/// payload is gzip-compressed. Tracks advertised with the Streaming spec's
/// `+gzip` spelling or with an EXI coding are logged and skipped: retina builds
/// no depacketizer for the first, and g2g has no EXI decoder for the second.
fn onvif_metadata_stream_index(streams: &[retina::client::Stream]) -> Option<(usize, bool)> {
    streams.iter().enumerate().find_map(|(i, s)| {
        if s.media() != "application" {
            return None;
        }
        Some((i, onvif_metadata_gzip(s.encoding_name())?))
    })
}

/// Whether an SDP encoding name is an ONVIF metadata track g2g plays, and
/// whether its payload is gzip-compressed. `None` for a coding it has to leave
/// alone, with a line saying which.
fn onvif_metadata_gzip(encoding_name: &str) -> Option<bool> {
    match encoding_name {
        ONVIF_METADATA_ENCODING => Some(false),
        ONVIF_METADATA_GZIP_ENCODING => Some(true),
        ONVIF_METADATA_SPEC_GZIP_ENCODING => {
            std::eprintln!(
                "rtspsrcn: skipping the {ONVIF_METADATA_SPEC_GZIP_ENCODING} track: retina has no \
                 depacketizer for that spelling and setting it up would fail the whole session",
            );
            None
        }
        name if ONVIF_METADATA_EXI_ENCODINGS.contains(&name) => {
            std::eprintln!("rtspsrcn: skipping the {name} track: no EXI decoder");
            None
        }
        _ => None,
    }
}

/// Inflate one RFC 1952 gzip member: the header (and its optional extra field,
/// name, comment and header CRC) is skipped, then the deflate stream is
/// inflated. `None` for anything that is not a gzip member or does not inflate
/// within [`MAX_INFLATED_METADATA_LEN`], so a truncated or crafted payload is
/// dropped rather than trusted.
fn inflate_gzip(data: &[u8]) -> Option<Vec<u8>> {
    /// Fixed part of the header: magic, compression method, flags, mtime, extra
    /// flags, OS.
    const FIXED_HEADER_LEN: usize = 10;
    const MAGIC: [u8; 2] = [0x1f, 0x8b];
    const DEFLATE_METHOD: u8 = 8;
    const FLAG_HEADER_CRC: u8 = 0x02;
    const FLAG_EXTRA: u8 = 0x04;
    const FLAG_NAME: u8 = 0x08;
    const FLAG_COMMENT: u8 = 0x10;

    if data.get(..2)? != MAGIC || *data.get(2)? != DEFLATE_METHOD {
        return None;
    }
    let flags = *data.get(3)?;
    let mut pos = FIXED_HEADER_LEN;
    if flags & FLAG_EXTRA != 0 {
        let len = u16::from_le_bytes([*data.get(pos)?, *data.get(pos + 1)?]) as usize;
        pos = pos.checked_add(2)?.checked_add(len)?;
    }
    for flag in [FLAG_NAME, FLAG_COMMENT] {
        if flags & flag != 0 {
            let end = data.get(pos..)?.iter().position(|&b| b == 0)?;
            pos = pos.checked_add(end)?.checked_add(1)?;
        }
    }
    if flags & FLAG_HEADER_CRC != 0 {
        pos = pos.checked_add(2)?;
    }
    let deflate = data.get(pos..)?;
    miniz_oxide::inflate::decompress_to_vec_with_limit(deflate, MAX_INFLATED_METADATA_LEN).ok()
}

/// An RTCP sender report's 64-bit NTP timestamp as nanoseconds since the Unix
/// epoch: seconds in the high word, a binary fraction of a second in the low.
fn ntp_to_unix_nanos(ntp: u64) -> i64 {
    const NANOS_PER_SEC: i64 = 1_000_000_000;
    let secs = (ntp >> 32) as i64 - NTP_TO_UNIX_EPOCH_SECS as i64;
    let frac_nanos = (((ntp & 0xFFFF_FFFF) * NANOS_PER_SEC as u64) >> 32) as i64;
    secs.saturating_mul(NANOS_PER_SEC)
        .saturating_add(frac_nanos)
}

/// The wall clock a stream's latest sender report pins its RTP clock to: an
/// instant on the sender's clock and the media timestamp it names.
#[derive(Debug, Clone, Copy)]
struct SenderClock {
    unix_nanos: i64,
    rtp_timestamp: u32,
    clock_rate_hz: u32,
}

impl SenderClock {
    /// The sender's wall-clock time for the frame stamped `rtp_timestamp`,
    /// nanoseconds since the Unix epoch. The RTP timestamp wraps at 2^32, so
    /// the distance from the report is taken as a signed 32-bit difference and
    /// a frame just before the report reads as earlier rather than as most of a
    /// wrap ahead.
    fn wall_clock_ns(&self, rtp_timestamp: u32) -> i64 {
        const NANOS_PER_SEC: i64 = 1_000_000_000;
        let delta = rtp_timestamp.wrapping_sub(self.rtp_timestamp) as i32;
        let rate = i64::from(self.clock_rate_hz.max(1));
        self.unix_nanos + i64::from(delta) * NANOS_PER_SEC / rate
    }
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
    let onvif_metadata = onvif_metadata_stream_index(streams).is_some();
    Some(RtspTracks {
        video,
        audio,
        onvif_metadata,
    })
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
        "onvif-metadata",
        PropKind::Bool,
        "subscribe the SDP's ONVIF analytics metadata track on a pad of its own",
    )
    .with_default("false"),
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
        1 + usize::from(self.audio.is_some()) + usize::from(self.onvif_metadata)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        if Some(output) == self.metadata_port() {
            return Ok(Caps::OnvifMetadata);
        }
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
            "onvif-metadata" => {
                self.onvif_metadata = value.as_bool().ok_or(PropError::Type)?;
                self.fill_audio_pad();
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
            "onvif-metadata" => Some(PropValue::Bool(self.onvif_metadata)),
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
    let metadata_port = src.metadata_port();
    let (session, video_idx, audio_idx, metadata) = match connect_describe_setup(
        &src.url,
        &src.user_agent,
        src.creds.as_ref(),
        &src.transports,
        want_audio,
        metadata_port.is_some(),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => return SessionOutcome::NetworkError(e),
    };
    let (metadata_idx, metadata_gzip) = match metadata {
        Some((idx, gzip)) => (Some(idx), gzip),
        None => (None, false),
    };

    let audio_clock_rate = audio_idx
        .map(|i| u64::from(session.streams()[i].clock_rate_hz()))
        .unwrap_or(1);
    // The sender report each stream was last pinned to, indexed by retina's
    // stream index, so a frame can carry the sender's wall clock.
    let mut sender_clocks: Vec<Option<SenderClock>> = alloc::vec![None; session.streams().len()];
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
    if let Some(port) = metadata_port {
        if let Err(e) = out
            .push_to(port, PipelinePacket::CapsChanged(Caps::OnvifMetadata))
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
            // The compound packet's sender report pins this stream's RTP clock
            // to the sender's wall clock, which is the only thing that lines the
            // metadata track up with the video (the metadata RTP timestamps
            // carry no meaning of their own).
            CodecItem::Rtcp(compound) => {
                let stream_id = compound.stream_id();
                let sr = compound
                    .pkts()
                    .next()
                    .and_then(|p| p.as_sender_report().ok().flatten());
                // RFC 3550 lets a sender without a wall clock report NTP zero.
                let sr = sr.filter(|sr| sr.ntp_timestamp().0 != 0);
                if let (Some(sr), Some(slot)) = (sr, sender_clocks.get_mut(stream_id)) {
                    *slot = Some(SenderClock {
                        unix_nanos: ntp_to_unix_nanos(sr.ntp_timestamp().0),
                        rtp_timestamp: sr.rtp_timestamp(),
                        clock_rate_hz: demuxed.streams()[stream_id].clock_rate_hz(),
                    });
                }
                continue;
            }
            CodecItem::MessageFrame(mf) if Some(mf.stream_id()) == metadata_idx => {
                let port = match metadata_port {
                    Some(port) => port,
                    None => continue,
                };
                let document = if metadata_gzip {
                    match inflate_gzip(mf.data()) {
                        Some(bytes) => bytes.into_boxed_slice(),
                        None => {
                            std::eprintln!(
                                "rtspsrcn: dropping a {}-byte ONVIF metadata document that does not inflate",
                                mf.data().len(),
                            );
                            continue;
                        }
                    }
                } else {
                    Box::from(mf.data())
                };
                (port, mf.timestamp(), 0, document, true)
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

        let stream_idx = match port {
            VIDEO_PORT => video_idx,
            _ if Some(port) == metadata_port => metadata_idx.unwrap_or(video_idx),
            _ => audio_idx.unwrap_or(video_idx),
        };
        let clock_rate = u64::from(demuxed.streams()[stream_idx].clock_rate_hz()).max(1);
        let raw_ns = (timestamp.elapsed().max(0) as u64).saturating_mul(1_000_000_000) / clock_rate;
        let origin = *origin_ns.get_or_insert(raw_ns);
        let pts_ns = pts_base_ns.saturating_add(raw_ns.saturating_sub(origin));
        *session_max_pts = (*session_max_pts).max(pts_ns);

        let mut frame = Frame {
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
        // Before the stream's first sender report there is no wall clock to
        // name, so the frame carries none.
        if let Some(clock) = sender_clocks[stream_idx] {
            attach_wall_clock(
                &mut frame,
                clock.wall_clock_ns(timestamp.timestamp() as u32),
            );
        }
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

/// Attach the sender's wall clock to a frame. A no-op without the `metadata`
/// feature, where a frame has nowhere to carry it.
#[cfg(feature = "metadata")]
fn attach_wall_clock(frame: &mut Frame, unix_nanos: i64) {
    frame.meta.attach(g2g_core::WallClockMeta { unix_nanos });
}

#[cfg(not(feature = "metadata"))]
fn attach_wall_clock(_frame: &mut Frame, _unix_nanos: i64) {}

/// DESCRIBE and SETUP the video stream plus, when asked for, the audio one and
/// the ONVIF metadata one. Every track takes the same lower transport, so a
/// server that refuses UDP moves the whole session to interleaved TCP rather
/// than splitting it. The metadata result is `(stream index, gzip)`.
async fn connect_describe_setup(
    url: &str,
    user_agent: &str,
    creds: Option<&retina::client::Credentials>,
    transports: &[LowerTransport],
    want_audio: Option<AudioFormat>,
    want_metadata: bool,
) -> Result<
    (
        Session<Described>,
        usize,
        Option<usize>,
        Option<(usize, bool)>,
    ),
    G2gError,
> {
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
    // A missing metadata track leaves the pad silent through to EOS rather than
    // failing the session: a camera can be configured to stream analytics on
    // one profile and not another.
    let metadata = want_metadata
        .then(|| onvif_metadata_stream_index(session.streams()))
        .flatten();

    let mut setup_err = None;
    for transport in transports {
        let options = || {
            SetupOptions::default()
                .frame_format(FrameFormat::SIMPLE)
                .transport(transport.retina())
        };
        let extra = audio_idx.into_iter().chain(metadata.map(|(idx, _)| idx));
        match session.setup(video_idx, options()).await {
            Ok(()) => {}
            Err(e) => {
                setup_err = Some(e);
                continue;
            }
        }
        let mut failed = None;
        for idx in extra {
            if let Err(e) = session.setup(idx, options()).await {
                failed = Some(e);
                break;
            }
        }
        match failed {
            None => return Ok((session, video_idx, audio_idx, metadata)),
            Some(e) => setup_err = Some(e),
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
            onvif_metadata: false,
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
    fn onvif_metadata_pad_is_opt_in_and_takes_the_last_slot() {
        // Off by default: an existing consumer sees exactly what it saw before.
        let src = RtspSrcN::new("rtsp://example/stream").with_outputs(2);
        assert_eq!(src.output_count(), 2);
        assert!(matches!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio { .. })
        ));

        // Two linked pads plus the metadata track means video and metadata: the
        // audio pad gives up its slot rather than the element growing past what
        // the launch line linked.
        let mut src = RtspSrcN::new("rtsp://example/stream").with_outputs(2);
        src.set_property("onvif-metadata", PropValue::Bool(true))
            .expect("onvif-metadata is a declared property");
        assert_eq!(src.output_count(), 2);
        assert_eq!(src.output_caps(1), Ok(Caps::OnvifMetadata));
        assert_eq!(
            src.get_property("onvif-metadata"),
            Some(PropValue::Bool(true))
        );

        // Three linked pads keep the audio one, with metadata after it.
        let src = RtspSrcN::new("rtsp://example/stream")
            .with_outputs(3)
            .with_onvif_metadata(true);
        assert_eq!(src.output_count(), 3);
        assert!(matches!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio { .. })
        ));
        assert_eq!(src.output_caps(2), Ok(Caps::OnvifMetadata));
    }

    #[test]
    fn sdp_encoding_names_select_the_metadata_track_and_its_compression() {
        assert_eq!(onvif_metadata_gzip(ONVIF_METADATA_ENCODING), Some(false));
        assert_eq!(
            onvif_metadata_gzip(ONVIF_METADATA_GZIP_ENCODING),
            Some(true)
        );
        // retina has no depacketizer for the specification's `+gzip` spelling,
        // and none for EXI, so those tracks are left alone.
        assert_eq!(onvif_metadata_gzip(ONVIF_METADATA_SPEC_GZIP_ENCODING), None);
        for exi in ONVIF_METADATA_EXI_ENCODINGS {
            assert_eq!(onvif_metadata_gzip(exi), None);
        }
        assert_eq!(onvif_metadata_gzip("vnd.onvif.metadata.other"), None);
    }

    /// 2008-10-10T12:24:57.321Z, the instant the ONVIF Analytics Specification's
    /// first example frame names (section 5.1.3.1, page 13).
    const SPEC_EXAMPLE_NANOS: i64 = 1_223_641_497_321_000_000;
    const VIDEO_CLOCK_HZ: u32 = 90_000;

    #[test]
    fn a_sender_report_puts_frames_on_the_senders_wall_clock() {
        // The same instant as an NTP timestamp: seconds since 1900 in the high
        // word, the .321 as a binary fraction in the low.
        let secs = (SPEC_EXAMPLE_NANOS / 1_000_000_000) as u64 + NTP_TO_UNIX_EPOCH_SECS;
        let frac = (321_000_000u64 << 32) / 1_000_000_000;
        let ntp = (secs << 32) | frac;
        // The 32-bit fraction names about a quarter of a nanosecond, so the
        // round trip is exact to one.
        assert!((ntp_to_unix_nanos(ntp) - SPEC_EXAMPLE_NANOS).abs() <= 1);

        let clock = SenderClock {
            unix_nanos: SPEC_EXAMPLE_NANOS,
            rtp_timestamp: 1_000_000,
            clock_rate_hz: VIDEO_CLOCK_HZ,
        };
        // The report's own timestamp is the report's own instant.
        assert_eq!(clock.wall_clock_ns(1_000_000), SPEC_EXAMPLE_NANOS);
        // One second of 90 kHz ticks after it, and one second before.
        assert_eq!(
            clock.wall_clock_ns(1_000_000 + VIDEO_CLOCK_HZ),
            SPEC_EXAMPLE_NANOS + 1_000_000_000
        );
        assert_eq!(
            clock.wall_clock_ns(1_000_000 - VIDEO_CLOCK_HZ),
            SPEC_EXAMPLE_NANOS - 1_000_000_000
        );
    }

    #[test]
    fn the_wall_clock_survives_an_rtp_timestamp_wrap() {
        // A report taken one second of ticks before the 32-bit RTP clock wraps.
        let clock = SenderClock {
            unix_nanos: SPEC_EXAMPLE_NANOS,
            rtp_timestamp: u32::MAX - VIDEO_CLOCK_HZ + 1,
            clock_rate_hz: VIDEO_CLOCK_HZ,
        };
        // A frame a second after it, on the far side of the wrap. Read as an
        // unsigned difference this would have come out most of a wrap ahead
        // (about 13 hours) instead of a second.
        assert_eq!(clock.wall_clock_ns(0), SPEC_EXAMPLE_NANOS + 1_000_000_000);
        // And one from a second before the report, on the near side.
        assert_eq!(
            clock.wall_clock_ns(u32::MAX - 2 * VIDEO_CLOCK_HZ + 1),
            SPEC_EXAMPLE_NANOS - 1_000_000_000,
        );
    }

    #[test]
    fn gzip_inflate_reads_a_member_and_refuses_anything_else() {
        const PAYLOAD: &[u8] = br#"<?xml version="1.0"?><tt:MetadataStream/>"#;
        let member = gzip_member(PAYLOAD);
        assert_eq!(inflate_gzip(&member).as_deref(), Some(PAYLOAD));

        // Not a gzip member, a header cut short, and a deflate stream cut
        // short: all refused rather than panicking.
        assert_eq!(inflate_gzip(PAYLOAD), None);
        assert_eq!(inflate_gzip(&member[..5]), None);
        assert_eq!(inflate_gzip(&member[..12]), None);
        assert_eq!(inflate_gzip(&[]), None);
    }

    /// An RFC 1952 member around `payload`: the fixed header, the raw deflate
    /// stream, then the CRC32 and length trailer.
    fn gzip_member(payload: &[u8]) -> Vec<u8> {
        let mut member = Vec::from([0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff]);
        member.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(payload, 6));
        member.extend_from_slice(&crc32(payload).to_le_bytes());
        member.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        member
    }

    /// CRC-32, so the member the test builds carries a real trailer.
    fn crc32(data: &[u8]) -> u32 {
        const POLYNOMIAL: u32 = 0xEDB8_8320;
        let mut crc = u32::MAX;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (POLYNOMIAL & mask);
            }
        }
        !crc
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
