// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::HashMap;
use std::io;

use g2g_plugins::opusdec::ChannelMapping;

use super::stream::StreamEncoder;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const HEAD_LIMIT: usize = 16 * 1024;
const BODY_LIMIT: usize = 16 * 1024;
const READ_CHUNK: usize = 2048;
const SESSION_HEADER: &str = "G2G;timeout = 90";
// moonlight's encryptionSupported and encryptionRequested bits
const ENCRYPT_CONTROL: u32 = 0x1;
const ENCRYPT_VIDEO: u32 = 0x2;
const ENCRYPT_AUDIO: u32 = 0x4;
// moonlight takes this sprop-parameter-sets value as the host offering HEVC
const HEVC_MARKER: &str = "sprop-parameter-sets=AAAAAU";
const BITSTREAM_FORMAT_H264: u32 = 0;
const BITSTREAM_FORMAT_HEVC: u32 = 1;
// sunshine's opus bitrates for stereo, 5.1 and 7.1
const AUDIO_BITRATES: [AudioBitrate; 3] = [
    AudioBitrate {
        channels: 2,
        bits_per_second: 96_000,
    },
    AudioBitrate {
        channels: 6,
        bits_per_second: 256_000,
    },
    AudioBitrate {
        channels: 8,
        bits_per_second: 450_000,
    },
];

#[derive(Debug, Clone, Copy)]
struct AudioBitrate {
    channels: u8,
    bits_per_second: u32,
}

// moonlight reads channels, streams, coupled streams and the mapping as one digit each, with lfe moved last
fn surround_params(channels: u8) -> String {
    let opus = ChannelMapping::default_for(channels).expect("opus maps up to eight channels");
    let mut mapping = opus.table;
    mapping[3..].rotate_left(1);
    let digits: String = [channels, opus.streams, opus.coupled_streams]
        .into_iter()
        .chain(mapping)
        .map(|digit| char::from(b'0' + digit))
        .collect();
    format!("a=fmtp:97 surround-params={digits}\r\n")
}

#[derive(Debug)]
pub(crate) struct RtspRequest {
    pub(crate) method: String,
    pub(crate) target: String,
    sequence: String,
    pub(crate) body: Vec<u8>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn head_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4)
}

pub(crate) async fn read_request<S: AsyncRead + Unpin>(stream: &mut S) -> io::Result<RtspRequest> {
    let mut buffer = Vec::with_capacity(READ_CHUNK);
    let mut chunk = [0u8; READ_CHUNK];
    let head_length = loop {
        if let Some(end) = head_end(&buffer) {
            break end;
        }
        if buffer.len() > HEAD_LIMIT {
            return Err(invalid("RTSP request head too long"));
        }
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buffer.extend_from_slice(&chunk[..read]);
    };
    let mut body = buffer.split_off(head_length);
    let head = std::str::from_utf8(&buffer).map_err(|_| invalid("RTSP head is not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split(' ');
    let (Some(method), Some(target), Some("RTSP/1.0")) = (
        request_line.next(),
        request_line.next(),
        request_line.next(),
    ) else {
        return Err(invalid("malformed RTSP request line"));
    };
    let headers: HashMap<String, String> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let sequence = headers
        .get("cseq")
        .ok_or_else(|| invalid("RTSP request has no CSeq"))?
        .clone();
    let body_length = match headers.get("content-length") {
        Some(length) => length
            .parse::<usize>()
            .map_err(|_| invalid("bad Content-Length"))?,
        None => 0,
    };
    if body_length > BODY_LIMIT {
        return Err(invalid("RTSP request body too long"));
    }
    while body.len() < body_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(body_length);
    Ok(RtspRequest {
        method: method.to_string(),
        target: target.to_string(),
        sequence,
        body,
    })
}

#[derive(Debug)]
pub(crate) struct RtspReply {
    status: &'static str,
    headers: Vec<(&'static str, String)>,
    body: String,
}

impl RtspReply {
    pub(crate) fn ok() -> Self {
        Self {
            status: "200 OK",
            headers: Vec::new(),
            body: String::new(),
        }
    }

    pub(crate) fn bad_request() -> Self {
        Self {
            status: "400 Bad Request",
            ..Self::ok()
        }
    }

    pub(crate) fn not_found() -> Self {
        Self {
            status: "404 Not Found",
            ..Self::ok()
        }
    }

    pub(crate) fn description(encoder: StreamEncoder) -> Self {
        let surround: String = AUDIO_BITRATES
            .iter()
            .filter(|bitrate| bitrate.channels > 2)
            .map(|bitrate| surround_params(bitrate.channels))
            .collect();
        // moonlight sends touch and pen events only to hosts that set this sunshine feature flag
        const PEN_TOUCH_EVENTS: u32 = 0x01;
        let supported = ENCRYPT_CONTROL | ENCRYPT_VIDEO | ENCRYPT_AUDIO;
        let requested = ENCRYPT_VIDEO | ENCRYPT_AUDIO;
        // nvenc invalidates references, the media foundation encoders answer an invalidation with a key frame
        let reference_invalidation = match encoder {
            StreamEncoder::Nvenc => "a=x-nv-video[0].refPicInvalidation:1\r\n",
            #[cfg(target_os = "windows")]
            StreamEncoder::MediaFoundation => "",
        };
        Self {
            body: format!(
                "a=x-ss-general.featureFlags:{PEN_TOUCH_EVENTS}\r\n\
                 a=x-ss-general.encryptionSupported:{supported}\r\n\
                 a=x-ss-general.encryptionRequested:{requested}\r\n\
                 {reference_invalidation}\
                 {surround}\
                 {HEVC_MARKER}\r\n"
            ),
            ..Self::ok()
        }
    }

    pub(crate) fn setup(server_port: u16) -> Self {
        Self {
            headers: Vec::from([
                ("Session", SESSION_HEADER.to_string()),
                ("Transport", format!("server_port={server_port}")),
            ]),
            ..Self::ok()
        }
    }
}

pub(crate) async fn write_reply<S: AsyncWrite + Unpin>(
    stream: &mut S,
    request: &RtspRequest,
    reply: RtspReply,
) -> io::Result<()> {
    let mut head = format!(
        "RTSP/1.0 {}\r\nCSeq: {}\r\n",
        reply.status, request.sequence
    );
    for (name, value) in &reply.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!("Content-Length: {}\r\n\r\n", reply.body.len()));
    stream.write_all(head.as_bytes()).await?;
    stream.write_all(reply.body.as_bytes()).await?;
    stream.shutdown().await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VideoCodec {
    H264,
    Hevc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamSettings {
    pub(crate) codec: VideoCodec,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) frames_per_second: u32,
    pub(crate) packet_size: u32,
    pub(crate) bitrate_kilobits: u32,
    pub(crate) minimum_fec_packets: u32,
    // moonlight sends 0, the host's choice, only when its decoder handles any count
    pub(crate) reference_frames: u32,
    pub(crate) encrypt_video: bool,
    pub(crate) audio_packet_duration_ms: u32,
    pub(crate) encrypt_audio: bool,
    pub(crate) audio_channels: u8,
    pub(crate) audio_bits_per_second: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnnounceError {
    Missing(&'static str),
    Unparsable(&'static str),
    UnsupportedCodec(u32),
    UnsupportedChannels(u32),
}

impl std::fmt::Display for AnnounceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "ANNOUNCE has no {name}"),
            Self::Unparsable(name) => write!(f, "ANNOUNCE has a bad {name}"),
            Self::UnsupportedCodec(format) => {
                write!(f, "ANNOUNCE asks for bitstream format {format}")
            }
            Self::UnsupportedChannels(channels) => {
                write!(
                    f,
                    "ANNOUNCE asks for {channels} audio channels, and the host sends stereo, 5.1 or 7.1"
                )
            }
        }
    }
}

pub(crate) fn parse_announce(body: &[u8]) -> Result<StreamSettings, AnnounceError> {
    let text = String::from_utf8_lossy(body);
    let attributes: HashMap<&str, &str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("a="))
        .filter_map(|attribute| attribute.split_once(':'))
        .map(|(name, value)| (name.trim(), value.trim()))
        .collect();
    let number = |name: &'static str| -> Result<u32, AnnounceError> {
        attributes
            .get(name)
            .ok_or(AnnounceError::Missing(name))?
            .parse()
            .map_err(|_| AnnounceError::Unparsable(name))
    };
    let codec = match number("x-nv-vqos[0].bitStreamFormat")? {
        BITSTREAM_FORMAT_H264 => VideoCodec::H264,
        BITSTREAM_FORMAT_HEVC => VideoCodec::Hevc,
        other => return Err(AnnounceError::UnsupportedCodec(other)),
    };
    let channels = number("x-nv-audio.surround.numChannels")?;
    let audio = AUDIO_BITRATES
        .into_iter()
        .find(|bitrate| u32::from(bitrate.channels) == channels)
        .ok_or(AnnounceError::UnsupportedChannels(channels))?;
    let encryption = match attributes.get("x-ss-general.encryptionEnabled") {
        Some(flags) => flags
            .parse::<u32>()
            .map_err(|_| AnnounceError::Unparsable("x-ss-general.encryptionEnabled"))?,
        None => 0,
    };
    Ok(StreamSettings {
        codec,
        width: number("x-nv-video[0].clientViewportWd")?,
        height: number("x-nv-video[0].clientViewportHt")?,
        frames_per_second: number("x-nv-video[0].maxFPS")?,
        packet_size: number("x-nv-video[0].packetSize")?,
        bitrate_kilobits: number("x-ml-video.configuredBitrateKbps")?,
        minimum_fec_packets: number("x-nv-vqos[0].fec.minRequiredFecPackets")?,
        reference_frames: number("x-nv-video[0].maxNumReferenceFrames")?,
        encrypt_video: encryption & ENCRYPT_VIDEO != 0,
        audio_packet_duration_ms: number("x-nv-aqos.packetDuration")?,
        encrypt_audio: encryption & ENCRYPT_AUDIO != 0,
        audio_channels: audio.channels,
        audio_bits_per_second: audio.bits_per_second,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_moonlight_setup_reads_its_target_and_sequence() {
        let wire = b"SETUP streamid=video/0/0 RTSP/1.0\r\nCSeq: 3\r\nX-GS-ClientVersion: 14\r\nHost: 10.0.0.2\r\n\r\n";
        let request = read_request(&mut &wire[..]).await.expect("parsed");
        assert_eq!(request.method, "SETUP");
        assert_eq!(request.target, "streamid=video/0/0");
        let mut reply = Vec::new();
        write_reply(&mut reply, &request, RtspReply::setup(47998))
            .await
            .unwrap();
        let reply = String::from_utf8(reply).unwrap();
        assert!(reply.starts_with("RTSP/1.0 200 OK\r\nCSeq: 3\r\n"));
        assert!(reply.contains("Session: G2G;timeout = 90\r\n"));
        assert!(reply.contains("Transport: server_port=47998\r\n"));
    }

    #[tokio::test]
    async fn an_announce_body_is_read_to_its_length() {
        let body = "v=0\r\na=x-nv-video[0].maxFPS:60 \r\n";
        let wire = format!(
            "ANNOUNCE streamid=control/13/0 RTSP/1.0\r\nCSeq: 6\r\nContent-length: {}\r\n\r\n{body}",
            body.len()
        );
        let request = read_request(&mut wire.as_bytes()).await.expect("parsed");
        assert_eq!(request.body, body.as_bytes());
    }

    #[test]
    fn the_description_asks_moonlight_for_touch_and_pen() {
        let body = RtspReply::description(StreamEncoder::Nvenc).body;
        assert!(body.contains("a=x-ss-general.featureFlags:1\r\n"), "{body}");
    }

    #[test]
    fn the_description_gives_moonlight_the_5_1_layout_opusenc_encodes() {
        let body = RtspReply::description(StreamEncoder::Nvenc).body;
        // the layout sunshine sends for 5.1, which libopus mapping family 1 produces too
        assert!(
            body.contains("a=fmtp:97 surround-params=642014235\r\n"),
            "{body}"
        );
    }

    #[test]
    fn the_description_offers_reference_invalidation_with_nvenc() {
        let body = RtspReply::description(StreamEncoder::Nvenc).body;
        assert!(
            body.contains("a=x-nv-video[0].refPicInvalidation:1\r\n"),
            "{body}"
        );
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn the_description_offers_no_reference_invalidation_with_media_foundation() {
        let body = RtspReply::description(StreamEncoder::MediaFoundation).body;
        assert!(!body.contains("refPicInvalidation"), "{body}");
    }

    #[test]
    fn an_announce_gives_the_stream_settings() {
        let sdp = "v=0\r\n\
            a=x-ss-general.encryptionEnabled:7\r\n\
            a=x-nv-video[0].clientViewportWd:1280 \r\n\
            a=x-nv-video[0].clientViewportHt:720 \r\n\
            a=x-nv-video[0].maxFPS:60 \r\n\
            a=x-nv-video[0].packetSize:1360 \r\n\
            a=x-ml-video.configuredBitrateKbps:20000 \r\n\
            a=x-nv-vqos[0].fec.minRequiredFecPackets:2 \r\n\
            a=x-nv-vqos[0].bitStreamFormat:1 \r\n\
            a=x-nv-video[0].maxNumReferenceFrames:1 \r\n\
            a=x-nv-aqos.packetDuration:5 \r\n\
            a=x-nv-audio.surround.numChannels:2 \r\n";
        assert_eq!(
            parse_announce(sdp.as_bytes()),
            Ok(StreamSettings {
                codec: VideoCodec::Hevc,
                width: 1280,
                height: 720,
                frames_per_second: 60,
                packet_size: 1360,
                bitrate_kilobits: 20_000,
                minimum_fec_packets: 2,
                reference_frames: 1,
                encrypt_video: true,
                audio_packet_duration_ms: 5,
                encrypt_audio: true,
                audio_channels: 2,
                audio_bits_per_second: 96_000,
            })
        );
        assert_eq!(
            parse_announce(b"a=x-nv-vqos[0].bitStreamFormat:2\r\n"),
            Err(AnnounceError::UnsupportedCodec(2))
        );
        let surround = parse_announce(sdp.replace("numChannels:2", "numChannels:6").as_bytes());
        assert_eq!(
            surround.map(|settings| (settings.audio_channels, settings.audio_bits_per_second)),
            Ok((6, 256_000))
        );
        let quadraphonic = sdp.replace("numChannels:2", "numChannels:4");
        assert_eq!(
            parse_announce(quadraphonic.as_bytes()),
            Err(AnnounceError::UnsupportedChannels(4))
        );
    }
}
