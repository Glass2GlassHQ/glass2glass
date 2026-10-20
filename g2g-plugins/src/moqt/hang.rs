use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use g2g_core::VideoCodec;
use serde::Deserialize;

use super::catalog::MAX_CATALOG_BYTES;
use super::coding::Reader;
use crate::annexb;

pub const CATALOG_TRACK: &str = "catalog.json";

const BROADCAST_SUFFIX: &str = ".hang";

// moq-hang-04, the newest draft whose catalog this reader understands
const NEWEST_CATALOG_VERSION: u64 = 4;

const LEGACY_CONTAINER: &str = "legacy";

const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;

pub const VIDEO_CODECS: [VideoCodec; 2] = [VideoCodec::H264, VideoCodec::H265];

pub fn is_hang_broadcast(namespace: &str) -> bool {
    namespace.ends_with(BROADCAST_SUFFIX)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HangError {
    CatalogTooLarge,
    CatalogJson,
    CatalogVersion(u64),
    NoVideoRendition,
    UnknownRendition(String),
    OtherBroadcast(String),
    Container(String),
    Codec(String),
    Description,
    Frame,
}

impl fmt::Display for HangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CatalogTooLarge => write!(f, "the catalog exceeds {MAX_CATALOG_BYTES} bytes"),
            Self::CatalogJson => write!(f, "the catalog is not hang JSON"),
            Self::CatalogVersion(v) => write!(
                f,
                "catalog version {v} is newer than {NEWEST_CATALOG_VERSION}"
            ),
            Self::NoVideoRendition => write!(f, "the catalog has no decodable video rendition"),
            Self::UnknownRendition(name) => write!(f, "the catalog has no video rendition {name}"),
            Self::OtherBroadcast(name) => {
                write!(f, "rendition {name} lives in another broadcast")
            }
            Self::Container(kind) => write!(f, "container {kind} is not the legacy container"),
            Self::Codec(codec) => write!(f, "codec {codec} is not H.264 or H.265"),
            Self::Description => write!(f, "the rendition description is not a decoder config"),
            Self::Frame => write!(f, "a media object has no timestamp"),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Catalog {
    version: Option<u64>,
    video: Option<VideoSection>,
}

#[derive(Debug, Deserialize)]
struct VideoSection {
    renditions: BTreeMap<String, VideoRendition>,
}

#[derive(Debug, Deserialize)]
struct VideoRendition {
    codec: String,
    description: Option<String>,
    container: Option<Container>,
    track: Option<String>,
    broadcast: Option<String>,
    enabled: Option<bool>,
    preference: Option<i32>,
    bitrate: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Container {
    kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HangVideoTrack {
    pub track: String,
    pub codec: VideoCodec,
    // the hex `description` decoded: present when the frames are length-prefixed
    config: Option<annexb::DecoderConfigRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HangAccessUnit {
    pub pts_ns: u64,
    pub annexb: Vec<u8>,
    pub keyframe: bool,
}

// with no name, hang's rule: the highest preference among enabled renditions, then bitrate
pub fn select_video_track(catalog: &[u8], wanted: &str) -> Result<HangVideoTrack, HangError> {
    if catalog.len() > MAX_CATALOG_BYTES {
        return Err(HangError::CatalogTooLarge);
    }
    let catalog: Catalog = serde_json::from_slice(catalog).map_err(|_| HangError::CatalogJson)?;
    if let Some(version) = catalog.version.filter(|v| *v > NEWEST_CATALOG_VERSION) {
        return Err(HangError::CatalogVersion(version));
    }
    let renditions = catalog.video.ok_or(HangError::NoVideoRendition)?.renditions;
    if !wanted.is_empty() {
        let (id, rendition) = renditions
            .iter()
            .find(|(id, r)| *id == wanted || r.track.as_deref() == Some(wanted))
            .ok_or_else(|| HangError::UnknownRendition(String::from(wanted)))?;
        return video_track(id, rendition);
    }
    renditions
        .iter()
        .filter(|(_, r)| r.enabled != Some(false))
        .filter_map(|(id, r)| Some((r, video_track(id, r).ok()?)))
        .max_by_key(|(r, _)| (r.preference.unwrap_or(0), r.bitrate.unwrap_or(0)))
        .map(|(_, track)| track)
        .ok_or(HangError::NoVideoRendition)
}

fn video_track(id: &str, rendition: &VideoRendition) -> Result<HangVideoTrack, HangError> {
    if let Some(broadcast) = rendition.broadcast.as_deref().filter(|b| !b.is_empty()) {
        return Err(HangError::OtherBroadcast(String::from(broadcast)));
    }
    let kind = rendition
        .container
        .as_ref()
        .map_or(LEGACY_CONTAINER, |c| c.kind.as_str());
    if kind != LEGACY_CONTAINER {
        return Err(HangError::Container(String::from(kind)));
    }
    let codec = match rendition.codec.split('.').next() {
        Some("avc1" | "avc3") => VideoCodec::H264,
        Some("hvc1" | "hev1") => VideoCodec::H265,
        _ => return Err(HangError::Codec(rendition.codec.clone())),
    };
    let config = rendition
        .description
        .as_deref()
        .map(|hex| decoder_config(codec, hex))
        .transpose()?;
    Ok(HangVideoTrack {
        track: rendition.track.clone().unwrap_or_else(|| String::from(id)),
        codec,
        config,
    })
}

fn decoder_config(codec: VideoCodec, hex: &str) -> Result<annexb::DecoderConfigRecord, HangError> {
    let record = crate::textencoding::hex_decode(hex).ok_or(HangError::Description)?;
    let parsed = match codec {
        VideoCodec::H265 => annexb::parse_hvcc(&record),
        _ => annexb::parse_avcc(&record),
    };
    parsed.map_err(|_| HangError::Description)
}

// an empty payload only marks where the previous frame ends
pub fn access_unit(
    track: &HangVideoTrack,
    object: &[u8],
) -> Result<Option<HangAccessUnit>, HangError> {
    let mut reader = Reader::new(object);
    let timestamp_us = reader.varint().map_err(|_| HangError::Frame)?;
    let payload = reader.rest();
    if payload.is_empty() {
        return Ok(None);
    }
    let mut annexb = match &track.config {
        Some(config) => annexb::length_prefixed_to_annexb(payload, config.nal_length_size)
            .ok_or(HangError::Frame)?,
        None => payload.to_vec(),
    };
    let keyframe = annexb::au_is_keyframe(track.codec, &annexb);
    if let Some(config) = track.config.as_ref().filter(|_| keyframe) {
        if !annexb::starts_with_param_set(&annexb, track.codec) {
            annexb = annexb::prepend_param_sets(&annexb, &config.parameter_sets, track.codec);
        }
    }
    Ok(Some(HangAccessUnit {
        pts_ns: timestamp_us.saturating_mul(NANOSECONDS_PER_MICROSECOND),
        annexb,
        keyframe,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_CATALOG: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/hang/catalog.json"
    ));

    fn catalog_with(renditions: &str) -> Vec<u8> {
        alloc::format!("{{\"video\":{{\"renditions\":{{{renditions}}}}}}}").into_bytes()
    }

    #[test]
    fn the_recorded_catalog_selects_its_h264_rendition() {
        let track = select_video_track(FIXTURE_CATALOG, "").unwrap();
        assert_eq!(track.track, "0.avc3");
        assert_eq!(track.codec, VideoCodec::H264);
        assert!(
            track.config.is_none(),
            "avc3 carries its parameter sets in band"
        );
    }

    #[test]
    fn selection_prefers_preference_then_bitrate_among_enabled_renditions() {
        let catalog = catalog_with(concat!(
            "\"hi\":{\"codec\":\"avc1.64001f\",\"bitrate\":4000,\"enabled\":false},",
            "\"mid\":{\"codec\":\"avc1.64001e\",\"bitrate\":2000},",
            "\"low\":{\"codec\":\"avc1.64001e\",\"bitrate\":1000},",
            "\"compat\":{\"codec\":\"avc1.64001e\",\"bitrate\":9000,\"preference\":-1},",
            "\"av1\":{\"codec\":\"av01.0.08M.08\",\"bitrate\":8000}"
        ));
        assert_eq!(select_video_track(&catalog, "").unwrap().track, "mid");
        assert_eq!(select_video_track(&catalog, "low").unwrap().track, "low");
    }

    #[test]
    fn a_track_field_names_the_subscription() {
        let catalog = catalog_with("\"720p\":{\"codec\":\"avc3.64001f\",\"track\":\"video-v2\"}");
        assert_eq!(select_video_track(&catalog, "").unwrap().track, "video-v2");
        assert_eq!(
            select_video_track(&catalog, "720p").unwrap().track,
            "video-v2"
        );
    }

    #[test]
    fn unreadable_renditions_are_refused() {
        let cmaf = catalog_with(
            "\"v\":{\"codec\":\"avc1.64001f\",\"container\":{\"kind\":\"cmaf\",\"init\":\"\"}}",
        );
        assert_eq!(
            select_video_track(&cmaf, "v"),
            Err(HangError::Container(String::from("cmaf")))
        );
        let elsewhere =
            catalog_with("\"v\":{\"codec\":\"avc3.64001f\",\"broadcast\":\"./source\"}");
        assert_eq!(
            select_video_track(&elsewhere, "v"),
            Err(HangError::OtherBroadcast(String::from("./source")))
        );
        let newer = b"{\"version\":5,\"video\":{\"renditions\":{}}}";
        assert_eq!(
            select_video_track(newer, ""),
            Err(HangError::CatalogVersion(5))
        );
        assert_eq!(
            select_video_track(b"[1,2]", ""),
            Err(HangError::CatalogJson)
        );
        assert_eq!(
            select_video_track(&catalog_with("\"v\":{\"codec\":\"avc3.64001f\"}"), "w"),
            Err(HangError::UnknownRendition(String::from("w")))
        );
    }

    #[test]
    fn a_description_turns_length_prefixed_frames_into_annexb_with_parameter_sets() {
        let sps = [0x67, 0x64, 0x00, 0x1e];
        let pps = [0x68, 0xee, 0x3c, 0x80];
        let idr = [0x65, 0x88, 0x84];
        let mut avcc = alloc::vec![0x01, 0x64, 0x00, 0x1e, 0xff, 0xe1, 0x00, sps.len() as u8];
        avcc.extend_from_slice(&sps);
        avcc.extend_from_slice(&[0x01, 0x00, pps.len() as u8]);
        avcc.extend_from_slice(&pps);
        let catalog = catalog_with(&alloc::format!(
            "\"v\":{{\"codec\":\"avc1.64001e\",\"description\":\"{}\"}}",
            crate::textencoding::hex_encode(&avcc)
        ));
        let track = select_video_track(&catalog, "").unwrap();

        let timestamp_us = 1_000_000u64;
        let mut object = Vec::new();
        super::super::coding::put_varint(&mut object, timestamp_us);
        object.extend_from_slice(&(idr.len() as u32).to_be_bytes());
        object.extend_from_slice(&idr);
        let unit = access_unit(&track, &object).unwrap().unwrap();

        let start = [0u8, 0, 0, 1];
        let expected: Vec<u8> = [&start[..], &sps, &start, &pps, &start, &idr].concat();
        assert_eq!(unit.annexb, expected);
        assert!(unit.keyframe);
        assert_eq!(unit.pts_ns, timestamp_us * NANOSECONDS_PER_MICROSECOND);
    }

    #[test]
    fn an_empty_payload_is_a_duration_marker_and_a_bare_object_is_refused() {
        let track = select_video_track(FIXTURE_CATALOG, "").unwrap();
        let mut marker = Vec::new();
        super::super::coding::put_varint(&mut marker, 42);
        assert_eq!(access_unit(&track, &marker), Ok(None));
        assert_eq!(access_unit(&track, &[]), Err(HangError::Frame));
    }
}
