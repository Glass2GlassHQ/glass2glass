// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! What the raw byte-stream carriers (TCP, SRT, RIST, raw UDP) share: the frame shape
//! a received chunk takes, the MPEG-TS packet geometry that decides how a
//! datagram is cut, and the container list a wire sink can carry.

use alloc::vec::Vec;

use g2g_core::{ByteStreamEncoding, Caps};
#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
use g2g_core::{PropError, PropKind, PropValue, PropertySpec};

#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
pub(crate) const BYTESTREAM_FORMAT_PROPERTY: PropertySpec = PropertySpec::new(
    "bytestream-format",
    PropKind::Str,
    "container of the byte stream: mpegts | matroska | ogg | flv | mp4",
)
.with_default("mpegts");

#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
pub(crate) const CAPS_PROPERTY: PropertySpec = PropertySpec::new(
    "caps",
    PropKind::Str,
    "caps to declare instead, gst-launch syntax: e.g. video/x-raw,format=I420,width=320,height=240",
);

#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
#[derive(Debug, Clone)]
pub(crate) struct DeclaredSourceCaps {
    caps: Caps,
    // the `caps` property as written, so it reads back unchanged
    caps_text: Option<alloc::string::String>,
}

#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
impl Default for DeclaredSourceCaps {
    fn default() -> Self {
        Self::of(Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
    }
}

#[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
impl DeclaredSourceCaps {
    pub(crate) fn of(caps: Caps) -> Self {
        Self {
            caps,
            caps_text: None,
        }
    }

    pub(crate) fn caps(&self) -> &Caps {
        &self.caps
    }

    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        let text = || value.as_str().ok_or(PropError::Type);
        let parsed = match name {
            "bytestream-format" => text().and_then(|text| {
                let encoding = crate::filesrc::encoding_from_str(text).ok_or(PropError::Value)?;
                Ok((Caps::ByteStream { encoding }, None))
            }),
            "caps" => text().and_then(|text| {
                let caps = crate::capsfilter::parse_caps(text).ok_or(PropError::Value)?;
                Ok((caps, Some(text.into())))
            }),
            _ => return None,
        };
        Some(parsed.map(|(caps, caps_text)| {
            self.caps = caps;
            self.caps_text = caps_text;
        }))
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match (name, &self.caps) {
            ("bytestream-format", Caps::ByteStream { encoding }) => Some(PropValue::Str(
                crate::filesrc::encoding_to_str(*encoding).into(),
            )),
            ("caps", _) => self.caps_text.clone().map(PropValue::Str),
            _ => None,
        }
    }
}

/// MPEG-TS packet size (ISO/IEC 13818-1).
pub const TS_PACKET_SIZE: usize = 188;
/// TS packets per datagram in the 1316-byte payload SRT and the broadcast UDP
/// senders use: the largest whole number that fits under a 1500-byte MTU.
pub const TS_PACKETS_PER_DATAGRAM: usize = 7;
/// That payload, 1316 bytes.
pub const TS_DATAGRAM_PAYLOAD: usize = TS_PACKET_SIZE * TS_PACKETS_PER_DATAGRAM;

/// Formats a raw wire sink carries. `ByteStreamEncoding` is `#[non_exhaustive]`
/// from another crate, so this is the sink's own enumeration of it: a variant
/// added later is simply not advertised until it is listed here.
pub const CARRIED_ENCODINGS: [ByteStreamEncoding; 18] = [
    ByteStreamEncoding::MpegTs,
    ByteStreamEncoding::Matroska,
    ByteStreamEncoding::Ogg,
    ByteStreamEncoding::Flv,
    ByteStreamEncoding::IsoBmff,
    ByteStreamEncoding::Mp4,
    ByteStreamEncoding::Ivf,
    ByteStreamEncoding::MpegPs,
    ByteStreamEncoding::Wav,
    ByteStreamEncoding::Aiff,
    ByteStreamEncoding::Au,
    ByteStreamEncoding::Avi,
    ByteStreamEncoding::Rtp,
    ByteStreamEncoding::Srtp,
    ByteStreamEncoding::Rtcp,
    ByteStreamEncoding::Srtcp,
    ByteStreamEncoding::Dtls,
    ByteStreamEncoding::Datagram,
];

/// Whether each frame is one complete network packet and must remain one
/// datagram.
pub fn is_packet_encoding(encoding: ByteStreamEncoding) -> bool {
    matches!(
        encoding,
        ByteStreamEncoding::Rtp
            | ByteStreamEncoding::Srtp
            | ByteStreamEncoding::Rtcp
            | ByteStreamEncoding::Srtcp
            | ByteStreamEncoding::Dtls
            | ByteStreamEncoding::Datagram
    )
}

/// The MPEG-TS byte stream the TS carriers (SRT, RIST) take and produce.
pub fn mpegts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// [`CARRIED_ENCODINGS`] as caps, for a pad template or an `Accepts` set.
pub fn carried_bytestream_caps() -> Vec<Caps> {
    CARRIED_ENCODINGS
        .iter()
        .map(|&encoding| Caps::ByteStream { encoding })
        .collect()
}

/// Bytes per datagram for `encoding` under a `max_payload` cap. An MPEG-TS
/// stream is cut on whole 188-byte packets, so a receiver never has to
/// reassemble a split one; with the 1400-byte default that is
/// [`TS_DATAGRAM_PAYLOAD`].
pub fn datagram_chunk(encoding: ByteStreamEncoding, max_payload: usize) -> usize {
    if encoding != ByteStreamEncoding::MpegTs {
        return max_payload.max(1);
    }
    (max_payload / TS_PACKET_SIZE).max(1) * TS_PACKET_SIZE
}

/// A frame carrying `bytes`, stamped and sequenced the way `FileSrc` stamps a
/// file chunk, so every byte source looks identical downstream.
#[cfg(any(feature = "tcp", feature = "udp-ingress", feature = "rist"))]
pub(crate) fn byte_frame(bytes: Vec<u8>, sequence: u64) -> g2g_core::frame::Frame {
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain};

    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "tcp", all(target_os = "linux", feature = "shm")))]
    #[test]
    fn caps_and_bytestream_format_each_replace_the_declared_caps() {
        let gdp = "application/x-gdp,original-caps=(caps)[meta/x-klv]";
        let mut declared = DeclaredSourceCaps::default();
        let set = |declared: &mut DeclaredSourceCaps, name: &str, text: &str| {
            declared
                .set_property(name, &PropValue::Str(text.into()))
                .expect("a declared-caps property")
        };
        set(&mut declared, "caps", gdp).expect("gdp caps parse");
        assert!(matches!(declared.caps(), Caps::Gdp { original: Some(_) }));
        assert_eq!(
            declared.get_property("caps"),
            Some(PropValue::Str(gdp.into()))
        );
        assert_eq!(declared.get_property("bytestream-format"), None);
        assert_eq!(
            set(&mut declared, "caps", "video/x-raw,format={NV12,I420}"),
            Err(PropError::Value)
        );
        set(&mut declared, "bytestream-format", "matroska").expect("a container");
        assert_eq!(
            *declared.caps(),
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska
            }
        );
        assert_eq!(declared.get_property("caps"), None);
        assert_eq!(declared.set_property("port", &PropValue::Uint(1)), None);
    }

    #[test]
    fn ts_datagrams_hold_whole_packets() {
        // The 1400-byte udpsink default has to land on the SRT payload size.
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 1400),
            TS_DATAGRAM_PAYLOAD
        );
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 1400) % TS_PACKET_SIZE,
            0
        );
        // Below one packet the chunk still carries a whole packet: a split TS
        // packet is unreadable to the receiver either way.
        assert_eq!(
            datagram_chunk(ByteStreamEncoding::MpegTs, 100),
            TS_PACKET_SIZE
        );
    }

    #[test]
    fn other_containers_use_the_whole_payload() {
        assert_eq!(datagram_chunk(ByteStreamEncoding::Matroska, 1400), 1400);
    }

    #[test]
    fn packet_formats_are_carried_without_container_conversion() {
        for encoding in [
            ByteStreamEncoding::Rtp,
            ByteStreamEncoding::Srtp,
            ByteStreamEncoding::Rtcp,
            ByteStreamEncoding::Srtcp,
            ByteStreamEncoding::Dtls,
        ] {
            assert!(CARRIED_ENCODINGS.contains(&encoding));
            assert!(is_packet_encoding(encoding));
        }
    }
}
