//! `data:` URI source (`dataurisrc`). Turns the payload carried inside a URI
//! (RFC 2397) into a byte stream, so a small media object that arrives in a
//! manifest, a config file or a launch line plays without a file on disk.
//!
//! The media type comes from the payload's own header
//! ([`crate::typefind::sniff_caps`]), not from the URI's declared MIME type: the
//! declaration is written by whoever built the URI and a wrong one would plug
//! the wrong demuxer, while the bytes are the thing being played. A payload
//! nothing recognizes is an untyped `ByteStream{Raw}`.
//!
//! Both encodings are read: `;base64` and the default percent-escaped text.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

/// What every `data:` URI opens with, and the marker separating the payload.
const SCHEME: &str = "data:";
const PAYLOAD_SEPARATOR: char = ',';
/// The parameter naming the base64 encoding of the payload.
const BASE64_PARAMETER: &str = ";base64";

/// Ceiling on a decoded payload. A URI is meant for a small object, and the
/// whole thing is held in memory at once.
const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Bytes per buffer, so a payload larger than one read still arrives as a
/// stream (and a parser downstream is exercised the same way a file is).
const DEFAULT_BLOCKSIZE: usize = 64 * 1024;

/// The same value as declared text, for `gst-inspect`.
const DEFAULT_BLOCKSIZE_TEXT: &str = "65536";

static DATAURISRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("uri", PropKind::Str, "the `data:` URI to play"),
    PropertySpec::new("blocksize", PropKind::Uint, "bytes to push per buffer")
        .with_default(DEFAULT_BLOCKSIZE_TEXT),
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::dataurisrc::DataUriSrc;
///
/// // gst-launch equivalent: dataurisrc uri="data:audio/mpeg;base64,SUQz..." ! decodebin
/// let src = DataUriSrc::new("data:text/plain,hello");
/// ```
#[derive(Debug)]
pub struct DataUriSrc {
    uri: String,
    blocksize: usize,
    /// The decoded payload and its type, resolved at negotiation (the caps are a
    /// property of the payload, so nothing can be declared before it is read).
    payload: Option<(Vec<u8>, Caps)>,
    configured: bool,
}

impl DataUriSrc {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            blocksize: DEFAULT_BLOCKSIZE,
            payload: None,
            configured: false,
        }
    }

    pub fn with_blocksize(mut self, bytes: usize) -> Self {
        if bytes > 0 {
            self.blocksize = bytes;
        }
        self
    }

    /// Decode the URI's payload and type it. `CapsMismatch` for anything that is
    /// not a readable `data:` URI, or a payload past
    /// [`MAX_PAYLOAD_BYTES`](MAX_PAYLOAD_BYTES).
    fn resolve(&self) -> Result<(Vec<u8>, Caps), G2gError> {
        let bytes = decode_data_uri(&self.uri).ok_or(G2gError::CapsMismatch)?;
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(G2gError::CapsMismatch);
        }
        let caps = crate::typefind::sniff_caps(&bytes).unwrap_or(Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        });
        Ok((bytes, caps))
    }

    fn caps(&self) -> Result<Caps, G2gError> {
        match &self.payload {
            Some((_, caps)) => Ok(caps.clone()),
            None => self.resolve().map(|(_, caps)| caps),
        }
    }
}

/// The payload bytes of a `data:` URI, or `None` when it is not one.
fn decode_data_uri(uri: &str) -> Option<Vec<u8>> {
    let rest = uri.strip_prefix(SCHEME)?;
    let (header, payload) = rest.split_once(PAYLOAD_SEPARATOR)?;
    if header.to_ascii_lowercase().ends_with(BASE64_PARAMETER) {
        return crate::sdp::base64_decode(payload.trim());
    }
    percent_decode(payload)
}

/// Undo the `%XX` escaping of a URI's text payload. `None` on a truncated or
/// non-hexadecimal escape, so a malformed URI fails rather than playing bytes
/// nobody wrote.
fn percent_decode(text: &str) -> Option<Vec<u8>> {
    const ESCAPE: u8 = b'%';
    const ESCAPE_DIGITS: usize = 2;
    let raw = text.as_bytes();
    let mut out = Vec::with_capacity(raw.len());
    let mut at = 0;
    while at < raw.len() {
        if raw[at] != ESCAPE {
            out.push(raw[at]);
            at += 1;
            continue;
        }
        let digits = raw.get(at + 1..at + 1 + ESCAPE_DIGITS)?;
        let text = core::str::from_utf8(digits).ok()?;
        out.push(u8::from_str_radix(text, 16).ok()?);
        at += 1 + ESCAPE_DIGITS;
    }
    Some(out)
}

impl SourceLoop for DataUriSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let caps = self.caps();
        core::future::ready(caps.map(|caps| CapsConstraint::Produces(CapsSet::one(caps))))
    }

    /// The payload is decoded here, so `run` only pushes. The solved caps are a
    /// fixation of what the payload sniffed as (a still's placeholder geometry
    /// gets fixed downstream), so they are intersected rather than compared.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (bytes, caps) = self.resolve()?;
        let solved = absolute_caps.intersect(&caps)?;
        self.payload = Some((bytes, solved));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// The payload is inside the URI, so its type is known without I/O: the
    /// `decodebin` parser reads this to pick the chain for what the URI carries.
    fn configured_output_caps(&self) -> Option<Caps> {
        self.caps().ok()
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let Some((bytes, _)) = &self.payload else {
                return Err(G2gError::NotConfigured);
            };
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut sequence = 0u64;
            for chunk in bytes.chunks(self.blocksize) {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        chunk.to_vec().into_boxed_slice(),
                    )),
                    FrameTiming::default(),
                    sequence,
                );
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DATAURISRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "data: URI source",
            "Source/Network",
            "Plays the payload carried inside a data: URI",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "uri" => {
                self.uri = value.as_str().ok_or(PropError::Type)?.into();
                // The payload and its type belong to the old URI.
                self.payload = None;
            }
            "blocksize" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes == 0 || bytes > u64::from(u32::MAX) {
                    return Err(PropError::Value);
                }
                self.blocksize = bytes as usize;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "uri" => Some(PropValue::Str(self.uri.clone())),
            "blocksize" => Some(PropValue::Uint(self.blocksize as u64)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    struct CollectSink {
        payload: Vec<u8>,
        buffers: usize,
        eos: bool,
    }

    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::DataFrame(f) => {
                    self.payload
                        .extend_from_slice(f.domain.as_system_slice().expect("system"));
                    self.buffers += 1;
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[test]
    fn reads_both_payload_encodings() {
        assert_eq!(
            decode_data_uri("data:text/plain;base64,aGk="),
            Some(vec![b'h', b'i'])
        );
        // The parameter is matched case-insensitively, as a URI's are.
        assert_eq!(
            decode_data_uri("data:text/plain;BASE64,aGk="),
            Some(vec![b'h', b'i'])
        );
        assert_eq!(
            decode_data_uri("data:,a%20b"),
            Some(b"a b".to_vec()),
            "the default encoding is percent-escaped text"
        );
        assert_eq!(
            decode_data_uri("data:text/plain,plain"),
            Some(b"plain".to_vec())
        );
    }

    #[test]
    fn refuses_a_malformed_uri() {
        assert_eq!(decode_data_uri("http://host/clip.ts"), None);
        assert_eq!(
            decode_data_uri("data:text/plain"),
            None,
            "no payload marker"
        );
        assert_eq!(decode_data_uri("data:,%2"), None, "a truncated escape");
        assert_eq!(decode_data_uri("data:,%zz"), None, "a non-hex escape");
        assert_eq!(decode_data_uri("data:audio/mpeg;base64,not base64!"), None);
    }

    #[tokio::test]
    async fn an_untyped_payload_is_a_raw_byte_stream() {
        let mut src = DataUriSrc::new("data:,hello");
        let caps = src.caps().expect("the payload decodes");
        assert_eq!(
            caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw
            },
            "nothing types five letters of text"
        );
        src.configure_pipeline(&caps).expect("its own caps");
        let mut out = CollectSink {
            payload: Vec::new(),
            buffers: 0,
            eos: false,
        };
        src.run(&mut out).await.expect("the payload plays");
        assert_eq!(out.payload, b"hello".to_vec());
        assert!(out.eos);
    }

    #[tokio::test]
    async fn blocksize_splits_a_larger_payload() {
        // A base64 payload of 12 bytes, pushed 5 at a time.
        let mut src = DataUriSrc::new("data:application/octet-stream;base64,YWJjZGVmZ2hpamts")
            .with_blocksize(5);
        let caps = src.caps().expect("the payload decodes");
        src.configure_pipeline(&caps).expect("its own caps");
        let mut out = CollectSink {
            payload: Vec::new(),
            buffers: 0,
            eos: false,
        };
        src.run(&mut out).await.expect("the payload plays");
        assert_eq!(out.payload, b"abcdefghijkl".to_vec());
        assert_eq!(out.buffers, 3, "5 + 5 + 2 bytes");
    }

    #[test]
    fn a_typed_payload_takes_its_own_media_type() {
        // A PNG signature: sniffing types it as a still image, whatever the URI
        // declares.
        let png = "data:application/octet-stream;base64,iVBORw0KGgo=";
        let src = DataUriSrc::new(png);
        assert_eq!(
            src.caps(),
            Ok(crate::typefind::still_image_caps(g2g_core::VideoCodec::Png)),
            "the bytes decide the type, not the declaration"
        );
    }
}
