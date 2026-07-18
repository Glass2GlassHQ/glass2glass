//! Multi-file source (`multifilesrc`). Reads a sequence of files named from a
//! printf-style `location` pattern (e.g. `img%05d.jpg`), emitting each whole file
//! as one `DataFrame`, until a file in the sequence is missing. The g2g analog of
//! GStreamer's `multifilesrc`, the canonical front of an image-sequence decode:
//! `multifilesrc location=img%05d.jpg ! mjpegdec ! ...`.
//!
//! Each file is one independently-decodable unit, so every frame is marked a
//! keyframe. The output media type defaults to Motion-JPEG (the common case); a
//! different sequence type is set at construction.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use std::fs::File;
use std::io::Read;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::filesink::io_err;

#[derive(Debug)]
pub struct MultiFileSrc {
    location: String,
    caps: Caps,
    start_index: i64,
    stop_index: i64,
    loop_seq: bool,
    configured: bool,
}

impl MultiFileSrc {
    /// A launch-registry `multifilesrc` defaulting to a Motion-JPEG sequence. The
    /// geometry is a fixable `Range` placeholder (never `Any`, which cannot
    /// fixate); the real per-image dimensions arrive from the decoder downstream.
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            caps: Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                width: Dim::Range {
                    min: 16,
                    max: 65535,
                },
                height: Dim::Range {
                    min: 16,
                    max: 65535,
                },
                framerate: Rate::Range {
                    min_q16: 1 << 16,
                    max_q16: 240 << 16,
                },
            },
            start_index: 0,
            // -1 means "until the first missing file".
            stop_index: -1,
            loop_seq: false,
            configured: false,
        }
    }

    /// Set the sequence's media type (e.g. a raw byte stream) explicitly.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }
}

impl SourceLoop for MultiFileSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.caps.clone(),
        ))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.caps.clone())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut sequence = 0u64;
            let mut index = self.start_index;
            loop {
                if self.stop_index >= 0 && index > self.stop_index {
                    if self.loop_seq && sequence > 0 {
                        index = self.start_index;
                        continue;
                    }
                    break;
                }
                let path = crate::multifilesink::expand(&self.location, index as u64);
                let mut file = match File::open(&path) {
                    Ok(f) => f,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // A gap ends the sequence (or restarts it when looping).
                        if self.loop_seq && sequence > 0 {
                            index = self.start_index;
                            continue;
                        }
                        break;
                    }
                    Err(e) => return Err(io_err(e)),
                };
                let mut buf = alloc::vec::Vec::new();
                file.read_to_end(&mut buf).map_err(io_err)?;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: FrameTiming {
                        keyframe: true,
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                index += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MULTIFILESRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Multi-file source",
            "Source/File",
            "Reads a sequence of files",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => self.location = value.as_str().ok_or(PropError::Type)?.into(),
            "start-index" => self.start_index = value.as_int().ok_or(PropError::Type)?,
            "stop-index" => self.stop_index = value.as_int().ok_or(PropError::Type)?,
            "loop" => self.loop_seq = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "start-index" => Some(PropValue::Int(self.start_index)),
            "stop-index" => Some(PropValue::Int(self.stop_index)),
            "loop" => Some(PropValue::Bool(self.loop_seq)),
            _ => None,
        }
    }
}

static MULTIFILESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "printf-style file pattern, e.g. img%05d.jpg",
    ),
    PropertySpec::new("start-index", PropKind::Int, "first index to read"),
    PropertySpec::new(
        "stop-index",
        PropKind::Int,
        "last index (-1 = until a file is missing)",
    ),
    PropertySpec::new("loop", PropKind::Bool, "restart the sequence at the end"),
];

#[cfg(test)]
mod tests {
    use super::*;

    struct CollectSink {
        frames: alloc::vec::Vec<alloc::vec::Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CollectSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Box::pin(async { Ok(g2g_core::PushOutcome::Accepted) })
        }
    }

    #[tokio::test]
    async fn reads_sequence_until_gap() {
        let dir = std::env::temp_dir();
        let pat = dir
            .join("g2g_mfsrc_%02d.bin")
            .to_string_lossy()
            .into_owned();
        std::fs::write(crate::multifilesink::expand(&pat, 0), b"one").unwrap();
        std::fs::write(crate::multifilesink::expand(&pat, 1), b"two").unwrap();
        // index 2 is missing -> the sequence ends after two frames.
        let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, 2));

        let mut src = MultiFileSrc::new(&pat);
        src.configure_pipeline(&src.caps.clone()).unwrap();
        let mut out = CollectSink {
            frames: alloc::vec::Vec::new(),
            eos: false,
        };
        let n = src.run(&mut out).await.unwrap();
        assert_eq!(n, 2);
        assert_eq!(out.frames, alloc::vec![b"one".to_vec(), b"two".to_vec()]);
        assert!(out.eos);
        let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, 0));
        let _ = std::fs::remove_file(crate::multifilesink::expand(&pat, 1));
    }
}
