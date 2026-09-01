//! Multi-file source (`multifilesrc`). Reads a sequence of files named from a
//! printf-style `location` pattern (e.g. `img%05d.jpg`), emitting each whole file
//! as one `DataFrame`, until a file in the sequence is missing. The g2g analog of
//! GStreamer's `multifilesrc`, the canonical front of an image-sequence decode:
//! `multifilesrc location=img%05d.jpg ! mjpegdec ! ...`.
//!
//! Each file is one independently-decodable unit, so every frame is marked a
//! keyframe. The output media type defaults to Motion-JPEG (the common case); a
//! `location` whose extension names a still-image format types the sequence
//! from it, and a different type can also be set at construction.
//!
//! With a `framerate` the source is gst's `imagesequencesrc`: the same file
//! walk, but the output rate is stated and each file is stamped on that grid,
//! so a sequence of stills plays as a clip. Without one the files carry no
//! timing, which is `multifilesrc`.

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

use crate::filesink::{io_err, path_io_err};
use g2g_core::log::short_type_name;

/// # Example
///
/// ```no_run
/// use g2g_plugins::multifilesrc::MultiFileSrc;
///
/// // multifilesrc location=img%05d.jpg ! mjpegdec ! ...
/// let src = MultiFileSrc::new("img%05d.jpg");
/// ```
#[derive(Debug)]
pub struct MultiFileSrc {
    location: String,
    caps: Caps,
    start_index: i64,
    stop_index: i64,
    loop_seq: bool,
    /// gst `imagesequencesrc`'s output rate. `None` (a plain `multifilesrc`)
    /// leaves the files unstamped.
    framerate: Option<(u32, u32)>,
    configured: bool,
}

impl MultiFileSrc {
    /// A launch-registry `multifilesrc` defaulting to a Motion-JPEG sequence. The
    /// geometry is a fixable `Range` placeholder (never `Any`, which cannot
    /// fixate); the real per-image dimensions arrive from the decoder downstream.
    pub fn new(location: impl Into<String>) -> Self {
        let mut src = Self {
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
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            start_index: 0,
            // -1 means "until the first missing file".
            stop_index: -1,
            loop_seq: false,
            framerate: None,
            configured: false,
        };
        src.type_from_location();
        src
    }

    /// Set the sequence's media type (e.g. a raw byte stream) explicitly.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// State the output framerate and stamp each file on that grid, which is
    /// what `imagesequencesrc` does.
    pub fn with_framerate(mut self, numerator: u32, denominator: u32) -> Self {
        if denominator > 0 {
            self.framerate = Some((numerator, denominator));
        }
        self
    }

    /// The stated rate in the Q16 fixed-point fps `Rate` carries, `None` when
    /// the files are unstamped.
    fn rate_q16(&self) -> Option<u32> {
        let (numerator, denominator) = self.framerate?;
        if denominator == 0 {
            return None;
        }
        u32::try_from((u64::from(numerator) << 16) / u64::from(denominator)).ok()
    }

    /// The declared output caps: the sequence's media type, at the stated rate
    /// when there is one.
    fn output_caps(&self) -> Caps {
        let Some(rate_q16) = self.rate_q16() else {
            return self.caps.clone();
        };
        let mut caps = self.caps.clone();
        if let Caps::CompressedVideo { framerate, .. } = &mut caps {
            *framerate = Rate::Fixed(rate_q16);
        }
        caps
    }

    /// Type the sequence from the `location` pattern's extension, so
    /// `img%05d.png` is a PNG sequence without a caps argument. Leaves the
    /// current type alone for a pattern with no still-image extension.
    fn type_from_location(&mut self) {
        if let Some(caps @ Caps::CompressedVideo { .. }) =
            crate::filesrc::caps_from_extension(std::path::Path::new(&self.location))
        {
            self.caps = caps;
        }
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
        core::future::ready(Ok(self.output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.output_caps(),
        ))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.output_caps())
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
                    Err(e) => return Err(path_io_err(short_type_name::<Self>(), "open", &path, e)),
                };
                let mut buf = alloc::vec::Vec::new();
                file.read_to_end(&mut buf).map_err(io_err)?;
                let period_ns = self
                    .rate_q16()
                    .map_or(0, crate::compositor::frame_period_ns);
                let pts_ns = sequence.saturating_mul(period_ns);
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: period_ns,
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
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.into();
                self.type_from_location();
            }
            "start-index" => self.start_index = value.as_int().ok_or(PropError::Type)?,
            "stop-index" => self.stop_index = value.as_int().ok_or(PropError::Type)?,
            "loop" => self.loop_seq = value.as_bool().ok_or(PropError::Type)?,
            "framerate" => {
                let (numerator, denominator) = value.as_fraction().ok_or(PropError::Type)?;
                if numerator <= 0 || denominator <= 0 {
                    return Err(PropError::Value);
                }
                self.framerate = Some((numerator as u32, denominator as u32));
            }
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
            "framerate" => {
                let (numerator, denominator) = self.framerate.unwrap_or((0, 1));
                Some(PropValue::Fraction(numerator as i32, denominator as i32))
            }
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
    PropertySpec::new(
        "framerate",
        PropKind::Fraction,
        "output framerate, stamping each file on that grid (0/1 = unstamped)",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    struct CollectSink {
        frames: alloc::vec::Vec<alloc::vec::Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");

            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
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
