//! Multi-file sink (`multifilesink`). Writes the stream to a series of files
//! named from a printf-style `location` pattern (e.g. `frame%05d.jpg`), starting
//! a new file per the `next-file` policy. The g2g analog of GStreamer's
//! `multifilesink`, for writing image sequences or segmenting a byte stream.
//!
//! `next-file` policies:
//! - `buffer` (default): one file per buffer (an image sequence).
//! - `key-frame`: a new file at each keyframe (segments starting on a keyframe).
//! - `max-size`: a new file once the current one would exceed `max-file-size`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;

use std::fs::File;
use std::io::{BufWriter, Write};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::filesink::io_err;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextFile {
    Buffer,
    KeyFrame,
    MaxSize,
}

impl NextFile {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "buffer" => Some(Self::Buffer),
            "key-frame" => Some(Self::KeyFrame),
            "max-size" => Some(Self::MaxSize),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::KeyFrame => "key-frame",
            Self::MaxSize => "max-size",
        }
    }
}

#[derive(Debug)]
pub struct MultiFileSink {
    location: String,
    next_file: NextFile,
    max_file_size: u64,
    index: u64,
    writer: Option<BufWriter<File>>,
    current_bytes: u64,
    files_written: u64,
}

impl MultiFileSink {
    /// `location` is a printf-style pattern with one integer field, e.g.
    /// `frame%05d.raw`; without a field the index is appended.
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            next_file: NextFile::Buffer,
            max_file_size: 2 * 1024 * 1024,
            index: 0,
            writer: None,
            current_bytes: 0,
            files_written: 0,
        }
    }

    pub fn files_written(&self) -> u64 {
        self.files_written
    }

    /// Close the current file (if any) and open the next in the sequence.
    fn open_next(&mut self) -> Result<(), G2gError> {
        if let Some(mut w) = self.writer.take() {
            w.flush().map_err(io_err)?;
        }
        let path = expand(&self.location, self.index);
        let file = File::create(&path).map_err(io_err)?;
        self.writer = Some(BufWriter::new(file));
        self.current_bytes = 0;
        self.index += 1;
        self.files_written += 1;
        Ok(())
    }

    fn write_frame(&mut self, bytes: &[u8], keyframe: bool) -> Result<(), G2gError> {
        match self.next_file {
            NextFile::Buffer => {
                self.open_next()?;
            }
            NextFile::KeyFrame => {
                if self.writer.is_none() || (keyframe && self.current_bytes > 0) {
                    self.open_next()?;
                }
            }
            NextFile::MaxSize => {
                if self.writer.is_none()
                    || self.current_bytes.saturating_add(bytes.len() as u64) > self.max_file_size
                {
                    self.open_next()?;
                }
            }
        }
        let w = self.writer.as_mut().ok_or(G2gError::NotConfigured)?;
        w.write_all(bytes).map_err(io_err)?;
        self.current_bytes += bytes.len() as u64;
        Ok(())
    }
}

impl AsyncElement for MultiFileSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.write_frame(slice.as_slice(), frame.timing.keyframe)?;
                }
                PipelinePacket::Eos => {
                    if let Some(w) = self.writer.as_mut() {
                        w.flush().map_err(io_err)?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MULTIFILESINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Multi-file sink", "Sink/File", "Writes buffers to a sequence of files", "g2g")
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => self.location = value.as_str().ok_or(PropError::Type)?.into(),
            "next-file" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.next_file = NextFile::from_str(s).ok_or(PropError::Value)?;
            }
            "max-file-size" => self.max_file_size = value.as_uint().ok_or(PropError::Type)?,
            "index" => self.index = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "next-file" => Some(PropValue::Str(self.next_file.as_str().into())),
            "max-file-size" => Some(PropValue::Uint(self.max_file_size)),
            "index" => Some(PropValue::Uint(self.index)),
            _ => None,
        }
    }
}

static MULTIFILESINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "printf-style file pattern, e.g. frame%05d.raw"),
    PropertySpec::new("next-file", PropKind::Str, "new-file policy: buffer | key-frame | max-size"),
    PropertySpec::new("max-file-size", PropKind::Uint, "max bytes per file in max-size mode"),
    PropertySpec::new("index", PropKind::Uint, "next file index"),
];

impl PadTemplates for MultiFileSink {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        alloc::vec::Vec::from([PadTemplate::sink_any()])
    }
}

/// Expand a printf-style integer pattern (`%d`, `%05d`) with `index`. A pattern
/// without a valid integer field gets the index appended. Only one integer field
/// is supported (the first), matching `multifilesink`'s `location`. Shared with
/// `multifilesrc` (the read side of the same pattern).
pub(crate) fn expand(pattern: &str, index: u64) -> String {
    let bytes = pattern.as_bytes();
    if let Some(pct) = pattern.find('%') {
        let mut i = pct + 1;
        let zero = bytes.get(i) == Some(&b'0');
        if zero {
            i += 1;
        }
        let width_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let width: usize = pattern[width_start..i].parse().unwrap_or(0);
        if bytes.get(i) == Some(&b'd') {
            let num = if zero {
                alloc::format!("{index:0width$}")
            } else {
                alloc::format!("{index:width$}")
            };
            return alloc::format!("{}{}{}", &pattern[..pct], num, &pattern[i + 1..]);
        }
    }
    alloc::format!("{pattern}{index}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_printf_patterns() {
        assert_eq!(expand("frame%05d.raw", 7), "frame00007.raw");
        assert_eq!(expand("img%d.jpg", 42), "img42.jpg");
        assert_eq!(expand("out", 3), "out3");
        assert_eq!(expand("a%3d.bin", 5), "a  5.bin");
    }

    #[tokio::test]
    async fn buffer_mode_writes_one_file_per_frame() {
        let dir = std::env::temp_dir();
        let pat = dir.join("g2g_mfs_%03d.bin").to_string_lossy().into_owned();
        let mut sink = MultiFileSink::new(&pat);
        let mut out = NullSink;
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .unwrap();
        sink.process(frame(b"aaa", false), &mut out).await.unwrap();
        sink.process(frame(b"bb", false), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(sink.files_written(), 2);
        assert_eq!(std::fs::read(expand(&pat, 0)).unwrap(), b"aaa");
        assert_eq!(std::fs::read(expand(&pat, 1)).unwrap(), b"bb");
        let _ = std::fs::remove_file(expand(&pat, 0));
        let _ = std::fs::remove_file(expand(&pat, 1));
    }

    #[tokio::test]
    async fn key_frame_mode_starts_a_file_at_each_keyframe() {
        let dir = std::env::temp_dir();
        let pat = dir.join("g2g_mfs_kf_%03d.bin").to_string_lossy().into_owned();
        let mut sink = MultiFileSink::new(&pat);
        sink.set_property("next-file", PropValue::Str("key-frame".into())).unwrap();
        let mut out = NullSink;
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        })
        .unwrap();
        // keyframe, delta, delta, keyframe -> two files.
        sink.process(frame(b"K1", true), &mut out).await.unwrap();
        sink.process(frame(b"d", false), &mut out).await.unwrap();
        sink.process(frame(b"K2", true), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(sink.files_written(), 2);
        assert_eq!(std::fs::read(expand(&pat, 0)).unwrap(), b"K1d");
        assert_eq!(std::fs::read(expand(&pat, 1)).unwrap(), b"K2");
        let _ = std::fs::remove_file(expand(&pat, 0));
        let _ = std::fs::remove_file(expand(&pat, 1));
    }

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(g2g_core::PushOutcome::Accepted) })
        }
    }

    fn frame(bytes: &[u8], keyframe: bool) -> PipelinePacket {
        use g2g_core::{Frame, FrameTiming, SystemSlice};
        let timing = FrameTiming { keyframe, ..FrameTiming::default() };
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            timing,
            sequence: 0,
            meta: Default::default(),
        })
    }
}
