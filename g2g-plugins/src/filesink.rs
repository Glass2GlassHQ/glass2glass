//! File sink. Writes every `DataFrame`'s system-memory bytes to a file in
//! arrival order, producing a raw byte stream (e.g. an Annex-B `.h264`
//! elementary stream when fed from `H264Parse` or `MfEncode`). M20: with
//! `FileSrc` this completes the record / playback path.
//!
//! Caps are not encoded in the output (a raw byte stream has no container),
//! so the sink accepts any caps and records `CapsChanged` packets only for
//! inspection. `Flush` is a no-op on the file: a raw stream has no seek
//! index to reset, and truncating mid-run would corrupt the recording.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, HardwareError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

/// Map a filesystem error to the structured `Hardware(Io)` variant, carrying
/// the raw OS error code. Shared with `FileSrc`.
pub(crate) fn io_err(e: std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Io(e.raw_os_error().unwrap_or(0)))
}

#[derive(Debug)]
pub struct FileSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    bytes_written: u64,
    frames_written: u64,
    eos_seen: bool,
}

impl FileSink {
    /// The file is created (truncating an existing one) in
    /// `configure_pipeline`, not here, so constructing the element has no
    /// filesystem side effects until the pipeline negotiates.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            writer: None,
            bytes_written: 0,
            frames_written: 0,
            eos_seen: false,
        }
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }
}

impl AsyncElement for FileSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard sink: a raw byte stream can record anything.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The runner re-invokes configure_pipeline on a mid-stream caps change.
        // Create (and truncate) the file only on the first negotiation so a
        // later re-negotiation keeps the open writer and what was already
        // recorded, instead of truncating it.
        if self.writer.is_none() {
            let file = File::create(&self.path).map_err(io_err)?;
            self.writer = Some(BufWriter::new(file));
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let writer = self.writer.as_mut().ok_or(G2gError::NotConfigured)?;
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(bytes) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    writer.write_all(bytes).map_err(io_err)?;
                    self.bytes_written += bytes.len() as u64;
                    self.frames_written += 1;
                }
                PipelinePacket::Eos => {
                    writer.flush().map_err(io_err)?;
                    self.eos_seen = true;
                }
                // No seek index to reset in a raw stream; data already
                // written stays written.
                PipelinePacket::Flush => {}
                // Caps aren't representable in a raw byte stream; the
                // recording continues under the new caps.
                PipelinePacket::CapsChanged(_) => {}
                // Segment is control: ignored at sink.
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FILESINK_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "File sink",
            "Sink/File",
            "Writes incoming buffers to a local file",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.to_string_lossy().into_owned())),
            _ => None,
        }
    }
}

/// `FileSink`'s settable properties (M107): the output file path.
static FILESINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "output file path",
)];

impl PadTemplates for FileSink {
    /// Wildcard sink, matching the runtime `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{
        ByteStreamEncoding, Frame, FrameTiming, MemoryDomain, PushOutcome, SystemSlice,
    };

    struct NullSink;
    impl OutputSink for NullSink {
        fn push<'a>(
            &'a mut self,
            _packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    fn frame(bytes: &[u8]) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    #[tokio::test]
    async fn midstream_caps_change_does_not_truncate() {
        let path = std::env::temp_dir().join("g2g_filesink_recfg.bin");
        let _ = std::fs::remove_file(&path);
        let mut sink = FileSink::new(&path);
        let caps = Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        };
        sink.configure_pipeline(&caps).unwrap();
        let mut out = NullSink;
        sink.process(frame(b"first"), &mut out).await.unwrap();
        // A mid-stream caps change re-invokes configure_pipeline; the already
        // written bytes must survive instead of being truncated away.
        sink.configure_pipeline(&caps).unwrap();
        sink.process(frame(b"second"), &mut out).await.unwrap();
        sink.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"firstsecond");
        let _ = std::fs::remove_file(&path);
    }
}
