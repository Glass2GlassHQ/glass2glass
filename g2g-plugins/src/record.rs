//! Record / replay for deterministic repro. `recordsink` writes the packet
//! stream crossing its input (the negotiated caps, then every `DataFrame`) to a
//! file as length-prefixed [`g2g_core::wire`] records; `replaysrc` reads that
//! file back as a source, re-emitting the caps and frames (optionally paced to
//! the recorded PTS), so a bug that needed a live camera can be reproduced from
//! a file.
//!
//! File format: a flat sequence of `[u32-le length][length bytes]` records, each
//! payload an `encode_packet` frame. The first record is the `CapsChanged` the
//! sink was configured with; the rest are `DataFrame`s in arrival order. EOS is
//! not stored (the replay source emits its own at end of file).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::wire::{decode_packet, encode_packet};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, HardwareError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

fn io_err(e: std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Io(e.raw_os_error().unwrap_or(0)))
}

/// Serialize `packet` and append it to `w` as a `[u32-le len][bytes]` record.
/// A non-`System` frame is not serializable and returns `UnsupportedDomain`.
fn write_record<W: Write>(w: &mut W, packet: &PipelinePacket) -> Result<(), G2gError> {
    let bytes = encode_packet(packet).map_err(|_| G2gError::UnsupportedDomain)?;
    let len = u32::try_from(bytes.len()).map_err(|_| G2gError::UnsupportedDomain)?;
    w.write_all(&len.to_le_bytes()).map_err(io_err)?;
    w.write_all(&bytes).map_err(io_err)?;
    Ok(())
}

/// Split a recording buffer into its packet records. A truncated trailing record
/// (a recording cut off mid-write) is dropped rather than failing the replay.
fn read_records(buf: &[u8]) -> Result<Vec<PipelinePacket>, G2gError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i + 4 <= buf.len() {
        let len = u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]) as usize;
        let start = i + 4;
        let end = match start.checked_add(len) {
            Some(e) if e <= buf.len() => e,
            _ => break, // truncated tail
        };
        out.push(decode_packet(&buf[start..end]).map_err(|_| G2gError::CapsMismatch)?);
        i = end;
    }
    Ok(out)
}

// --- recordsink ---------------------------------------------------------

#[derive(Debug)]
pub struct RecordSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    frames: u64,
}

impl RecordSink {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), writer: None, frames: 0 }
    }

    pub fn frames_recorded(&self) -> u64 {
        self.frames
    }
}

impl AsyncElement for RecordSink {
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Create (truncating) on first negotiation; a mid-stream re-negotiation
        // keeps the open writer and appends the new caps as a record so the
        // replay reproduces the change at the right point.
        if self.writer.is_none() {
            let file = File::create(&self.path).map_err(io_err)?;
            self.writer = Some(BufWriter::new(file));
        }
        let caps = absolute_caps.clone();
        let writer = self.writer.as_mut().expect("writer created above");
        write_record(writer, &PipelinePacket::CapsChanged(caps))?;
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
                PipelinePacket::DataFrame(_) => {
                    write_record(writer, &packet)?;
                    self.frames += 1;
                }
                PipelinePacket::Eos => writer.flush().map_err(io_err)?,
                // Caps changes arrive via configure_pipeline (recorded there);
                // Flush / Segment are control, not part of the replayable stream.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RECORD_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Recording sink",
            "Sink/File",
            "Records the packet stream to a file for deterministic replay",
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

static RECORD_PROPS: &[PropertySpec] =
    &[PropertySpec::new("location", PropKind::Str, "recording file path")];

impl PadTemplates for RecordSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

// --- replaysrc ----------------------------------------------------------

#[derive(Debug)]
pub struct ReplaySrc {
    path: PathBuf,
    /// Pace playback to the recorded frame PTS deltas; default off (replay as
    /// fast as possible, for deterministic tests).
    sync: bool,
    configured: bool,
}

impl ReplaySrc {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), sync: false, configured: false }
    }

    /// The caps stored as the recording's leading record.
    fn leading_caps(&self) -> Result<Caps, G2gError> {
        let buf = std::fs::read(&self.path).map_err(io_err)?;
        match read_records(&buf)?.into_iter().next() {
            Some(PipelinePacket::CapsChanged(caps)) => Ok(caps),
            // An empty or non-caps-led recording cannot type the stream.
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

impl SourceLoop for ReplaySrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.leading_caps())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let buf = std::fs::read(&self.path).map_err(io_err)?;
            let records = read_records(&buf)?;
            let mut frames = 0u64;
            let mut prev_pts: Option<u64> = None;
            for packet in records {
                if let PipelinePacket::DataFrame(frame) = &packet {
                    if self.sync {
                        let pts = frame.timing.pts_ns;
                        if let Some(p) = prev_pts {
                            if pts > p {
                                tokio::time::sleep(Duration::from_nanos(pts - p)).await;
                            }
                        }
                        prev_pts = Some(pts);
                    }
                    frames += 1;
                }
                out.push(packet).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(frames)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        REPLAY_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "sync" => {
                self.sync = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.to_string_lossy().into_owned())),
            "sync" => Some(PropValue::Bool(self.sync)),
            _ => None,
        }
    }
}

static REPLAY_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "recording file path"),
    PropertySpec::new("sync", PropKind::Bool, "pace playback to recorded PTS (default off)"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{ByteStreamEncoding, Frame, FrameTiming, MemoryDomain, SystemSlice};

    fn frame(bytes: &[u8], seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            seq,
        ))
    }

    #[test]
    fn records_round_trip() {
        let mut buf = Vec::new();
        let caps = PipelinePacket::CapsChanged(Caps::ByteStream { encoding: ByteStreamEncoding::Ogg });
        write_record(&mut buf, &caps).unwrap();
        write_record(&mut buf, &frame(b"abc", 0)).unwrap();
        write_record(&mut buf, &frame(b"defg", 1)).unwrap();

        let records = read_records(&buf).unwrap();
        assert_eq!(records.len(), 3);
        assert!(matches!(records[0], PipelinePacket::CapsChanged(_)));
        match &records[2] {
            PipelinePacket::DataFrame(f) => {
                assert_eq!(f.sequence, 1);
                let MemoryDomain::System(s) = &f.domain else { panic!("system") };
                assert_eq!(s.as_slice(), b"defg");
            }
            _ => panic!("expected DataFrame"),
        }
    }

    #[test]
    fn truncated_tail_is_dropped_not_fatal() {
        let mut buf = Vec::new();
        write_record(&mut buf, &frame(b"whole", 0)).unwrap();
        let full_len = buf.len();
        // Append a half-written second record (length header claims more than present).
        write_record(&mut buf, &frame(b"partial", 1)).unwrap();
        buf.truncate(full_len + 6); // header + a couple bytes of the payload

        let records = read_records(&buf).unwrap();
        assert_eq!(records.len(), 1, "only the intact record survives; no error");
    }
}
