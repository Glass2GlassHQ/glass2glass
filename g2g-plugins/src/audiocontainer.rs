//! Shared PCM-file plumbing for AIFF and AU: caps sets, sample emission, and
//! the muxer loop that writes a header then converts each buffer onto the wire.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::pcmendian::{convert_wire_layout, sample_width, PcmWire, CARRIED};

/// Size fields of a stream whose length is not known when the header is written.
pub(crate) const STREAMING_SIZE: u32 = u32::MAX;
/// Sample rates above this are not audio.
pub(crate) const MAX_SAMPLE_RATE: u32 = 384_000;
/// Widest channel count these containers will accept.
pub(crate) const MAX_CHANNELS: u16 = 16;
/// AIFF and AU both write multi-byte samples big-endian.
const MUX_BIG_ENDIAN: bool = true;

/// What tells one PCM container apart from another. AIFF and AU each define a
/// single `static` of this and hand it to the shared muxer.
#[derive(Debug)]
pub(crate) struct PcmContainer {
    /// Byte-stream encoding of the file written.
    pub encoding: ByteStreamEncoding,
    /// Whether this container can store a graph format.
    pub supported: fn(AudioFormat) -> bool,
    /// Header bytes for a (format, channels, sample rate, payload size) file.
    pub header: fn(AudioFormat, u16, u32, usize) -> Option<Vec<u8>>,
    /// Hold the samples until EOS so the header can carry final sizes.
    pub finalize_at_eos: bool,
    /// The sample payload is padded up to a multiple of this many bytes.
    pub data_alignment: usize,
}

pub(crate) fn container_caps(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
    }
}

/// Source-pad alternatives a parser advertises before the header is read.
pub(crate) fn parse_output_alternatives() -> CapsSet {
    CapsSet::from_alternatives(
        CARRIED
            .map(|format| audio_caps(format, ANY_CHANNELS, ANY_SAMPLE_RATE))
            .to_vec(),
    )
}

/// Sink-pad alternatives a muxer accepts.
pub(crate) fn mux_input_alternatives() -> Vec<Caps> {
    CARRIED.map(|format| audio_caps(format, 0, 0)).to_vec()
}

/// Parser transform constraint: PCM at a fixable default rate first, then the
/// wildcard, then G.711, matching `wavparse`.
pub(crate) fn parse_constraint(encoding: ByteStreamEncoding) -> CapsConstraint<'static> {
    CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
        Caps::ByteStream { encoding: e } if *e == encoding => {
            let pcm = |sample_rate| audio_caps(AudioFormat::PcmS16Le, ANY_CHANNELS, sample_rate);
            let mut alternatives = alloc::vec![pcm(48_000), pcm(ANY_SAMPLE_RATE)];
            alternatives.extend(
                CARRIED
                    .iter()
                    .copied()
                    .filter(|f| *f != AudioFormat::PcmS16Le)
                    .map(|format| audio_caps(format, ANY_CHANNELS, ANY_SAMPLE_RATE)),
            );
            CapsSet::from_alternatives(alternatives)
        }
        _ => CapsSet::from_alternatives(Vec::new()),
    }))
}

pub(crate) fn mux_constraint(container: &'static PcmContainer) -> CapsConstraint<'static> {
    CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
        Caps::Audio { format, .. } if (container.supported)(*format) => {
            CapsSet::one(container_caps(container.encoding))
        }
        _ => CapsSet::from_alternatives(Vec::new()),
    }))
}

/// Take whole interleaved sample frames off `buf`, convert them off the wire,
/// and push one data frame. Leaves a trailing partial frame in `buf`.
pub(crate) async fn emit_pcm(
    buf: &mut Vec<u8>,
    remaining: &mut Option<usize>,
    wire: &PcmWire,
    sequence: &mut u64,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    let width = sample_width(wire.format).ok_or(G2gError::CapsMismatch)?;
    let frame_bytes = width
        .checked_mul(wire.channels as usize)
        .ok_or(G2gError::CapsMismatch)?;
    if frame_bytes == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let available = remaining.map_or(buf.len(), |bytes| bytes.min(buf.len()));
    let whole = available - (available % frame_bytes);
    if whole == 0 {
        return Ok(());
    }
    let mut samples: Vec<u8> = buf.drain(..whole).collect();
    if let Some(bytes) = remaining {
        *bytes -= whole;
    }
    convert_wire_layout(&mut samples, wire.format, wire.big_endian);
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(samples.into_boxed_slice())),
        FrameTiming::default(),
        *sequence,
    );
    *sequence += 1;
    out.push(PipelinePacket::DataFrame(frame)).await?;
    Ok(())
}

/// Muxer state shared by AIFF and AU: header once, then converted samples.
#[derive(Debug, Default)]
pub(crate) struct PcmMux {
    input: Option<(AudioFormat, u16, u32)>,
    header_written: bool,
    configured: bool,
    emitted: u64,
    pending: Vec<u8>,
}

impl PcmMux {
    pub(crate) fn configure(
        &mut self,
        caps: &Caps,
        container: &PcmContainer,
    ) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format,
            channels,
            sample_rate,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !(container.supported)(*format) {
            return Err(G2gError::CapsMismatch);
        }
        self.input = Some((*format, *channels as u16, *sample_rate));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    pub(crate) fn intercept(upstream: &Caps, container: &PcmContainer) -> Result<Caps, G2gError> {
        match upstream {
            Caps::Audio { format, .. } if (container.supported)(*format) => Ok(upstream.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn frame(&mut self, bytes: Vec<u8>) -> PipelinePacket {
        self.emitted += 1;
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: Default::default(),
            sequence: self.emitted,
            meta: Default::default(),
        })
    }

    async fn emit_header(
        &mut self,
        out: &mut dyn OutputSink,
        caps: Caps,
        bytes: Vec<u8>,
    ) -> Result<(), G2gError> {
        out.push(PipelinePacket::CapsChanged(caps)).await?;
        let packet = self.frame(bytes);
        out.push(packet).await?;
        self.header_written = true;
        Ok(())
    }

    pub(crate) fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
        container: &'static PcmContainer,
    ) -> Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> {
        Box::pin(async move {
            let caps = container_caps(container.encoding);
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let (format, channels, sample_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    let samples = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let frame_bytes = sample_width(format)
                        .and_then(|width| width.checked_mul(channels as usize))
                        .ok_or(G2gError::CapsMismatch)?;
                    if frame_bytes == 0 || samples.len() % frame_bytes != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    let mut data = samples.to_vec();
                    convert_wire_layout(&mut data, format, MUX_BIG_ENDIAN);
                    if container.finalize_at_eos {
                        self.pending.extend_from_slice(&data);
                    } else {
                        if !self.header_written {
                            if channels == 0 || sample_rate == 0 {
                                return Err(G2gError::CapsMismatch);
                            }
                            let bytes = (container.header)(format, channels, sample_rate, 0)
                                .ok_or(G2gError::CapsMismatch)?;
                            self.emit_header(out, caps, bytes).await?;
                        }
                        let packet = self.frame(data);
                        out.push(packet).await?;
                    }
                }
                PipelinePacket::CapsChanged(caps) => {
                    if !self.header_written {
                        if let Caps::Audio {
                            format,
                            channels,
                            sample_rate,
                        } = caps
                        {
                            if (container.supported)(format) {
                                self.input = Some((format, channels as u16, sample_rate));
                            }
                        }
                    }
                }
                PipelinePacket::Eos if self.header_written => {}
                PipelinePacket::Eos if container.finalize_at_eos => {
                    let (format, channels, sample_rate) =
                        self.input.ok_or(G2gError::NotConfigured)?;
                    let alignment = container.data_alignment;
                    if channels == 0 || sample_rate == 0 || alignment == 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    let payload_size = self.pending.len();
                    let bytes = (container.header)(format, channels, sample_rate, payload_size)
                        .ok_or(G2gError::CapsMismatch)?;
                    self.emit_header(out, caps, bytes).await?;
                    let padding = (alignment - (payload_size % alignment)) % alignment;
                    let padded_size = payload_size
                        .checked_add(padding)
                        .ok_or(G2gError::CapsMismatch)?;
                    self.pending.resize(padded_size, 0);
                    let data = core::mem::take(&mut self.pending);
                    if !data.is_empty() {
                        let packet = self.frame(data);
                        out.push(packet).await?;
                    }
                }
                PipelinePacket::Eos => {
                    if !self.header_written {
                        let (format, channels, sample_rate) =
                            self.input.ok_or(G2gError::NotConfigured)?;
                        if channels == 0 || sample_rate == 0 {
                            return Err(G2gError::CapsMismatch);
                        }
                        let bytes = (container.header)(format, channels, sample_rate, 0)
                            .ok_or(G2gError::CapsMismatch)?;
                        self.emit_header(out, caps, bytes).await?;
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
