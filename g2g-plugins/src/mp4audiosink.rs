//! Audio-only fragmented-MP4 muxer sink (M37). The audio counterpart of
//! `Mp4Mux`: wraps a raw AAC-LC elementary stream in a single-`soun`-track
//! fMP4 (`ftyp` + `moov` once, then one `moof`+`mdat` fragment per access
//! unit), so an encoded audio stream is a playable, durable `.m4a`. Pairs with
//! `MfAacEncode` upstream; `WavSink` remains the uncompressed alternative.
//!
//! The `moov`'s `mp4a`/`esds` sample entry needs the stream's
//! AudioSpecificConfig, which AAC access units do not carry in-band, so it is
//! supplied with [`Mp4AudioSink::with_audio_specific_config`] (the encoder
//! exposes it). Audio access units are stored verbatim in the `mdat` (no
//! length prefix); the media timescale is the sample rate and each AAC-LC
//! access unit is 1024 samples.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::filesink::{io_err, path_io_err};
use g2g_core::log::short_type_name;
use g2g_core::time::ns_to_ticks;

/// Samples per AAC-LC access unit (the fragment sample duration in media-time).
const AAC_FRAME_SAMPLES: u32 = 1024;

/// # Example
///
/// ```no_run
/// use g2g_plugins::mp4audiosink::Mp4AudioSink;
///
/// // AudioSpecificConfig for 48 kHz stereo AAC-LC
/// let sink = Mp4AudioSink::new("out.mp4").with_audio_specific_config([0x11, 0x90]);
/// ```
#[derive(Debug)]
pub struct Mp4AudioSink {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    channels: u8,
    sample_rate: u32,
    asc: Vec<u8>,
    header_written: bool,
    fragments: u64,
    decode_time: u64,
    // durations are cut from the running total so rounding never accumulates
    elapsed_ns: u64,
    eos_seen: bool,
}

impl Mp4AudioSink {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            writer: None,
            channels: 0,
            sample_rate: 0,
            asc: Vec::new(),
            header_written: false,
            fragments: 0,
            decode_time: 0,
            elapsed_ns: 0,
            eos_seen: false,
        }
    }

    /// Supply the stream's AudioSpecificConfig (from `MfAacEncode`), written
    /// into the `mp4a`/`esds`. Required before `configure_pipeline`.
    pub fn with_audio_specific_config(mut self, asc: impl Into<Vec<u8>>) -> Self {
        self.asc = asc.into();
        self
    }

    pub fn fragments_written(&self) -> u64 {
        self.fragments
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    fn accept_caps(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::Audio {
            format: AudioFormat::Aac,
            channels,
            sample_rate,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if *channels == 0 || *sample_rate == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.channels = *channels;
        self.sample_rate = *sample_rate;
        Ok(())
    }
}

impl AsyncElement for Mp4AudioSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| match c {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(c.clone()),
            _ => Err(G2gError::CapsMismatch),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.accept_caps(absolute_caps)?;
        if self.asc.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        let file = File::create(&self.path)
            .map_err(|e| path_io_err(short_type_name::<Self>(), "create", &self.path, e))?;
        self.writer = Some(BufWriter::new(file));
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.writer.is_none() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let au = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if au.is_empty() {
                        return Err(G2gError::CapsMismatch);
                    }

                    if !self.header_written {
                        let track = TrackInit::Audio {
                            format: AudioFormat::Aac,
                            channels: self.channels,
                            rate: self.sample_rate,
                            config: self.asc.clone(),
                        };
                        let moov = av_moov(&[Some(track)], None, &MoovMetadata::default(), &[]);
                        let header = [ftyp(), moov].concat();
                        let w = self.writer.as_mut().expect("checked above");
                        w.write_all(&header).map_err(io_err)?;
                        self.header_written = true;
                    }

                    // duration in media-time ticks: AAC-LC frame, or derived
                    // from the frame's explicit duration if present.
                    let duration = if frame.timing.duration_ns != 0 {
                        let end_ns = self.elapsed_ns.saturating_add(frame.timing.duration_ns);
                        let ticks = ns_to_ticks(end_ns, self.sample_rate)
                            - ns_to_ticks(self.elapsed_ns, self.sample_rate);
                        self.elapsed_ns = end_ns;
                        ticks as u32
                    } else {
                        AAC_FRAME_SAMPLES
                    };
                    let sample = FragmentSample {
                        bytes: au,
                        duration,
                        is_sync: true,
                    };
                    let mut frag = Vec::new();
                    av_fragment(
                        &mut frag,
                        TRACK_ID,
                        self.fragments + 1,
                        self.decode_time,
                        false,
                        &[sample],
                    );
                    let w = self.writer.as_mut().expect("checked above");
                    w.write_all(&frag).map_err(io_err)?;
                    self.fragments += 1;
                    self.decode_time += duration as u64;
                }
                PipelinePacket::Eos => {
                    let w = self.writer.as_mut().expect("checked above");
                    w.flush().map_err(io_err)?;
                    self.eos_seen = true;
                }
                PipelinePacket::CapsChanged(c) => {
                    self.accept_caps(&c)?;
                }
                PipelinePacket::Flush => {}
                // Segment is control: ignored at sink.
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for Mp4AudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }))])
    }
}

use crate::mp4box::{ftyp, full_box};
use crate::mp4muxn::{av_fragment, av_moov, FragmentSample, MoovMetadata, TrackInit};

/// The one track's `track_ID`.
const TRACK_ID: u32 = 1;

// --- box writers -----------------------------------------------------------

/// An MPEG-4 descriptor: tag, expandable size (single byte, payloads here are
/// small), then payload.
fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    debug_assert!(
        payload.len() < 128,
        "descriptor payload exceeds single-byte size"
    );
    let mut d = Vec::with_capacity(2 + payload.len());
    d.push(tag);
    d.push(payload.len() as u8);
    d.extend_from_slice(payload);
    d
}

/// The `esds` (ES_Descriptor) box carrying the AAC AudioSpecificConfig.
pub(crate) fn esds(asc: &[u8]) -> Vec<u8> {
    let dec_specific = descriptor(0x05, asc);
    let mut dcd = Vec::new();
    dcd.push(0x40); // objectTypeIndication: Audio ISO/IEC 14496-3 (AAC)
    dcd.push(0x15); // streamType audio (0x05 << 2) | upstream 0 | reserved 1
    dcd.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    dcd.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate
    dcd.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate
    dcd.extend_from_slice(&dec_specific);
    let dec_config = descriptor(0x04, &dcd);

    let sl = descriptor(0x06, &[0x02]); // SLConfigDescriptor: predefined 2

    let mut es = Vec::new();
    es.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es.push(0); // flags
    es.extend_from_slice(&dec_config);
    es.extend_from_slice(&sl);
    let es_descriptor = descriptor(0x03, &es);

    full_box(b"esds", 0, 0, &es_descriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn esds_carries_the_asc() {
        // AAC-LC 48 kHz stereo ASC is typically [0x11, 0x90].
        let asc = [0x11u8, 0x90];
        let b = esds(&asc);
        assert_eq!(&b[4..8], b"esds");
        // the ASC appears inside a DecoderSpecificInfo (tag 0x05).
        let pos = b.windows(2).position(|w| w == asc).expect("asc present");
        assert_eq!(b[pos - 2], 0x05, "ASC tagged as DecoderSpecificInfo");
        assert_eq!(b[pos - 1], asc.len() as u8, "ASC length byte");
    }

    #[test]
    fn rejects_non_aac_caps() {
        let mut sink = Mp4AudioSink::new("x.m4a").with_audio_specific_config(vec![0x11, 0x90]);
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(matches!(
            sink.configure_pipeline(&pcm),
            Err(G2gError::CapsMismatch)
        ));
    }

    #[test]
    fn requires_audio_specific_config() {
        let mut sink = Mp4AudioSink::new("x.m4a");
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(matches!(
            sink.configure_pipeline(&aac),
            Err(G2gError::CapsMismatch)
        ));
    }
}
