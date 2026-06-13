//! Audio-only fragmented-MP4 muxer sink (M37). The audio counterpart of
//! `Mp4Sink`: wraps a raw AAC-LC elementary stream in a single-`soun`-track
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
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::filesink::io_err;

/// Samples per AAC-LC access unit (the fragment sample duration in media-time).
const AAC_FRAME_SAMPLES: u32 = 1024;

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
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
        let file = File::create(&self.path).map_err(io_err)?;
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let au = slice.as_slice();
                    if au.is_empty() {
                        return Err(G2gError::CapsMismatch);
                    }

                    if !self.header_written {
                        let header = [
                            ftyp(),
                            moov(self.channels, self.sample_rate, &self.asc),
                        ]
                        .concat();
                        let w = self.writer.as_mut().expect("checked above");
                        w.write_all(&header).map_err(io_err)?;
                        self.header_written = true;
                    }

                    // duration in media-time ticks: AAC-LC frame, or derived
                    // from the frame's explicit duration if present.
                    let duration = if frame.timing.duration_ns != 0 {
                        ((frame.timing.duration_ns as u128 * self.sample_rate as u128)
                            / 1_000_000_000) as u32
                    } else {
                        AAC_FRAME_SAMPLES
                    };
                    let frag = fragment(self.fragments + 1, self.decode_time, duration, au);
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
        }))])
    }
}

// --- box writers -----------------------------------------------------------

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + payload.len());
    b.extend_from_slice(&((payload.len() as u32 + 8).to_be_bytes()));
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}

fn full_box(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut p = Vec::with_capacity(4 + payload.len());
    p.push(version);
    p.extend_from_slice(&flags.to_be_bytes()[1..]);
    p.extend_from_slice(payload);
    mp4_box(kind, &p)
}

/// An MPEG-4 descriptor: tag, expandable size (single byte, payloads here are
/// small), then payload.
fn descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    debug_assert!(payload.len() < 128, "descriptor payload exceeds single-byte size");
    let mut d = Vec::with_capacity(2 + payload.len());
    d.push(tag);
    d.push(payload.len() as u8);
    d.extend_from_slice(payload);
    d
}

fn ftyp() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(b"iso5");
    p.extend_from_slice(&512u32.to_be_bytes());
    p.extend_from_slice(b"iso5");
    p.extend_from_slice(b"isom");
    mp4_box(b"ftyp", &p)
}

const MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000];

/// The `esds` (ES_Descriptor) box carrying the AAC AudioSpecificConfig.
fn esds(asc: &[u8]) -> Vec<u8> {
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

fn moov(channels: u8, rate: u32, asc: &[u8]) -> Vec<u8> {
    let mvhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&1000u32.to_be_bytes()); // timescale
        p.extend_from_slice(&0u32.to_be_bytes()); // duration (fragmented)
        p.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate 1.0
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
        p.extend_from_slice(&[0u8; 10]);
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&[0u8; 24]);
        p.extend_from_slice(&2u32.to_be_bytes()); // next track id
        full_box(b"mvhd", 0, 0, &p)
    };

    let tkhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&1u32.to_be_bytes()); // track id
        p.extend_from_slice(&[0u8; 4]);
        p.extend_from_slice(&0u32.to_be_bytes()); // duration
        p.extend_from_slice(&[0u8; 8]); // reserved
        p.extend_from_slice(&0u16.to_be_bytes()); // layer
        p.extend_from_slice(&0u16.to_be_bytes()); // alternate group
        p.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0 (audio track)
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        for m in MATRIX {
            p.extend_from_slice(&m.to_be_bytes());
        }
        p.extend_from_slice(&0u32.to_be_bytes()); // width
        p.extend_from_slice(&0u32.to_be_bytes()); // height
        full_box(b"tkhd", 0, 3, &p) // enabled | in_movie
    };

    let mdhd = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 8]);
        p.extend_from_slice(&rate.to_be_bytes()); // timescale = sample rate
        p.extend_from_slice(&0u32.to_be_bytes()); // duration
        p.extend_from_slice(&0x55C4u16.to_be_bytes()); // language "und"
        p.extend_from_slice(&[0u8; 2]);
        full_box(b"mdhd", 0, 0, &p)
    };

    let hdlr = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 4]);
        p.extend_from_slice(b"soun");
        p.extend_from_slice(&[0u8; 12]);
        p.extend_from_slice(b"g2g\0");
        full_box(b"hdlr", 0, 0, &p)
    };

    let mp4a = {
        let mut p = Vec::new();
        p.extend_from_slice(&[0u8; 6]); // reserved
        p.extend_from_slice(&1u16.to_be_bytes()); // data reference index
        p.extend_from_slice(&[0u8; 8]); // reserved (version/revision/vendor)
        p.extend_from_slice(&(channels as u16).to_be_bytes());
        p.extend_from_slice(&16u16.to_be_bytes()); // sample size
        p.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
        p.extend_from_slice(&0u16.to_be_bytes()); // reserved
        p.extend_from_slice(&(rate << 16).to_be_bytes()); // 16.16 sample rate
        p.extend_from_slice(&esds(asc));
        mp4_box(b"mp4a", &p)
    };

    let stbl = {
        let stsd = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes());
            p.extend_from_slice(&mp4a);
            full_box(b"stsd", 0, 0, &p)
        };
        let empty4 = 0u32.to_be_bytes();
        let stts = full_box(b"stts", 0, 0, &empty4);
        let stsc = full_box(b"stsc", 0, 0, &empty4);
        let stsz = full_box(b"stsz", 0, 0, &[0u8; 8]);
        let stco = full_box(b"stco", 0, 0, &empty4);
        mp4_box(b"stbl", &[stsd, stts, stsc, stsz, stco].concat())
    };

    let minf = {
        // smhd: sound media header (balance 0).
        let smhd = full_box(b"smhd", 0, 0, &[0u8; 4]);
        let dref = {
            let url = full_box(b"url ", 0, 1, &[]);
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes());
            p.extend_from_slice(&url);
            full_box(b"dref", 0, 0, &p)
        };
        let dinf = mp4_box(b"dinf", &dref);
        mp4_box(b"minf", &[smhd, dinf, stbl].concat())
    };

    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());
    let trak = mp4_box(b"trak", &[tkhd, mdia].concat());

    let mvex = {
        let mut p = Vec::new();
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&[0u8; 12]);
        let trex = full_box(b"trex", 0, 0, &p);
        mp4_box(b"mvex", &trex)
    };

    mp4_box(b"moov", &[mvhd, trak, mvex].concat())
}

/// One `moof`+`mdat` fragment holding a single audio access unit (always a
/// sync sample).
fn fragment(sequence: u64, decode_time: u64, duration: u32, sample: &[u8]) -> Vec<u8> {
    let sample_flags: u32 = 0x0200_0000; // sync sample

    let build_moof = |data_offset: u32| -> Vec<u8> {
        let mfhd = full_box(b"mfhd", 0, 0, &(sequence as u32).to_be_bytes());
        let tfhd = full_box(b"tfhd", 0, 0x020000, &1u32.to_be_bytes()); // default-base-is-moof
        let tfdt = full_box(b"tfdt", 1, 0, &decode_time.to_be_bytes());
        let trun = {
            let mut p = Vec::new();
            p.extend_from_slice(&1u32.to_be_bytes()); // sample count
            p.extend_from_slice(&data_offset.to_be_bytes());
            p.extend_from_slice(&duration.to_be_bytes());
            p.extend_from_slice(&(sample.len() as u32).to_be_bytes());
            p.extend_from_slice(&sample_flags.to_be_bytes());
            full_box(b"trun", 0, 0x000701, &p)
        };
        let traf = mp4_box(b"traf", &[tfhd, tfdt, trun].concat());
        mp4_box(b"moof", &[mfhd, traf].concat())
    };

    let moof_len = build_moof(0).len() as u32;
    let moof = build_moof(moof_len + 8);
    let mdat = mp4_box(b"mdat", sample);
    [moof, mdat].concat()
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
    fn fragment_data_offset_points_at_the_au() {
        let frag = fragment(1, 0, 1024, &[7, 8, 9]);
        let moof_len = u32::from_be_bytes(frag[..4].try_into().unwrap()) as usize;
        let payload_at = moof_len + 8;
        assert_eq!(&frag[payload_at..payload_at + 3], &[7, 8, 9]);
        let pos = frag.windows(4).position(|w| w == b"trun").unwrap();
        let data_offset = u32::from_be_bytes(frag[pos + 12..pos + 16].try_into().unwrap()) as usize;
        assert_eq!(data_offset, payload_at);
    }

    #[test]
    fn rejects_non_aac_caps() {
        let mut sink = Mp4AudioSink::new("x.m4a").with_audio_specific_config(vec![0x11, 0x90]);
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
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
        };
        assert!(matches!(
            sink.configure_pipeline(&aac),
            Err(G2gError::CapsMismatch)
        ));
    }
}
