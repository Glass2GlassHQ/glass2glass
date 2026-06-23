//! Audio-only fragmented-MP4 demuxer source (M37), the read-side counterpart of
//! `Mp4AudioSink`: parses a single-`soun`-track AAC fMP4 and emits the raw AAC
//! access units with recovered timing, so a recorded `.m4a` plays back through
//! `MfAacDecode` like a live stream.
//!
//! The caps probe (`intercept_caps`) reads the codec/channels/rate and the
//! AudioSpecificConfig from the `mp4a`/`esds` sample entry; the ASC is exposed
//! via [`Mp4AudioSrc::audio_specific_config`] so a decoder can be configured.
//! Audio access units are stored verbatim in the `mdat` (no length prefix),
//! recovered by `trun` sample sizes.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket,
};

use crate::filesink::io_err;

#[derive(Debug)]
struct Header {
    channels: u8,
    sample_rate: u32,
    asc: Vec<u8>,
}

#[derive(Debug)]
pub struct Mp4AudioSrc {
    path: PathBuf,
    header: Option<Header>,
    configured: bool,
}

impl Mp4AudioSrc {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: None,
            configured: false,
        }
    }

    /// The recorded stream's AudioSpecificConfig, available after a probe (or
    /// `run`). Use it to configure an `MfAacDecode`.
    pub fn audio_specific_config(&self) -> Option<&[u8]> {
        self.header.as_ref().map(|h| h.asc.as_slice())
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.header.is_none() {
            let data = std::fs::read(&self.path).map_err(io_err)?;
            self.header = Some(parse_header(&data)?);
        }
        let h = self.header.as_ref().expect("just parsed");
        Ok(Caps::Audio {
            format: AudioFormat::Aac,
            channels: h.channels,
            sample_rate: h.sample_rate,
        })
    }
}

impl SourceLoop for Mp4AudioSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.probe())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.probe()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
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
            let data = std::fs::read(&self.path).map_err(io_err)?;
            if self.header.is_none() {
                self.header = Some(parse_header(&data)?);
            }
            let header = self.header.as_ref().expect("parsed above");
            let samples = parse_fragments(&data, header.sample_rate)?;

            let mut sequence = 0u64;
            for s in samples {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(s.au.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: s.pts_ns,
                        dts_ns: s.pts_ns,
                        duration_ns: s.duration_ns,
                        capture_ns: s.pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        keyframe: false, // audio: every sample is independent
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

#[derive(Debug)]
struct Sample {
    au: Vec<u8>,
    pts_ns: u64,
    duration_ns: u64,
}

// box read primitives are shared across the MP4 elements.
use crate::mp4box::{be32, be64, boxes, find_box, find_path};

fn parse_header(data: &[u8]) -> Result<Header, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;

    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let mdhd = find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?;
    if mdhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let sample_rate = be32(mdhd, 12)?; // timescale = sample rate
    if sample_rate == 0 {
        return Err(G2gError::CapsMismatch);
    }

    let stsd = find_path(mdia, &[b"minf", b"stbl", b"stsd"]).ok_or(G2gError::CapsMismatch)?;
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
    let mp4a = find_box(entries, b"mp4a").ok_or(G2gError::CapsMismatch)?;
    // AudioSampleEntry: channelcount at offset 16, then 28 bytes before the
    // nested esds.
    let channels = u16::from_be_bytes(
        mp4a.get(16..18)
            .ok_or(G2gError::CapsMismatch)?
            .try_into()
            .expect("2 bytes"),
    ) as u8;
    let mp4a_children = mp4a.get(28..).ok_or(G2gError::CapsMismatch)?;
    let esds = find_box(mp4a_children, b"esds").ok_or(G2gError::CapsMismatch)?;
    let asc = parse_esds(esds)?;
    if channels == 0 {
        return Err(G2gError::CapsMismatch);
    }

    Ok(Header {
        channels,
        sample_rate,
        asc,
    })
}

/// Extract the AudioSpecificConfig from an `esds` payload by descending the
/// descriptor tree: ES (0x03) -> DecoderConfig (0x04) -> DecoderSpecific (0x05).
fn parse_esds(esds: &[u8]) -> Result<Vec<u8>, G2gError> {
    // skip the full-box version/flags (4 bytes).
    let es = find_descriptor(esds.get(4..).ok_or(G2gError::CapsMismatch)?, 0x03)
        .ok_or(G2gError::CapsMismatch)?;
    // ES_Descriptor payload: ES_ID (2) + flags (1), then sub-descriptors.
    let dcd = find_descriptor(es.get(3..).ok_or(G2gError::CapsMismatch)?, 0x04)
        .ok_or(G2gError::CapsMismatch)?;
    // DecoderConfigDescriptor: 13 fixed bytes, then DecoderSpecificInfo.
    let asc = find_descriptor(dcd.get(13..).ok_or(G2gError::CapsMismatch)?, 0x05)
        .ok_or(G2gError::CapsMismatch)?;
    if asc.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(asc.to_vec())
}

/// Find the first descriptor with `tag` among the descriptors laid out at the
/// start of `data`, returning its payload. Handles the expandable size encoding
/// (7 bits per byte, high bit a continuation flag).
fn find_descriptor(data: &[u8], tag: u8) -> Option<&[u8]> {
    let mut i = 0usize;
    while i < data.len() {
        let t = data[i];
        i += 1;
        let mut size = 0usize;
        loop {
            let b = *data.get(i)?;
            i += 1;
            size = (size << 7) | (b & 0x7F) as usize;
            if b & 0x80 == 0 {
                break;
            }
        }
        let payload = data.get(i..i + size)?;
        if t == tag {
            return Some(payload);
        }
        i += size;
    }
    None
}

/// Walk `moof`+`mdat` pairs and split every raw AAC access unit out of its
/// `mdat`.
fn parse_fragments(data: &[u8], timescale: u32) -> Result<Vec<Sample>, G2gError> {
    let mut samples = Vec::new();
    let mut pending: Option<Vec<(u32, u64)>> = None; // (size, pts_ns)
    let mut durations: Vec<u64> = Vec::new();

    for (kind, payload) in boxes(data) {
        match kind {
            b"moof" => {
                let traf = find_box(payload, b"traf").ok_or(G2gError::CapsMismatch)?;
                let tfdt = find_box(traf, b"tfdt").ok_or(G2gError::CapsMismatch)?;
                let base_time = match tfdt.first() {
                    Some(1) => be64(tfdt, 4)?,
                    Some(0) => be32(tfdt, 4)? as u64,
                    _ => return Err(G2gError::CapsMismatch),
                };
                let trun = find_box(traf, b"trun").ok_or(G2gError::CapsMismatch)?;
                let (sizes, durs) = parse_trun(trun)?;
                let mut t = base_time;
                let mut tagged = Vec::with_capacity(sizes.len());
                durations.clear();
                for (size, dur) in sizes.iter().zip(&durs) {
                    tagged.push((*size, timescale_to_ns(t, timescale)));
                    durations.push(timescale_to_ns(*dur as u64, timescale));
                    t += *dur as u64;
                }
                pending = Some(tagged);
            }
            b"mdat" => {
                let Some(tagged) = pending.take() else {
                    return Err(G2gError::CapsMismatch);
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let au = payload
                        .get(at..at + *size as usize)
                        .ok_or(G2gError::CapsMismatch)?
                        .to_vec();
                    samples.push(Sample {
                        au,
                        pts_ns: *pts_ns,
                        duration_ns: durations[i],
                    });
                    at += *size as usize;
                }
            }
            _ => {}
        }
    }
    if pending.is_some() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(samples)
}

fn parse_trun(trun: &[u8]) -> Result<(Vec<u32>, Vec<u32>), G2gError> {
    if trun.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    if flags & 0x200 == 0 {
        return Err(G2gError::CapsMismatch); // sizes must be explicit
    }
    let count = be32(trun, 4)? as usize;
    let mut at = 8usize;
    if flags & 0x1 != 0 {
        at += 4; // data offset
    }
    if flags & 0x4 != 0 {
        at += 4; // first sample flags
    }
    let mut sizes = Vec::with_capacity(count);
    let mut durations = Vec::with_capacity(count);
    for _ in 0..count {
        let mut duration = 0u32;
        if flags & 0x100 != 0 {
            duration = be32(trun, at)?;
            at += 4;
        }
        sizes.push(be32(trun, at)?);
        at += 4;
        if flags & 0x400 != 0 {
            at += 4; // per-sample flags
        }
        if flags & 0x800 != 0 {
            at += 4; // composition time offset
        }
        durations.push(duration);
    }
    Ok((sizes, durations))
}

fn timescale_to_ns(t: u64, timescale: u32) -> u64 {
    t.saturating_mul(1_000_000_000) / timescale as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn descriptor_reader_handles_single_byte_sizes() {
        // tag 0x05 with a 2-byte payload preceded by a 0x04 wrapper.
        let inner = [0x05u8, 2, 0xAA, 0xBB];
        let outer = {
            let mut v = vec![0x04u8, inner.len() as u8];
            v.extend_from_slice(&inner);
            v
        };
        let dcd = find_descriptor(&outer, 0x04).unwrap();
        let asc = find_descriptor(dcd, 0x05).unwrap();
        assert_eq!(asc, &[0xAA, 0xBB]);
    }

    #[test]
    fn descriptor_reader_handles_expandable_sizes() {
        // size 130 encoded as 0x81 0x02 (continuation).
        let mut payload = vec![0u8; 130];
        payload[0] = 1;
        let mut d = vec![0x03u8, 0x81, 0x02];
        d.extend_from_slice(&payload);
        let got = find_descriptor(&d, 0x03).unwrap();
        assert_eq!(got.len(), 130);
        assert_eq!(got[0], 1);
    }

    #[test]
    fn timescale_conversion() {
        // 48 kHz timescale: 1024 ticks is one AAC frame ~21.33 ms.
        assert_eq!(timescale_to_ns(1024, 48_000), 21_333_333);
    }
}
