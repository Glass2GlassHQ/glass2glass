//! Fragmented-MP4 demuxer source (M28), the read-side counterpart of
//! `Mp4Sink`: parses a single-video-track fMP4 and emits Annex-B H.264
//! access units with their recovered timing, so a recording plays back
//! through `MfDecode` / `FfmpegH264Dec` exactly like a live stream.
//!
//! Caps discovery is the M18 async-source path: `intercept_caps` reads the
//! file's `ftyp`/`moov` (dims from `tkhd`, SPS/PPS from `avcC`, timescale
//! from `mdhd`) before negotiation, so downstream solves against the real
//! geometry. The fragment scan happens in `run`.
//!
//! Supported profile: what `Mp4Sink` writes and CMAF-style single-track
//! files generally share: one video track, `trun` v0 with explicit sample
//! sizes, `default-base-is-moof` data offsets landing on the following
//! `mdat`'s payload. Anything else fails loud rather than emitting a
//! corrupt bitstream. If the first sample carries no in-band SPS, the
//! `avcC` parameter sets are prepended so a decoder can start.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;

#[derive(Debug)]
struct Header {
    width: u32,
    height: u32,
    timescale: u32,
    sps: Vec<u8>,
    pps: Vec<u8>,
}

#[derive(Debug)]
pub struct Mp4Src {
    path: PathBuf,
    header: Option<Header>,
    configured: bool,
}

impl Mp4Src {
    /// The file is read during caps probing and `run`; construction has no
    /// filesystem side effects.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: None,
            configured: false,
        }
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.header.is_none() {
            let data = std::fs::read(&self.path).map_err(io_err)?;
            self.header = Some(parse_header(&data)?);
        }
        let h = self.header.as_ref().expect("just parsed");
        Ok(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(h.width),
            height: Dim::Fixed(h.height),
            framerate: Rate::Any,
        })
    }
}

impl SourceLoop for Mp4Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    /// Header probe during negotiation (file I/O is synchronous, so a
    /// ready future carries the result).
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
            let samples = parse_fragments(&data, header.timescale)?;

            let mut sequence = 0u64;
            for s in samples {
                let mut annexb = s.annexb;
                if sequence == 0 && !has_sps(&annexb) {
                    // out-of-band parameter sets: prepend so a decoder can
                    // start (our own writer keeps them in-band).
                    let mut with_sets = Vec::new();
                    for set in [&header.sps, &header.pps] {
                        with_sets.extend_from_slice(&[0, 0, 0, 1]);
                        with_sets.extend_from_slice(set);
                    }
                    with_sets.extend_from_slice(&annexb);
                    annexb = with_sets;
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        annexb.into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: s.pts_ns,
                        dts_ns: s.pts_ns,
                        duration_ns: s.duration_ns,
                        capture_ns: s.pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                    },
                    sequence,
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
    annexb: Vec<u8>,
    pts_ns: u64,
    duration_ns: u64,
}

// --- box parsing -----------------------------------------------------------

fn be32(data: &[u8], at: usize) -> Result<u32, G2gError> {
    data.get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().expect("4 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

fn be64(data: &[u8], at: usize) -> Result<u64, G2gError> {
    data.get(at..at + 8)
        .map(|b| u64::from_be_bytes(b.try_into().expect("8 bytes")))
        .ok_or(G2gError::CapsMismatch)
}

/// Iterate the child boxes of `data`, yielding `(fourcc, payload)`.
fn boxes(data: &[u8]) -> impl Iterator<Item = (&[u8; 4], &[u8])> {
    let mut i = 0usize;
    core::iter::from_fn(move || {
        if i + 8 > data.len() {
            return None;
        }
        let size = u32::from_be_bytes(data[i..i + 4].try_into().expect("4 bytes")) as usize;
        if size < 8 || i + size > data.len() {
            return None;
        }
        let kind: &[u8; 4] = data[i + 4..i + 8].try_into().expect("4 bytes");
        let payload = &data[i + 8..i + size];
        i += size;
        Some((kind, payload))
    })
}

fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, p)| p)
}

/// Descend a path of nested boxes.
fn find_path<'a>(mut data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    for kind in path {
        data = find_box(data, kind)?;
    }
    Some(data)
}

fn parse_header(data: &[u8]) -> Result<Header, G2gError> {
    let moov = find_box(data, b"moov").ok_or(G2gError::CapsMismatch)?;
    let trak = find_box(moov, b"trak").ok_or(G2gError::CapsMismatch)?;

    // tkhd v0: width/height as 16.16 at payload offset 76/80 (after the
    // 4-byte version/flags).
    let tkhd = find_box(trak, b"tkhd").ok_or(G2gError::CapsMismatch)?;
    if tkhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let width = be32(tkhd, 76)? >> 16;
    let height = be32(tkhd, 80)? >> 16;

    // mdhd v0: timescale at payload offset 12.
    let mdia = find_box(trak, b"mdia").ok_or(G2gError::CapsMismatch)?;
    let mdhd = find_box(mdia, b"mdhd").ok_or(G2gError::CapsMismatch)?;
    if mdhd.first() != Some(&0) {
        return Err(G2gError::CapsMismatch);
    }
    let timescale = be32(mdhd, 12)?;
    if timescale == 0 {
        return Err(G2gError::CapsMismatch);
    }

    // stsd's first entry must be avc1; its avcC carries the parameter sets.
    let stsd = find_path(mdia, &[b"minf", b"stbl", b"stsd"]).ok_or(G2gError::CapsMismatch)?;
    // full box: version/flags + entry count, then the first sample entry.
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
    let avc1 = find_box(entries, b"avc1").ok_or(G2gError::CapsMismatch)?;
    // avc1 sample entry: 78 bytes of fields before the nested boxes.
    let avc1_children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
    let avcc = find_box(avc1_children, b"avcC").ok_or(G2gError::CapsMismatch)?;
    let (sps, pps) = parse_avcc(avcc)?;

    Ok(Header {
        width,
        height,
        timescale,
        sps,
        pps,
    })
}

/// First SPS and PPS out of an `avcC` payload.
fn parse_avcc(avcc: &[u8]) -> Result<(Vec<u8>, Vec<u8>), G2gError> {
    // 5 fixed bytes, then SPS count (low 5 bits).
    let sps_count = avcc.get(5).map(|b| b & 0x1F).ok_or(G2gError::CapsMismatch)?;
    if sps_count == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let sps_len = u16::from_be_bytes(
        avcc.get(6..8)
            .ok_or(G2gError::CapsMismatch)?
            .try_into()
            .expect("2 bytes"),
    ) as usize;
    let sps = avcc
        .get(8..8 + sps_len)
        .ok_or(G2gError::CapsMismatch)?
        .to_vec();
    let mut at = 8 + sps_len;
    let pps_count = *avcc.get(at).ok_or(G2gError::CapsMismatch)?;
    if pps_count == 0 {
        return Err(G2gError::CapsMismatch);
    }
    at += 1;
    let pps_len = u16::from_be_bytes(
        avcc.get(at..at + 2)
            .ok_or(G2gError::CapsMismatch)?
            .try_into()
            .expect("2 bytes"),
    ) as usize;
    at += 2;
    let pps = avcc
        .get(at..at + pps_len)
        .ok_or(G2gError::CapsMismatch)?
        .to_vec();
    Ok((sps, pps))
}

/// Walk the `moof`+`mdat` pairs and split every sample out of its `mdat`,
/// converting AVCC framing back to Annex-B.
fn parse_fragments(data: &[u8], timescale: u32) -> Result<Vec<Sample>, G2gError> {
    let mut samples = Vec::new();
    let mut pending: Option<Vec<(u32, u64)>> = None; // (size, pts_ns) per sample
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
                    return Err(G2gError::CapsMismatch); // mdat without moof
                };
                let mut at = 0usize;
                for (i, (size, pts_ns)) in tagged.iter().enumerate() {
                    let raw = payload
                        .get(at..at + *size as usize)
                        .ok_or(G2gError::CapsMismatch)?;
                    samples.push(Sample {
                        annexb: avcc_to_annexb(raw)?,
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
        return Err(G2gError::CapsMismatch); // trailing moof without mdat
    }
    Ok(samples)
}

/// `trun` v0 with explicit sample sizes; returns (sizes, durations) with a
/// zero duration when the stream omits it.
fn parse_trun(trun: &[u8]) -> Result<(Vec<u32>, Vec<u32>), G2gError> {
    if trun.first() != Some(&0) {
        return Err(G2gError::CapsMismatch); // v1 (signed cts) unsupported
    }
    let flags = be32(trun, 0)? & 0x00FF_FFFF;
    if flags & 0x200 == 0 {
        return Err(G2gError::CapsMismatch); // sizes must be explicit
    }
    let count = be32(trun, 4)? as usize;
    let mut at = 8usize;
    if flags & 0x1 != 0 {
        at += 4; // data offset (sequential mdat split makes it redundant)
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

/// 4-byte-length-prefixed AVCC NALUs back to Annex-B start codes.
fn avcc_to_annexb(avcc: &[u8]) -> Result<Vec<u8>, G2gError> {
    let mut out = Vec::with_capacity(avcc.len());
    let mut at = 0usize;
    while at < avcc.len() {
        let len = be32(avcc, at)? as usize;
        at += 4;
        let nalu = avcc.get(at..at + len).ok_or(G2gError::CapsMismatch)?;
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nalu);
        at += len;
    }
    Ok(out)
}

fn has_sps(annexb: &[u8]) -> bool {
    // a leading start code followed by an SPS NAL (type 7)
    annexb.len() > 4 && annexb[..4] == [0, 0, 0, 1] && annexb[4] & 0x1F == 7
}

fn timescale_to_ns(t: u64, timescale: u32) -> u64 {
    t.saturating_mul(1_000_000_000) / timescale as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn avcc_to_annexb_round_trips_length_prefixes() {
        let avcc = [0, 0, 0, 3, 0x67, 1, 2, 0, 0, 0, 2, 0x65, 3];
        let annexb = avcc_to_annexb(&avcc).unwrap();
        assert_eq!(annexb, vec![0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 3]);
        // a truncated NALU fails loud
        assert!(avcc_to_annexb(&[0, 0, 0, 9, 1]).is_err());
    }

    #[test]
    fn trun_parser_reads_the_writer_profile() {
        // flags 0x701: data offset + duration + size + flags, one sample.
        let mut p = vec![0u8, 0, 7, 1];
        p.extend_from_slice(&1u32.to_be_bytes()); // count
        p.extend_from_slice(&120u32.to_be_bytes()); // data offset
        p.extend_from_slice(&3000u32.to_be_bytes()); // duration
        p.extend_from_slice(&77u32.to_be_bytes()); // size
        p.extend_from_slice(&0x0200_0000u32.to_be_bytes()); // sample flags
        let (sizes, durs) = parse_trun(&p).unwrap();
        assert_eq!(sizes, vec![77]);
        assert_eq!(durs, vec![3000]);
    }

    #[test]
    fn timescale_conversion_inverts_the_sink() {
        // the sink writes 90 kHz; 2999 ticks is the 33.33 ms frame
        assert_eq!(timescale_to_ns(90_000, 90_000), 1_000_000_000);
        assert_eq!(timescale_to_ns(2999, 90_000), 33_322_222);
    }

    #[test]
    fn sps_detection_reads_the_first_nal_type() {
        assert!(has_sps(&[0, 0, 0, 1, 0x67, 0xAA]));
        assert!(!has_sps(&[0, 0, 0, 1, 0x65, 0xAA]));
    }
}
