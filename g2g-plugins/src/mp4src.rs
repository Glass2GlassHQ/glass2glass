//! Fragmented-MP4 demuxer source (M28, HEVC in M31), the read-side counterpart
//! of `Mp4Sink`: parses a single-video-track fMP4 and emits Annex-B H.264 or
//! H.265 access units with their recovered timing, so a recording plays back
//! through `MfDecode` / `FfmpegH264Dec` exactly like a live stream.
//!
//! Caps discovery is the M18 async-source path: `intercept_caps` reads the
//! file's `ftyp`/`moov` (dims from `tkhd`, codec + parameter sets from the
//! `avc1`/`avcC` or `hvc1`/`hvcC` sample entry, timescale from `mdhd`) before
//! negotiation, so downstream solves against the real geometry. The fragment
//! scan happens in `run`.
//!
//! Supported profile: what `Mp4Sink` writes and CMAF-style single-track
//! files generally share: one video track, `trun` v0 with explicit sample
//! sizes, `default-base-is-moof` data offsets landing on the following
//! `mdat`'s payload. Anything else fails loud rather than emitting a
//! corrupt bitstream. If the first sample carries no in-band parameter sets,
//! the ones from the config record are prepended so a decoder can start.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    BusHandle, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, Rate, Segment, VideoCodec,
};

use crate::filesink::io_err;

#[derive(Debug)]
struct Header {
    codec: VideoCodec,
    width: u32,
    height: u32,
    timescale: u32,
    /// Parameter-set NALUs in container order (SPS,PPS for H.264; VPS,SPS,PPS
    /// for H.265), prepended to the first sample if it carries none in-band.
    param_sets: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub struct Mp4Src {
    path: PathBuf,
    header: Option<Header>,
    configured: bool,
    bus: Option<BusHandle>,
    seek: Option<SeekController>,
}

impl Mp4Src {
    /// The file is read during caps probing and `run`; construction has no
    /// filesystem side effects.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: None,
            configured: false,
            bus: None,
            seek: None,
        }
    }

    /// Attach the pipeline bus so the file's `moov/udta/meta/ilst` metadata posts
    /// as a [`BusMessage::Tag`] once read.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Make the source seekable: `run` polls `controller` between frames and, on a
    /// flushing seek, emits `Flush`, repositions to the keyframe at or before the
    /// target, emits the post-flush `Segment`, and resumes. The application keeps a
    /// clone of the controller to drive scrubbing / editing.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        if self.header.is_none() {
            let data = std::fs::read(&self.path).map_err(io_err)?;
            self.header = Some(parse_header(&data)?);
        }
        let h = self.header.as_ref().expect("just parsed");
        Ok(Caps::CompressedVideo {
            codec: h.codec,
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
            // Surface the file's metadata once, before the samples flow.
            if let Some(bus) = &self.bus {
                if let Some(moov) = find_box(&data, b"moov") {
                    let tags = parse_ilst_tags(moov);
                    if !tags.is_empty() {
                        bus.try_post(BusMessage::Tag(tags));
                    }
                }
            }
            let header = self.header.as_ref().expect("parsed above");
            let samples = parse_fragments(&data, header.timescale, header.codec)?;

            let mut sequence = 0u64;
            // The next emitted frame is a (re)start: prepend the out-of-band
            // parameter sets if it lacks them, so a decoder can resume. Set again
            // after every seek, since the landed keyframe also needs them.
            let mut need_param_sets = true;
            let mut i = 0usize;
            while i < samples.len() {
                // A flushing seek repositions to the keyframe at or before the
                // target before the next frame is produced (GStreamer-style:
                // upstream to the source, latest-wins).
                if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                    if seek.is_flush() {
                        out.push(PipelinePacket::Flush).await?;
                        i = keyframe_index_for(&samples, seek.start);
                        need_param_sets = true;
                        out.push(PipelinePacket::Segment(Segment::for_flush_seek(&seek, None)))
                            .await?;
                    }
                    continue; // re-evaluate from the repositioned index
                }

                let s = &samples[i];
                let mut annexb = s.annexb.clone();
                if need_param_sets && !starts_with_param_set(&annexb, header.codec) {
                    // out-of-band parameter sets: prepend so a decoder can
                    // start (our own writer keeps them in-band).
                    let mut with_sets = Vec::new();
                    for set in &header.param_sets {
                        with_sets.extend_from_slice(&[0, 0, 0, 1]);
                        with_sets.extend_from_slice(set);
                    }
                    with_sets.extend_from_slice(&annexb);
                    annexb = with_sets;
                }
                need_param_sets = false;
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
                    meta: Default::default(),
                };
                sequence += 1;
                i += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }
}

/// The index of the keyframe at or before `target_ns` (GStreamer `SNAP_BEFORE`,
/// so a decoder can resume from a clean reference); 0 when none precedes it.
fn keyframe_index_for(samples: &[Sample], target_ns: u64) -> usize {
    samples
        .iter()
        .enumerate()
        .rfind(|(_, s)| s.keyframe && s.pts_ns <= target_ns)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Whether an Annex-B access unit contains an IDR picture (the keyframe a seek
/// snaps to). Mp4Src emits 4-byte start codes (from `avcc_to_annexb`), so NAL
/// boundaries are `00 00 00 01`. H.264 IDR is NAL type 5; H.265 IDR is 19/20.
fn contains_keyframe(annexb: &[u8], codec: VideoCodec) -> bool {
    annexb
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == [0, 0, 0, 1])
        .any(|(at, _)| {
            annexb.get(at + 4).is_some_and(|&b| match codec {
                VideoCodec::H265 => matches!((b >> 1) & 0x3F, 19 | 20),
                _ => b & 0x1F == 5,
            })
        })
}

#[derive(Debug)]
struct Sample {
    annexb: Vec<u8>,
    pts_ns: u64,
    duration_ns: u64,
    /// Whether the access unit carries an IDR picture (a seek snap point).
    keyframe: bool,
}

// box read primitives are shared across the MP4 elements.
use crate::mp4box::{be32, be64, boxes, find_box, find_path, parse_ilst_tags};

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

    // stsd's first entry is the visual sample entry: avc1/avcC (H.264) or
    // hvc1/hev1 with hvcC (H.265). Its config record carries the parameter sets.
    let stsd = find_path(mdia, &[b"minf", b"stbl", b"stsd"]).ok_or(G2gError::CapsMismatch)?;
    // full box: version/flags + entry count, then the first sample entry.
    let entries = stsd.get(8..).ok_or(G2gError::CapsMismatch)?;
    // visual sample entry: 78 bytes of fixed fields before the nested boxes.
    let (codec, param_sets) = if let Some(avc1) = find_box(entries, b"avc1") {
        let children = avc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let avcc = find_box(children, b"avcC").ok_or(G2gError::CapsMismatch)?;
        let (sps, pps) = parse_avcc(avcc)?;
        (VideoCodec::H264, Vec::from([sps, pps]))
    } else if let Some(hvc1) =
        find_box(entries, b"hvc1").or_else(|| find_box(entries, b"hev1"))
    {
        let children = hvc1.get(78..).ok_or(G2gError::CapsMismatch)?;
        let hvcc = find_box(children, b"hvcC").ok_or(G2gError::CapsMismatch)?;
        (VideoCodec::H265, parse_hvcc(hvcc)?)
    } else {
        return Err(G2gError::CapsMismatch);
    };

    Ok(Header {
        codec,
        width,
        height,
        timescale,
        param_sets,
    })
}

/// Parameter-set NALUs out of an `hvcC` payload, in array order (VPS, SPS,
/// PPS). Fixed 22-byte prefix (config version + 12-byte general PTL +
/// descriptive fields), then `numOfArrays`, then per-array NAL lists.
fn parse_hvcc(hvcc: &[u8]) -> Result<Vec<Vec<u8>>, G2gError> {
    let num_arrays = *hvcc.get(22).ok_or(G2gError::CapsMismatch)?;
    let mut at = 23usize;
    let mut sets = Vec::new();
    for _ in 0..num_arrays {
        // array header byte: array_completeness | reserved | NAL_unit_type.
        at += 1;
        let num_nalus = u16::from_be_bytes(
            hvcc.get(at..at + 2)
                .ok_or(G2gError::CapsMismatch)?
                .try_into()
                .expect("2 bytes"),
        );
        at += 2;
        for _ in 0..num_nalus {
            let len = u16::from_be_bytes(
                hvcc.get(at..at + 2)
                    .ok_or(G2gError::CapsMismatch)?
                    .try_into()
                    .expect("2 bytes"),
            ) as usize;
            at += 2;
            let nalu = hvcc.get(at..at + len).ok_or(G2gError::CapsMismatch)?;
            sets.push(nalu.to_vec());
            at += len;
        }
    }
    if sets.is_empty() {
        return Err(G2gError::CapsMismatch);
    }
    Ok(sets)
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
/// converting AVCC framing back to Annex-B. `codec` selects the IDR NAL type used
/// to flag keyframes (the seek snap points).
fn parse_fragments(data: &[u8], timescale: u32, codec: VideoCodec) -> Result<Vec<Sample>, G2gError> {
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
                    let annexb = avcc_to_annexb(raw)?;
                    let keyframe = contains_keyframe(&annexb, codec);
                    samples.push(Sample {
                        annexb,
                        pts_ns: *pts_ns,
                        duration_ns: durations[i],
                        keyframe,
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

/// Whether the access unit already opens with a parameter-set NAL (so the
/// config-record sets need not be prepended): H.264 SPS(7), H.265 VPS(32).
fn starts_with_param_set(annexb: &[u8], codec: VideoCodec) -> bool {
    if annexb.len() <= 4 || annexb[..4] != [0, 0, 0, 1] {
        return false;
    }
    match codec {
        VideoCodec::H265 => (annexb[4] >> 1) & 0x3F == 32,
        _ => annexb[4] & 0x1F == 7,
    }
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
        // H.264: SPS is type 7 (0x67), an IDR slice (0x65) is not a param set.
        assert!(starts_with_param_set(&[0, 0, 0, 1, 0x67, 0xAA], VideoCodec::H264));
        assert!(!starts_with_param_set(&[0, 0, 0, 1, 0x65, 0xAA], VideoCodec::H264));
        // H.265: VPS is type 32 (0x40), an IDR (0x26) is not a param set.
        assert!(starts_with_param_set(&[0, 0, 0, 1, 0x40, 0x01], VideoCodec::H265));
        assert!(!starts_with_param_set(&[0, 0, 0, 1, 0x26, 0x01], VideoCodec::H265));
    }

    #[test]
    fn hvcc_parser_recovers_arrays_in_order() {
        // build an hvcC the way Mp4Sink does: 22-byte prefix, numOfArrays,
        // then VPS/SPS/PPS arrays of one NALU each.
        let vps: &[u8] = &[0x40, 0x01, 0xAA];
        let sps: &[u8] = &[0x42, 0x01, 0xBB, 0xCC];
        let pps: &[u8] = &[0x44, 0x01, 0xDD];
        let mut p = vec![0u8; 22];
        p[22 - 22] = 1; // configuration version at offset 0
        p.push(3); // numOfArrays at offset 22
        for (ty, nalu) in [(32u8, vps), (33u8, sps), (34u8, pps)] {
            p.push(0x80 | ty);
            p.extend_from_slice(&1u16.to_be_bytes());
            p.extend_from_slice(&(nalu.len() as u16).to_be_bytes());
            p.extend_from_slice(nalu);
        }
        let sets = parse_hvcc(&p).unwrap();
        assert_eq!(sets, vec![vps.to_vec(), sps.to_vec(), pps.to_vec()]);
    }
}
