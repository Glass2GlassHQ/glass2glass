//! Ogg demuxer element (M116): `Caps::ByteStream{Ogg}` in, the selected audio
//! elementary stream out (`Caps::Audio{Opus}` default, `stream=flac` for the
//! Ogg-FLAC mapping (M775), `stream=vorbis` for Vorbis (M777)).
//!
//! Wraps the pure [`crate::ogg::OggDemuxer`], the Ogg sibling of
//! [`crate::mkvdemux::MkvDemux`]: it reassembles the logical bitstream's packets,
//! skips the codec setup headers, and forwards the audio packets. Once the
//! identification header is parsed the channels / rate are known, so the demuxer
//! refines the caps via `CapsChanged` before the first frame. The codec header
//! goes downstream in-band first (`OpusHead` for the decoder's pre-skip; the
//! native `fLaC` STREAMINFO as the [`crate::ffmpegaudiodec`] extradata). CPU,
//! `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.opus, caps=ByteStream{Ogg}) ! oggdemux ! <opus decoder>
//! filesrc(location=x.oga) ! oggdemux stream=flac ! <flac decoder>
//! ```
//!
//! Scope: one logical bitstream; a stream not matching the `stream` selection is
//! parsed but not forwarded.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SeekController;
use g2g_core::{
    AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Seek, Segment, Tag,
    TagList,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::ogg::{OggCodec, OggDemuxer};

/// Number of 48 kHz samples an Opus packet decodes to, from its TOC byte
/// (RFC 6716 §3.1): config (top 5 bits) gives the per-frame duration, the frame
/// count code (low 2 bits) the frame count. Opus is always 48 kHz, so this maps
/// directly to a duration. `0` for an empty packet.
fn opus_packet_samples(pkt: &[u8]) -> u32 {
    let Some(&toc) = pkt.first() else {
        return 0;
    };
    let frame_samples: u32 = match toc >> 3 {
        // SILK NB/MB/WB and Hybrid SWB/FB: 10 / 20 / 40 / 60 ms.
        0 | 4 | 8 => 480,
        1 | 5 | 9 => 960,
        2 | 6 | 10 => 1920,
        3 | 7 | 11 => 2880,
        12 | 14 => 480,
        13 | 15 => 960,
        // CELT NB/WB/SWB/FB: 2.5 / 5 / 10 / 20 ms.
        16 | 20 | 24 | 28 => 120,
        17 | 21 | 25 | 29 => 240,
        18 | 22 | 26 | 30 => 480,
        _ => 960,
    };
    let frames: u32 = match toc & 0x3 {
        0 => 1,
        1 | 2 => 2,
        // Code 3: the frame count is the low 6 bits of the following byte.
        _ => pkt.get(1).map(|b| (b & 0x3F) as u32).unwrap_or(1).max(1),
    };
    frame_samples.saturating_mul(frames)
}

/// Convert a 48 kHz sample count to nanoseconds.
fn opus_samples_to_ns(samples: u64) -> u64 {
    samples.saturating_mul(1_000_000_000) / 48_000
}

/// Demuxes an Ogg byte stream into its Opus audio elementary stream.
#[derive(Debug)]
pub struct OggDemux {
    demux: OggDemuxer,
    configured: bool,
    emitted: u64,
    last_caps: Option<Caps>,
    bus: Option<BusHandle>,
    tags_posted: bool,
    /// Running stream-time (ns) of the next audio packet, accumulated from each
    /// Opus packet's decoded duration (the demuxer carries no per-packet PTS).
    /// FLAC times from `decoded_samples` instead.
    pts_ns: u64,
    /// Running count of decoded samples (per channel; 48 kHz incl. pre-skip for
    /// Opus, the STREAMINFO rate for FLAC) over the audio packets seen so far.
    /// For Opus, compared against the end-of-stream granule position to trim
    /// the encoder padding off the final packet(s).
    decoded_samples: u64,
    /// Whether the in-band codec header (`OpusHead` / the native `fLaC`
    /// STREAMINFO) was forwarded to the decoder. Reset on a flush so the
    /// re-read stream re-sends it.
    head_forwarded: bool,
    /// The logical stream to emit (`stream` property): Opus by default, Flac
    /// for the Ogg-FLAC mapping, or Vorbis. A stream of any other codec is
    /// dropped.
    stream: OggCodec,
    /// Vorbis packet-duration tables (M778), parsed from the ident + setup
    /// headers once both land. `None` until then / for other codecs / when the
    /// setup scan fails (packets then ride untimed and untrimmed).
    vorbis: Option<crate::ogg::VorbisTiming>,
    /// The previous Vorbis audio packet's block size (`None` before the first,
    /// whose window has no predecessor: it primes the overlap and decodes to
    /// nothing, counting `blocksize / 2` on the timeline).
    prev_blocksize: Option<u32>,
    /// Vorbis timeline anchor (M778): the excess of the natural packet
    /// durations over the first audio page's granule position, clipped off the
    /// front of the emitted timeline (initial encoder priming). `Some(0)` when
    /// the first page is the EOS page (an end clamp, not an anchor).
    anchor_offset: Option<u64>,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
}

impl Default for OggDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl OggDemux {
    pub fn new() -> Self {
        Self {
            demux: OggDemuxer::new(),
            configured: false,
            emitted: 0,
            last_caps: None,
            bus: None,
            tags_posted: false,
            pts_ns: 0,
            decoded_samples: 0,
            head_forwarded: false,
            stream: OggCodec::Opus,
            vorbis: None,
            prev_blocksize: None,
            anchor_offset: None,
            seek: DemuxSeek::default(),
        }
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the packet
    /// at/after the target (every audio packet is a resync point).
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop the Ogg
    /// page/packet state and the running PTS, which the re-read stream
    /// re-establishes from its first page. The caps are unchanged (same file), so
    /// `last_caps` is kept (no redundant `CapsChanged`).
    fn reset_parser(&mut self) {
        self.demux = OggDemuxer::new();
        self.pts_ns = 0;
        self.decoded_samples = 0;
        self.head_forwarded = false;
        self.vorbis = None;
        self.prev_blocksize = None;
        self.anchor_offset = None;
    }

    /// Attach the pipeline bus so the stream's VorbisComment metadata posts as a
    /// [`BusMessage::Tag`] once the comment header is parsed.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Count of audio packets forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        }
    }

    /// The placeholder output for the selected stream: a sentinel channels/rate,
    /// refined from the identification header via `CapsChanged` once parsed.
    fn output_caps(stream: OggCodec) -> Caps {
        Caps::Audio {
            format: match stream {
                OggCodec::Flac => AudioFormat::Flac,
                OggCodec::Vorbis => AudioFormat::Vorbis,
                _ => AudioFormat::Opus,
            },
            channels: 0,
            sample_rate: 0,
        }
    }

    /// The concrete caps once the identification header is parsed, or `None`
    /// until then / when the stream's codec is not the selected one.
    fn concrete_caps(&self) -> Option<Caps> {
        let info = self.demux.info()?;
        if info.codec != self.stream || info.sample_rate == 0 {
            return None;
        }
        let format = match info.codec {
            OggCodec::Opus => AudioFormat::Opus,
            OggCodec::Flac => AudioFormat::Flac,
            OggCodec::Vorbis => AudioFormat::Vorbis,
            _ => return None,
        };
        Some(Caps::Audio {
            format,
            channels: info.channels.max(1),
            sample_rate: info.sample_rate,
        })
    }

    /// Emit a `CapsChanged` once the stream parameters are known, then forward
    /// each audio packet. A stream whose codec is not the `stream` selection is
    /// drained and dropped (the output pad is typed by the selection).
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        // Surface the stream's metadata once, as soon as the comment header lands.
        if !self.tags_posted && self.bus.is_some() {
            if let Some(comment) = self.demux.comment_header() {
                let tags = parse_vorbis_comment(comment);
                self.tags_posted = true;
                if !tags.is_empty() {
                    if let Some(bus) = &self.bus {
                        bus.try_post(BusMessage::Tag(tags));
                    }
                }
            }
        }
        if let Some(caps) = self.concrete_caps() {
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
        }
        let selected = self.demux.info().map(|i| i.codec) == Some(self.stream);
        // Forward the codec config in-band once, before the first audio packet:
        // OpusHead as-is (the decoder reads its pre-skip from it); for FLAC the
        // native `fLaC` STREAMINFO the mapping's first packet embeds at offset 9,
        // with the last-metadata-block flag set so the standalone header
        // terminates (detect() already validated the layout); for Vorbis all
        // three headers (ident / comment / setup, each with its unambiguous
        // `\x0Nvorbis` prefix; the decoder needs ident + setup). Codec config,
        // not audio, so the decoder consumes it without emitting PCM.
        if selected && !self.head_forwarded {
            let mut heads: Vec<Vec<u8>> = Vec::new();
            match self.stream {
                OggCodec::Vorbis => {
                    if let (Some(h), Some(c), Some(s)) = (
                        self.demux.head_header(),
                        self.demux.comment_header(),
                        self.demux.setup_header(),
                    ) {
                        heads.extend([h.to_vec(), c.to_vec(), s.to_vec()]);
                    }
                }
                OggCodec::Flac => {
                    if let Some(head) = self.demux.head_header() {
                        let mut native = head[9..].to_vec();
                        native[4] |= 0x80;
                        heads.push(native);
                    }
                }
                _ => {
                    if let Some(head) = self.demux.head_header() {
                        heads.push(head.to_vec());
                    }
                }
            }
            if !heads.is_empty() {
                self.head_forwarded = true;
                for head in heads {
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(head.into_boxed_slice())),
                        FrameTiming::default(),
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
            }
        }
        // Vorbis timing tables (M778): parse once ident + setup have landed.
        if self.stream == OggCodec::Vorbis && self.vorbis.is_none() {
            if let (Some(h), Some(s)) = (self.demux.head_header(), self.demux.setup_header()) {
                self.vorbis = crate::ogg::VorbisTiming::parse(h, s);
            }
        }
        // Total decodable samples (incl. pre-skip); the tail beyond it is padding.
        let end_granule = self.demux.end_granule();
        let sample_rate = self.demux.info().map(|i| i.sample_rate).unwrap_or(0);
        let packets = self.demux.take_packets();
        // Vorbis timeline anchor (M778): the first audio page's granule names
        // the timeline position after its last packet, so the excess of the
        // natural durations over it is initial priming, clipped off the front
        // (ffmpeg anchors the same way, as a negative first pts). An EOS first
        // page is an end clamp instead: anchor at zero.
        if self.stream == OggCodec::Vorbis && self.anchor_offset.is_none() {
            if let (Some(t), Some((gp, n, eos))) =
                (self.vorbis.as_ref(), self.demux.first_data_granule())
            {
                self.anchor_offset = if eos || self.decoded_samples > 0 {
                    // Already streaming (a headerless mid-join): no anchor.
                    Some(0)
                } else {
                    // Lapped natural durations over the first page's packets.
                    let mut prev: Option<u32> = None;
                    let mut cum = 0u64;
                    for p in packets.iter().take(n as usize) {
                        if let Some(c) = t.packet_blocksize(p) {
                            cum = cum.saturating_add(u64::from((prev.unwrap_or(c) + c) / 4));
                            prev = Some(c);
                        }
                    }
                    Some(cum.saturating_sub(gp))
                };
            }
        }
        for packet in packets {
            if !selected {
                continue;
            }
            let pkt_samples = match self.stream {
                // Each Ogg-FLAC audio packet is one whole frame; its header
                // carries the block size.
                OggCodec::Flac => crate::flacparse::parse_frame_header(&packet)
                    .map(|h| u64::from(h.block_size))
                    .unwrap_or(0),
                OggCodec::Opus => opus_packet_samples(&packet) as u64,
                // Vorbis (M778): the lapped `(prev + cur) / 4` from the mode
                // tables (the stream's first packet gets its nominal `bs / 2`
                // while decoding to nothing; the anchor clips it off the front).
                OggCodec::Vorbis => match self
                    .vorbis
                    .as_ref()
                    .and_then(|t| t.packet_blocksize(&packet))
                {
                    Some(c) => {
                        let s = u64::from((self.prev_blocksize.unwrap_or(c) + c) / 4);
                        self.prev_blocksize = Some(c);
                        s
                    }
                    None => 0,
                },
                _ => 0,
            };
            let decoded_before = self.decoded_samples;
            self.decoded_samples = decoded_before.saturating_add(pkt_samples);
            // Anchored timeline position (M778): the Vorbis initial-priming
            // offset is clipped off the front; Opus / FLAC anchor at zero.
            let off = self.anchor_offset.unwrap_or(0);
            let anchored_before = decoded_before.saturating_sub(off);
            let anchored_after = decoded_before
                .saturating_add(pkt_samples)
                .saturating_sub(off);
            // End-of-stream trim (Opus + Vorbis; FLAC carries no padding
            // convention): keep only the samples up to the final granule
            // position. A packet wholly past it is pure padding, so drop it; a
            // straddling packet is kept but marked short via `duration_ns`,
            // which the decoder honors. Without a known end granule keep the
            // packet whole.
            let span = anchored_after.saturating_sub(anchored_before);
            let keep = match end_granule {
                Some(gp) if matches!(self.stream, OggCodec::Opus | OggCodec::Vorbis) => {
                    gp.saturating_sub(anchored_before).min(span)
                }
                _ => span,
            };
            let (pts_ns, duration_ns) = match self.stream {
                OggCodec::Flac | OggCodec::Vorbis => {
                    // Sample-accurate at the header rate (per-packet rounding
                    // would drift at non-48 kHz rates).
                    let ns =
                        |s: u64| (s as u128 * 1_000_000_000 / sample_rate.max(1) as u128) as u64;
                    let pts = ns(anchored_before);
                    (
                        pts,
                        ns(anchored_before.saturating_add(keep)).saturating_sub(pts),
                    )
                }
                OggCodec::Opus => {
                    let pts = self.pts_ns;
                    self.pts_ns = pts.saturating_add(opus_samples_to_ns(pkt_samples));
                    (pts, opus_samples_to_ns(keep))
                }
                _ => (0, 0),
            };
            // Drop a timed packet wholly past the end granule (Opus padding /
            // Vorbis tail). A head-clipped packet (anchored position still 0,
            // e.g. the Vorbis priming packet) and an untimed one (pkt_samples
            // 0: a Vorbis stream whose setup scan failed) always flow, since
            // the decoder needs them to prime its window.
            if keep == 0 && anchored_before > 0 && pkt_samples > 0 {
                continue;
            }
            // M362 seek: every audio packet is a resync point, so drop until the
            // first packet at/after the target, which emits a fresh segment.
            match self.seek.admit(pts_ns, true) {
                Admit::Drop => continue,
                Admit::Resume(start) => {
                    let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                Admit::Emit => {}
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(packet.into_boxed_slice())),
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns,
                    ..FrameTiming::default()
                },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for OggDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Ogg,
            } => CapsSet::one(Self::output_caps(stream)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Ogg
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            // M362: a pending app seek triggers an upstream byte-seek; until its
            // `Flush` returns, drop input so no stale pre-seek packets are emitted.
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice);
                    self.emit_ready(out).await?;
                }
                // The upstream byte-seek's flush: reset the parser, then re-sync
                // from the re-read stream. Forward it downstream.
                PipelinePacket::Flush => {
                    self.seek.on_flush();
                    self.reset_parser();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    // Emit any final packets; the runner's transform arm forwards EOS.
                    self.emit_ready(out).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        OGGDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                self.stream = match value.as_str().ok_or(PropError::Type)? {
                    "opus" => OggCodec::Opus,
                    "flac" => OggCodec::Flac,
                    "vorbis" => OggCodec::Vorbis,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(
                match self.stream {
                    OggCodec::Flac => "flac",
                    OggCodec::Vorbis => "vorbis",
                    _ => "opus",
                }
                .into(),
            )),
            _ => None,
        }
    }
}

/// `OggDemux`'s settable properties (M775).
static OGGDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "logical stream to emit: opus | flac | vorbis",
)];

impl PadTemplates for OggDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                Self::output_caps(OggCodec::Opus),
                Self::output_caps(OggCodec::Flac),
                Self::output_caps(OggCodec::Vorbis),
            ]))),
        ])
    }
}

/// Parse a VorbisComment metadata block into a [`TagList`]. Accepts the comment
/// header with its codec prefix (`OpusTags`, the Vorbis `\x03vorbis`, or a FLAC
/// VORBIS_COMMENT metadata block, type 4, whose 4-byte block header wraps the
/// same body): vendor string, then a count-prefixed list of `KEY=VALUE` UTF-8
/// fields (RFC 7845 §5.2 for Opus). Unparseable / truncated input yields
/// whatever was read so far.
fn parse_vorbis_comment(packet: &[u8]) -> TagList {
    let body = if let Some(rest) = packet.strip_prefix(b"OpusTags".as_slice()) {
        rest
    } else if let Some(rest) = packet.strip_prefix(b"\x03vorbis".as_slice()) {
        rest
    } else if packet.len() >= 4 && packet[0] & 0x7F == 4 {
        &packet[4..]
    } else {
        return TagList::new();
    };

    fn read_u32_le(b: &[u8], pos: &mut usize) -> Option<u32> {
        let s = b.get(*pos..*pos + 4)?;
        *pos += 4;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    let mut list = TagList::new();
    let mut pos = 0usize;
    let Some(vendor_len) = read_u32_le(body, &mut pos) else {
        return list;
    };
    pos = match pos.checked_add(vendor_len as usize) {
        Some(p) if p <= body.len() => p, // skip the vendor string
        _ => return list,
    };
    let Some(count) = read_u32_le(body, &mut pos) else {
        return list;
    };
    for _ in 0..count {
        let Some(len) = read_u32_le(body, &mut pos) else {
            break;
        };
        let Some(end) = pos.checked_add(len as usize) else {
            break;
        };
        let Some(field) = body.get(pos..end) else {
            break;
        };
        pos = end;
        if let Ok(s) = core::str::from_utf8(field) {
            if let Some((key, value)) = s.split_once('=') {
                list.push(Tag::from_key_value(key, value));
            }
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Dim, PushOutcome, Rate, RawVideoFormat};

    /// Build one Ogg page carrying `packets` for `serial` (mirrors the parser
    /// test helper).
    fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
        let mut table = Vec::new();
        let mut body = Vec::new();
        for p in packets {
            let mut n = p.len();
            loop {
                let seg = n.min(255);
                table.push(seg as u8);
                n -= seg;
                if seg < 255 {
                    break;
                }
            }
            body.extend_from_slice(p);
        }
        let mut out = b"OggS".to_vec();
        out.push(0);
        out.push(header_type);
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.push(table.len() as u8);
        out.extend_from_slice(&table);
        out.extend_from_slice(&body);
        out
    }

    fn opus_head(channels: u8) -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.push(1);
        h.push(channels);
        h.extend_from_slice(&[0, 0]);
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&[0, 0, 0]);
        h
    }

    /// An `OpusTags` comment header carrying `comments` (a "g2g" vendor string).
    fn opus_tags(comments: &[(&str, &str)]) -> Vec<u8> {
        let mut p = b"OpusTags".to_vec();
        let vendor: &[u8] = b"g2g";
        p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        p.extend_from_slice(vendor);
        p.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for (k, v) in comments {
            let field = [k.as_bytes(), b"=", v.as_bytes()].concat();
            p.extend_from_slice(&(field.len() as u32).to_le_bytes());
            p.extend_from_slice(&field);
        }
        p
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
        eos: bool,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
                        }
                    }
                    PipelinePacket::Eos => self.eos = true,
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[test]
    fn caps_byte_stream_in_opus_out() {
        let d = OggDemux::new();
        assert!(d.intercept_caps(&OggDemux::input_caps()).is_ok());
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
        // The Matroska byte stream is the wrong container.
        let mkv = Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        };
        assert!(d.intercept_caps(&mkv).is_err());
    }

    #[tokio::test]
    async fn demuxes_opus_with_refined_caps() {
        let serial = 7;
        let mut stream = Vec::new();
        stream.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
        stream.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags"]));
        stream.extend_from_slice(&page(0x00, serial, 2, &[&[0x11, 0x22], &[0x33]]));

        let mut d = OggDemux::new();
        d.configure_pipeline(&OggDemux::input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        d.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(
            sink.caps,
            alloc::vec![Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000
            }]
        );
        // OpusHead is forwarded in-band (the decoder's pre-skip source), ahead of
        // the two audio packets.
        assert_eq!(
            sink.frames,
            alloc::vec![opus_head(2), alloc::vec![0x11, 0x22], alloc::vec![0x33]]
        );
        assert!(
            !sink.eos,
            "EOS is forwarded by the runner's arm, not the element"
        );
        assert_eq!(d.emitted(), 3);
    }

    #[test]
    fn parse_vorbis_comment_reads_fields_and_rejects_non_comment() {
        let tags = parse_vorbis_comment(&opus_tags(&[("TITLE", "Song"), ("ENCODER", "libopus")]));
        assert_eq!(
            tags.tags(),
            &[Tag::Title("Song".into()), Tag::Encoder("libopus".into())]
        );
        // The identification header (OpusHead) is not a comment block.
        assert!(parse_vorbis_comment(&opus_head(2)).is_empty());
    }

    #[tokio::test]
    async fn posts_vorbis_comment_tags_on_the_bus() {
        use g2g_core::Bus;
        let (bus, handle) = Bus::new(8);
        let serial = 9;
        let mut stream = Vec::new();
        stream.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
        stream.extend_from_slice(&page(
            0x00,
            serial,
            1,
            &[&opus_tags(&[("TITLE", "Song"), ("ARTIST", "Band")])],
        ));
        stream.extend_from_slice(&page(0x00, serial, 2, &[&[0x10, 0x11]]));

        let mut d = OggDemux::new().with_bus(handle);
        d.configure_pipeline(&OggDemux::input_caps()).unwrap();
        let mut sink = CaptureSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let mut posted = None;
        while let Some(m) = bus.try_recv() {
            if let BusMessage::Tag(t) = m {
                posted = Some(t);
            }
        }
        let tags = posted.expect("a Tag message was posted");
        assert_eq!(
            tags.tags(),
            &[Tag::Title("Song".into()), Tag::Artist("Band".into())]
        );
    }
}
