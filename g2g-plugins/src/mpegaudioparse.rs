//! MPEG audio stream parser element (`mpegaudioparse`): an MPEG-1/2/2.5 Layer II
//! or Layer III byte stream in (`Caps::Audio{Mp2|Mp3}`, arbitrary chunks from
//! `filesrc`), one frame per buffer out.
//!
//! The audio sibling of [`crate::flacparse`]: an `.mp3` / `.mp2` file is a bare
//! sequence of self-syncing frames with a tag block glued to each end, and
//! [`crate::ffmpegaudiodec`] takes one frame per packet, so something has to
//! split the byte stream. Frame lengths come from
//! [`mpa_header`](crate::audioframe::mpa_header), shared with the program-stream
//! demuxer.
//!
//! Audio bytes alias the 11-bit frame sync often, so a candidate header is
//! trusted only when the header at its frame length is valid too and describes
//! the same stream (same version, layer and sample rate), or the stream ends
//! there. That is also how the parser resynchronizes after garbage.
//!
//! Three things in the stream are not audio and are dropped: an ID3v2 tag ahead
//! of the first frame, an ID3v1 block after the last (held back until `Eos`,
//! since only the end of the stream can tell a trailer from audio), and the
//! Xing / Info / VBRI header frame, a real MPEG frame whose payload is a VBR
//! seek table rather than sound. The tags reach the application on the bus.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_error, g2g_warn, AsyncElement, AudioFormat, BusHandle, BusMessage, Caps, CapsConstraint,
    CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, TagList,
};

use crate::audioframe::{
    is_vbr_header_frame, locate_frame, mpa_header, Located, MpaHeader, MpaLayer,
};
use crate::id3::{id3v2_len, parse_id3v1, parse_id3v2, ID3V1_LEN, ID3V2_HEADER_LEN};

/// Nanoseconds per second, the presentation-time unit.
const NS_PER_SECOND: u128 = 1_000_000_000;

/// The largest ID3v2 tag whose text frames are read. A tag is skipped whatever
/// its size, but reading one means holding it whole, and a tag past this is
/// carrying artwork rather than text.
const MAX_ID3V2_TAG_PARSED: usize = 1 << 20;

/// # Example
///
/// ```no_run
/// use g2g_plugins::mpegaudioparse::MpegAudioParse;
///
/// let parser = MpegAudioParse::new();
/// assert_eq!(parser.frames_emitted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct MpegAudioParse {
    configured: bool,
    bus: Option<BusHandle>,
    /// Unconsumed input bytes, starting at stream offset `buf_offset`.
    buf: Vec<u8>,
    buf_offset: u64,
    /// Whether the start of the stream has been examined for an ID3v2 tag.
    head_examined: bool,
    /// Bytes of ID3v2 tag still to drop from the front of the stream.
    id3v2_skip: usize,
    tags: TagList,
    tags_posted: bool,
    /// Whether a frame has been emitted, so the VBR header frame (only ever the
    /// first) is looked for once.
    first_frame_seen: bool,
    last_caps: Option<Caps>,
    /// Samples emitted since the last time base, the presentation-time counter.
    samples: u64,
    base_ns: u64,
    /// A presentation time from upstream and the stream offset its buffer began
    /// at, applied to the first frame that starts at or past it.
    pending_rebase: Option<(u64, u64)>,
    sequence: u64,
    log_name: LogName,
}

impl MpegAudioParse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the pipeline bus so the stream's ID3 tags reach the application as
    /// a [`BusMessage::Tag`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Count of audio frames emitted (the VBR header frame is not one).
    pub fn frames_emitted(&self) -> u64 {
        self.sequence
    }

    /// The ID3 tags read from the stream, empty until one is parsed.
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    /// Post the tags read so far, once. A stream carrying both an ID3v2 tag and
    /// an ID3v1 trailer posts the v2 one: the v1 block is the same metadata cut
    /// to 30 bytes a field.
    fn post_tags(&mut self) {
        if self.tags_posted || self.tags.is_empty() {
            return;
        }
        self.tags_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::Tag {
                tags: self.tags.clone(),
                program: None,
            });
        }
    }

    /// Drop `n` bytes from the front of the buffer.
    fn consume(&mut self, n: usize) {
        self.buf.drain(..n);
        self.buf_offset += n as u64;
    }

    /// Skip the ID3v2 tag at the head of the stream, reading its text frames on
    /// the way past. Returns whether the audio has been reached.
    fn skip_id3v2(&mut self, eos: bool) -> bool {
        if !self.head_examined {
            if self.buf.len() < ID3V2_HEADER_LEN && !eos {
                return false;
            }
            self.head_examined = true;
            self.id3v2_skip = id3v2_len(&self.buf).unwrap_or(0);
        }
        if self.id3v2_skip == 0 {
            return true;
        }
        let readable = self.id3v2_skip <= MAX_ID3V2_TAG_PARSED;
        if readable && self.buf.len() < self.id3v2_skip {
            return false; // the tags are read from the whole tag: wait for it
        }
        if readable {
            self.tags = parse_id3v2(&self.buf[..self.id3v2_skip]);
        }
        let drop = self.id3v2_skip.min(self.buf.len());
        self.consume(drop);
        self.id3v2_skip -= drop;
        self.id3v2_skip == 0
    }

    /// Drop the ID3v1 block at the end of the stream, reading its fields. Called
    /// once the last byte has arrived, the only point at which a trailing `TAG`
    /// block can be told from audio.
    fn strip_id3v1(&mut self) {
        let Some(start) = self.buf.len().checked_sub(ID3V1_LEN) else {
            return;
        };
        let Some(tags) = parse_id3v1(&self.buf[start..]) else {
            return;
        };
        if self.tags.is_empty() {
            self.tags = tags;
        }
        self.buf.truncate(start);
    }

    /// Emit every frame the buffer holds whole. Until `eos`, the last
    /// [`ID3V1_LEN`] bytes are held back: they may be the trailer.
    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if !self.skip_id3v2(eos) {
            return Ok(());
        }
        self.post_tags();
        loop {
            let limit = if eos {
                self.buf.len()
            } else {
                self.buf.len().saturating_sub(ID3V1_LEN)
            };
            let Located::Frame { start, len } = locate_frame::<MpaHeader>(&self.buf[..limit], eos)
            else {
                return Ok(());
            };
            if start > 0 {
                g2g_warn!(self, "resynchronized past {start} bytes of non-audio");
                self.consume(start);
            }
            let header = mpa_header(&self.buf).ok_or(G2gError::CapsMismatch)?;
            if header.layer == MpaLayer::One {
                g2g_error!(self, "MPEG audio Layer I: nothing here decodes it");
                return Err(G2gError::CapsMismatch);
            }
            let start_offset = self.buf_offset;
            let data: Vec<u8> = self.buf.drain(..len).collect();
            self.buf_offset += len as u64;
            if !self.first_frame_seen {
                self.first_frame_seen = true;
                if is_vbr_header_frame(&data, &header) {
                    continue; // a seek table, not sound
                }
            }
            self.emit(data, &header, start_offset, out).await?;
        }
    }

    /// Push one frame with the caps its header declares and a presentation time
    /// from the running sample count.
    async fn emit(
        &mut self,
        data: Vec<u8>,
        header: &MpaHeader,
        start_offset: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = Caps::Audio {
            format: audio_format(header.layer),
            channels: header.channels,
            sample_rate: header.sample_rate,
        };
        if self.last_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_caps = Some(caps);
        }
        if let Some((at, pts_ns)) = self.pending_rebase {
            if start_offset >= at {
                self.base_ns = pts_ns;
                self.samples = 0;
                self.pending_rebase = None;
            }
        }
        let rate = u128::from(header.sample_rate);
        let ns = |samples: u64| (u128::from(samples) * NS_PER_SECOND / rate) as u64;
        let pts_ns = self.base_ns + ns(self.samples);
        let duration_ns = ns(u64::from(header.samples_per_frame));
        self.samples += u64::from(header.samples_per_frame);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                // Every MPEG audio frame decodes on its own.
                keyframe: true,
                ..FrameTiming::default()
            },
            self.sequence,
        );
        self.sequence += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

/// The caps format a layer codes to. Layer I never reaches here (the parser
/// rejects it), so it maps to the `Mp2` its header is closest to.
fn audio_format(layer: MpaLayer) -> AudioFormat {
    match layer {
        MpaLayer::Three => AudioFormat::Mp3,
        _ => AudioFormat::Mp2,
    }
}

impl AsyncElement for MpegAudioParse {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Mp2 | AudioFormat::Mp3,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over MPEG audio of any channels/rate: the frame
    /// header refines them mid-stream, and `Caps::Audio` cannot express "MP3 at
    /// any channels/rate" in a single `Caps` (the same reason `aacparse` gives).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Mp2 | AudioFormat::Mp3,
                ..
            } => {
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG audio parser",
            "Codec/Parser/Audio",
            "Splits an MPEG audio stream into frames and refines caps",
            "g2g",
        )
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    // A byte source stamps every chunk 0, so only a real time
                    // (a demuxer's) re-bases the counter.
                    if let Some(pts) = frame.timing.pts().filter(|pts| *pts != 0) {
                        self.pending_rebase = Some((self.buf_offset + self.buf.len() as u64, pts));
                    }
                    self.buf.extend_from_slice(slice);
                    self.drain(false, out).await?;
                }
                // The last frame ends at the end of the stream, and only now can
                // a trailing `TAG` block be told from audio.
                PipelinePacket::Eos => {
                    self.strip_id3v1();
                    self.drain(true, out).await?;
                    self.post_tags();
                }
                PipelinePacket::Flush => {
                    self.buf.clear();
                    self.pending_rebase = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // The refined caps this element emits replace the upstream
                // declaration, so an incoming one is not forwarded.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for MpegAudioParse {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

impl PadTemplates for MpegAudioParse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo / 44.1 kHz shape.
        let mp3 = Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(mp3.clone())),
            PadTemplate::source(CapsSet::one(mp3)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use g2g_core::PushOutcome;

    use crate::audioframe::test_frames::{mp3_frame, MP3_128K_44100_LEN};
    use crate::audioframe::MPA_HEADER_LEN;

    /// The sample rate and channel count [`mp3_frame`] codes.
    const FRAME_RATE_HZ: u64 = 44_100;
    const FRAME_CHANNELS: u8 = 2;
    /// Side information of a stereo MPEG-1 Layer III frame, what a Xing tag sits
    /// behind.
    const STEREO_SIDE_INFO: usize = 32;

    /// One stereo frame with a payload that tells it from its neighbours.
    fn frame_bytes(fill: u8) -> Vec<u8> {
        mp3_frame(false, fill)
    }

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    impl RecordingSink {
        fn frames(&self) -> Vec<&Frame> {
            self.packets
                .iter()
                .filter_map(|p| match p {
                    PipelinePacket::DataFrame(f) => Some(f),
                    _ => None,
                })
                .collect()
        }
    }

    fn mp3_caps() -> Caps {
        // Sentinel pre-parse caps: format pinned, channels/rate unknown.
        Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 0,
            sample_rate: 0,
        }
    }

    /// Push `bytes` as one buffer, then end the stream.
    async fn parse(bytes: Vec<u8>) -> RecordingSink {
        let mut parser = MpegAudioParse::new();
        parser.configure_pipeline(&mp3_caps()).expect("mp3 caps");
        let mut sink = RecordingSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        parser
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("the buffer parses");
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        sink
    }

    #[tokio::test]
    async fn resynchronizes_past_leading_garbage() {
        let mut stream = vec![0u8; 20];
        // A lone sync byte pair that no valid header follows: skipped too.
        stream.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x00]);
        for fill in 0..3u8 {
            stream.extend(frame_bytes(fill));
        }
        let sink = parse(stream).await;
        assert_eq!(sink.frames().len(), 3, "every frame behind the garbage");
        assert!(sink
            .frames()
            .iter()
            .all(|f| f.domain.as_system_slice().map(<[u8]>::len) == Some(MP3_128K_44100_LEN)));
    }

    #[tokio::test]
    async fn drops_the_xing_header_frame() {
        let mut xing = frame_bytes(0);
        xing[MPA_HEADER_LEN + STEREO_SIDE_INFO..][..4].copy_from_slice(b"Xing");
        let mut stream = xing;
        for fill in 1..3u8 {
            stream.extend(frame_bytes(fill));
        }
        let sink = parse(stream).await;
        assert_eq!(sink.frames().len(), 2, "the seek table is not audio");
    }

    #[tokio::test]
    async fn stamps_pts_from_the_sample_count() {
        let mut stream = Vec::new();
        for fill in 0..3u8 {
            stream.extend(frame_bytes(fill));
        }
        let sink = parse(stream).await;
        // An MPEG-1 Layer III frame is 1152 samples. The presentation time comes
        // off the running sample count, not a sum of rounded durations.
        const SAMPLES_PER_FRAME: u64 = 1152;
        let pts = |frames: u64| frames * SAMPLES_PER_FRAME * NS_PER_SECOND as u64 / FRAME_RATE_HZ;
        let duration = pts(1);
        let times: Vec<(u64, u64)> = sink
            .frames()
            .iter()
            .map(|f| (f.timing.pts_ns, f.timing.duration_ns))
            .collect();
        assert_eq!(
            times,
            vec![(pts(0), duration), (pts(1), duration), (pts(2), duration)]
        );
    }

    #[tokio::test]
    async fn emits_concrete_caps_once() {
        let mut stream = Vec::new();
        for fill in 0..3u8 {
            stream.extend(frame_bytes(fill));
        }
        let sink = parse(stream).await;
        let caps: Vec<&Caps> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(
            caps,
            vec![&Caps::Audio {
                format: AudioFormat::Mp3,
                channels: FRAME_CHANNELS,
                sample_rate: FRAME_RATE_HZ as u32,
            }]
        );
    }

    #[tokio::test]
    async fn rejects_layer_one() {
        // 288 kbit/s (bitrate index 9 of the MPEG-1 Layer I table) at 44100 Hz
        // is 312 bytes a frame, in 4-byte slots.
        const FRAME_LEN_288K_44100: usize = 312;
        let mut frame = frame_bytes(0);
        frame[1] = 0xFF; // layer bits 11 = Layer I
        frame.truncate(FRAME_LEN_288K_44100);
        let mut parser = MpegAudioParse::new();
        parser.configure_pipeline(&mp3_caps()).expect("mp3 caps");
        let mut sink = RecordingSink::default();
        let mut stream = frame.clone();
        stream.extend(frame);
        let data = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        let err = parser
            .process(PipelinePacket::DataFrame(data), &mut sink)
            .await
            .expect_err("Layer I has no decoder here");
        assert_eq!(err, G2gError::CapsMismatch);
    }
}
