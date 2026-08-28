//! Xing / Info VBR header writer element (`xingmux`): an MPEG audio byte stream
//! in, the same stream with a Xing header frame at its head out.
//!
//! A variable-bitrate `.mp3` has no constant bytes-per-second, so a player
//! cannot turn a seek time into a byte offset by division. The Xing header is
//! the answer: a real MPEG frame whose payload is a seek table rather than
//! sound, carrying the stream's frame count, its byte count, and 100 offsets
//! sampled across it. Without one a VBR file seeks by guesswork.
//!
//! The header can only be written once the counts are known, and it goes at the
//! head, so the whole stream is held until EOS. A leading ID3v2 tag stays where
//! it is and the header goes behind it; a Xing / Info / VBRI header frame
//! already there is replaced.
//!
//! The frame is marked `Xing` when the stream's frames differ in bitrate and
//! `Info` when they do not, which is what a player reads to tell VBR from CBR.
//!
//! # Example
//!
//! ```no_run
//! use g2g_plugins::xingmux::XingMux;
//!
//! // gst-launch equivalent:
//! //   filesrc location=in.mp3 ! xingmux ! filesink location=out.mp3
//! let element = XingMux::new();
//! ```

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audioframe::{
    is_vbr_header_frame, locate_frame, mpa_header, side_info_len, Located, MpaHeader,
    MPA_HEADER_LEN, XING_TAG_CBR, XING_TAG_VBR,
};
use crate::id3::id3v2_len;

/// Xing flag bits, in the 4-byte big-endian word behind the tag: the frame
/// count, the byte count and the 100-entry seek table are present. Bit 3 marks a
/// quality indicator, which this does not write.
const XING_FLAG_FRAMES: u32 = 0x0001;
const XING_FLAG_BYTES: u32 = 0x0002;
const XING_FLAG_TOC: u32 = 0x0004;

/// Entries in the Xing seek table, one per percent of the stream.
const XING_TOC_ENTRIES: usize = 100;
/// A table entry is the byte offset scaled into a single byte, so the whole
/// stream spans this many steps.
const XING_TOC_SCALE: u64 = 256;

/// Bytes of Xing payload behind the side information: the 4-byte tag, the flag
/// word, the frame count, the byte count, and the seek table.
const XING_PAYLOAD_LEN: usize = 4 + 4 + 4 + 4 + XING_TOC_ENTRIES;

/// Bitrate indices a header may code, 1 through 14 (0 is free format and 15 is
/// invalid). The Xing frame takes the lowest one long enough to hold the
/// payload, so the header costs as few bytes as it can.
const BITRATE_INDEX_RANGE: core::ops::RangeInclusive<u8> = 1..=14;

/// # Example
///
/// ```no_run
/// use g2g_plugins::xingmux::XingMux;
///
/// let element = XingMux::new();
/// ```
#[derive(Debug, Default)]
pub struct XingMux {
    configured: bool,
    /// The whole stream: the counts the header carries are only known at EOS,
    /// and the header goes ahead of everything.
    buf: Vec<u8>,
    log_name: LogName,
}

impl XingMux {
    pub fn new() -> Self {
        Self::default()
    }
}

/// One MPEG audio frame's place in the stream.
struct FrameSpan {
    start: usize,
    header: MpaHeader,
}

/// Walk the frames of `audio`, in order. Stops at the first byte run that is not
/// a frame, so a truncated tail is left out of the counts rather than guessed at.
fn walk_frames(audio: &[u8]) -> Vec<FrameSpan> {
    let mut frames = Vec::new();
    let mut at = 0usize;
    while at < audio.len() {
        let Located::Frame { start, len } = locate_frame::<MpaHeader>(&audio[at..], true) else {
            break;
        };
        let Some(begin) = at.checked_add(start) else {
            break;
        };
        let Some(header) = mpa_header(&audio[begin..]) else {
            break;
        };
        frames.push(FrameSpan {
            start: begin,
            header,
        });
        let Some(next) = begin.checked_add(len) else {
            break;
        };
        at = next;
    }
    frames
}

/// Build the Xing header frame for a stream of `frames` frames occupying
/// `audio_bytes` bytes behind it. `None` when no codable bitrate gives a frame
/// long enough to hold the payload, which leaves the stream as it was.
fn xing_frame(
    first: &MpaHeader,
    template: [u8; MPA_HEADER_LEN],
    spans: &[FrameSpan],
    audio_bytes: usize,
) -> Option<Vec<u8>> {
    let side_info = side_info_len(first);
    let needed = MPA_HEADER_LEN + side_info + XING_PAYLOAD_LEN;
    let (header, frame_len) = BITRATE_INDEX_RANGE.into_iter().find_map(|index| {
        let mut header = template;
        // Swap in the bitrate index and clear the padding bit: the frame's
        // length is whatever that index codes, with no slot added.
        header[2] = (index << 4) | (header[2] & 0x0D);
        let len = mpa_header(&header)?.frame_len;
        (len >= needed).then_some((header, len))
    })?;

    let total_bytes = frame_len.checked_add(audio_bytes)?;
    let variable = spans.iter().any(|s| s.header.bitrate != first.bitrate);
    let tag: &[u8] = if variable { XING_TAG_VBR } else { XING_TAG_CBR };
    let mut frame = Vec::from(header);
    frame.resize(MPA_HEADER_LEN + side_info, 0);
    frame.extend_from_slice(tag);
    frame.extend_from_slice(&(XING_FLAG_FRAMES | XING_FLAG_BYTES | XING_FLAG_TOC).to_be_bytes());
    frame.extend_from_slice(&(spans.len() as u32).to_be_bytes());
    frame.extend_from_slice(&(total_bytes as u32).to_be_bytes());
    frame.extend_from_slice(&seek_table(spans, frame_len, total_bytes));
    frame.resize(frame_len, 0);
    Some(frame)
}

/// The 100-entry seek table: entry `i` is the byte offset of the frame `i`
/// percent of the way through the stream, scaled to 0..255 of the total. The
/// offsets count from the Xing frame itself, which is where a player seeking to
/// zero lands.
fn seek_table(spans: &[FrameSpan], xing_len: usize, total_bytes: usize) -> Vec<u8> {
    let mut toc = Vec::with_capacity(XING_TOC_ENTRIES);
    for i in 0..XING_TOC_ENTRIES {
        let at = i * spans.len() / XING_TOC_ENTRIES;
        let offset = spans.get(at).map_or(0, |s| xing_len + s.start) as u64;
        let scaled = offset.saturating_mul(XING_TOC_SCALE) / (total_bytes.max(1) as u64);
        toc.push(scaled.min(XING_TOC_SCALE - 1) as u8);
    }
    toc
}

impl XingMux {
    /// Rewrite the held stream: the leading ID3v2 tag, then the Xing frame, then
    /// the audio.
    async fn finish(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let stream: Vec<u8> = core::mem::take(&mut self.buf);
        let tag_len = id3v2_len(&stream).unwrap_or(0).min(stream.len());
        let mut audio = &stream[tag_len..];
        let mut spans = walk_frames(audio);
        // A VBR header frame already there is a seek table, not sound: replace it.
        let existing = spans.first().and_then(|first| {
            is_vbr_header_frame(&audio[first.start..], &first.header)
                .then_some(first.start + first.header.frame_len)
        });
        if let Some(drop) = existing {
            audio = &audio[drop.min(audio.len())..];
            spans = walk_frames(audio);
        }
        let mut written = Vec::from(&stream[..tag_len]);
        match spans.first() {
            Some(first) => {
                let template: [u8; MPA_HEADER_LEN] = audio[first.start..][..MPA_HEADER_LEN]
                    .try_into()
                    .expect("a located frame carries a whole header");
                let header = first.header;
                match xing_frame(&header, template, &spans, audio.len()) {
                    Some(frame) => written.extend_from_slice(&frame),
                    None => g2g_warn!(
                        self,
                        "no MPEG frame length holds a Xing header at {} Hz: the stream is unchanged",
                        header.sample_rate
                    ),
                }
            }
            None => g2g_warn!(self, "no MPEG audio frame in the stream: nothing to index"),
        }
        written.extend_from_slice(audio);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(written.into_boxed_slice())),
            Default::default(),
            0,
        );
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for XingMux {
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
                format: AudioFormat::Mp3,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over MP3 of any channels / rate: the frame header
    /// carries them and `Caps::Audio` cannot express "MP3 at any channels/rate",
    /// the same reason `mpegaudioparse` gives.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Mp3,
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
            "Xing VBR header writer",
            "Formatter/Metadata",
            "Writes the Xing / Info seek header at the head of an MPEG audio stream",
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
                    self.buf.extend_from_slice(slice);
                }
                PipelinePacket::Eos => self.finish(out).await?,
                PipelinePacket::Flush => {
                    self.buf.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
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

impl LogSource for XingMux {
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

impl PadTemplates for XingMux {
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
    use g2g_core::PushOutcome;

    use crate::audioframe::test_frames::{mp3_frame, mp3_frame_at, MP3_128K_44100_LEN};
    use crate::id3::{write_id3v2, ID3V2_VERSION_2_3};

    /// Bitrate indices 9 and 11 are 128 and 192 kbit/s, so a stream mixing them
    /// is variable-bitrate and its frames differ in length.
    const BITRATE_INDEX_128K: u8 = 9;
    const BITRATE_INDEX_192K: u8 = 11;
    /// Side information of a stereo MPEG-1 Layer III frame, what the Xing tag
    /// sits behind.
    const STEREO_SIDE_INFO: usize = 32;
    /// Offsets within the Xing frame, from the tag onwards.
    const TAG_AT: usize = MPA_HEADER_LEN + STEREO_SIDE_INFO;
    const FLAGS_AT: usize = TAG_AT + 4;
    const FRAMES_AT: usize = FLAGS_AT + 4;
    const BYTES_AT: usize = FRAMES_AT + 4;
    const TOC_AT: usize = BYTES_AT + 4;

    #[derive(Default)]
    struct RecordingSink {
        bytes: Vec<u8>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let PipelinePacket::DataFrame(frame) =
                packet_slot.take().expect("poll_push without a packet")
            {
                let slice = frame.domain.require_system_slice("test").expect("system");
                self.bytes.extend_from_slice(slice);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn mp3_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
        }
    }

    async fn run(input: &[u8], chunk: usize) -> Vec<u8> {
        let mut element = XingMux::new();
        let mut out = RecordingSink::default();
        element.configure_pipeline(&mp3_caps()).expect("mp3 in");
        for piece in input.chunks(chunk) {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(Vec::from(piece).into_boxed_slice())),
                Default::default(),
                0,
            );
            element
                .process(PipelinePacket::DataFrame(frame), &mut out)
                .await
                .expect("process");
        }
        element
            .process(PipelinePacket::Eos, &mut out)
            .await
            .expect("eos");
        out.bytes
    }

    fn u32_be_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    /// A constant-bitrate stream: the header is marked `Info`, and its counts
    /// name the frames and bytes that follow.
    #[tokio::test]
    async fn writes_the_counts_a_seek_needs() {
        const FRAMES: usize = 6;
        let mut stream = Vec::new();
        for i in 0..FRAMES {
            stream.extend(mp3_frame(false, i as u8));
        }
        let written = run(&stream, 64).await;
        let xing_len = written.len() - stream.len();
        assert_eq!(
            &written[xing_len..],
            &stream[..],
            "the audio behind the header is byte-for-byte what came in"
        );
        assert_eq!(&written[TAG_AT..TAG_AT + 4], XING_TAG_CBR.as_slice());
        assert_eq!(
            u32_be_at(&written, FLAGS_AT),
            XING_FLAG_FRAMES | XING_FLAG_BYTES | XING_FLAG_TOC
        );
        assert_eq!(u32_be_at(&written, FRAMES_AT) as usize, FRAMES);
        assert_eq!(u32_be_at(&written, BYTES_AT) as usize, written.len());
        // The frame is a real MPEG frame, long enough to hold what it carries.
        let header = mpa_header(&written).expect("the header parses");
        assert_eq!(header.frame_len, xing_len);
        assert!(header.frame_len >= TOC_AT + XING_TOC_ENTRIES);
    }

    /// Frames of differing bitrate make the stream VBR, which the mark says.
    #[tokio::test]
    async fn a_variable_bitrate_stream_is_marked_xing() {
        let mut stream = mp3_frame_at(BITRATE_INDEX_128K, 1);
        stream.extend(mp3_frame_at(BITRATE_INDEX_192K, 2));
        stream.extend(mp3_frame_at(BITRATE_INDEX_128K, 3));
        let written = run(&stream, stream.len()).await;
        assert_eq!(&written[TAG_AT..TAG_AT + 4], XING_TAG_VBR.as_slice());
        assert_eq!(u32_be_at(&written, FRAMES_AT), 3);
    }

    /// The seek table maps the whole stream: it opens on the header's own offset
    /// and rises to the last frame, never past the 256-step scale.
    #[tokio::test]
    async fn the_seek_table_spans_the_stream() {
        let mut stream = Vec::new();
        for i in 0..50u8 {
            stream.extend(mp3_frame(false, i));
        }
        let written = run(&stream, 512).await;
        let toc = &written[TOC_AT..TOC_AT + XING_TOC_ENTRIES];
        let xing_len = written.len() - stream.len();
        let expected_first = (xing_len as u64 * XING_TOC_SCALE) / written.len() as u64;
        assert_eq!(u64::from(toc[0]), expected_first);
        assert!(toc.windows(2).all(|w| w[0] <= w[1]), "offsets only advance");
        let last_frame_at = xing_len + stream.len() - MP3_128K_44100_LEN;
        let expected_last = (last_frame_at as u64 * XING_TOC_SCALE) / written.len() as u64;
        assert_eq!(u64::from(toc[XING_TOC_ENTRIES - 1]), expected_last);
    }

    /// Running the writer twice leaves one header, not two.
    #[tokio::test]
    async fn an_existing_header_frame_is_replaced() {
        let mut stream = Vec::new();
        for i in 0..4u8 {
            stream.extend(mp3_frame(false, i));
        }
        let once = run(&stream, 128).await;
        let twice = run(&once, 128).await;
        assert_eq!(twice, once);
    }

    /// A leading ID3v2 tag is metadata, not audio: it stays at the head and the
    /// Xing frame goes behind it.
    #[tokio::test]
    async fn a_leading_id3v2_tag_keeps_its_place() {
        let tag = write_id3v2(
            &[g2g_core::Tag::Title("Sine".into())].into_iter().collect(),
            ID3V2_VERSION_2_3,
        );
        let mut stream = tag.clone();
        stream.extend(mp3_frame(false, 1));
        stream.extend(mp3_frame(false, 2));
        let written = run(&stream, 16).await;
        assert_eq!(&written[..tag.len()], &tag[..]);
        assert_eq!(u32_be_at(&written, tag.len() + FRAMES_AT), 2);
        let xing_len = mpa_header(&written[tag.len()..])
            .expect("the Xing frame follows the tag")
            .frame_len;
        assert_eq!(&written[tag.len() + xing_len..], &stream[tag.len()..]);
    }
}
