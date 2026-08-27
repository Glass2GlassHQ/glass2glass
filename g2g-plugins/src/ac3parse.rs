//! AC-3 stream parser element (`ac3parse`): an AC-3 byte stream in
//! (`Caps::Audio{Ac3}`, arbitrary chunks from `filesrc`), one syncframe per
//! buffer out.
//!
//! The AC-3 sibling of [`crate::mpegaudioparse`]: an `.ac3` file is a bare
//! sequence of self-syncing frames, and [`crate::ffmpegaudiodec`] takes one
//! frame per packet, so something has to split the byte stream. Frame lengths
//! and the channel / rate fields come from
//! [`ac3_header`](crate::audioframe::ac3_header), shared with the program-stream
//! demuxer, and the resync rule is the two-header one in
//! [`locate_frame`](crate::audioframe::locate_frame).
//!
//! E-AC-3 is rejected rather than parsed. Its syncframe carries `bsid` 16 and
//! packs the frame size, sample rate and channel fields differently, so the
//! plain AC-3 decode above would read nonsense out of it; nothing here decodes
//! E-AC-3 either, so a stream whose head syncframe declares a `bsid` past
//! [`AC3_MAX_BSID`](crate::audioframe::AC3_MAX_BSID) fails the parse.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_error, g2g_warn, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket,
};

use crate::audioframe::{
    ac3_bsid, ac3_header, locate_frame, Ac3Header, Located, AC3_MAX_BSID, AC3_SAMPLES_PER_FRAME,
};

/// Nanoseconds per second, the presentation-time unit.
const NS_PER_SECOND: u128 = 1_000_000_000;

/// # Example
///
/// ```no_run
/// use g2g_plugins::ac3parse::Ac3Parse;
///
/// let parser = Ac3Parse::new();
/// assert_eq!(parser.frames_emitted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct Ac3Parse {
    configured: bool,
    /// Unconsumed input bytes, starting at stream offset `buf_offset`.
    buf: Vec<u8>,
    buf_offset: u64,
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

impl Ac3Parse {
    pub fn new() -> Self {
        Self::default()
    }

    /// Count of syncframes emitted.
    pub fn frames_emitted(&self) -> u64 {
        self.sequence
    }

    /// Emit every frame the buffer holds whole.
    async fn drain(&mut self, eos: bool, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            // Only the head is tested: the buffer is frame-aligned there once
            // the first frame has been found, so an E-AC-3 syncword cannot be a
            // chance byte pair in data the resync is skipping over.
            if ac3_bsid(&self.buf).is_some_and(|bsid| bsid > AC3_MAX_BSID) {
                g2g_error!(self, "E-AC-3 stream: nothing here decodes it");
                return Err(G2gError::CapsMismatch);
            }
            let Located::Frame { start, len } = locate_frame::<Ac3Header>(&self.buf, eos) else {
                return Ok(());
            };
            if start > 0 {
                g2g_warn!(self, "resynchronized past {start} bytes of non-audio");
                self.buf.drain(..start);
                self.buf_offset += start as u64;
                continue; // the head moved: re-test it for E-AC-3
            }
            let header = ac3_header(&self.buf).ok_or(G2gError::CapsMismatch)?;
            let start_offset = self.buf_offset;
            let data: Vec<u8> = self.buf.drain(..len).collect();
            self.buf_offset += len as u64;
            self.emit(data, &header, start_offset, out).await?;
        }
    }

    /// Push one frame with the caps its header declares and a presentation time
    /// from the running sample count.
    async fn emit(
        &mut self,
        data: Vec<u8>,
        header: &Ac3Header,
        start_offset: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = Caps::Audio {
            format: AudioFormat::Ac3,
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
        let duration_ns = ns(u64::from(AC3_SAMPLES_PER_FRAME));
        self.samples += u64::from(AC3_SAMPLES_PER_FRAME);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns,
                // Every AC-3 syncframe decodes on its own.
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

impl AsyncElement for Ac3Parse {
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
                format: AudioFormat::Ac3,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Pass-through identity over AC-3 of any channels/rate: the frame header
    /// refines them mid-stream, and `Caps::Audio` cannot express "AC-3 at any
    /// channels/rate" in a single `Caps` (the same reason `aacparse` gives).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Ac3,
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
            "AC-3 parser",
            "Codec/Parser/Audio",
            "Splits an AC-3 stream into syncframes and refines caps",
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
                // The last frame ends at the end of the stream, so it has no
                // successor to confirm its sync against.
                PipelinePacket::Eos => {
                    self.drain(true, out).await?;
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

impl LogSource for Ac3Parse {
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

impl PadTemplates for Ac3Parse {
    fn pad_templates() -> Vec<PadTemplate> {
        // `Caps::Audio` has no open dims; pin the common stereo / 48 kHz shape.
        let ac3 = Caps::Audio {
            format: AudioFormat::Ac3,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(ac3.clone())),
            PadTemplate::source(CapsSet::one(ac3)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    use g2g_core::PushOutcome;

    use crate::audioframe::test_frames::{ac3_frame, AC3_192K_48000_LEN};

    /// The sample rate and channel count [`ac3_frame`] codes.
    const FRAME_RATE_HZ: u64 = 48_000;
    const FRAME_CHANNELS: u8 = 2;

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

    fn ac3_caps() -> Caps {
        // Sentinel pre-parse caps: format pinned, channels/rate unknown.
        Caps::Audio {
            format: AudioFormat::Ac3,
            channels: 0,
            sample_rate: 0,
        }
    }

    fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    /// Push `bytes` as one buffer, then end the stream.
    async fn parse(bytes: Vec<u8>) -> RecordingSink {
        let mut parser = Ac3Parse::new();
        parser.configure_pipeline(&ac3_caps()).expect("AC-3 caps");
        let mut sink = RecordingSink::default();
        parser
            .process(data_frame(bytes), &mut sink)
            .await
            .expect("the buffer parses");
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        sink
    }

    fn stream(frames: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        for fill in 0..frames {
            bytes.extend(ac3_frame(fill));
        }
        bytes
    }

    #[tokio::test]
    async fn resynchronizes_past_leading_garbage() {
        let mut bytes = vec![0u8; 20];
        // A lone syncword that no valid header follows: skipped too.
        bytes.extend_from_slice(&[0x0B, 0x77, 0x00, 0x00, 0xC0, 0x00, 0x00]);
        bytes.extend(stream(3));
        let sink = parse(bytes).await;
        assert_eq!(sink.frames().len(), 3, "every frame behind the garbage");
        assert!(sink
            .frames()
            .iter()
            .all(|f| f.domain.as_system_slice().map(<[u8]>::len) == Some(AC3_192K_48000_LEN)));
    }

    #[tokio::test]
    async fn drops_a_truncated_tail() {
        let mut bytes = stream(2);
        bytes.extend_from_slice(&ac3_frame(2)[..AC3_192K_48000_LEN / 2]);
        let sink = parse(bytes).await;
        assert_eq!(sink.frames().len(), 2, "the half frame is not emitted");
    }

    #[tokio::test]
    async fn frames_split_across_input_buffers() {
        // An odd chunk size, so frames straddle buffer boundaries.
        const CHUNK_LEN: usize = 397;
        let bytes = stream(4);
        let mut parser = Ac3Parse::new();
        parser.configure_pipeline(&ac3_caps()).expect("AC-3 caps");
        let mut sink = RecordingSink::default();
        for piece in bytes.chunks(CHUNK_LEN) {
            parser
                .process(data_frame(piece.to_vec()), &mut sink)
                .await
                .expect("the chunk parses");
        }
        parser
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the tail flushes");
        assert_eq!(sink.frames().len(), 4);
        assert_eq!(parser.frames_emitted(), 4);
    }

    #[tokio::test]
    async fn stamps_pts_from_the_sample_count() {
        let sink = parse(stream(3)).await;
        let pts = |frames: u64| {
            frames * u64::from(AC3_SAMPLES_PER_FRAME) * NS_PER_SECOND as u64 / FRAME_RATE_HZ
        };
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
        let sink = parse(stream(3)).await;
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
                format: AudioFormat::Ac3,
                channels: FRAME_CHANNELS,
                sample_rate: FRAME_RATE_HZ as u32,
            }]
        );
    }

    #[tokio::test]
    async fn rejects_eac3() {
        /// `bsid` 16 is E-AC-3, whatever the rest of the header says.
        const BSID_EAC3: u8 = 16;
        let mut frame = ac3_frame(0);
        frame[5] = BSID_EAC3 << 3;
        let mut parser = Ac3Parse::new();
        parser.configure_pipeline(&ac3_caps()).expect("AC-3 caps");
        let mut sink = RecordingSink::default();
        let err = parser
            .process(data_frame(frame), &mut sink)
            .await
            .expect_err("E-AC-3 has no decoder here");
        assert_eq!(err, G2gError::CapsMismatch);
    }

    #[tokio::test]
    async fn re_bases_on_an_upstream_presentation_time() {
        const DEMUXER_PTS_NS: u64 = 3_000_000_000;
        let mut parser = Ac3Parse::new();
        parser.configure_pipeline(&ac3_caps()).expect("AC-3 caps");
        let mut sink = RecordingSink::default();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream(2).into_boxed_slice())),
            FrameTiming {
                pts_ns: DEMUXER_PTS_NS,
                ..FrameTiming::default()
            },
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
        assert_eq!(sink.frames()[0].timing.pts_ns, DEMUXER_PTS_NS);
    }

    #[test]
    fn decodes_the_channel_count_from_acmod_and_lfeon() {
        // acmod 7 (3/2) plus the LFE bit, which sits seven bits into byte 6.
        let mut frame = ac3_frame(0);
        frame[6] = (7 << 5) | 1;
        let header = ac3_header(&frame).expect("a 5.1 header parses");
        assert_eq!(header.channels, 6);
        assert_eq!(header.frame_len, AC3_192K_48000_LEN);
        // A reserved fscod has no frame length at all.
        let mut reserved = ac3_frame(0);
        reserved[4] |= 0xC0;
        assert!(ac3_header(&reserved).is_none());
        assert!(ac3_header(&[0x0B, 0x77]).is_none());
    }
}
