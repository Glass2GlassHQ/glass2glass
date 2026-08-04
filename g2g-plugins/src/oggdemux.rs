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
//! Scope: one logical bitstream per physical stream; a stream not matching the
//! `stream` selection is parsed but not forwarded.
//!
//! **Seeking (M362, M862).** Ogg carries no index, so a time seek guesses the
//! byte offset to land at: the page granules seen while playing give
//! `(byte offset, stream time)` anchors, the target interpolates through them,
//! and the landing is backed off so it sits before the target and, when the
//! source reports a byte length, short of the end of the stream. The parser
//! re-syncs there keeping what it knows of the bitstream (codec, headers), and
//! re-times from the granule of the first page it lands on, so the packets from
//! the target on are exactly the ones a re-scan from the file start delivers.
//! A landing past the target, in another physical stream (a chained file), or
//! in an unparseable region falls back: one corrected guess, then the plain
//! re-scan from offset `0`.
//!
//! **Chained files (M827).** A chain (a second physical stream after the first
//! one's end-of-stream page, the radio-stream form) is *sequential*: the same
//! output pad continues with the next chain's stream of the selected codec,
//! distinct from the *concurrent* grouped streams (M790) that
//! [`OggDemuxN`] splits onto separate pads. Each chain is announced with a
//! [`PipelinePacket::Segment`] whose `start` / `base` is the summed **playable**
//! duration of the chains before it (the end-granule-clamped decode position,
//! less the Opus pre-skip the decoder discards), so the chains concatenate
//! sample-exactly: a chain plays exactly as it would alone, shifted by what
//! came before, and ffmpeg / GStreamer report the same total. Stream time
//! restarts at zero per chain, running time carries on. The chain's own codec
//! headers go downstream in-band ahead of its audio (so a decoder re-inits on
//! the new pre-skip / codebooks) and its parameters re-announce via
//! `CapsChanged`. A chain that carries no stream of the selected codec fails
//! loud with [`G2gError::CapsMismatch`]: one output pad names one codec, so a
//! cross-codec chain has nowhere to go.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SeekController;
use g2g_core::{
    g2g_debug, g2g_error, AsyncElement, AudioFormat, BusHandle, BusMessage, ByteStreamEncoding,
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    MultiOutputElement, MultiOutputSink, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Seek, Segment, Stream, StreamCollection,
    StreamType, Tag, TagList,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::ogg::{OggCodec, OggDemuxer, OggLogicalStream};
use crate::opusparse::packet_samples as opus_packet_samples;

/// Convert a 48 kHz sample count to nanoseconds.
fn opus_samples_to_ns(samples: u64) -> u64 {
    samples.saturating_mul(1_000_000_000) / 48_000
}

/// Convert a sample count at `sample_rate` to nanoseconds. Sample-accurate at
/// the header rate (per-packet rounding would drift at non-48 kHz rates).
fn samples_to_ns(samples: u64, sample_rate: u32) -> u64 {
    (samples as u128 * 1_000_000_000 / sample_rate.max(1) as u128) as u64
}

/// Fraction of a proportional seek guess to land early, plus a fixed slack: a
/// stretch of the file denser than the observed average would otherwise land
/// past the target, which costs a second seek (M862).
const GUESS_BACKOFF_DIV: u64 = 8;
const GUESS_SLACK_BYTES: u64 = 8 * 1024;
/// Below this a re-scan from the start reads about as much, so it is not worth
/// a guess.
const MIN_GUESS_BYTES: u64 = 64 * 1024;
/// Bytes a landing stays clear of the end of the stream: it must still have a
/// granule-bearing page ahead of it (an Ogg page is at most ~64 kB), and landing
/// early only costs a scan.
const GUESS_EOF_MARGIN: u64 = 64 * 1024;
/// Span the two interpolation anchors must cover before a guess extrapolates
/// from them.
const MIN_ANCHOR_NS: u64 = 1_000_000_000;
const MIN_ANCHOR_BYTES: u64 = 32 * 1024;
/// Bytes read past a landing without reaching a granule-bearing page before the
/// landing is abandoned (a garbage / unparseable region).
const RESUME_SCAN_LIMIT: u64 = 4 * 1024 * 1024;
/// Byte-offset guesses per app seek: the proportional one plus one correction
/// from the time the landing turned out to be at.
const MAX_GUESSES: u8 = 2;

/// The byte offset to land at for `target_ns`, interpolated through two
/// observed `(byte offset, stream time ns)` points and backed off so a denser
/// stretch still lands before the target. Two points rather than a ratio
/// against the origin, so the header pages' fixed cost cancels. `stream_len`,
/// when the source published one, keeps the landing inside the stream: a guess
/// past the end reaches EOF instead of the target. `None` when the pair is too
/// short to extrapolate from, the target is behind it, or the landing would not
/// save a useful number of bytes.
fn guess_offset(
    first: (u64, u64),
    last: (u64, u64),
    target_ns: u64,
    stream_len: Option<u64>,
) -> Option<u64> {
    let (b1, t1) = first;
    let (b2, t2) = last;
    let bytes = b2.checked_sub(b1)?;
    let span = t2.checked_sub(t1)?;
    if span < MIN_ANCHOR_NS || bytes < MIN_ANCHOR_BYTES || target_ns <= t1 {
        return None;
    }
    // The anchors come from the file, so fold the extrapolation in u128.
    let ahead = u128::from(target_ns - t1) * u128::from(bytes) / u128::from(span);
    let raw = (u128::from(b1) + ahead).min(u128::from(u64::MAX)) as u64;
    let mut off = raw
        .saturating_sub(raw / GUESS_BACKOFF_DIV)
        .saturating_sub(GUESS_SLACK_BYTES);
    if let Some(len) = stream_len {
        off = off.min(len.saturating_sub(GUESS_EOF_MARGIN));
    }
    (off >= MIN_GUESS_BYTES).then_some(off)
}

/// A byte-offset guess in flight (M862).
#[derive(Debug, Default, Clone, Copy)]
struct GuessedSeek {
    /// Offset of the landing, `None` when the seek in flight is a plain re-scan
    /// from the file start.
    landing: Option<u64>,
    /// Guesses issued for the current app seek, capped at [`MAX_GUESSES`].
    attempts: u8,
    target_ns: u64,
    /// Chain the guess was planned in. A landing in another one is a chained
    /// file, which a single global proportion cannot describe.
    chain: u32,
}

/// Demuxes an Ogg byte stream into its Opus audio elementary stream.
#[derive(Debug)]
pub struct OggDemux {
    demux: OggDemuxer,
    configured: bool,
    emitted: u64,
    bus: Option<BusHandle>,
    /// The chain whose VorbisComment tags have been posted, so each physical
    /// stream's metadata posts once.
    tags_posted: Option<u32>,
    /// The logical stream to emit (`stream` property): Opus by default, Flac
    /// for the Ogg-FLAC mapping, or Vorbis. A file with no stream of that codec
    /// forwards nothing.
    stream: OggCodec,
    /// Caps, in-band config and packet timing for the selected bitstream.
    emitter: StreamEmitter,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
    /// Byte offset of the next source byte (M862): the seek guess is
    /// proportional in bytes, so the element tracks where in the file it reads.
    read_offset: u64,
    /// Earliest and latest observed `(parsed byte offset, stream time ns)`, the
    /// pair a seek guess interpolates through.
    anchor_first: Option<(u64, u64)>,
    anchor_last: Option<(u64, u64)>,
    /// The byte-offset guess in flight, if any.
    guess: GuessedSeek,
    /// Runner-assigned instance name plus any category override, for logging.
    log_name: LogName,
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
            bus: None,
            tags_posted: None,
            stream: OggCodec::Opus,
            emitter: StreamEmitter::default(),
            seek: DemuxSeek::default(),
            read_offset: 0,
            anchor_first: None,
            anchor_last: None,
            guess: GuessedSeek::default(),
            log_name: LogName::new(),
        }
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer repositions the source and re-syncs from the
    /// packet at/after the target (every audio packet is a resync point).
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Service a pending app time seek (M862). Ogg carries no index, so the
    /// byte offset to land at is guessed proportionally: the granule positions
    /// seen so far give `(byte offset, stream time)` anchors, and the target
    /// interpolates through them. The landing is deliberately early, and the
    /// scan from it to the first packet at/after the target delivers exactly
    /// what a re-scan from the file start would. Without usable anchors (too
    /// little played, a chained file, headers not yet in hand) it re-scans from
    /// offset `0`, the M362 path.
    fn poll_seek(&mut self) {
        if self.seek.is_seeking() {
            return;
        }
        // Planned before the request is taken: `poll_request_indexed` borrows
        // the seek state, so its closure cannot look at the rest of the element.
        let anchors = self.guess_anchors();
        let stream_len = self.seek.upstream_len();
        let chain = self.demux.chain();
        let mut chosen: Option<(u64, u64)> = None;
        let started = self.seek.poll_request_indexed(|target_ns| {
            let (first, last) = anchors?;
            let offset = guess_offset(first, last, target_ns, stream_len)?;
            chosen = Some((target_ns, offset));
            Some(offset)
        });
        if !started {
            return;
        }
        self.guess = match chosen {
            Some((target_ns, offset)) => GuessedSeek {
                landing: Some(offset),
                attempts: 1,
                target_ns,
                chain,
            },
            None => GuessedSeek::default(),
        };
    }

    /// The two anchors a seek guess interpolates through, or `None` when this
    /// file cannot take one: a chained file (a chain boundary breaks a single
    /// global proportion, so it re-scans), a stream whose codec headers are not
    /// in hand (a mid-file landing has no beginning-of-stream page to re-read
    /// them from), or nothing played yet.
    fn guess_anchors(&self) -> Option<((u64, u64), (u64, u64))> {
        if self.demux.chain() != 0 {
            return None;
        }
        let stream = &self.demux.streams()[self.selected()?];
        let codec = stream.info()?.codec;
        if in_band_headers(codec, stream).is_empty() || !self.emitter.can_resume(codec) {
            return None;
        }
        Some((self.anchor_first?, self.anchor_last?))
    }

    /// Note where the parse has reached, as the `(byte offset, stream time)`
    /// pair a later seek guess interpolates through. Only while playing
    /// normally: mid-seek the two are not the same position.
    fn note_anchor(&mut self) {
        if self.seek.is_seeking() || self.demux.chain() != 0 {
            return;
        }
        let time_ns = self.emitter.position_ns();
        let offset = self
            .read_offset
            .saturating_sub(self.demux.buffered() as u64);
        if time_ns == 0 || offset == 0 {
            return;
        }
        if self.anchor_first.is_none() {
            self.anchor_first = Some((offset, time_ns));
        }
        self.anchor_last = Some((offset, time_ns));
    }

    /// Whether a landing that cannot serve the seek has been reached: it left
    /// the physical stream that was resumed (a chained file), or it is scanning
    /// through a region with no parseable page.
    fn landing_lost(&self) -> bool {
        if !self.seek.is_seeking() || self.guess.landing.is_none() {
            return false;
        }
        if self.demux.foreign_page() || self.demux.chain() != self.guess.chain {
            return true;
        }
        let scanned = self
            .read_offset
            .saturating_sub(self.guess.landing.unwrap_or(0));
        self.emitter.awaiting_anchor() && scanned > RESUME_SCAN_LIMIT
    }

    /// Re-seek after a landing that cannot serve the target. `landed` is the
    /// stream time the landing turned out to be at, when known: it makes one
    /// corrected guess possible (the same interpolation against a true point
    /// this time). Otherwise, and after the correction, re-scan from the file
    /// start, which always works.
    fn reseek(&mut self, landed: Option<u64>) {
        let target_ns = self.guess.target_ns;
        let corrected = match (self.guess.landing, landed, self.anchor_first) {
            (Some(landing), Some(at), Some(first)) if self.guess.attempts < MAX_GUESSES => {
                guess_offset(first, (landing, at), target_ns, self.seek.upstream_len())
                    .filter(|off| *off < landing)
            }
            _ => None,
        };
        match corrected {
            Some(offset) => {
                g2g_debug!(
                    self,
                    "seek guess landed past {target_ns} ns, retrying at {offset}"
                );
                self.guess.landing = Some(offset);
                self.guess.attempts = self.guess.attempts.saturating_add(1);
                self.seek.begin_seek_at(target_ns, offset, true);
            }
            None => {
                g2g_debug!(self, "seek guess abandoned, re-scanning from the start");
                self.guess = GuessedSeek::default();
                self.seek.begin_seek_at(target_ns, 0, false);
            }
        }
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop the Ogg
    /// page/packet state and the running PTS, which the re-read stream
    /// re-establishes from its first page.
    fn reset_parser(&mut self) {
        self.demux = OggDemuxer::new();
        self.emitter.reset();
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

    /// The logical bitstream this element forwards out of the chain being
    /// parsed: the first one whose codec is the `stream` selection. A grouped
    /// file (M790) carries several, so the selection picks among them; a
    /// single-stream file has only one candidate.
    fn selected(&self) -> Option<usize> {
        self.demux.stream_of(self.stream)
    }

    /// The selected bitstream of every retained chain, in play order (M827): a
    /// chained file continues on this pad with the next physical stream's
    /// stream of the same codec.
    fn selected_streams(&self) -> Vec<usize> {
        let mut picked: Option<u32> = None;
        let mut out = Vec::new();
        for (index, stream) in self.demux.streams().iter().enumerate() {
            if picked == Some(stream.chain()) {
                continue;
            }
            if stream.info().map(|i| i.codec) == Some(self.stream) {
                picked = Some(stream.chain());
                out.push(index);
            }
        }
        out
    }

    /// A chained physical stream carries the selected codec, or this pad cannot
    /// present it: the output caps name one codec, so a chain that switches
    /// codecs (or carries only mappings g2g does not read) fails loud instead of
    /// silently going quiet for the rest of the file. Waits while the chain's
    /// opening group is still arriving.
    fn check_chain_codec(&self) -> Result<(), G2gError> {
        let chain = self.demux.chain();
        if chain == 0 || !self.demux.grouping_done() {
            return Ok(());
        }
        let group = || self.demux.streams().iter().filter(|s| s.chain() == chain);
        // Still arriving: an identification packet of the group is unparsed.
        if group().next().is_none() || group().any(|s| s.info().is_none()) {
            return Ok(());
        }
        if group().any(|s| s.info().map(|i| i.codec) == Some(self.stream)) {
            return Ok(());
        }
        g2g_error!(
            self,
            "chained physical stream {chain} carries no {:?} bitstream: a chain that changes codec cannot ride one output pad",
            self.stream
        );
        Err(G2gError::CapsMismatch)
    }

    /// Emit a `CapsChanged` once the stream parameters are known, then forward
    /// each audio packet. A file with no stream of the selected codec is drained
    /// and dropped (the output pad is typed by the selection).
    async fn emit_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        // Surface the stream's metadata once per chain, as soon as its comment
        // header lands (a chained file's next physical stream has its own).
        if self.tags_posted != Some(self.demux.chain()) && self.bus.is_some() {
            let comment = self
                .selected()
                .and_then(|i| self.demux.streams()[i].comment_header());
            if let Some(comment) = comment {
                let tags = parse_vorbis_comment(comment);
                self.tags_posted = Some(self.demux.chain());
                if !tags.is_empty() {
                    if let Some(bus) = &self.bus {
                        bus.try_post(BusMessage::Tag {
                            tags,
                            program: None,
                        });
                    }
                }
            }
        }
        // Drain every logical bitstream (an unselected one must not accumulate
        // packets for the length of the file), keeping the selected one of each
        // chain: a chained file's physical streams play back to back on this
        // pad, in order.
        let chosen = self.selected_streams();
        let mut batches: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
        for index in 0..self.demux.streams().len() {
            let taken = self
                .demux
                .stream_mut(index)
                .map(|s| s.take_packets())
                .unwrap_or_default();
            if chosen.contains(&index) {
                batches.push((index, taken));
            }
        }
        for (index, packets) in batches {
            let stream = &self.demux.streams()[index];
            // A finished chain's leftovers are drained above; only its own or a
            // later chain drives the timeline forward.
            if stream.chain() < self.emitter.chain() {
                continue;
            }
            let ready = self.emitter.step(stream, packets);
            // M862: a landing past the target cannot deliver the packets a
            // re-scan would, so nothing of this batch goes out (the codec
            // headers included: the re-seek sends them again) and the seek
            // starts over closer to the target.
            if let Some(at) = ready.resumed_at {
                if self.guess.landing.is_some() && at > self.guess.target_ns {
                    self.reseek(Some(at));
                    return Ok(());
                }
            }
            if let Some(caps) = ready.caps {
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
            if let Some(seg) = ready.segment {
                out.push(PipelinePacket::Segment(seg)).await?;
            }
            for (payload, timing, config) in ready.frames {
                // M362 seek: every audio packet is a resync point, so drop until
                // the first packet at/after the target, which emits a fresh
                // segment. The in-band codec config always flows: the decoder
                // needs it whatever the seek target.
                if !config {
                    match self.seek.admit(timing.pts_ns, true) {
                        Admit::Drop => continue,
                        Admit::Resume(start) => {
                            let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                            out.push(PipelinePacket::Segment(seg)).await?;
                        }
                        Admit::Emit => {}
                    }
                }
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                    timing,
                    self.emitted,
                );
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
        }
        self.check_chain_codec()
    }
}

/// What one demux step produced for a logical bitstream: the refined caps (when
/// they changed), the segment opening a chained physical stream (M827), and the
/// frames to forward, the in-band codec config first.
#[derive(Debug, Default)]
struct StreamReady {
    caps: Option<Caps>,
    segment: Option<Segment>,
    /// The frames to forward, each flagged when it is codec config rather than
    /// audio (config rides the same pad and flows whatever a seek is doing).
    frames: Vec<(Vec<u8>, FrameTiming, bool)>,
    /// Stream time a mid-file landing anchored at, in the batch that anchored it
    /// (M862). The caller compares it against the seek target.
    resumed_at: Option<u64>,
}

/// The per-logical-bitstream half of the demux elements: caps announcement,
/// in-band codec-config forwarding, and packet timing for each of the three
/// mappings. [`OggDemux`] owns one for its selected stream, [`OggDemuxN`] one
/// per output port, so this mapping logic is written once.
#[derive(Debug, Default)]
struct StreamEmitter {
    /// Last caps announced, so an unchanged refinement is not re-emitted.
    last_caps: Option<Caps>,
    /// Running stream-time (ns) of the next audio packet, accumulated from each
    /// Opus packet's decoded duration (the demuxer carries no per-packet PTS).
    /// FLAC and Vorbis time from `decoded_samples` instead.
    pts_ns: u64,
    /// Running count of decoded samples (per channel; 48 kHz incl. pre-skip for
    /// Opus, the header rate otherwise) over the audio packets seen so far.
    /// Compared against the end-of-stream granule position to trim the encoder
    /// padding off the final packet(s).
    decoded_samples: u64,
    /// Whether the in-band codec header (`OpusHead` / the native `fLaC`
    /// STREAMINFO / the three Vorbis headers) was forwarded to the decoder.
    /// Reset on a flush so the re-read stream re-sends it.
    head_forwarded: bool,
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
    /// The chain (physical stream, M827) this timing state belongs to. A chained
    /// file's next chain restarts the state on top of `chain_offset_ns`.
    chain: u32,
    /// Where the current chain starts on the output timeline (ns): the summed
    /// playable duration of the chains before it.
    chain_offset_ns: u64,
    /// Playable end reached in the current chain (ns, on its own timeline): the
    /// end-granule-clamped decode position, less the Opus pre-skip the decoder
    /// discards. Fixes where the next chain starts.
    playable_end_ns: u64,
    /// Timeline position (ns) after the last packet emitted, the time axis of
    /// the proportional seek guess (M862).
    position_ns: u64,
    /// Waiting for a mid-file landing's granule anchor (M862): packets are not
    /// timed from the file start any more, so nothing is emitted until the
    /// landing's first granule-bearing page says where it is.
    awaiting_anchor: bool,
}

impl StreamEmitter {
    /// Drop the timing state for a discontinuity (a `Flush` / seek), which the
    /// re-read stream re-establishes from its first page. The caps are unchanged
    /// (same file), so `last_caps` is kept (no redundant `CapsChanged`).
    fn reset(&mut self) {
        let caps = self.last_caps.take();
        *self = Self::default();
        self.last_caps = caps;
    }

    /// The chain (physical stream) this emitter is timing.
    fn chain(&self) -> u32 {
        self.chain
    }

    /// Timeline position (ns) after the last packet emitted.
    fn position_ns(&self) -> u64 {
        self.position_ns
    }

    /// Whether a mid-file landing is still waiting for its granule anchor.
    fn awaiting_anchor(&self) -> bool {
        self.awaiting_anchor
    }

    /// Whether this emitter can time packets from a mid-file landing (M862).
    /// Vorbis also needs its block-size tables and the priming anchor, both
    /// learned from the start of the stream the landing skips past.
    fn can_resume(&self, codec: OggCodec) -> bool {
        codec != OggCodec::Vorbis || (self.vorbis.is_some() && self.anchor_offset.is_some())
    }

    /// Arm a mid-file landing (M862): the position state is re-established from
    /// the landing's page granule, while the caps, the codec headers to re-send,
    /// the Vorbis tables and the priming anchor carry over (same stream, same
    /// file).
    fn begin_resume(&mut self) {
        self.pts_ns = 0;
        self.decoded_samples = 0;
        self.position_ns = 0;
        self.head_forwarded = false;
        self.prev_blocksize = None;
        self.awaiting_anchor = true;
    }

    /// Rebase onto a chained physical stream (M827): the previous chain's
    /// playable end becomes the new chain's zero and the per-chain timing state
    /// restarts. The returned segment carries that offset as both `start` (the
    /// new chain's earliest stream timestamp) and `base` (its running time), so
    /// running time stays continuous while stream time restarts at zero. The
    /// caps are kept, so an identically-parameterized chain re-announces
    /// nothing.
    fn begin_chain(&mut self, chain: u32) -> Segment {
        let offset = self.chain_offset_ns.saturating_add(self.playable_end_ns);
        let caps = self.last_caps.take();
        *self = Self {
            last_caps: caps,
            chain,
            chain_offset_ns: offset,
            ..Self::default()
        };
        Segment {
            start: offset,
            base: offset,
            position: offset,
            ..Segment::new()
        }
    }

    /// Turn this batch of `packets` from `stream` into the caps + frames to
    /// forward.
    fn step(&mut self, stream: &OggLogicalStream, packets: Vec<Vec<u8>>) -> StreamReady {
        let mut ready = StreamReady::default();
        let Some(info) = stream.info() else {
            return ready;
        };
        if stream.chain() != self.chain {
            ready.segment = Some(self.begin_chain(stream.chain()));
        }
        let codec = info.codec;
        // Mid-file landing (M862): the granule position of the page the parser
        // resumed on is the decode position the packets after it start from, so
        // the timeline continues exactly where a scan from the file start would
        // have it. Nothing is emitted before that anchor lands.
        if self.awaiting_anchor {
            let Some((granule, prev)) = stream.resume_anchor() else {
                return ready;
            };
            let at = match codec {
                OggCodec::Opus => opus_samples_to_ns(granule),
                _ => samples_to_ns(granule, info.sample_rate),
            };
            // The Vorbis timeline is the natural decode count less the initial
            // priming the anchor clips off the front, so add it back to get the
            // count the granule stands for.
            self.decoded_samples = granule.saturating_add(self.anchor_offset.unwrap_or(0));
            self.pts_ns = at;
            self.position_ns = at.saturating_add(self.chain_offset_ns);
            // The first emitted packet laps against the packet the anchor page
            // ended on.
            self.prev_blocksize = self.vorbis.as_ref().and_then(|t| t.packet_blocksize(prev));
            self.awaiting_anchor = false;
            ready.resumed_at = Some(self.position_ns);
        }
        if let Some(caps) = concrete_caps(info) {
            if self.last_caps.as_ref() != Some(&caps) {
                self.last_caps = Some(caps.clone());
                ready.caps = Some(caps);
            }
        }
        // Forward the codec config in-band once, before the first audio packet.
        // Codec config, not audio, so the decoder consumes it without emitting
        // PCM.
        if !self.head_forwarded {
            let heads = in_band_headers(codec, stream);
            if !heads.is_empty() {
                self.head_forwarded = true;
                ready
                    .frames
                    .extend(heads.into_iter().map(|h| (h, FrameTiming::default(), true)));
            }
        }
        // Vorbis timing tables (M778): parse once ident + setup have landed.
        if codec == OggCodec::Vorbis && self.vorbis.is_none() {
            if let (Some(h), Some(s)) = (stream.head_header(), stream.setup_header()) {
                self.vorbis = crate::ogg::VorbisTiming::parse(h, s);
            }
        }
        // Total decodable samples (incl. pre-skip); the tail beyond it is padding.
        let end_granule = stream.end_granule();
        let sample_rate = info.sample_rate;
        // Vorbis timeline anchor (M778): the first audio page's granule names
        // the timeline position after its last packet, so the excess of the
        // natural durations over it is initial priming, clipped off the front
        // (ffmpeg anchors the same way, as a negative first pts). An EOS first
        // page is an end clamp instead: anchor at zero.
        if codec == OggCodec::Vorbis && self.anchor_offset.is_none() {
            if let (Some(t), Some((gp, n, eos))) =
                (self.vorbis.as_ref(), stream.first_data_granule())
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
            let pkt_samples = match codec {
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
                Some(gp) if matches!(codec, OggCodec::Opus | OggCodec::Vorbis) => {
                    gp.saturating_sub(anchored_before).min(span)
                }
                _ => span,
            };
            // Sample-accurate at the header rate (per-packet rounding would
            // drift at non-48 kHz rates).
            let ns = |s: u64| samples_to_ns(s, sample_rate);
            let (pts_ns, duration_ns) = match codec {
                OggCodec::Flac | OggCodec::Vorbis => {
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
            // Where this chain's playable content ends on its own timeline: the
            // clamped decode position, less the pre-skip the decoder discards.
            // The next chain of a chained file starts there (M827).
            let playable = anchored_before
                .saturating_add(keep)
                .saturating_sub(u64::from(info.pre_skip));
            self.playable_end_ns = match codec {
                OggCodec::Opus => opus_samples_to_ns(playable),
                _ => ns(playable),
            };
            // Chained physical streams share the output pad, so each chain's
            // timeline rides on the summed playable duration before it.
            let pts_ns = pts_ns.saturating_add(self.chain_offset_ns);
            // Drop a timed packet wholly past the end granule (Opus padding /
            // Vorbis tail). A head-clipped packet (anchored position still 0,
            // e.g. the Vorbis priming packet) and an untimed one (pkt_samples
            // 0: a Vorbis stream whose setup scan failed) always flow, since
            // the decoder needs them to prime its window.
            if keep == 0 && anchored_before > 0 && pkt_samples > 0 {
                continue;
            }
            self.position_ns = pts_ns.saturating_add(duration_ns);
            ready.frames.push((
                packet,
                FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns,
                    ..FrameTiming::default()
                },
                false,
            ));
        }
        ready
    }
}

/// The concrete caps of a parsed stream, or `None` for a codec with no g2g
/// mapping / before the identification header lands.
fn concrete_caps(info: crate::ogg::OggStreamInfo) -> Option<Caps> {
    if info.sample_rate == 0 {
        return None;
    }
    Some(Caps::Audio {
        format: match info.codec {
            OggCodec::Opus => AudioFormat::Opus,
            OggCodec::Flac => AudioFormat::Flac,
            OggCodec::Vorbis => AudioFormat::Vorbis,
            _ => return None,
        },
        channels: info.channels.max(1),
        sample_rate: info.sample_rate,
    })
}

/// The codec config to forward in-band ahead of the audio packets: `OpusHead`
/// as-is (the decoder reads its pre-skip from it); for FLAC the native `fLaC`
/// STREAMINFO the mapping's first packet embeds at offset 9, with the
/// last-metadata-block flag set so the standalone header terminates (`detect()`
/// already validated the layout); for Vorbis all three headers (ident / comment
/// / setup, each with its unambiguous `\x0Nvorbis` prefix; the decoder needs
/// ident + setup). Empty until the headers have parsed.
fn in_band_headers(codec: OggCodec, stream: &OggLogicalStream) -> Vec<Vec<u8>> {
    let mut heads: Vec<Vec<u8>> = Vec::new();
    match codec {
        OggCodec::Vorbis => {
            if let (Some(h), Some(c), Some(s)) = (
                stream.head_header(),
                stream.comment_header(),
                stream.setup_header(),
            ) {
                heads.extend([h.to_vec(), c.to_vec(), s.to_vec()]);
            }
        }
        OggCodec::Flac => {
            if let Some(head) = stream.head_header() {
                let mut native = head[9..].to_vec();
                native[4] |= 0x80;
                heads.push(native);
            }
        }
        _ => {
            if let Some(head) = stream.head_header() {
                heads.push(head.to_vec());
            }
        }
    }
    heads
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
            self.poll_seek();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.read_offset = self.read_offset.saturating_add(slice.len() as u64);
                    self.demux.push_data(slice);
                    self.emit_ready(out).await?;
                    if self.landing_lost() {
                        self.reseek(None);
                    } else {
                        self.note_anchor();
                    }
                }
                // The upstream byte-seek's flush: re-sync from the repositioned
                // stream. A mid-file landing keeps what is known about the
                // bitstream (M862); a re-scan from the start resets outright.
                // Forward it downstream.
                PipelinePacket::Flush => {
                    let ours = self.seek.dropping_input();
                    self.seek.on_flush();
                    if ours {
                        self.read_offset = self.guess.landing.unwrap_or(0);
                    } else {
                        // An upstream discontinuity we did not ask for: the byte
                        // position is no longer known, so the anchors go too.
                        self.read_offset = 0;
                        self.anchor_first = None;
                        self.anchor_last = None;
                    }
                    if ours && self.seek.keeps_state() {
                        self.demux.resume_mid_stream();
                        self.emitter.begin_resume();
                    } else {
                        self.reset_parser();
                    }
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

    fn set_instance_name(&mut self, name: alloc::string::String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: alloc::string::String) {
        self.log_name.set_category(category);
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

/// M845: log identity. Category is the short type name (what the runner derives
/// for the instance name, so one `G2G_DEBUG` entry filters both).
impl LogSource for OggDemux {
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

/// `OggDemux`'s settable properties (M775).
static OGGDEMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "stream",
    PropKind::Str,
    "logical stream to emit: opus | flac | vorbis",
)];

/// The published stream id of the bitstream in slot `slot`. Positional,
/// matching the output-port order of [`OggDemuxN`].
fn stream_id(slot: usize) -> alloc::string::String {
    alloc::format!("ogg-stream-{slot}")
}

/// One output port of [`OggDemuxN`]: which logical bitstream it carries and the
/// codec that types it at negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OggPort {
    /// Slot of the logical bitstream within its physical stream, in
    /// beginning-of-stream order. A chained file's next chain continues on the
    /// same port through the same slot (M827).
    pub stream: usize,
    /// Expected codec, from the file probe. The port's runtime `CapsChanged`
    /// carries the concrete channels / rate.
    pub codec: OggCodec,
}

impl OggPort {
    pub fn new(stream: usize, codec: OggCodec) -> Self {
        Self { stream, codec }
    }
}

/// Multi-output Ogg demuxer (M790): one grouped Ogg byte stream in, N audio
/// elementary streams out, one per output port. The multi-output counterpart of
/// [`OggDemux`] (which forwards a single selected stream); the read-side analog
/// of [`crate::oggmuxn::OggMuxN`].
///
/// A [`MultiOutputElement`] driven by
/// [`run_source_fanout`](g2g_core::runtime::run_source_fanout). Routing is
/// **positional**: each [`OggPort`] names the slot of the logical bitstream it
/// carries, in beginning-of-stream order. Ogg groups streams rather than typing
/// them, and a file with two streams of the same codec is ordinary, so there is
/// no codec-keyed selection to route by; a port's codec is only the
/// negotiation-time expectation from the file probe, which its runtime
/// `CapsChanged` then refines. A stream no port names is drained and dropped.
///
/// A chained file (M827) continues each port with the same slot of the next
/// physical stream, on the rebased timeline [`OggDemux`] documents, and
/// re-announces the collection and tags per chain.
///
/// The container's per-stream metadata posts as
/// [`BusMessage::StreamTag`] under the same ids as the announced
/// [`StreamCollection`](BusMessage::StreamCollection).
#[derive(Debug)]
pub struct OggDemuxN {
    demux: OggDemuxer,
    /// The logical bitstream each output port carries, in port order.
    ports: Vec<OggPort>,
    /// Caps, in-band config and packet timing per port.
    emitters: Vec<StreamEmitter>,
    bus: Option<BusHandle>,
    /// The chain whose `StreamCollection` has been announced, so each physical
    /// stream's collection posts once.
    collection_posted: Option<u32>,
    /// The chain whose tags `tags_posted` tracks.
    tags_chain: Option<u32>,
    /// Whether the tags of the bitstream in each slot of that chain have been
    /// posted.
    tags_posted: Vec<bool>,
    emitted: u64,
}

impl OggDemuxN {
    /// A demuxer with one output port per entry of `ports`. Panics if `ports` is
    /// empty (a fan-out needs a port).
    pub fn new(ports: Vec<OggPort>) -> Self {
        assert!(
            !ports.is_empty(),
            "OggDemuxN needs at least one output port"
        );
        let emitters = (0..ports.len()).map(|_| StreamEmitter::default()).collect();
        Self {
            demux: OggDemuxer::new(),
            ports,
            emitters,
            bus: None,
            collection_posted: None,
            tags_chain: None,
            tags_posted: Vec::new(),
            emitted: 0,
        }
    }

    /// Attach the pipeline bus so the file's `StreamCollection` and per-stream
    /// VorbisComment tags post once parsed.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Number of output ports.
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Count of frames forwarded across all ports.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Announce the chain's logical bitstreams as a
    /// [`BusMessage::StreamCollection`], once its beginning-of-stream block has
    /// parsed (so the list is complete). Lists all of them, not just the ported
    /// ones; a chained file announces one collection per physical stream, under
    /// the id `ogg-<chain>`.
    fn post_stream_collection(&mut self) {
        let chain = self.demux.chain();
        if self.collection_posted == Some(chain) || !self.demux.grouping_done() {
            return;
        }
        let streams: Vec<Stream> = self
            .demux
            .streams()
            .iter()
            .filter(|s| s.chain() == chain)
            .filter_map(|s| {
                Some(Stream::new(
                    stream_id(s.slot()),
                    StreamType::Audio,
                    concrete_caps(s.info()?)?,
                ))
            })
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = Some(chain);
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new(
                alloc::format!("ogg-{chain}"),
                streams,
            )));
        }
    }

    /// Post each stream's VorbisComment metadata as a [`BusMessage::StreamTag`],
    /// once per chain, as its comment header lands.
    fn post_tags(&mut self) {
        let chain = self.demux.chain();
        if self.tags_chain != Some(chain) {
            self.tags_chain = Some(chain);
            self.tags_posted.clear();
        }
        let mut fresh: Vec<(alloc::string::String, TagList)> = Vec::new();
        for stream in self.demux.streams().iter().filter(|s| s.chain() == chain) {
            let slot = stream.slot();
            if self.tags_posted.len() <= slot {
                self.tags_posted.resize(slot + 1, false);
            }
            if self.tags_posted[slot] {
                continue;
            }
            let Some(comment) = stream.comment_header() else {
                continue;
            };
            self.tags_posted[slot] = true;
            let tags = parse_vorbis_comment(comment);
            if !tags.is_empty() {
                fresh.push((stream_id(slot), tags));
            }
        }
        if let Some(bus) = &self.bus {
            for (stream_id, tags) in fresh {
                bus.try_post(BusMessage::StreamTag { stream_id, tags });
            }
        }
    }
}

impl MultiOutputElement for OggDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&OggDemux::input_caps())
    }

    /// Declare each port's elementary-stream caps, so the solver negotiates each
    /// branch against its codec at startup; the concrete channels / rate arrive
    /// at runtime via the port's `CapsChanged`.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.ports.get(port).map(|p| OggDemux::output_caps(p.codec))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps
            .intersect(&OggDemux::input_caps())
            .map(|_| ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.demux.push_data(slice);
                }
                // Emit any final packets; the runner forwards EOS to every port.
                PipelinePacket::Eos => {}
                PipelinePacket::CapsChanged(_) => return Ok(()),
                _ => return Ok(()),
            }
            if self.bus.is_some() {
                self.post_stream_collection();
                self.post_tags();
            }
            // Drain every logical bitstream first: one no port carries must not
            // accumulate packets for the length of the file.
            let drained: Vec<(usize, Vec<Vec<u8>>)> = (0..self.demux.streams().len())
                .map(|i| {
                    let packets = self
                        .demux
                        .stream_mut(i)
                        .map(|s| s.take_packets())
                        .unwrap_or_default();
                    (i, packets)
                })
                .collect();
            // Route by slot, so a chained file's next physical stream continues
            // on the same ports (M827); a finished chain's leftovers are drained
            // above but no longer drive a port's timeline.
            for (index, packets) in drained {
                let (chain, slot) = {
                    let stream = &self.demux.streams()[index];
                    (stream.chain(), stream.slot())
                };
                let Some(port) = self.ports.iter().position(|p| p.stream == slot) else {
                    continue;
                };
                if chain < self.emitters[port].chain() {
                    continue;
                }
                let ready = self.emitters[port].step(&self.demux.streams()[index], packets);
                if let Some(caps) = ready.caps {
                    out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
                }
                if let Some(seg) = ready.segment {
                    out.push_to(port, PipelinePacket::Segment(seg)).await?;
                }
                for (payload, timing, _) in ready.frames {
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                        timing,
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                }
            }
            Ok(())
        })
    }
}

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
    fn guess_offset_interpolates_and_lands_early() {
        // 10 kB per second from 2 s on, with 4 kB of headers ahead of it: the
        // two anchors cancel the fixed cost, so 20 s sits at 204 kB, and the
        // guess lands short of it.
        let first = (24_000, 2_000_000_000);
        let last = (124_000, 12_000_000_000);
        let off = guess_offset(first, last, 20_000_000_000, None).expect("a guess");
        assert!(off < 204_000, "the landing is early, got {off}");
        assert!(off > 170_000, "but not by much, got {off}");
        // Behind the pair, too short a span, and too small a saving: no guess.
        assert_eq!(guess_offset(first, last, 1_000_000_000, None), None);
        assert_eq!(
            guess_offset(first, (24_500, 2_500_000_000), 20_000_000_000, None),
            None
        );
        assert_eq!(
            guess_offset((0, 0), (40_000, 4_000_000_000), 5_000_000_000, None),
            None,
            "a landing under 64 kB saves nothing worth a seek"
        );
    }

    #[test]
    fn guess_offset_clamps_to_the_stream_length() {
        // The same front-measured proportion, but the file ends at 200 kB: the
        // landing sits a page-max short of the end rather than past it.
        let first = (24_000, 2_000_000_000);
        let last = (124_000, 12_000_000_000);
        assert_eq!(
            guess_offset(first, last, 20_000_000_000, Some(200_000)),
            Some(200_000 - GUESS_EOF_MARGIN)
        );
        // Clamped below the minimum saving: fall back to the plain re-scan.
        assert_eq!(
            guess_offset(first, last, 20_000_000_000, Some(100_000)),
            None,
            "no runway left between the clamp and the file start"
        );
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
            if let BusMessage::Tag { tags: t, .. } = m {
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
