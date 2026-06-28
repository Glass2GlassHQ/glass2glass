//! MPEG-TS demuxer element (M108): `Caps::ByteStream{MpegTs}` in, one selected
//! elementary stream out. H.264 / H.265 video leave as `Caps::CompressedVideo`
//! (Annex-B); AAC audio leaves as `Caps::Audio` (ADTS).
//!
//! Wraps the pure [`crate::mpegts::TsDemuxer`] parser. Incoming byte frames are
//! resynchronized to 188-byte TS packets and fed to the demuxer; the reassembled
//! PES access units of the selected stream ([`TsStream`], default H.264) are
//! forwarded with their PTS, ready for the matching parser. CPU, `no_std`
//! baseline.
//!
//! ```text
//! filesrc(location=x.ts, caps=ByteStream{MpegTs}) ! tsdemux ! h264parse ! <decoder> ! <sink>
//! tsdemux stream=aac ! aacparse ! <audio>
//! ```
//!
//! One output pad carries one elementary stream: the parser reassembles every
//! stream the PMT names, and the [`TsStream`] selection picks which to emit, so a
//! second `tsdemux` selecting another stream demuxes the rest of the multiplex.
//! The choice is by codec, not a runtime-discovered "first video", because the
//! output caps are fixed at negotiation before any packet is parsed (M109).
//! Scope (v1): the first stream of the selected codec; multi-program selection
//! and a muxer are follow-ups (DESIGN.md §4.17).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SeekController;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, Seek, Segment, VideoCodec,
};

use crate::demuxseek::{Admit, DemuxSeek};
use crate::mpegts::{
    EsUnit, TsDemuxer, STREAM_TYPE_AAC, STREAM_TYPE_H264, STREAM_TYPE_H265, TS_PACKET_LEN,
};

const TS_SYNC: u8 = 0x47;

/// Which elementary stream a [`TsDemux`] instance forwards. A TS multiplex
/// carries several (video + audio); this element has one output pad, so it emits
/// exactly one, chosen here. The choice is by codec because the output caps are
/// fixed at negotiation, before any packet is parsed: H.264 and H.265 are
/// distinct downstream decoders, not a geometry refinement of one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TsStream {
    /// The first H.264 (AVC) video elementary stream. The default.
    #[default]
    H264,
    /// The first H.265 (HEVC) video elementary stream.
    H265,
    /// The first AAC (ADTS) audio elementary stream.
    Aac,
}

/// Demuxes an MPEG-TS byte stream into one selected elementary stream.
#[derive(Debug)]
pub struct TsDemux {
    demux: TsDemuxer,
    /// The elementary stream this instance forwards (the single output pad).
    stream: TsStream,
    /// Bytes not yet consumed as whole TS packets (packet realignment across
    /// input frames).
    buf: Vec<u8>,
    configured: bool,
    emitted: u64,
    /// Seek support (M362): app time seeks drive an upstream byte-seek and a
    /// re-sync. Inert unless `with_seek` wired the controllers.
    seek: DemuxSeek,
}

impl Default for TsDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl TsDemux {
    pub fn new() -> Self {
        Self {
            demux: TsDemuxer::new(),
            stream: TsStream::H264,
            buf: Vec::new(),
            configured: false,
            emitted: 0,
            seek: DemuxSeek::default(),
        }
    }

    /// Select which elementary stream to forward (default [`TsStream::H264`]).
    pub fn with_stream(mut self, stream: TsStream) -> Self {
        self.stream = stream;
        self
    }

    /// Make the demuxer seekable (M362): `app` carries app time seeks; `upstream`
    /// is the byte source's ([`FileSrc`](crate::filesrc)) byte-seek controller.
    /// On a time seek the demuxer rewinds the source and re-syncs from the
    /// keyframe at/after the target.
    pub fn with_seek(mut self, app: SeekController, upstream: SeekController) -> Self {
        self.seek.with(app, upstream);
        self
    }

    /// Reset the parser for a discontinuity (a `Flush` / seek): drop buffered
    /// bytes and the demuxer's PAT/PMT/PES state, which the re-read stream
    /// re-establishes from its start.
    fn reset_parser(&mut self) {
        self.buf.clear();
        self.demux = TsDemuxer::new();
    }

    /// The elementary stream this instance forwards.
    pub fn stream(&self) -> TsStream {
        self.stream
    }

    /// Count of frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an MPEG-TS byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
    }

    /// The output caps for the selected elementary stream. Video geometry is
    /// unknown until the bitstream parser reads the SPS, so H.264 / H.265
    /// advertise a fixatable placeholder `Range` (`Dim::Any` would fail Phase-2
    /// fixate) refined downstream via `CapsChanged`, the pattern `RtspSrc` uses
    /// for async-discovered dims. AAC has no open `Caps` field, so it advertises
    /// the sentinel channels/rate 0 that `aacparse` accepts pre-header and
    /// refines from the ADTS header.
    fn output_caps(stream: TsStream) -> Caps {
        match stream {
            TsStream::H264 => Self::compressed_video(VideoCodec::H264),
            TsStream::H265 => Self::compressed_video(VideoCodec::H265),
            TsStream::Aac => Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 },
        }
    }

    fn compressed_video(codec: VideoCodec) -> Caps {
        Caps::CompressedVideo {
            codec,
            width: Dim::Range { min: 16, max: 65_535 },
            height: Dim::Range { min: 16, max: 65_535 },
            framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
        }
    }

    /// The PMT `stream_type` the selected output corresponds to.
    fn selected_stream_type(stream: TsStream) -> u8 {
        match stream {
            TsStream::H264 => STREAM_TYPE_H264,
            TsStream::H265 => STREAM_TYPE_H265,
            TsStream::Aac => STREAM_TYPE_AAC,
        }
    }

    /// Consume whole 188-byte TS packets from `buf`, resyncing to the sync byte,
    /// feeding each to the demuxer. Leaves any trailing partial packet in `buf`.
    fn drain_packets(&mut self) {
        loop {
            // Resync: drop bytes before the next sync byte.
            if self.buf.first() != Some(&TS_SYNC) {
                match self.buf.iter().position(|&b| b == TS_SYNC) {
                    Some(pos) => {
                        self.buf.drain(..pos);
                    }
                    None => {
                        self.buf.clear();
                        return;
                    }
                }
            }
            if self.buf.len() < TS_PACKET_LEN {
                return;
            }
            // Feed one packet. (A copy keeps the borrow off `self.buf` so the
            // drain below is clean.)
            let mut pkt = [0u8; TS_PACKET_LEN];
            pkt.copy_from_slice(&self.buf[..TS_PACKET_LEN]);
            self.demux.push_packet(&pkt);
            self.buf.drain(..TS_PACKET_LEN);
        }
    }

    /// Emit each completed access unit of the selected elementary stream as a
    /// frame (Annex-B for H.264 / H.265, ADTS for AAC), carrying its PTS.
    async fn emit_units(
        &mut self,
        units: Vec<EsUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let want = Self::selected_stream_type(self.stream);
        for u in units {
            if u.stream_type != want {
                continue; // a stream other than the selected one
            }
            let pts_ns = u
                .pts_90khz
                .map(|p| (p as u128 * 1_000_000_000 / 90_000) as u64)
                .unwrap_or(0);
            // M362 seek: an audio frame is always a resync point; a video AU is
            // one only if it carries an IDR/IRAP. Drop until the target keyframe.
            let keyframe = match self.stream {
                TsStream::H264 => crate::annexb::au_is_keyframe(VideoCodec::H264, &u.data),
                TsStream::H265 => crate::annexb::au_is_keyframe(VideoCodec::H265, &u.data),
                TsStream::Aac => true,
            };
            match self.seek.admit(pts_ns, keyframe) {
                Admit::Drop => continue,
                Admit::Resume(start) => {
                    let seg = Segment::for_flush_seek(&Seek::flush_to(start), None);
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                Admit::Emit => {}
            }
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(u.data.into_boxed_slice())),
                FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl AsyncElement for TsDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // ByteStream{MpegTs} in -> the selected elementary stream out. The solver
        // hands downstream the chosen caps; the bitstream parser refines video
        // geometry / audio params from the stream via CapsChanged.
        let stream = self.stream;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs } => {
                CapsSet::one(Self::output_caps(stream))
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }) {
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
            // `Flush` returns, drop input so no stale pre-seek units are emitted.
            self.seek.poll_request();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    if self.seek.dropping_input() {
                        return Ok(());
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buf.extend_from_slice(slice.as_slice());
                    self.drain_packets();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // The upstream byte-seek's flush: reset the parser, then re-sync
                // from the re-read stream. Forward it downstream.
                PipelinePacket::Flush => {
                    self.seek.on_flush();
                    self.reset_parser();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final in-flight PES and emit it; the runner's
                    // transform arm forwards the EOS itself.
                    self.demux.flush();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                // ByteStream caps don't carry geometry; nothing to forward, and
                // a Segment passes through.
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TSDEMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stream" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stream = ts_stream_from_str(s).ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "stream" => Some(PropValue::Str(ts_stream_to_str(self.stream).into())),
            _ => None,
        }
    }
}

/// `TsDemux`'s settable properties (M109).
static TSDEMUX_PROPS: &[PropertySpec] =
    &[PropertySpec::new("stream", PropKind::Str, "elementary stream to emit: h264 | h265 | aac")];

/// Parse a `stream` property string to a [`TsStream`].
fn ts_stream_from_str(s: &str) -> Option<TsStream> {
    match s {
        "h264" => Some(TsStream::H264),
        "h265" => Some(TsStream::H265),
        "aac" => Some(TsStream::Aac),
        _ => None,
    }
}

/// The `stream` property string for a [`TsStream`].
fn ts_stream_to_str(stream: TsStream) -> &'static str {
    match stream {
        TsStream::H264 => "h264",
        TsStream::H265 => "h265",
        TsStream::Aac => "aac",
    }
}

impl PadTemplates for TsDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        // One sink (the TS byte stream); the source pad can carry any of the
        // selectable elementary streams (an instance pins one via with_stream).
        let source = CapsSet::from_alternatives(Vec::from([
            Self::output_caps(TsStream::H264),
            Self::output_caps(TsStream::H265),
            Self::output_caps(TsStream::Aac),
        ]));
        Vec::from([PadTemplate::sink(CapsSet::one(Self::input_caps())), PadTemplate::source(source)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{PushOutcome, RawVideoFormat};

    // Re-use the synthetic TS builders by constructing equivalent packets here.
    fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
        const ROOM: usize = TS_PACKET_LEN - 4;
        let mut p = alloc::vec![0u8; TS_PACKET_LEN];
        p[0] = TS_SYNC;
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
        p[2] = (pid & 0xFF) as u8;
        let l = payload.len();
        if l == ROOM {
            p[3] = 0x10;
            p[4..].copy_from_slice(payload);
        } else {
            p[3] = 0x30;
            let af_len = ROOM - 1 - l;
            p[4] = af_len as u8;
            if af_len >= 1 {
                p[5] = 0x00;
                for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                    *b = 0xFF;
                }
            }
            p[5 + af_len..].copy_from_slice(payload);
        }
        p
    }

    fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
        let section_length = body.len() + 4;
        let mut s = alloc::vec![table_id, 0xB0 | ((section_length >> 8) as u8 & 0x0F), (section_length & 0xFF) as u8];
        s.extend_from_slice(body);
        s.extend_from_slice(&[0, 0, 0, 0]);
        let mut payload = alloc::vec![0u8];
        payload.extend_from_slice(&s);
        ts_packet(pid, true, &payload)
    }

    fn pat(pmt_pid: u16) -> Vec<u8> {
        psi(0x0000, 0x00, &[0, 1, 0xC1, 0, 0, 0, 1, 0xE0 | (pmt_pid >> 8) as u8 & 0x1F, pmt_pid as u8])
    }

    fn pmt(es_pid: u16) -> Vec<u8> {
        psi(
            0x1000,
            0x02,
            &[
                0x00, 0x01, 0xC1, 0x00, 0x00,
                0xE0 | (es_pid >> 8) as u8 & 0x1F, es_pid as u8,
                0xF0, 0x00,
                STREAM_TYPE_H264,
                0xE0 | (es_pid >> 8) as u8 & 0x1F, es_pid as u8,
                0xF0, 0x00,
            ],
        )
    }

    fn pes(es: &[u8]) -> Vec<u8> {
        pes_id(0xE0, es)
    }

    /// A PES with an explicit `stream_id` (video 0xE0, audio 0xC0), no PTS.
    fn pes_id(stream_id: u8, es: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0x00, 0x00, 0x01, stream_id];
        let header = [0x80u8, 0x00, 0x00];
        let len = header.len() + es.len();
        p.push((len >> 8) as u8);
        p.push((len & 0xFF) as u8);
        p.extend_from_slice(&header);
        p.extend_from_slice(es);
        p
    }

    /// A two-stream PMT (one video, one audio), the common A/V multiplex shape.
    fn pmt2(v_pid: u16, v_type: u8, a_pid: u16, a_type: u8) -> Vec<u8> {
        psi(
            0x1000,
            0x02,
            &[
                0x00, 0x01, 0xC1, 0x00, 0x00,
                0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8, // PCR_PID
                0xF0, 0x00,
                v_type, 0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8, 0xF0, 0x00,
                a_type, 0xE0 | (a_pid >> 8) as u8 & 0x1F, a_pid as u8, 0xF0, 0x00,
            ],
        )
    }

    #[derive(Default)]
    struct CaptureSink {
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
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
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
    fn caps_byte_stream_in_h264_out() {
        let d = TsDemux::new();
        assert!(d.intercept_caps(&TsDemux::input_caps()).is_ok());
        // A non-TS byte stream / other caps is rejected.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(d.intercept_caps(&raw).is_err());
    }

    #[tokio::test]
    async fn demuxes_h264_frames_from_ts_bytes() {
        let es_pid = 0x0100;
        let mut d = TsDemux::new();
        d.configure_pipeline(&TsDemux::input_caps()).unwrap();

        // Build a TS byte stream: PAT, PMT, then two H.264 access units, each its
        // own PES (PUSI). The first flushes when the second's PES starts; the
        // second flushes on EOS.
        let au0 = [0u8, 0, 0, 1, 0x65, 0xAA];
        let au1 = [0u8, 0, 0, 1, 0x41, 0xBB, 0xCC];
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt(es_pid));
        stream.extend_from_slice(&ts_packet(es_pid, true, &pes(&au0)));
        stream.extend_from_slice(&ts_packet(es_pid, true, &pes(&au1)));

        let mut sink = CaptureSink::default();
        // Feed the whole stream as one System frame.
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        d.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(sink.frames.len(), 2, "two H.264 access units demuxed");
        assert_eq!(sink.frames[0], au0, "first AU bytes intact (PES header stripped)");
        assert_eq!(sink.frames[1], au1);
        assert!(!sink.eos, "EOS is forwarded by the runner's arm, not the element");
        assert_eq!(d.emitted(), 2);
    }

    /// Feed the whole stream as one frame, then EOS, capturing the output.
    async fn run_demux(d: &mut TsDemux, stream: &[u8], sink: &mut CaptureSink) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(stream.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        d.process(PipelinePacket::DataFrame(frame), sink).await.unwrap();
        d.process(PipelinePacket::Eos, sink).await.unwrap();
    }

    #[tokio::test]
    async fn selects_audio_or_video_from_a_multiplex() {
        let v_pid = 0x0100u16;
        let a_pid = 0x0101u16;
        // PAT, a 2-stream PMT (H.264 video + AAC audio), then interleaved access
        // units, each its own PES. The same bytes feed two demuxers that pick
        // apart the multiplex: one the video, one the audio.
        let v0 = [0u8, 0, 0, 1, 0x65, 0x11];
        let v1 = [0u8, 0, 0, 1, 0x41, 0x22];
        let a0 = [0xFFu8, 0xF1, 0x50, 0x80, 0x01, 0x23];
        let a1 = [0xFFu8, 0xF1, 0x50, 0x80, 0x02, 0x45];
        let mut stream = Vec::new();
        stream.extend_from_slice(&pat(0x1000));
        stream.extend_from_slice(&pmt2(v_pid, STREAM_TYPE_H264, a_pid, STREAM_TYPE_AAC));
        stream.extend_from_slice(&ts_packet(v_pid, true, &pes_id(0xE0, &v0)));
        stream.extend_from_slice(&ts_packet(a_pid, true, &pes_id(0xC0, &a0)));
        stream.extend_from_slice(&ts_packet(v_pid, true, &pes_id(0xE0, &v1)));
        stream.extend_from_slice(&ts_packet(a_pid, true, &pes_id(0xC0, &a1)));

        // Default selects H.264: only the two video AUs come out.
        let mut video = TsDemux::new();
        video.configure_pipeline(&TsDemux::input_caps()).unwrap();
        let mut vsink = CaptureSink::default();
        run_demux(&mut video, &stream, &mut vsink).await;
        assert_eq!(vsink.frames, alloc::vec![v0.to_vec(), v1.to_vec()], "video only");

        // stream=aac selects AAC: only the two audio AUs come out (ADTS payload).
        let mut audio = TsDemux::new().with_stream(TsStream::Aac);
        audio.configure_pipeline(&TsDemux::input_caps()).unwrap();
        let mut asink = CaptureSink::default();
        run_demux(&mut audio, &stream, &mut asink).await;
        assert_eq!(asink.frames, alloc::vec![a0.to_vec(), a1.to_vec()], "audio only");
    }

    #[test]
    fn output_caps_track_the_selection() {
        assert!(matches!(
            TsDemux::output_caps(TsStream::H264),
            Caps::CompressedVideo { codec: VideoCodec::H264, .. }
        ));
        assert!(matches!(
            TsDemux::output_caps(TsStream::H265),
            Caps::CompressedVideo { codec: VideoCodec::H265, .. }
        ));
        assert!(matches!(
            TsDemux::output_caps(TsStream::Aac),
            Caps::Audio { format: AudioFormat::Aac, .. }
        ));
    }

    #[test]
    fn stream_property_round_trips_and_drives_output() {
        let mut d = TsDemux::new();
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("h264".into())));

        d.set_property("stream", PropValue::Str("aac".into())).unwrap();
        assert_eq!(d.stream(), TsStream::Aac);
        assert_eq!(d.get_property("stream"), Some(PropValue::Str("aac".into())));

        // An unsupported codec name is rejected (leaving the selection unchanged).
        assert_eq!(d.set_property("stream", PropValue::Str("vp9".into())), Err(PropError::Value));
        assert_eq!(d.stream(), TsStream::Aac);

        // DerivedOutput now maps the TS byte stream to AAC audio.
        let CapsConstraint::DerivedOutput(f) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&TsDemux::input_caps());
        assert!(matches!(out.alternatives(), [Caps::Audio { format: AudioFormat::Aac, .. }]));
    }
}
