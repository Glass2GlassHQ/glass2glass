//! MPEG-TS demuxer element (M108): `Caps::ByteStream{MpegTs}` in, the H.264
//! video elementary stream out (`Caps::CompressedVideo{H264}`, Annex-B).
//!
//! Wraps the pure [`crate::mpegts::TsDemuxer`] parser. Incoming byte frames are
//! resynchronized to 188-byte TS packets and fed to the demuxer; the reassembled
//! PES access units of the first video stream are forwarded as H.264 frames with
//! their PTS, ready for `h264parse`. CPU, `no_std` baseline.
//!
//! ```text
//! filesrc(location=x.ts, caps=ByteStream{MpegTs}) ! tsdemux ! h264parse ! <decoder> ! <sink>
//! ```
//!
//! Scope (v1): one video stream (H.264). Audio elementary streams and H.265 are
//! parsed by the core demuxer but not yet emitted here (the output pad is typed
//! H.264); selecting them is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, VideoCodec,
};

use crate::mpegts::{EsUnit, TsDemuxer, STREAM_TYPE_H264, TS_PACKET_LEN};

const TS_SYNC: u8 = 0x47;

/// Demuxes an MPEG-TS byte stream into its H.264 video elementary stream.
#[derive(Debug)]
pub struct TsDemux {
    demux: TsDemuxer,
    /// Bytes not yet consumed as whole TS packets (packet realignment across
    /// input frames).
    buf: Vec<u8>,
    configured: bool,
    emitted: u64,
}

impl Default for TsDemux {
    fn default() -> Self {
        Self::new()
    }
}

impl TsDemux {
    pub fn new() -> Self {
        Self { demux: TsDemuxer::new(), buf: Vec::new(), configured: false, emitted: 0 }
    }

    /// Count of H.264 frames forwarded.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The input this element accepts: an MPEG-TS byte stream.
    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }
    }

    /// The output it produces: H.264. The geometry is unknown until `h264parse`
    /// reads the SPS, so this advertises a fixatable placeholder `Range`
    /// (`Dim::Any` would fail Phase-2 fixate) that `h264parse` refines via
    /// `CapsChanged`, the same pattern `RtspSrc` uses for async-discovered dims.
    fn output_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Range { min: 16, max: 65_535 },
            height: Dim::Range { min: 16, max: 65_535 },
            framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
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

    /// Emit each completed video access unit as an H.264 frame.
    async fn emit_units(
        &mut self,
        units: Vec<EsUnit>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for u in units {
            if u.stream_type != STREAM_TYPE_H264 {
                continue; // v1: only the H.264 video stream is forwarded
            }
            let pts_ns = u
                .pts_90khz
                .map(|p| (p as u128 * 1_000_000_000 / 90_000) as u64)
                .unwrap_or(0);
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
        // ByteStream{MpegTs} in -> H.264 out. The solver hands downstream the
        // H.264 caps; h264parse refines geometry from the SPS via CapsChanged.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs } => {
                CapsSet::one(Self::output_caps())
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buf.extend_from_slice(slice.as_slice());
                    self.drain_packets();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the final in-flight PES, emit it, then forward EOS.
                    self.demux.flush();
                    let units = self.demux.take_units();
                    self.emit_units(units, out).await?;
                    out.push(PipelinePacket::Eos).await?;
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
}

impl PadTemplates for TsDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_caps())),
            PadTemplate::source(CapsSet::one(Self::output_caps())),
        ])
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
        let mut p = alloc::vec![0x00, 0x00, 0x01, 0xE0];
        let header = [0x80u8, 0x00, 0x00]; // no PTS
        let len = header.len() + es.len();
        p.push((len >> 8) as u8);
        p.push((len & 0xFF) as u8);
        p.extend_from_slice(&header);
        p.extend_from_slice(es);
        p
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
        assert!(sink.eos, "EOS forwarded");
        assert_eq!(d.emitted(), 2);
    }
}
