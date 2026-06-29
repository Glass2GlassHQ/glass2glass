//! Multi-output fragmented-MP4 demuxer element (M391): one ISO-BMFF byte stream
//! in, N elementary streams out, one track per output port. The MP4 sibling of
//! [`MkvDemuxN`](crate::mkvdemux::MkvDemuxN) and
//! [`TsDemuxN`](crate::tsdemux::TsDemuxN), and the multi-track read-side analog
//! of [`Mp4Src`](crate::mp4src::Mp4Src) (which forwards a single video track).
//!
//! A [`MultiOutputElement`] driven by
//! [`run_source_fanout`](g2g_core::runtime::run_source_fanout): it buffers the
//! whole byte stream and, on `Eos`, parses every `moov/trak`
//! ([`parse_all_tracks`]) and routes each `moof`+`mdat` fragment to the port
//! matching its `track_ID` ([`parse_fragments_multi`]), so one demuxer feeds
//! several decode branches (audio + video together). Port `i` emits its
//! elementary [`Caps`] ([`PipelinePacket::CapsChanged`]) before its first frame.
//! With a bus, announces the file's tracks as a `StreamCollection` (M386).
//!
//! Scope (v1): fragmented multi-track files (what [`Mp4MuxN`](crate::mp4muxn)
//! writes and CMAF shares); clear (unencrypted) H.264 / H.265 video + AAC audio.
//! A progressive (non-fragmented) multi-track file carries no `moof`, so it
//! yields no fragments here; single-track progressive playback stays on `Mp4Src`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    BusHandle, BusMessage, ByteStreamEncoding, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, MultiOutputElement, MultiOutputSink, PipelinePacket, Rate, Stream,
    StreamCollection, StreamType,
};

use crate::fmp4::{parse_all_tracks, parse_fragments_multi, starts_with_param_set, TrackHeader, TrackKind};

/// One output port: the `track_ID` it forwards and the elementary [`Caps`] the
/// downstream decode branch plugs from (parsed from the file's `moov` when the
/// fan-out is built). The MP4 analog of a [`TsStream`](crate::tsdemux::TsStream)
/// selection, but a track is named by its numeric id rather than a codec enum.
#[derive(Debug, Clone)]
pub struct Mp4Port {
    pub track_id: u32,
    pub caps: Caps,
}

/// A forwardable track for the `playbin uri=*.mp4` fan-out (M392): the
/// `track_ID` a demux port forwards, the elementary caps the decode branch plugs
/// from, and whether it is video (vs audio). The MP4 analog of
/// [`TsStreamInfo`](crate::tsdemux::TsStreamInfo).
#[derive(Debug, Clone)]
pub struct Mp4StreamInfo {
    pub track_id: u32,
    pub caps: Caps,
    pub video: bool,
    /// The AAC AudioSpecificConfig for an audio track (empty for video), which an
    /// AAC decode branch needs out-of-band (the muxed access units are raw AAC).
    pub asc: Vec<u8>,
}

/// The track's **negotiation** caps (and video flag): what the decode branch
/// solves against at startup. Video advertises `Fixed` geometry and a `Range`
/// framerate (per-frame PTS carries the real timing); compressed audio uses the
/// `0/0` "unknown until parsed" form (AAC caps intersect by strict equality, so
/// concrete channels/rate would not match a decoder's wildcard sink), the same
/// convention [`TsDemux`](crate::tsdemux::TsDemux) uses. Refined at runtime by
/// [`real_caps`] via the port's `CapsChanged`.
fn nego_caps(kind: &TrackKind) -> (Caps, bool) {
    match kind {
        TrackKind::Video { codec, width, height, .. } => (
            Caps::CompressedVideo {
                codec: *codec,
                width: Dim::Fixed(*width),
                height: Dim::Fixed(*height),
                framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
            },
            true,
        ),
        TrackKind::Audio { format, .. } => {
            (Caps::Audio { format: *format, channels: 0, sample_rate: 0 }, false)
        }
    }
}

/// The track's **real** caps: the concrete channel layout / sample rate (audio)
/// for the runtime `CapsChanged` refinement and the discovery `StreamCollection`.
/// For video this equals [`nego_caps`] (geometry is already concrete).
fn real_caps(kind: &TrackKind) -> Caps {
    match kind {
        TrackKind::Video { .. } => nego_caps(kind).0,
        TrackKind::Audio { format, channels, sample_rate, .. } => {
            Caps::Audio { format: *format, channels: *channels, sample_rate: *sample_rate }
        }
    }
}

/// The forwardable tracks an MP4 carries, in `moov` order (M392): one
/// [`Mp4StreamInfo`] per A/V `trak`, carrying the negotiation caps (what a decode
/// branch plugs from). `data` must hold the `moov` (a file prefix covering the
/// init is enough); returns empty for a non-MP4 or unparseable input, which the
/// `playbin` hook reads as "decline, fall through".
pub fn forwardable_streams(data: &[u8]) -> Vec<Mp4StreamInfo> {
    parse_all_tracks(data)
        .map(|tracks| {
            tracks
                .iter()
                .map(|t| {
                    let (caps, video) = nego_caps(&t.kind);
                    let asc = match &t.kind {
                        TrackKind::Audio { asc, .. } => asc.clone(),
                        TrackKind::Video { .. } => Vec::new(),
                    };
                    Mp4StreamInfo { track_id: t.track_id, caps, video, asc }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Multi-output fragmented-MP4 demuxer: one ISO-BMFF byte stream in, N
/// elementary streams out, one track per port.
#[derive(Debug)]
pub struct Mp4DemuxN {
    /// The whole byte stream, accumulated until `Eos` (the `moov` may sit after
    /// the fragments in a progressive file; buffering keeps the parse simple).
    buf: Vec<u8>,
    /// Port `i` forwards `ports[i].track_id` with `ports[i].caps`.
    ports: Vec<Mp4Port>,
    /// Whether port `i` has emitted its opening `CapsChanged` yet.
    announced: Vec<bool>,
    bus: Option<BusHandle>,
    /// Set once the `StreamCollection` (M386) has been announced, so it posts once.
    collection_posted: bool,
    emitted: u64,
}

impl Mp4DemuxN {
    /// A demuxer with one output port per entry of `ports`, in port order. Panics
    /// if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<Mp4Port>) -> Self {
        assert!(!ports.is_empty(), "Mp4DemuxN needs at least one output port");
        let announced = alloc::vec![false; ports.len()];
        Self { buf: Vec::new(), ports, announced, bus: None, collection_posted: false, emitted: 0 }
    }

    /// Attach the pipeline bus so the file's tracks post as a `StreamCollection`
    /// (M386), the way [`Mp4Src::with_bus`](crate::mp4src::Mp4Src::with_bus) does.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Number of output ports (the forwarded-track count).
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Count of frames forwarded across all ports.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn input_caps() -> Caps {
        Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff }
    }

    /// Announce every forwardable track as a `StreamCollection` (M386), once. A
    /// no-op without a bus or once posted.
    fn post_stream_collection(&mut self, tracks: &[TrackHeader]) {
        if self.collection_posted {
            return;
        }
        let streams: Vec<Stream> = tracks
            .iter()
            .map(|t| {
                let video = matches!(t.kind, TrackKind::Video { .. });
                let ty = if video { StreamType::Video } else { StreamType::Audio };
                let name = alloc::format!("mp4-track-{}", t.track_id);
                Stream::new(name, ty, real_caps(&t.kind))
            })
            .collect();
        if streams.is_empty() {
            return;
        }
        self.collection_posted = true;
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::StreamCollection(StreamCollection::new("mp4-0", streams)));
        }
    }

    /// Parse the buffered file and route every sample to its track's port, each
    /// port's opening `CapsChanged` first. Video samples that lack in-band
    /// parameter sets get the `moov`'s sets prepended to their first frame, so a
    /// decoder can start (matching [`Mp4Src`](crate::mp4src::Mp4Src)).
    async fn parse_and_emit(&mut self, out: &mut dyn MultiOutputSink) -> Result<(), G2gError> {
        let tracks = parse_all_tracks(&self.buf)?;
        if self.bus.is_some() {
            self.post_stream_collection(&tracks);
        }
        let samples = parse_fragments_multi(&self.buf, &tracks)?;
        let mut need_sets = alloc::vec![true; self.ports.len()];

        for (track_id, sample) in samples {
            let Some(port) = self.ports.iter().position(|p| p.track_id == track_id) else {
                continue; // a track no port forwards
            };
            let kind = tracks.iter().find(|t| t.track_id == track_id).map(|t| &t.kind);
            // Announce the port's real caps once (refining the looser negotiation
            // caps the branch solved against, e.g. concrete AAC channels/rate).
            if !self.announced[port] {
                let caps = kind.map(real_caps).unwrap_or_else(|| self.ports[port].caps.clone());
                out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
                self.announced[port] = true;
            }

            let mut data = sample.annexb;
            // Prepend out-of-band parameter sets to the first video frame if it
            // carries none (our own muxer keeps them in-band; CMAF may not).
            if need_sets[port] {
                if let Some(TrackKind::Video { codec, param_sets, .. }) = kind {
                    if !starts_with_param_set(&data, *codec) {
                        let mut with = Vec::new();
                        for set in param_sets {
                            with.extend_from_slice(&[0, 0, 0, 1]);
                            with.extend_from_slice(set);
                        }
                        with.extend_from_slice(&data);
                        data = with;
                    }
                }
            }
            need_sets[port] = false;

            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                timing: FrameTiming {
                    pts_ns: sample.pts_ns,
                    dts_ns: sample.pts_ns,
                    duration_ns: sample.duration_ns,
                    capture_ns: sample.pts_ns,
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    keyframe: sample.keyframe,
                },
                sequence: self.emitted,
                meta: Default::default(),
            };
            self.emitted += 1;
            out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl MultiOutputElement for Mp4DemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_caps())
    }

    /// Declare each port's elementary caps (M380), so the solver negotiates each
    /// branch against its track at startup. `None` for an out-of-range port.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.ports.get(port).map(|p| p.caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        absolute_caps.intersect(&Self::input_caps()).map(|_| ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                // Buffer the byte stream; the parse waits for the whole file.
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.buf.extend_from_slice(slice.as_slice());
                }
                // End of stream: parse the buffered file and emit every sample;
                // the runner broadcasts the merged Eos to every port after this.
                PipelinePacket::Eos => {
                    self.parse_and_emit(out).await?;
                }
                // A flush discards the buffer (a re-read restarts the file).
                PipelinePacket::Flush => {
                    self.buf.clear();
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.ports.len() {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                // The input's (byte-stream) CapsChanged is consumed: each port
                // defines its own caps, announced per port in parse_and_emit.
                PipelinePacket::CapsChanged(_) => {}
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4muxn::Mp4MuxN;
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::runtime::block_on;
    use g2g_core::{
        AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate,
        VideoCodec,
    };

    /// An OutputSink that captures the muxer's byte stream.
    #[derive(Default)]
    struct ByteCapture {
        bytes: Vec<u8>,
    }
    impl OutputSink for ByteCapture {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(f) = packet {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes.extend_from_slice(s.as_slice());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    /// A MultiOutputSink that records, per port, the CapsChanged and DataFrame
    /// payloads it receives.
    #[derive(Default)]
    struct PortCapture {
        caps: Vec<Option<Caps>>,
        frames: Vec<Vec<Vec<u8>>>,
    }
    impl PortCapture {
        fn new(ports: usize) -> Self {
            Self { caps: alloc::vec![None; ports], frames: alloc::vec![Vec::new(); ports] }
        }
    }
    impl MultiOutputSink for PortCapture {
        fn port_count(&self) -> usize {
            self.frames.len()
        }

        fn push_to<'a>(
            &'a mut self,
            port: usize,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps[port] = Some(c),
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames[port].push(s.as_slice().to_vec());
                        }
                    }
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    fn adts_au(payload: &[u8]) -> Vec<u8> {
        let frame_len = payload.len() + 7;
        let mut au = alloc::vec![
            0xFF,
            0xF1,
            (1 << 6) | (3 << 2), // 48 kHz
            ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
            ((frame_len >> 3) & 0xFF) as u8,
            (((frame_len & 7) << 5) as u8) | 0x1F,
            0xFC,
        ];
        au.extend_from_slice(payload);
        au
    }

    fn vframe(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
            0,
        ))
    }

    /// Build a two-track (H.264 + AAC) fragmented MP4 by driving `Mp4MuxN`, then
    /// fan it back out with `Mp4DemuxN`: each track's samples must land on its own
    /// port with the right caps, and the video's parameter sets must survive.
    #[test]
    fn fans_a_muxed_av_file_out_to_two_ports() {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];

        // --- mux an A/V file ---------------------------------------------
        let bytes = block_on(async {
            let mut mux = Mp4MuxN::new(2);
            mux.configure_pipeline(
                0,
                &Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    width: Dim::Fixed(320),
                    height: Dim::Fixed(240),
                    framerate: Rate::Fixed(30 << 16),
                },
            )
            .unwrap();
            mux.configure_pipeline(
                1,
                &Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48000 },
            )
            .unwrap();
            let mut sink = ByteCapture::default();
            mux.process(0, vframe(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
            mux.process(1, vframe(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink).await.unwrap();
            mux.process(0, vframe(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
            mux.process(1, vframe(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink).await.unwrap();
            mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
            mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
            sink.bytes
        });

        // --- discover the tracks and fan back out ------------------------
        let streams = forwardable_streams(&bytes);
        assert_eq!(streams.len(), 2, "video + audio tracks discovered");
        assert!(streams[0].video, "track 0 is video");
        assert!(!streams[1].video, "track 1 is audio");

        let ports: Vec<Mp4Port> =
            streams.iter().map(|s| Mp4Port { track_id: s.track_id, caps: s.caps.clone() }).collect();
        let mut demux = Mp4DemuxN::new(ports);
        let mut out = PortCapture::new(2);
        block_on(async {
            demux.process(vframe(bytes, 0), &mut out).await.unwrap();
            demux.process(PipelinePacket::Eos, &mut out).await.unwrap();
        });

        // Port 0 (video) got both frames; the first opens with parameter sets.
        assert!(matches!(out.caps[0], Some(Caps::CompressedVideo { codec: VideoCodec::H264, .. })));
        assert_eq!(out.frames[0].len(), 2, "two video access units");
        assert!(
            starts_with_param_set(&out.frames[0][0], VideoCodec::H264),
            "first video frame opens with an SPS"
        );
        // Port 1 (audio) got both AAC access units.
        assert!(matches!(out.caps[1], Some(Caps::Audio { format: AudioFormat::Aac, .. })));
        assert_eq!(out.frames[1].len(), 2, "two audio access units");
        assert_eq!(demux.emitted(), 4, "four frames forwarded across both ports");
    }
}
