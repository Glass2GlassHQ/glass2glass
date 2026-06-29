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

use crate::fmp4::{
    parse_all_tracks, parse_fragments_multi, parse_progressive_multi, starts_with_param_set, Sample,
    TrackHeader, TrackKind,
};
use crate::mp4box::find_box;

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
        // A timed-text track plugs as its cue format directly (the container
        // carries the timing); not video, so the fan-out flag is false.
        TrackKind::Text { format } => (Caps::Text { format: *format }, false),
    }
}

/// The track's **real** caps: the concrete channel layout / sample rate (audio)
/// for the runtime `CapsChanged` refinement and the discovery `StreamCollection`.
/// For video this equals [`nego_caps`] (geometry is already concrete).
fn real_caps(kind: &TrackKind) -> Caps {
    match kind {
        TrackKind::Video { .. } | TrackKind::Text { .. } => nego_caps(kind).0,
        TrackKind::Audio { format, channels, sample_rate, .. } => {
            Caps::Audio { format: *format, channels: *channels, sample_rate: *sample_rate }
        }
    }
}

/// Build an ADTS-framed AAC access unit from the track's 2-byte
/// AudioSpecificConfig and the raw access unit: a 7-byte ADTS header (no CRC)
/// derived from the ASC's audio-object-type, sampling-frequency index, and
/// channel configuration, then the AU. The inverse of the muxer's de-ADTS write,
/// so the demuxed audio is self-describing. `None` when the ASC is too short, the
/// rate index / channel config is out of range, or the frame exceeds the 13-bit
/// ADTS length (then the AU is forwarded raw).
fn adts_from_asc(asc: &[u8], au: &[u8]) -> Option<Vec<u8>> {
    if asc.len() < 2 {
        return None;
    }
    let aot = asc[0] >> 3; // audio object type (5 bits)
    let sr_index = ((asc[0] & 0x07) << 1) | (asc[1] >> 7);
    let channel_config = (asc[1] >> 3) & 0x0F;
    if sr_index > 12 || channel_config == 0 {
        return None; // reserved/explicit rate or "config in stream": not ADTS-able
    }
    let profile = aot.saturating_sub(1) & 0x03; // ADTS profile = AOT - 1
    let frame_len = au.len() + 7;
    if frame_len > 0x1FFF {
        return None; // ADTS frame_length is 13 bits
    }
    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&[
        0xFF,
        0xF1, // syncword | MPEG-4 | layer 0 | protection_absent (no CRC)
        (profile << 6) | (sr_index << 2) | ((channel_config >> 2) & 1),
        ((channel_config & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F, // buffer fullness (top bits)
        0xFC, // buffer fullness (low) | num_raw_data_blocks = 0
    ]);
    out.extend_from_slice(au);
    Some(out)
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
                // Text tracks are discovered and forwardable by an explicit
                // [`Mp4Port`], but `playbin` has no text-branch auto-plug yet, so
                // omit them from the fan-out (an unplumbed `Caps::Text` branch
                // would fail negotiation). Wired when the text auto-plug lands.
                .filter(|t| !matches!(t.kind, TrackKind::Text { .. }))
                .map(|t| {
                    let (caps, video) = nego_caps(&t.kind);
                    let asc = match &t.kind {
                        TrackKind::Audio { asc, .. } => asc.clone(),
                        TrackKind::Video { .. } | TrackKind::Text { .. } => Vec::new(),
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
    /// Clear-key cbcs decryption key (M398), supplied by the app for an encrypted
    /// file; the constant IV + crypt/skip pattern come from each track's `tenc`.
    /// Without it an encrypted track fails loud.
    #[cfg(feature = "mp4-cenc")]
    cenc_key: Option<[u8; 16]>,
    emitted: u64,
}

impl Mp4DemuxN {
    /// A demuxer with one output port per entry of `ports`, in port order. Panics
    /// if `ports` is empty (a fan-out needs a port).
    pub fn new(ports: Vec<Mp4Port>) -> Self {
        assert!(!ports.is_empty(), "Mp4DemuxN needs at least one output port");
        let announced = alloc::vec![false; ports.len()];
        Self {
            buf: Vec::new(),
            ports,
            announced,
            bus: None,
            collection_posted: false,
            #[cfg(feature = "mp4-cenc")]
            cenc_key: None,
            emitted: 0,
        }
    }

    /// Supply the clear-key cbcs decryption key (M398) for an encrypted file. The
    /// constant IV and crypt/skip pattern come from each track's `tenc`; this is
    /// the 16-byte content key. Without it an encrypted track fails loud.
    #[cfg(feature = "mp4-cenc")]
    pub fn with_cenc_key(mut self, key: [u8; 16]) -> Self {
        self.cenc_key = Some(key);
        self
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
                let ty = match t.kind {
                    TrackKind::Video { .. } => StreamType::Video,
                    TrackKind::Audio { .. } => StreamType::Audio,
                    TrackKind::Text { .. } => StreamType::Text,
                };
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

    /// Parse the fragmented file's samples, decrypting an encrypted track in place
    /// with the supplied cbcs key (M398). Without the `mp4-cenc` feature (or a
    /// key), an encrypted track fails loud inside `parse_fragments_multi`.
    fn parse_fragments(&self, tracks: &[TrackHeader]) -> Result<Vec<(u32, Sample)>, G2gError> {
        #[cfg(feature = "mp4-cenc")]
        if let Some(key) = self.cenc_key {
            let mut decrypt = move |cenc: &crate::fmp4::CencDefaults, buf: &mut [u8], subs: &[crate::fmp4::Subsample]| {
                // cbcs constant IV (per-sample IV size 0); a malformed IV leaves
                // the sample untouched rather than panicking on the slice convert.
                if let Ok(iv) = <[u8; 16]>::try_from(cenc.constant_iv.as_slice()) {
                    crate::cenc::cbcs_decrypt_sample(
                        buf, subs, &key, &iv, cenc.crypt_byte_block, cenc.skip_byte_block,
                    );
                }
            };
            return parse_fragments_multi(&self.buf, tracks, Some(&mut decrypt));
        }
        parse_fragments_multi(&self.buf, tracks, None)
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
        // A `moof` marks a fragmented / CMAF file; its absence is the classic
        // progressive `moov`+`mdat` layout, walked from the sample tables instead.
        let samples = if find_box(&self.buf, b"moof").is_some() {
            self.parse_fragments(&tracks)?
        } else {
            parse_progressive_multi(&self.buf, &tracks)?
        };
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
            match kind {
                // Prepend out-of-band parameter sets to the first video frame if
                // it carries none (our own muxer keeps them in-band; CMAF may not).
                Some(TrackKind::Video { codec, param_sets, .. }) => {
                    if need_sets[port] && !starts_with_param_set(&data, *codec) {
                        let mut with = Vec::new();
                        for set in param_sets {
                            with.extend_from_slice(&[0, 0, 0, 1]);
                            with.extend_from_slice(set);
                        }
                        with.extend_from_slice(&data);
                        data = with;
                    }
                }
                // ADTS-frame every AAC access unit from the track's ASC, so the
                // audio elementary stream is self-describing (carries its profile
                // / rate / channels per frame) and a decoder can start without the
                // out-of-band config, symmetric with the in-band video param sets.
                Some(TrackKind::Audio { asc, .. }) => {
                    if let Some(framed) = adts_from_asc(asc, &data) {
                        data = framed;
                    }
                }
                // Text cues arrive already de-framed (the tx3g length prefix is
                // stripped in the sample parse); forward the UTF-8 payload as is.
                Some(TrackKind::Text { .. }) | None => {}
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
        ptss: Vec<Vec<u64>>,
    }
    impl PortCapture {
        fn new(ports: usize) -> Self {
            Self {
                caps: alloc::vec![None; ports],
                frames: alloc::vec![Vec::new(); ports],
                ptss: alloc::vec![Vec::new(); ports],
            }
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
                        self.ptss[port].push(f.timing.pts_ns);
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

    /// Mux a minimal two-track (H.264 + AAC) fragmented MP4 via `Mp4MuxN`.
    fn mux_av() -> Vec<u8> {
        let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
        let pps = [0x68u8, 0xce, 0x3c, 0x80];
        let idr = [0x65u8, 0x88, 0x84, 0x00];
        block_on(async {
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
        })
    }

    #[test]
    fn adts_from_asc_builds_a_valid_lc_header() {
        // ASC [0x11, 0x90]: AOT 2 (LC), sr_index 3 (48 kHz), channel config 2.
        let framed = adts_from_asc(&[0x11, 0x90], &[0xAA, 0xBB]).expect("ADTS built");
        assert_eq!(framed[0], 0xFF);
        assert_eq!(framed[1], 0xF1);
        // profile = AOT-1 = 1; sr_index 3; channel high bit 0.
        assert_eq!(framed[2], (1 << 6) | (3 << 2));
        // channel low 2 bits (2) in the top, then frame_len (9) high bits.
        let frame_len: u32 = 2 + 7;
        assert_eq!(framed[3], ((2u8 & 3) << 6) | (((frame_len >> 11) & 3) as u8));
        assert_eq!(&framed[7..], &[0xAA, 0xBB], "the AU follows the 7-byte header");
        // A too-short ASC declines (the AU would be forwarded raw).
        assert!(adts_from_asc(&[0x11], &[0xAA]).is_none());
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

    /// The audio port emits self-describing ADTS (the ASC wired in-band), so a
    /// real `AacParse` accepts the demuxed stream and recovers the concrete
    /// channel layout / sample rate from the headers, the proof that audio decodes.
    #[test]
    fn audio_port_emits_decodable_adts() {
        use crate::aacparse::AacParse;
        use g2g_core::AsyncElement;

        let bytes = mux_av();
        let streams = forwardable_streams(&bytes);
        let ports: Vec<Mp4Port> =
            streams.iter().map(|s| Mp4Port { track_id: s.track_id, caps: s.caps.clone() }).collect();
        let mut demux = Mp4DemuxN::new(ports);
        let mut out = PortCapture::new(2);
        block_on(async {
            demux.process(vframe(bytes, 0), &mut out).await.unwrap();
            demux.process(PipelinePacket::Eos, &mut out).await.unwrap();
        });

        // Every audio frame opens with an ADTS syncword (0xFFF).
        for au in &out.frames[1] {
            assert!(au.len() >= 7 && au[0] == 0xFF && (au[1] & 0xF0) == 0xF0, "ADTS-framed AAC");
        }

        // A real AAC parser recovers 48 kHz / stereo from the ADTS headers.
        #[derive(Default)]
        struct CapsCapture {
            last: Option<Caps>,
        }
        impl OutputSink for CapsCapture {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    if let PipelinePacket::CapsChanged(c) = packet {
                        self.last = Some(c);
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let recovered = block_on(async {
            let mut parser = AacParse::new();
            parser
                .configure_pipeline(&Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 })
                .unwrap();
            let mut sink = CapsCapture::default();
            for au in &out.frames[1] {
                parser.process(vframe(au.clone(), 0), &mut sink).await.unwrap();
            }
            sink.last
        });
        assert_eq!(
            recovered,
            Some(Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }),
            "AacParse recovers the real channel layout / sample rate from the wired-in ASC"
        );
    }

    /// A progressive MP4 carrying a `tx3g` 3GPP-timed-text subtitle track fans its
    /// cues out as a `Caps::Text { Utf8 }` port: each cue's UTF-8 payload (the
    /// 2-byte length prefix stripped) lands on the text port with the container's
    /// per-cue PTS. Proves subtitle-track extraction end to end. The text track is
    /// also kept out of `forwardable_streams` (no playbin text-branch auto-plug
    /// yet), so existing A/V auto-plug is unaffected.
    #[test]
    fn extracts_a_tx3g_subtitle_track() {
        use crate::fmp4::parse_all_tracks;
        use crate::mp4box::{full_box, mp4_box};
        use g2g_core::TextFormat;

        // --- box builders (same offsets the parser reads) -----------------
        let tkhd = |track_id: u32| {
            let mut c = alloc::vec![0u8; 80];
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = |timescale: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&timescale.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        let hdlr = |handler: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(handler);
            full_box(b"hdlr", 0, 0, &c)
        };
        let stsd = |entry: &[u8]| {
            let mut p = 1u32.to_be_bytes().to_vec();
            p.extend_from_slice(entry);
            full_box(b"stsd", 0, 0, &p)
        };
        let stsz = |sizes: &[u32]| {
            let mut b = alloc::vec![0u8; 8]; // default_size 0, then count
            b[4..8].copy_from_slice(&(sizes.len() as u32).to_be_bytes());
            for s in sizes {
                b.extend_from_slice(&s.to_be_bytes());
            }
            full_box(b"stsz", 0, 0, &b)
        };
        let stts = |count: u32, delta: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&count.to_be_bytes());
            b.extend_from_slice(&delta.to_be_bytes());
            full_box(b"stts", 0, 0, &b)
        };
        let stsc = |spc: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
            b.extend_from_slice(&spc.to_be_bytes()); // samples_per_chunk
            b.extend_from_slice(&1u32.to_be_bytes()); // sample_desc_index
            full_box(b"stsc", 0, 0, &b)
        };
        let stco = |offset: u32| {
            let mut b = 1u32.to_be_bytes().to_vec();
            b.extend_from_slice(&offset.to_be_bytes());
            full_box(b"stco", 0, 0, &b)
        };

        // A tx3g sample = 2-byte big-endian text length + that many UTF-8 bytes.
        let cue = |text: &str| {
            let mut s = (text.len() as u16).to_be_bytes().to_vec();
            s.extend_from_slice(text.as_bytes());
            s
        };
        let c0 = cue("Hello");
        let c1 = cue("World");
        let sizes = [c0.len() as u32, c1.len() as u32];

        // Minimal tx3g sample entry: the 8-byte SampleEntry header is all the
        // parser inspects (it only needs the `tx3g` box to be present).
        let tx3g = mp4_box(b"tx3g", &[0u8; 8]);

        // mdat first, so the chunk offset is the constant 8 (after the box header)
        // and the moov needs no placeholder rebuild.
        let mut mdat_body = c0.clone();
        mdat_body.extend_from_slice(&c1);
        let mdat = mp4_box(b"mdat", &mdat_body);

        // timescale 1000 (ms): cue 0 at t=0, cue 1 at t=1000 (1 s), each 1 s long.
        let stbl = [stsd(&tx3g), stsz(&sizes), stts(2, 1000), stsc(2), stco(8)].concat();
        let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stbl));
        let mdia = mp4_box(b"mdia", &[mdhd(1000), hdlr(b"text"), minf].concat());
        let trak = mp4_box(b"trak", &[tkhd(1), mdia].concat());
        let moov = mp4_box(b"moov", &trak);

        let mut file = mdat;
        file.extend_from_slice(&moov);

        // --- parse: one Utf8 text track --------------------------------------
        let tracks = parse_all_tracks(&file).expect("text track parses");
        assert_eq!(tracks.len(), 1, "the subtitle track is discovered");
        assert!(
            matches!(tracks[0].kind, TrackKind::Text { format: TextFormat::Utf8 }),
            "tx3g maps to Caps::Text {{ Utf8 }}"
        );

        // The playbin fan-out omits text (no text-branch auto-plug yet).
        assert!(forwardable_streams(&file).is_empty(), "text excluded from auto-plug");

        // --- fan out: the cues land on the text port -------------------------
        let ports = alloc::vec![Mp4Port { track_id: 1, caps: real_caps(&tracks[0].kind) }];
        let mut demux = Mp4DemuxN::new(ports);
        let mut out = PortCapture::new(1);
        block_on(async {
            demux.process(vframe(file, 0), &mut out).await.unwrap();
            demux.process(PipelinePacket::Eos, &mut out).await.unwrap();
        });

        assert_eq!(out.caps[0], Some(Caps::Text { format: TextFormat::Utf8 }));
        assert_eq!(out.frames[0].len(), 2, "two cues");
        assert_eq!(out.frames[0][0], b"Hello", "tx3g length prefix stripped to the cue");
        assert_eq!(out.frames[0][1], b"World");
        assert_eq!(out.ptss[0], alloc::vec![0, 1_000_000_000], "per-cue PTS from the container");
    }

    /// An encrypted (cbcs) H.264 track fans out and decrypts: a hand-built `encv`
    /// fragment whose sample is AES-128-CBC encrypted under a known key + constant
    /// IV is routed through `Mp4DemuxN::with_cenc_key`, and the demuxed Annex-B must
    /// match the original clear access unit. Without the key the encrypted track
    /// fails loud. Proves the per-track cbcs path the multi-track demuxer adds.
    #[cfg(feature = "mp4-cenc")]
    #[test]
    fn encrypted_track_decrypts_with_the_cenc_key() {
        use crate::mp4box::{full_box, mp4_box};

        let key = [0x11u8; 16];
        let iv = [0x22u8; 16];
        let (crypt, skip) = (1u8, 9u8);

        // The clear AVCC sample: a 4-byte length prefix + a 12-byte IDR NAL, 16
        // bytes total so the whole sample is one cbcs block (no clear remainder).
        let nal: [u8; 12] = [0x65, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut clear = alloc::vec![0u8, 0, 0, 12];
        clear.extend_from_slice(&nal);
        assert_eq!(clear.len(), 16);
        let cipher = cbc_encrypt_block(&clear, &key, &iv);

        // --- box builders --------------------------------------------------
        let tkhd = |track_id: u32, w: u32, h: u32| {
            let mut c = alloc::vec![0u8; 80];
            c[8..12].copy_from_slice(&track_id.to_be_bytes());
            c[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            c[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            full_box(b"tkhd", 0, 0, &c)
        };
        let mdhd = |ts: u32| {
            let mut c = alloc::vec![0u8; 16];
            c[8..12].copy_from_slice(&ts.to_be_bytes());
            full_box(b"mdhd", 0, 0, &c)
        };
        let hdlr = |h: &[u8; 4]| {
            let mut c = alloc::vec![0u8; 20];
            c[4..8].copy_from_slice(h);
            full_box(b"hdlr", 0, 0, &c)
        };
        let avcc = {
            let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1e];
            let pps: &[u8] = &[0x68, 0xce];
            let mut p = alloc::vec![0u8; 5];
            p.push(0xE1);
            p.extend_from_slice(&(sps.len() as u16).to_be_bytes());
            p.extend_from_slice(sps);
            p.push(1);
            p.extend_from_slice(&(pps.len() as u16).to_be_bytes());
            p.extend_from_slice(pps);
            mp4_box(b"avcC", &p)
        };
        // sinf: frma=avc1 + schm=cbcs + schi[tenc v1: pattern + constant IV].
        let tenc = {
            // [reserved, packed pattern, isProtected, per_sample_IV_size]
            let mut c = alloc::vec![0u8, (crypt << 4) | skip, 1, 0];
            c.extend_from_slice(&[0u8; 16]); // default_KID
            c.push(16); // default_constant_IV_size
            c.extend_from_slice(&iv); // constant IV
            full_box(b"tenc", 1, 0, &c)
        };
        let schm = {
            let mut c = Vec::new();
            c.extend_from_slice(b"cbcs"); // scheme_type (schm[4..8])
            c.extend_from_slice(&0u32.to_be_bytes()); // scheme_version
            full_box(b"schm", 0, 0, &c)
        };
        let sinf = mp4_box(
            b"sinf",
            &[mp4_box(b"frma", b"avc1"), schm, mp4_box(b"schi", &tenc)].concat(),
        );
        let encv = {
            let mut p = alloc::vec![0u8; 78];
            p.extend_from_slice(&avcc);
            p.extend_from_slice(&sinf);
            mp4_box(b"encv", &p)
        };
        let stsd = {
            let mut p = 1u32.to_be_bytes().to_vec();
            p.extend_from_slice(&encv);
            full_box(b"stsd", 0, 0, &p)
        };
        let trak = {
            let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stsd));
            let mdia = mp4_box(b"mdia", &[mdhd(90_000), hdlr(b"vide"), minf].concat());
            mp4_box(b"trak", &[tkhd(1, 320, 240), mdia].concat())
        };
        let moov = mp4_box(b"moov", &trak);

        // moof: tfhd(track 1) + tfdt(0) + trun(1 sample) + senc(1 sample, whole-AU).
        let tfhd = full_box(b"tfhd", 0, 0, &1u32.to_be_bytes());
        let tfdt = full_box(b"tfdt", 1, 0, &0u64.to_be_bytes());
        let trun = {
            let mut p = 1u32.to_be_bytes().to_vec(); // sample count
            p.extend_from_slice(&0u32.to_be_bytes()); // data offset
            p.extend_from_slice(&3000u32.to_be_bytes()); // duration
            p.extend_from_slice(&(cipher.len() as u32).to_be_bytes()); // size
            full_box(b"trun", 0, 0x000301, &p)
        };
        // senc with a subsample map (flags 0x2): one sample, one subsample with no
        // clear bytes so the whole 16-byte AU is the protected range.
        let senc = {
            let mut c = 1u32.to_be_bytes().to_vec(); // sample count
            c.extend_from_slice(&1u16.to_be_bytes()); // subsample count
            c.extend_from_slice(&0u16.to_be_bytes()); // clear bytes
            c.extend_from_slice(&(clear.len() as u32).to_be_bytes()); // protected bytes
            full_box(b"senc", 0, 0x2, &c)
        };
        let moof = mp4_box(b"moof", &mp4_box(b"traf", &[tfhd, tfdt, trun, senc].concat()));

        let mut file = Vec::new();
        file.extend_from_slice(&moov);
        file.extend_from_slice(&moof);
        file.extend_from_slice(&mp4_box(b"mdat", &cipher));

        // forwardable_streams discovers the (encrypted) video track.
        let streams = forwardable_streams(&file);
        assert_eq!(streams.len(), 1, "the encrypted video track is discovered");
        let ports: Vec<Mp4Port> =
            streams.iter().map(|s| Mp4Port { track_id: s.track_id, caps: s.caps.clone() }).collect();

        // With the key the sample decrypts back to the original Annex-B.
        let mut demux = Mp4DemuxN::new(ports.clone()).with_cenc_key(key);
        let mut out = PortCapture::new(1);
        block_on(async {
            demux.process(vframe(file.clone(), 0), &mut out).await.unwrap();
            demux.process(PipelinePacket::Eos, &mut out).await.unwrap();
        });
        assert_eq!(out.frames[0].len(), 1, "one video access unit");
        let frame = &out.frames[0][0];
        // The decrypted IDR rides at the tail (the parameter sets are prepended);
        // its bytes matching the clear NAL is the proof decryption succeeded.
        let mut idr = alloc::vec![0u8, 0, 0, 1];
        idr.extend_from_slice(&nal);
        assert!(frame.ends_with(&idr), "decrypted to the clear Annex-B IDR");
        assert!(starts_with_param_set(frame, VideoCodec::H264), "param sets prepended");

        // Without a key, the encrypted track fails loud rather than emitting garbage.
        let mut keyless = Mp4DemuxN::new(ports);
        let mut out2 = PortCapture::new(1);
        let result = block_on(async {
            keyless.process(vframe(file, 0), &mut out2).await?;
            keyless.process(PipelinePacket::Eos, &mut out2).await
        });
        assert!(result.is_err(), "an encrypted track without a key fails loud");
    }

    /// AES-128-CBC encrypt one 16-byte block (the fixture side of the cbcs
    /// decrypt), so the test data round-trips through the demuxer's decryptor.
    #[cfg(feature = "mp4-cenc")]
    fn cbc_encrypt_block(clear: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Vec<u8> {
        use aes::cipher::{block_padding::NoPadding, BlockEncryptMut, KeyIvInit};
        type Enc = cbc::Encryptor<aes::Aes128>;
        let mut buf = clear.to_vec();
        let len = buf.len();
        let ct = Enc::new(&(*key).into(), &(*iv).into())
            .encrypt_padded_mut::<NoPadding>(&mut buf, len)
            .expect("block-aligned");
        ct.to_vec()
    }
}
