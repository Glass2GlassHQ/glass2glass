//! Multi-stream Ogg multiplexer element (M790): N audio elementary streams in
//! (`Caps::Audio{Opus|Vorbis|Flac}` per pad), one grouped Ogg byte stream out.
//! The fan-in analog of the single-input [`crate::oggmux::OggMux`], and the
//! write side of the grouped files [`crate::oggdemux::OggDemuxN`] reads.
//!
//! A [`MultiInputElement`]: each pad becomes its own logical bitstream with its
//! own serial number, and the codec mapping per stream is the shared
//! [`OggStreamMux`] the single-input muxer also uses. Packets interleave by
//! presentation timestamp through the M204
//! [`InputAggregator::take_earliest_by`](g2g_core::InputAggregator::take_earliest_by)
//! merge, so the data pages of the streams alternate the way a player expects.
//!
//! Page order follows RFC 3533 §4 grouping: every stream's beginning-of-stream
//! page first, in pad order, then the remaining header pages per stream, then
//! the interleaved data pages. That block is written when the merge first
//! releases a packet, which is also the first moment every pad's in-band codec
//! config is known to have arrived.
//!
//! ```text
//! a.! m.  b.! m.  oggmux name=m ! filesink location=out.ogg
//! ```
//!
//! Registered as the `oggmux` muxer in
//! [`default_registry`](crate::registry::default_registry), so the one name
//! covers `! oggmux !` (the single-stream element) and the fan-in shape above,
//! the way `mpegtsmux` does.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, InputAggregator,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::oggmux::{format_of, ogg_caps, OggStreamMux, DEFAULT_SERIAL};

/// Muxes N audio elementary streams into one grouped Ogg byte stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::oggmuxn::OggMuxN;
///
/// let mux = OggMuxN::new(2).with_serial(0x1234);
/// assert_eq!(mux.emitted(), 0);
/// ```
#[derive(Debug)]
pub struct OggMuxN {
    inputs: usize,
    /// One logical bitstream per input pad, in pad order.
    streams: Vec<OggStreamMux>,
    /// Per-input packet buffer; releases the globally earliest-PTS packet (M204).
    agg: InputAggregator<Frame>,
    /// Serial of the first stream; pad `i` takes `serial + i`.
    serial: u32,
    /// Whether the grouped header block has been written.
    headers_written: bool,
    /// Whether the end-of-stream pages have been written, so the close happens
    /// once however many per-input EOS packets arrive.
    closed: bool,
    emitted: u64,
}

impl OggMuxN {
    /// A muxer with `inputs` input pads. Pad `i` becomes the `i`-th logical
    /// bitstream of the file, and its codec mapping comes from its negotiated
    /// caps.
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "OggMuxN needs at least one input");
        Self {
            inputs,
            streams: (0..inputs)
                .map(|i| OggStreamMux::new(serial_for(DEFAULT_SERIAL, i)))
                .collect(),
            agg: InputAggregator::new(inputs),
            serial: DEFAULT_SERIAL,
            headers_written: false,
            closed: false,
            emitted: 0,
        }
    }

    /// Set the first stream's serial number; pad `i` takes `serial + i`.
    pub fn with_serial(mut self, serial: u32) -> Self {
        self.set_serials(serial);
        self
    }

    /// Count of Ogg byte frames emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The serial numbers of the logical bitstreams, in pad order.
    pub fn serials(&self) -> Vec<u32> {
        self.streams.iter().map(|s| s.serial()).collect()
    }

    fn set_serials(&mut self, serial: u32) {
        self.serial = serial;
        for (i, stream) in self.streams.iter_mut().enumerate() {
            stream.set_serial(serial_for(serial, i));
        }
    }

    /// Write the grouped header block: every stream's beginning-of-stream page in
    /// pad order, then each stream's remaining header pages (RFC 3533 §4).
    fn write_headers(&mut self) -> Result<Vec<u8>, G2gError> {
        let mut out = Vec::new();
        for stream in &mut self.streams {
            out.extend_from_slice(&stream.write_bos()?);
        }
        for stream in &mut self.streams {
            out.extend_from_slice(&stream.write_rest());
        }
        self.headers_written = true;
        Ok(out)
    }

    /// Wrap muxed bytes as an output frame, or `None` when nothing was produced.
    fn byte_frame(&mut self, bytes: Vec<u8>) -> Option<PipelinePacket> {
        if bytes.is_empty() {
            return None;
        }
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            FrameTiming::default(),
            self.emitted,
        );
        self.emitted += 1;
        Some(PipelinePacket::DataFrame(frame))
    }
}

/// The serial of pad `index` given the base. Wrapping, so a base near the top of
/// the range still yields distinct serials.
fn serial_for(base: u32, index: usize) -> u32 {
    base.wrapping_add(index as u32)
}

impl MultiInputElement for OggMuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    /// Named request pads (M481): every input is an audio slot, so `audio_%u` /
    /// `sink_%u` each claim the next positional one.
    fn input_pad_index(
        &self,
        _req: &g2g_core::runtime::PadRequest,
        ordinal: usize,
    ) -> Option<usize> {
        (ordinal < self.inputs).then_some(ordinal)
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if format_of(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        // Each pad forwards its stream verbatim; `configure_pipeline` rejects an
        // unsupported caps. `AcceptsAny` is the native muxer-input shape.
        CapsConstraint::AcceptsAny
    }

    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(ogg_caps())))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.streams[input].configure(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(ogg_caps())
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "serial",
            PropKind::Uint,
            "serial number of the first logical bitstream; pad i takes serial + i",
        )];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "serial" => {
                // The serials are baked into every page, so this only takes effect
                // before the header block is written.
                if self.headers_written {
                    return Err(PropError::ReadOnly);
                }
                let raw = value.as_uint().ok_or(PropError::Type)?;
                self.set_serials(u32::try_from(raw).map_err(|_| PropError::Value)?);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "serial" => Some(PropValue::Uint(u64::from(self.serial))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Codec config arrives in-band ahead of the audio. It never
                    // enters the merge: the header block needs every stream's
                    // config before any of it can be written.
                    if !self.headers_written && self.streams[input].is_header(slice) {
                        self.streams[input].push_header(slice);
                        return Ok(());
                    }
                    self.agg.push(input, frame);
                }
                // A per-input Eos lets the merge release packets held waiting on
                // this input; the runner emits the merged Eos.
                PipelinePacket::Eos => self.agg.mark_ended(input),
                // Channels / rate only feed a synthesized header, written before
                // any audio flows.
                PipelinePacket::CapsChanged(caps) => {
                    self.streams[input].refine_caps(&caps);
                    return Ok(());
                }
                // A per-input `Segment` maps that stream to running time; a muxed
                // container carries its own timestamps, so it is consumed rather
                // than forwarded into the byte stream.
                PipelinePacket::Segment(_) => return Ok(()),
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            let mut bytes = Vec::new();
            // Drain every packet now safe to emit, in global PTS order. The first
            // release means every contributing pad has delivered audio, so every
            // stream's in-band config has landed and the header block can go out.
            while let Some((stream, frame)) = self.agg.take_earliest_by(|f| f.timing.pts_ns) {
                if !self.headers_written {
                    bytes.extend_from_slice(&self.write_headers()?);
                }
                let Some(slice) = frame.domain.as_system_slice() else {
                    return Err(G2gError::UnsupportedDomain);
                };
                bytes.extend_from_slice(
                    &self.streams[stream].push_audio(slice, frame.timing.duration_ns),
                );
            }
            // Once every pad has ended and drained, close each logical bitstream.
            if self.headers_written
                && !self.closed
                && (0..self.inputs).all(|i| self.agg.is_ended(i))
            {
                self.closed = true;
                for stream in &mut self.streams {
                    bytes.extend_from_slice(&stream.flush(true));
                }
            }
            if let Some(p) = self.byte_frame(bytes) {
                out.push(p).await?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg::{page_flags, OggCodec, OggDemuxer};
    use alloc::vec;
    use g2g_core::{AudioFormat, PushOutcome};

    #[derive(Default)]
    struct CaptureSink {
        bytes: Vec<u8>,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
        }
    }

    fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    /// A 20 ms CELT-FB stereo Opus packet (TOC config 31, one frame).
    fn opus_packet(fill: u8) -> Vec<u8> {
        vec![(31 << 3) | 0x04, fill, fill]
    }

    /// A native FLAC header (`fLaC` + STREAMINFO) for `rate`, stereo.
    fn flac_header(rate: u32) -> Vec<u8> {
        let mut native = Vec::from(*b"fLaC");
        native.extend_from_slice(&[0x80, 0, 0, 34]);
        let mut body = [0u8; 34];
        body[10] = (rate >> 12) as u8;
        body[11] = (rate >> 4) as u8;
        body[12] = (((rate & 0xF) as u8) << 4) | (1 << 1);
        native.extend_from_slice(&body);
        native
    }

    /// One 4096-sample FLAC frame header at 44.1 kHz stereo, with its CRC-8.
    fn flac_frame() -> Vec<u8> {
        vec![0xFFu8, 0xF8, 0xC9, 0x18, 0x00, 0xC2]
    }

    /// Mux an Opus stream on pad 0 and an Ogg-FLAC stream on pad 1.
    async fn mux_opus_and_flac() -> Vec<u8> {
        let mut mux = OggMuxN::new(2);
        mux.configure_pipeline(0, &audio_caps(AudioFormat::Opus, 2, 48_000))
            .unwrap();
        mux.configure_pipeline(1, &audio_caps(AudioFormat::Flac, 2, 44_100))
            .unwrap();
        let mut sink = CaptureSink::default();
        // FLAC's in-band header, then interleaved audio (Opus has no in-band
        // header here, so its `OpusHead` is synthesized).
        mux.process(1, frame(flac_header(44_100), 0), &mut sink)
            .await
            .unwrap();
        for i in 0..3u64 {
            mux.process(0, frame(opus_packet(i as u8), i * 20_000_000), &mut sink)
                .await
                .unwrap();
            mux.process(1, frame(flac_frame(), i * 92_879_818), &mut sink)
                .await
                .unwrap();
        }
        mux.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        mux.process(1, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
        sink.bytes
    }

    #[tokio::test]
    async fn beginning_of_stream_pages_lead_and_serials_are_distinct() {
        let bytes = mux_opus_and_flac().await;
        let pages = page_flags(&bytes);
        let bos: Vec<u32> = pages
            .iter()
            .filter(|(_, ht)| ht & 0x02 != 0)
            .map(|(s, _)| *s)
            .collect();
        assert_eq!(bos.len(), 2, "one beginning-of-stream page per stream");
        assert_ne!(bos[0], bos[1], "distinct serial numbers");
        assert_eq!(
            &pages[..2].iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            &bos,
            "RFC 3533 grouping: both BOS pages precede every other page"
        );
        // Exactly one end-of-stream page per stream, and both streams close.
        let eos: Vec<u32> = pages
            .iter()
            .filter(|(_, ht)| ht & 0x04 != 0)
            .map(|(s, _)| *s)
            .collect();
        assert_eq!(eos.len(), 2);
        assert_ne!(eos[0], eos[1]);
    }

    #[tokio::test]
    async fn both_streams_round_trip_through_the_demuxer() {
        let bytes = mux_opus_and_flac().await;
        let mut d = OggDemuxer::new();
        d.push_data(&bytes);

        assert_eq!(d.streams().len(), 2);
        assert_eq!(d.streams()[0].info().unwrap().codec, OggCodec::Opus);
        assert_eq!(d.streams()[1].info().unwrap().codec, OggCodec::Flac);
        assert_eq!(d.streams()[1].info().unwrap().sample_rate, 44_100);
        assert_eq!(
            d.stream_mut(0).unwrap().take_packets(),
            (0..3).map(|i| opus_packet(i as u8)).collect::<Vec<_>>()
        );
        assert_eq!(
            d.stream_mut(1).unwrap().take_packets(),
            vec![flac_frame(); 3]
        );
        // Per-stream granules: three 20 ms Opus packets, three 4096-sample blocks.
        assert_eq!(d.streams()[0].end_granule(), Some(3 * 960));
        assert_eq!(d.streams()[1].end_granule(), Some(3 * 4096));
    }

    #[tokio::test]
    async fn the_serial_property_re_keys_every_stream() {
        let mut mux = OggMuxN::new(2);
        mux.set_property("serial", PropValue::Uint(1000)).unwrap();
        assert_eq!(mux.serials(), vec![1000, 1001]);
        assert_eq!(mux.get_property("serial"), Some(PropValue::Uint(1000)));
        assert!(
            mux.set_property("serial", PropValue::Uint(1 << 40))
                .is_err(),
            "a serial number is 32 bits"
        );
    }
}
