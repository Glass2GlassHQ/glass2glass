//! Channel splitter (`deinterleave`), the fan-out that turns one N-channel PCM
//! stream into N mono streams: channel `i` of every sample frame becomes output
//! pad `i`, keeping the input's pts and duration. CPU-only, `no_std` baseline.
//!
//! Two forms, picked the way the demuxers' are (by how many branches link in):
//! [`DeinterleaveN`] is the fan-out (`deinterleave name=d  d.src_0 ! ...
//! d.src_1 ! ...`), and [`Deinterleave`] is the single-output form that emits
//! one chosen channel (`... ! deinterleave channel=1 ! ...`) so a pipeline that
//! wants one channel needs no pad syntax.
//!
//! A fan-out declares its ports' caps before its input negotiates, so
//! [`DeinterleaveN`] names the PCM shape on the element rather than learning it
//! from the input: `deinterleave format=F32LE rate=44100`, with one channel per
//! port. The defaults match `audiotestsrc` (S16LE, 48 kHz). Each port also
//! announces its mono [`Caps::Audio`] with a [`PipelinePacket::CapsChanged`]
//! ahead of its first frame, and again after a flush. Anything but the declared
//! shape is rejected, mid-stream included: a new channel count would need a pad
//! that a running graph cannot grow. The single-output [`Deinterleave`] is an
//! ordinary transform, so its output caps are derived from whatever PCM input it
//! negotiates and it needs no shape of its own.
//!
//! gst's `keep-positions` is not exposed: [`Caps::Audio`] carries a channel
//! count and no positions, so a mono output has nothing to keep.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

use crate::audioconvert::{
    get_pcm_shape_property, pcm_formats, sample_bytes, set_pcm_shape_property, DEFAULT_PCM_FORMAT,
    DEFAULT_PCM_RATE, PCM_SHAPE_PROPS,
};

/// The channel a bare `deinterleave` emits in its single-output form.
const DEFAULT_CHANNEL: u8 = 0;

/// The same value as declared text, for `gst-inspect`.
const DEFAULT_CHANNEL_TEXT: &str = "0";

/// The negotiated input shape: format, channel count, sample rate.
type PcmShape = (AudioFormat, u8, u32);

/// The input caps a splitter takes: any fixed PCM stream.
fn accept_input(caps: &Caps) -> Result<PcmShape, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    if !pcm_formats().contains(format)
        || *channels == ANY_CHANNELS
        || *sample_rate == ANY_SAMPLE_RATE
    {
        return Err(G2gError::CapsMismatch);
    }
    Ok((*format, *channels, *sample_rate))
}

/// The interleaved caps the input carries.
fn interleaved_caps(shape: PcmShape) -> Caps {
    let (format, channels, sample_rate) = shape;
    Caps::Audio {
        format,
        channels,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// The mono caps one output pad carries.
fn mono_caps(shape: PcmShape) -> Caps {
    let (format, _, sample_rate) = shape;
    Caps::Audio {
        format,
        channels: 1,
        sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

/// Copy one channel's samples out of an interleaved buffer. A trailing partial
/// sample frame is left behind.
fn extract_channel(bytes: &[u8], shape: PcmShape, channel: usize) -> Result<Box<[u8]>, G2gError> {
    let (format, channels, _) = shape;
    let per_sample = sample_bytes(format);
    let stride = per_sample * channels as usize;
    if stride == 0 || channel >= channels as usize {
        return Err(G2gError::CapsMismatch);
    }
    let samples = bytes.len() / stride;
    let mut out = vec![0u8; samples * per_sample].into_boxed_slice();
    for sample in 0..samples {
        let src = sample * stride + channel * per_sample;
        let dst = sample * per_sample;
        out[dst..dst + per_sample].copy_from_slice(&bytes[src..src + per_sample]);
    }
    Ok(out)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::deinterleave::DeinterleaveN;
///
/// // one stereo stream in, two mono streams out.
/// let element = DeinterleaveN::new(2);
/// ```
#[derive(Debug)]
pub struct DeinterleaveN {
    outputs: usize,
    format: AudioFormat,
    sample_rate: u32,
    /// Whether port `i` has emitted its mono `CapsChanged` yet. Re-armed on a
    /// flush, as a demuxer's port caps are.
    announced: Vec<bool>,
    emitted: u64,
}

impl DeinterleaveN {
    /// A splitter with `outputs` mono output ports, fed by an `outputs`-channel
    /// stream at the default shape. Panics if `outputs` is zero (a fan-out needs
    /// a port).
    pub fn new(outputs: usize) -> Self {
        assert!(
            outputs > 0 && outputs <= u8::MAX as usize,
            "DeinterleaveN takes 1 to {} outputs, one per input channel",
            u8::MAX
        );
        Self {
            outputs,
            format: DEFAULT_PCM_FORMAT,
            sample_rate: DEFAULT_PCM_RATE,
            announced: vec![false; outputs],
            emitted: 0,
        }
    }

    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Number of output ports (the input's channel count).
    pub fn port_count(&self) -> usize {
        self.outputs
    }

    /// The shape the input carries: one channel per output port.
    fn shape(&self) -> PcmShape {
        (self.format, self.outputs as u8, self.sample_rate)
    }

    /// Accept exactly the declared input shape. Another channel count has no pad
    /// to route its extra channels to, and another format or rate would leave the
    /// ports announcing caps the samples no longer have.
    fn accept_for_ports(&self, caps: &Caps) -> Result<PcmShape, G2gError> {
        let shape = accept_input(caps)?;
        if shape != self.shape() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(shape)
    }
}

impl MultiOutputElement for DeinterleaveN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host samples, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_for_ports(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// Every port carries the same mono shape, so a source that would deliver a
    /// different format, rate, or channel count fails to negotiate rather than
    /// being silently converted.
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(interleaved_caps(self.shape())))
    }

    /// Declare each port's mono caps, so a branch negotiates against the channel
    /// it will carry instead of against the input's channel count.
    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        (port < self.outputs).then(|| mono_caps(self.shape()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.accept_for_ports(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let shape = self.shape();
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    for port in 0..self.outputs {
                        if !self.announced[port] {
                            self.announced[port] = true;
                            out.push_to(port, PipelinePacket::CapsChanged(mono_caps(shape)))
                                .await?;
                        }
                        let channel = extract_channel(bytes, shape, port)?;
                        let out_frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(channel)),
                            frame.timing,
                            self.emitted,
                        );
                        self.emitted += 1;
                        out.push_to(port, PipelinePacket::DataFrame(out_frame))
                            .await?;
                    }
                }
                // Only the declared shape: a channel-count change would need a
                // pad per new channel, which a running graph cannot grow.
                PipelinePacket::CapsChanged(caps) => {
                    self.accept_for_ports(&caps)?;
                }
                PipelinePacket::Flush => {
                    self.announced.iter_mut().for_each(|a| *a = false);
                    for port in 0..self.outputs {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    for port in 0..self.outputs {
                        out.push_to(port, PipelinePacket::Segment(seg)).await?;
                    }
                }
                // The runner broadcasts the single Eos to every port.
                PipelinePacket::Eos => {}
                // future PipelinePacket variants: no-op.
                _ => {}
            }
            Ok(())
        })
    }

    /// The declared input shape (M1072). gst reads format and rate from the sink
    /// pad's caps, which a fan-out here cannot read before it declares its ports.
    fn properties(&self) -> &'static [PropertySpec] {
        PCM_SHAPE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        set_pcm_shape_property(&mut self.format, &mut self.sample_rate, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        get_pcm_shape_property(self.format, self.sample_rate, name)
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::deinterleave::Deinterleave;
///
/// // the right channel of a stereo stream.
/// let element = Deinterleave::new().with_channel(1);
/// ```
#[derive(Debug)]
pub struct Deinterleave {
    channel: u8,
    input: Option<PcmShape>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for Deinterleave {
    fn default() -> Self {
        Self::new()
    }
}

impl Deinterleave {
    /// The single-output form, emitting the first channel.
    pub fn new() -> Self {
        Self {
            channel: DEFAULT_CHANNEL,
            input: None,
            configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    pub fn with_channel(mut self, channel: u8) -> Self {
        self.channel = channel;
        self
    }

    /// Accept caps that carry the selected channel.
    fn accept_for_channel(&self, caps: &Caps) -> Result<PcmShape, G2gError> {
        let shape = accept_input(caps)?;
        if self.channel >= shape.1 {
            return Err(G2gError::CapsMismatch);
        }
        Ok(shape)
    }
}

impl AsyncElement for Deinterleave {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host samples, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.accept_for_channel(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// One channel out of any PCM input: format and rate pass through, the
    /// channel count becomes 1.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let channel = self.channel;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match accept_input(input) {
            Ok(shape) if channel < shape.1 => CapsSet::one(mono_caps(shape)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(self.accept_for_channel(absolute_caps)?);
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
                    let shape = self.input.ok_or(G2gError::NotConfigured)?;
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let channel = extract_channel(bytes, shape, self.channel as usize)?;

                    let caps = mono_caps(shape);
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let out_frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(channel)),
                        frame.timing,
                        self.emitted,
                    );
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The input's own caps arrive here; the mono output caps are
                // re-derived from them and re-announced above.
                PipelinePacket::CapsChanged(caps) => {
                    self.input = Some(self.accept_for_channel(&caps)?);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // the transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Audio channel picker",
            "Filter/Converter/Audio",
            "Emits one channel of an N-channel PCM stream as mono",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DEINTERLEAVE_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "channel" => self.channel = value.as_uint().ok_or(PropError::Type)? as u8,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "channel" => Some(PropValue::Uint(self.channel as u64)),
            _ => None,
        }
    }
}

/// `Deinterleave`'s properties (M1072): which channel the single-output form
/// emits. The fan-out form takes one channel per pad instead, so it has none.
static DEINTERLEAVE_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "channel",
    PropKind::Uint,
    "index of the channel to emit as mono",
)
.with_default(DEFAULT_CHANNEL_TEXT)];

impl PadTemplates for Deinterleave {
    fn pad_templates() -> Vec<PadTemplate> {
        let any_pcm = |format| Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let mono_pcm = |format| Caps::Audio {
            format,
            channels: 1,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(
                pcm_formats().map(any_pcm).to_vec(),
            )),
            PadTemplate::source(CapsSet::from_alternatives(
                pcm_formats().map(mono_pcm).to_vec(),
            )),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn stereo(format: AudioFormat) -> Caps {
        Caps::Audio {
            format,
            channels: 2,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    #[test]
    fn a_channel_is_every_nth_sample() {
        let shape = (AudioFormat::PcmS16Le, 2, RATE);
        // interleaved i16 pairs: (1, -1), (2, -2), (3, -3).
        let bytes: Vec<u8> = [1i16, -1, 2, -2, 3, -3]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let left = extract_channel(&bytes, shape, 0).unwrap();
        let right = extract_channel(&bytes, shape, 1).unwrap();
        let unpack = |b: &[u8]| -> Vec<i16> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        assert_eq!(unpack(&left), [1, 2, 3]);
        assert_eq!(unpack(&right), [-1, -2, -3]);
    }

    #[test]
    fn a_wide_sample_format_splits_by_its_sample_size() {
        let shape = (AudioFormat::PcmS24Le, 2, RATE);
        let bytes: Vec<u8> = (0..12).collect();
        assert_eq!(
            &*extract_channel(&bytes, shape, 0).unwrap(),
            &[0, 1, 2, 6, 7, 8]
        );
        assert_eq!(
            &*extract_channel(&bytes, shape, 1).unwrap(),
            &[3, 4, 5, 9, 10, 11]
        );
    }

    #[test]
    fn the_port_count_has_to_be_the_channel_count() {
        let mut element = DeinterleaveN::new(2);
        assert!(element
            .configure_pipeline(&stereo(AudioFormat::PcmS16Le))
            .is_ok());
        let mono = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(
            MultiOutputElement::configure_pipeline(&mut element, &mono).unwrap_err(),
            G2gError::CapsMismatch,
            "a channel with no pad to leave on is a mismatch"
        );
    }

    #[test]
    fn the_single_output_form_needs_the_channel_to_exist() {
        let element = Deinterleave::new().with_channel(2);
        assert_eq!(
            AsyncElement::intercept_caps(&element, &stereo(AudioFormat::PcmS16Le)).unwrap_err(),
            G2gError::CapsMismatch
        );
        assert_eq!(
            Deinterleave::new().with_channel(1).get_property("channel"),
            Some(PropValue::Uint(1))
        );
        assert_eq!(DEFAULT_CHANNEL_TEXT.parse::<u8>(), Ok(DEFAULT_CHANNEL));
    }
}
