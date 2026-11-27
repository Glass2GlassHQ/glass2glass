// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::time::ticks_to_ns;
use g2g_core::{
    pcm_caps_any_rate_and_channels, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet,
    ChannelLayout, ChannelPosition, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::mhas::{MhasAccessUnit, MhasFramer, MPEGH_SAMPLES_PER_ACCESS_UNIT};
use crate::mpeghconfig::{cicp_channel_count, mha_config_record};
use crate::pcmoutput::{pcm_decoder_output, DecodedPcm, PcmOutput};

#[allow(unreachable_pub)]
mod ffi {
    use core::ffi::c_int;

    #[repr(C)]
    pub struct MpeghDecoderContext {
        _opaque: [u8; 0],
    }

    #[repr(C)]
    pub struct MpeghDecoderOutputInfo {
        pub num_samples_per_channel: c_int,
        pub num_channels: c_int,
        pub sample_rate: c_int,
        pub ticks: u64,
        pub pts: u64,
        pub loudness: c_int,
        pub is_concealed: bool,
    }
    // sizeof(MPEGH_DECODER_OUTPUT_INFO) on x86_64 Linux
    const _: () = assert!(core::mem::size_of::<MpeghDecoderOutputInfo>() == 40);

    pub const MPEGH_DEC_OK: c_int = 0;
    pub const MPEGH_DEC_FEED_DATA: c_int = 1;

    pub const MPEGH_DEC_PARAM_TARGET_REFERENCE_LEVEL: c_int = 0;
    pub const MPEGH_DEC_PARAM_EFFECT_TYPE: c_int = 1;
    pub const MPEGH_DEC_PARAM_BOOST_FACTOR: c_int = 2;
    pub const MPEGH_DEC_PARAM_ATTENUATION_FACTOR: c_int = 3;
    pub const MPEGH_DEC_PARAM_ALBUM_MODE: c_int = 4;

    extern "C" {
        pub fn mpeghdecoder_init(cicp_setup: i32) -> *mut MpeghDecoderContext;
        pub fn mpeghdecoder_setMhaConfig(
            context: *mut MpeghDecoderContext,
            config: *const u8,
            config_size: u32,
        ) -> c_int;
        pub fn mpeghdecoder_destroy(context: *mut MpeghDecoderContext);
        pub fn mpeghdecoder_process(
            context: *mut MpeghDecoderContext,
            in_data: *const u8,
            in_length: u32,
            timestamp: u64,
        ) -> c_int;
        pub fn mpeghdecoder_getSamples(
            context: *mut MpeghDecoderContext,
            out_data: *mut i32,
            out_length: u32,
            out_info: *mut MpeghDecoderOutputInfo,
        ) -> c_int;
        pub fn mpeghdecoder_flushAndGet(context: *mut MpeghDecoderContext) -> c_int;
        pub fn mpeghdecoder_flush(context: *mut MpeghDecoderContext) -> c_int;
        pub fn mpeghdecoder_setParam(
            context: *mut MpeghDecoderContext,
            param: c_int,
            value: c_int,
        ) -> c_int;
    }
}

fn check(code: core::ffi::c_int) -> Result<(), G2gError> {
    if code == ffi::MPEGH_DEC_OK {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Other))
    }
}

// mpeghdec renders up to 24 speakers and holds up to 3072 samples per channel between calls
const MAX_OUTPUT_SAMPLES: usize = 24 * 3072;

#[derive(Debug)]
struct Decoder {
    context: NonNull<ffi::MpeghDecoderContext>,
    samples: Vec<i32>,
}

// SAFETY: the context is owned by one `Decoder` and only reached through `&mut self`, and
// mpeghdec keeps no thread-local state, so moving it to another thread is sound.
unsafe impl Send for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the context came from mpeghdecoder_init and is destroyed once, here.
        unsafe { ffi::mpeghdecoder_destroy(self.context.as_ptr()) };
    }
}

impl Decoder {
    fn open(target_layout: u8) -> Result<Self, G2gError> {
        // SAFETY: mpeghdecoder_init takes a plain integer and returns null on failure.
        let context = unsafe { ffi::mpeghdecoder_init(i32::from(target_layout)) };
        let context = NonNull::new(context).ok_or(G2gError::Hardware(HardwareError::Other))?;
        Ok(Decoder {
            context,
            samples: alloc::vec![0; MAX_OUTPUT_SAMPLES],
        })
    }

    fn set_param(&mut self, param: core::ffi::c_int, value: i32) -> Result<(), G2gError> {
        // SAFETY: the context is live and the parameter id is one mpeghdecoder.h defines.
        check(unsafe { ffi::mpeghdecoder_setParam(self.context.as_ptr(), param, value) })
    }

    fn set_mha_config(&mut self, config: &[u8]) -> Result<(), G2gError> {
        let length = u32::try_from(config.len()).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: the context is live and `config` is valid for `length` bytes.
        check(unsafe {
            ffi::mpeghdecoder_setMhaConfig(self.context.as_ptr(), config.as_ptr(), length)
        })
    }

    fn process(&mut self, access_unit: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let length = u32::try_from(access_unit.len()).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: the context is live and `access_unit` is valid for `length` bytes.
        check(unsafe {
            ffi::mpeghdecoder_process(self.context.as_ptr(), access_unit.as_ptr(), length, pts_ns)
        })
    }

    fn flush_and_get(&mut self) -> Result<(), G2gError> {
        // SAFETY: the context is live.
        check(unsafe { ffi::mpeghdecoder_flushAndGet(self.context.as_ptr()) })
    }

    fn flush(&mut self) -> Result<(), G2gError> {
        // SAFETY: the context is live.
        check(unsafe { ffi::mpeghdecoder_flush(self.context.as_ptr()) })
    }

    // each block of samples the decoder has ready, interleaved, with its pts
    fn take_samples(&mut self) -> Result<Vec<DecodedBlock>, G2gError> {
        let mut blocks = Vec::new();
        loop {
            // SAFETY: an all-zero bit pattern is valid for this integer and bool struct.
            let mut info: ffi::MpeghDecoderOutputInfo = unsafe { core::mem::zeroed() };
            // SAFETY: the context is live, `samples` has room for MAX_OUTPUT_SAMPLES values and
            // `info` is a writable out parameter.
            let code = unsafe {
                ffi::mpeghdecoder_getSamples(
                    self.context.as_ptr(),
                    self.samples.as_mut_ptr(),
                    MAX_OUTPUT_SAMPLES as u32,
                    &mut info,
                )
            };
            if code == ffi::MPEGH_DEC_FEED_DATA {
                return Ok(blocks);
            }
            check(code)?;
            let channels = u8::try_from(info.num_channels).map_err(|_| G2gError::CapsMismatch)?;
            let per_channel = usize::try_from(info.num_samples_per_channel)
                .map_err(|_| G2gError::CapsMismatch)?;
            if per_channel == 0 {
                continue;
            }
            blocks.push(DecodedBlock {
                samples: self.samples[..per_channel * usize::from(channels)].to_vec(),
                channels,
                sample_rate: u32::try_from(info.sample_rate).map_err(|_| G2gError::CapsMismatch)?,
                pts_ns: info.pts,
            });
        }
    }
}

struct DecodedBlock {
    samples: Vec<i32>,
    channels: u8,
    sample_rate: u32,
    pts_ns: u64,
}

// the speaker layout of a CICP target layout g2g names every speaker of, and the
// decoded channel behind each interleaved one
fn cicp_speaker_layout(index: u8) -> Option<(ChannelLayout, &'static [usize])> {
    use ChannelPosition::*;
    Some(match index {
        1 => (ChannelLayout::MONO, &[0]),
        2 => (ChannelLayout::STEREO, &[0, 1]),
        3 => (ChannelLayout::of(&[Fl, Fr, Fc]), &[0, 1, 2]),
        4 => (ChannelLayout::of(&[Fl, Fr, Fc, Bc]), &[0, 1, 2, 3]),
        5 => (ChannelLayout::of(&[Fl, Fr, Fc, Bl, Br]), &[0, 1, 2, 3, 4]),
        6 => (ChannelLayout::SURROUND_5_1, &[0, 1, 2, 3, 4, 5]),
        9 => (ChannelLayout::of(&[Fl, Fr, Bc]), &[0, 1, 2]),
        10 => (ChannelLayout::of(&[Fl, Fr, Bl, Br]), &[0, 1, 2, 3]),
        11 => (
            ChannelLayout::of(&[Fl, Fr, Fc, Lfe, Bl, Br, Bc]),
            &[0, 1, 2, 3, 4, 5, 6],
        ),
        // mpeghdec puts the side pair ahead of the rear pair
        12 => (ChannelLayout::SURROUND_7_1, &[0, 1, 2, 3, 6, 7, 4, 5]),
        // wide or height speakers
        _ => return None,
    })
}

// gst mpeghdec's drc-effect-type nicks and the MPEG-D DRC effect type each requests
const DRC_EFFECT_TYPES: [(&str, i32); 8] = [
    ("off", -1),
    ("none", 0),
    ("night", 1),
    ("noisy", 2),
    ("limited", 3),
    ("lowlevel", 4),
    ("dialog", 5),
    ("general", 6),
];

// mpeghdec takes the target loudness in steps of -0.25 LKFS
const LOUDNESS_STEPS_PER_LKFS: f64 = -4.0;
// mpeghdec takes the DRC scaling factors as 0 to 127
const DRC_FACTOR_FULL_SCALE: f64 = 127.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct MpeghDecSettings {
    target_layout: u8,
    target_reference_level: f64,
    drc_effect_type: i32,
    drc_attenuation_factor: f64,
    drc_boost_factor: f64,
    album_mode: bool,
}

impl Default for MpeghDecSettings {
    fn default() -> Self {
        // gst mpeghdec's defaults
        Self {
            target_layout: 6,
            target_reference_level: -24.0,
            drc_effect_type: 6,
            drc_attenuation_factor: 1.0,
            drc_boost_factor: 1.0,
            album_mode: false,
        }
    }
}

impl MpeghDecSettings {
    fn apply(&self, decoder: &mut Decoder) -> Result<(), G2gError> {
        decoder.set_param(
            ffi::MPEGH_DEC_PARAM_TARGET_REFERENCE_LEVEL,
            (self.target_reference_level * LOUDNESS_STEPS_PER_LKFS) as i32,
        )?;
        decoder.set_param(ffi::MPEGH_DEC_PARAM_EFFECT_TYPE, self.drc_effect_type)?;
        decoder.set_param(
            ffi::MPEGH_DEC_PARAM_ATTENUATION_FACTOR,
            (self.drc_attenuation_factor * DRC_FACTOR_FULL_SCALE) as i32,
        )?;
        decoder.set_param(
            ffi::MPEGH_DEC_PARAM_BOOST_FACTOR,
            (self.drc_boost_factor * DRC_FACTOR_FULL_SCALE) as i32,
        )?;
        decoder.set_param(ffi::MPEGH_DEC_PARAM_ALBUM_MODE, i32::from(self.album_mode))
    }

    fn output_caps(&self) -> CapsSet {
        pcm_decoder_output(
            AudioFormat::PcmS32Le,
            cicp_channel_count(self.target_layout).expect("validated by set_property"),
            cicp_speaker_layout(self.target_layout)
                .map_or(ChannelLayout::UNSPECIFIED, |(layout, _)| layout),
        )
    }
}

// a box header followed by the `mhaC` fourcc, which no MPEG-H access unit starts with
fn is_mhac_box(data: &[u8]) -> bool {
    data.get(4..8) == Some(b"mhaC")
}

#[derive(Default)]
pub struct MpeghDec {
    settings: MpeghDecSettings,
    input_format: Option<AudioFormat>,
    decoder: Option<Decoder>,
    framer: MhasFramer,
    // the pts of the next access unit of a stream that carries none
    next_untimed_pts_ns: u64,
    output: PcmOutput,
}

impl core::fmt::Debug for MpeghDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MpeghDec")
            .field("settings", &self.settings)
            .field("input_format", &self.input_format)
            .field("emitted", &self.output.frames_pushed())
            .finish()
    }
}

impl MpeghDec {
    pub fn new() -> Self {
        Self::default()
    }

    fn opened_decoder(&mut self) -> Result<&mut Decoder, G2gError> {
        if self.decoder.is_none() {
            let mut decoder = Decoder::open(self.settings.target_layout)?;
            self.settings.apply(&mut decoder)?;
            self.decoder = Some(decoder);
        }
        Ok(self.decoder.as_mut().expect("opened above"))
    }

    fn decode_access_unit(
        &mut self,
        access_unit: MhasAccessUnit,
    ) -> Result<Vec<DecodedBlock>, G2gError> {
        let pts_ns = access_unit.pts_ns.unwrap_or(self.next_untimed_pts_ns);
        let decoder = self.opened_decoder()?;
        decoder.process(&access_unit.data, pts_ns)?;
        let blocks = decoder.take_samples()?;
        if let Some(last) = blocks.last() {
            self.next_untimed_pts_ns =
                last.pts_ns + ticks_to_ns(MPEGH_SAMPLES_PER_ACCESS_UNIT, last.sample_rate);
        }
        Ok(blocks)
    }

    fn decode_frame(
        &mut self,
        data: &[u8],
        pts_ns: Option<u64>,
    ) -> Result<Vec<DecodedBlock>, G2gError> {
        match self.input_format {
            Some(AudioFormat::MpeghRaw) if is_mhac_box(data) => {
                let record = mha_config_record(&data[8..]).ok_or(G2gError::CapsMismatch)?;
                self.opened_decoder()?.set_mha_config(record.config)?;
                Ok(Vec::new())
            }
            Some(AudioFormat::MpeghRaw) => {
                if self.decoder.is_none() {
                    // mha1 access units decode only after the mhaC that leads them
                    return Err(G2gError::NotConfigured);
                }
                self.decode_access_unit(MhasAccessUnit {
                    data: data.to_vec(),
                    pts_ns,
                })
            }
            Some(AudioFormat::MpeghMhas) => {
                let mut blocks = Vec::new();
                for access_unit in self.framer.push(data, pts_ns)? {
                    blocks.extend(self.decode_access_unit(access_unit)?);
                }
                Ok(blocks)
            }
            _ => Err(G2gError::NotConfigured),
        }
    }

    fn drain(&mut self) -> Result<Vec<DecodedBlock>, G2gError> {
        let mut blocks = Vec::new();
        if let Some(access_unit) = self.framer.finish() {
            blocks.extend(self.decode_access_unit(access_unit)?);
        }
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.flush_and_get()?;
            blocks.extend(decoder.take_samples()?);
        }
        Ok(blocks)
    }

    fn to_pcm(&self, block: DecodedBlock) -> Result<(FrameTiming, DecodedPcm), G2gError> {
        let channels = block.channels;
        if Some(channels) != cicp_channel_count(self.settings.target_layout) {
            return Err(G2gError::CapsMismatch);
        }
        let speakers = cicp_speaker_layout(self.settings.target_layout);
        let channel_layout = speakers.map_or(ChannelLayout::UNSPECIFIED, |(layout, _)| layout);
        let order: Vec<usize> = speakers.map_or_else(
            || (0..usize::from(channels)).collect(),
            |(_, order)| order.to_vec(),
        );
        let samples_per_channel = block.samples.len() / usize::from(channels);
        let mut pcm = Vec::with_capacity(block.samples.len() * 4);
        for frame in block.samples.chunks_exact(usize::from(channels)) {
            for &source in &order {
                pcm.extend_from_slice(&frame[source].to_le_bytes());
            }
        }
        let timing = FrameTiming {
            pts_ns: block.pts_ns,
            dts_ns: block.pts_ns,
            duration_ns: ticks_to_ns(samples_per_channel as u64, block.sample_rate),
            ..FrameTiming::default()
        };
        let decoded = DecodedPcm {
            pcm,
            format: AudioFormat::PcmS32Le,
            channels,
            sample_rate: block.sample_rate,
            channel_layout,
        };
        Ok((timing, decoded))
    }

    async fn push_blocks(
        &mut self,
        blocks: Vec<DecodedBlock>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for block in blocks {
            let (timing, decoded) = self.to_pcm(block)?;
            self.output.push(timing, decoded, out).await?;
        }
        Ok(())
    }

    fn set_decoder_param(&mut self, param: core::ffi::c_int, value: i32) -> Result<(), PropError> {
        match self.decoder.as_mut() {
            Some(decoder) => decoder
                .set_param(param, value)
                .map_err(|_| PropError::Value),
            None => Ok(()),
        }
    }
}

fn mpegh_input(caps: &Caps) -> Option<AudioFormat> {
    match caps {
        Caps::Audio {
            format: format @ (AudioFormat::MpeghMhas | AudioFormat::MpeghRaw),
            ..
        } => Some(*format),
        _ => None,
    }
}

impl AsyncElement for MpeghDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        mpegh_input(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let settings = self.settings;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match mpegh_input(input) {
            Some(_) => settings.output_caps(),
            None => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let format = mpegh_input(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        if self.input_format != Some(format) {
            self.decoder = None;
            self.framer.clear();
            self.input_format = Some(format);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG-H audio decoder",
            "Codec/Decoder/Audio",
            "Decodes MPEG-H 3D Audio to interleaved PcmS32Le with Fraunhofer's mpeghdec",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let data = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let blocks = self.decode_frame(data, frame.timing.pts())?;
                    self.push_blocks(blocks, out).await?;
                }
                PipelinePacket::Eos => {
                    let blocks = self.drain()?;
                    self.push_blocks(blocks, out).await?;
                }
                PipelinePacket::Flush => {
                    if let Some(decoder) = self.decoder.as_mut() {
                        decoder.flush()?;
                    }
                    self.framer.clear();
                    self.next_untimed_pts_ns = 0;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(caps) => match &caps {
                    // the demuxer's input refine, the decoder reads its config from the stream
                    _ if mpegh_input(&caps).is_some() => {}
                    Caps::Audio {
                        format: AudioFormat::PcmS32Le,
                        ..
                    } => self.output.forward_caps(caps, out).await?,
                    _ => return Err(G2gError::CapsMismatch),
                },
                PipelinePacket::Segment(segment) => {
                    self.output.forward_segment(segment, out).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MPEGHDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let settings = &mut self.settings;
        match name {
            // like gst, a new layout takes effect when the next stream opens the decoder
            "target-layout" => {
                let layout = value.as_int().ok_or(PropError::Type)?;
                let layout = u8::try_from(layout).map_err(|_| PropError::Value)?;
                cicp_channel_count(layout).ok_or(PropError::Value)?;
                settings.target_layout = layout;
            }
            "target-ref-level" => {
                let level = value.as_double().ok_or(PropError::Type)?;
                if !(-31.75..=-10.0).contains(&level) {
                    return Err(PropError::Value);
                }
                settings.target_reference_level = level;
                let steps = (level * LOUDNESS_STEPS_PER_LKFS) as i32;
                self.set_decoder_param(ffi::MPEGH_DEC_PARAM_TARGET_REFERENCE_LEVEL, steps)?;
            }
            "drc-effect-type" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                let (_, effect) = DRC_EFFECT_TYPES
                    .into_iter()
                    .find(|(known, _)| *known == nick)
                    .ok_or(PropError::Value)?;
                settings.drc_effect_type = effect;
                self.set_decoder_param(ffi::MPEGH_DEC_PARAM_EFFECT_TYPE, effect)?;
            }
            "drc-cut-level" | "drc-boost-level" => {
                let factor = value.as_double().ok_or(PropError::Type)?;
                if !(0.0..=1.0).contains(&factor) {
                    return Err(PropError::Value);
                }
                let param = if name == "drc-cut-level" {
                    settings.drc_attenuation_factor = factor;
                    ffi::MPEGH_DEC_PARAM_ATTENUATION_FACTOR
                } else {
                    settings.drc_boost_factor = factor;
                    ffi::MPEGH_DEC_PARAM_BOOST_FACTOR
                };
                self.set_decoder_param(param, (factor * DRC_FACTOR_FULL_SCALE) as i32)?;
            }
            "album-mode" => {
                let album_mode = value.as_bool().ok_or(PropError::Type)?;
                settings.album_mode = album_mode;
                self.set_decoder_param(ffi::MPEGH_DEC_PARAM_ALBUM_MODE, i32::from(album_mode))?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let settings = &self.settings;
        Some(match name {
            "target-layout" => PropValue::Int(i64::from(settings.target_layout)),
            "target-ref-level" => PropValue::Double(settings.target_reference_level),
            "drc-effect-type" => {
                let (nick, _) = DRC_EFFECT_TYPES
                    .into_iter()
                    .find(|(_, effect)| *effect == settings.drc_effect_type)
                    .expect("set from the table");
                PropValue::Str(nick.into())
            }
            "drc-cut-level" => PropValue::Double(settings.drc_attenuation_factor),
            "drc-boost-level" => PropValue::Double(settings.drc_boost_factor),
            "album-mode" => PropValue::Bool(settings.album_mode),
            _ => return None,
        })
    }
}

static MPEGHDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "target-layout",
        PropKind::Int,
        "CICP index of the speaker layout to render to, applied when the decoder opens",
    )
    .with_default("6")
    .with_range("1", "20"),
    PropertySpec::new(
        "target-ref-level",
        PropKind::Double,
        "loudness in LKFS the MPEG-D DRC normalizes the output to",
    )
    .with_default("-24")
    .with_range("-31.75", "-10"),
    PropertySpec::new(
        "drc-effect-type",
        PropKind::Str,
        "the MPEG-D DRC effect to request",
    )
    .with_default("general")
    .with_enum_values("off | none | night | noisy | limited | lowlevel | dialog | general"),
    PropertySpec::new(
        "drc-cut-level",
        PropKind::Double,
        "share of the stream's attenuating DRC gains to apply",
    )
    .with_default("1")
    .with_range("0", "1"),
    PropertySpec::new(
        "drc-boost-level",
        PropKind::Double,
        "share of the stream's boosting DRC gains to apply",
    )
    .with_default("1")
    .with_range("0", "1"),
    PropertySpec::new(
        "album-mode",
        PropKind::Bool,
        "use the stream's album loudness instead of its fades",
    )
    .with_default("false"),
];

impl PadTemplates for MpeghDec {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(alloc::vec![
                crate::typefind::elementary_audio_caps(AudioFormat::MpeghMhas),
                crate::typefind::elementary_audio_caps(AudioFormat::MpeghRaw),
            ])),
            PadTemplate::source(pcm_caps_any_rate_and_channels(|format| {
                format == AudioFormat::PcmS32Le
            })),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_target_layout_reorders_each_decoded_channel_once() {
        for index in 1..=20 {
            let Some((layout, order)) = cicp_speaker_layout(index) else {
                continue;
            };
            assert_eq!(
                Some(layout.channels()),
                cicp_channel_count(index),
                "CICP {index}"
            );
            let mut sorted = order.to_vec();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..usize::from(layout.channels())).collect::<Vec<_>>(),
                "CICP {index}"
            );
        }
    }
}
