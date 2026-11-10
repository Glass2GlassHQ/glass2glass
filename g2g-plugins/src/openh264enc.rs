use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::log::short_type_name;
use g2g_core::{
    AsyncElement, BitstreamAlignment, Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome,
    Dim, ElementMetadata, FrameTiming, G2gError, HardwareError, Interlace, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PixelAspect, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat, VideoCodec,
};

use openh264::encoder::{
    BitRate, Complexity, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, QpRange,
    RateControlMode, UsageType,
};
use openh264::formats::YUVSlices;
use openh264::{OpenH264API, Timestamp};

// gst openh264enc defaults
const DEFAULT_BITRATE_BPS: u32 = 128_000;
const DEFAULT_GOP_SIZE: u32 = 90;
const MAX_QP: u8 = 51;

const RATE_CONTROL_NAMES: &[(&str, RateControlMode)] = &[
    ("quality", RateControlMode::Quality),
    ("bitrate", RateControlMode::Bitrate),
    ("buffer", RateControlMode::Bufferbased),
    ("off", RateControlMode::Off),
];
const COMPLEXITY_NAMES: &[(&str, Complexity)] = &[
    ("low", Complexity::Low),
    ("medium", Complexity::Medium),
    ("high", Complexity::High),
];
const USAGE_TYPE_NAMES: &[(&str, UsageType)] = &[
    ("camera", UsageType::CameraVideoRealTime),
    ("screen", UsageType::ScreenContentRealTime),
];

fn named_value<T: Copy>(table: &[(&'static str, T)], name: &str) -> Option<(&'static str, T)> {
    table.iter().find(|(entry, _)| *entry == name).copied()
}

fn property_name<T: Copy>(
    table: &[(&'static str, T)],
    value: PropValue,
) -> Result<&'static str, PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    named_value(table, name)
        .map(|(name, _)| name)
        .ok_or(PropError::Value)
}

fn table_value<T: Copy>(table: &[(&'static str, T)], name: &str) -> T {
    named_value(table, name)
        .map(|(_, value)| value)
        .expect("settings hold only names from the table")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Settings {
    bitrate_bps: u32,
    gop_size: u32,
    rate_control: &'static str,
    complexity: &'static str,
    usage_type: &'static str,
    qp_min: u8,
    qp_max: u8,
    threads: u16,
    slices: u32,
    frame_skip: bool,
    scene_change_detection: bool,
    adaptive_quantization: bool,
    background_detection: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            bitrate_bps: DEFAULT_BITRATE_BPS,
            gop_size: DEFAULT_GOP_SIZE,
            rate_control: "quality",
            complexity: "medium",
            usage_type: "camera",
            qp_min: 0,
            qp_max: MAX_QP,
            threads: 0,
            slices: 1,
            frame_skip: false,
            scene_change_detection: true,
            adaptive_quantization: true,
            background_detection: true,
        }
    }
}

fn open_encoder(settings: &Settings, framerate: &Rate) -> Result<Encoder, G2gError> {
    let mut config = EncoderConfig::new()
        .bitrate(BitRate::from_bps(settings.bitrate_bps))
        .intra_frame_period(IntraFramePeriod::from_num_frames(settings.gop_size))
        .rate_control_mode(table_value(RATE_CONTROL_NAMES, settings.rate_control))
        .complexity(table_value(COMPLEXITY_NAMES, settings.complexity))
        .usage_type(table_value(USAGE_TYPE_NAMES, settings.usage_type))
        .qp(QpRange::new(settings.qp_min, settings.qp_max))
        .num_threads(settings.threads)
        .skip_frames(settings.frame_skip)
        .scene_change_detect(settings.scene_change_detection)
        .adaptive_quantization(settings.adaptive_quantization)
        .background_detection(settings.background_detection);
    if let Rate::Fixed(q16) = framerate {
        config = config.max_frame_rate(FrameRate::from_hz(*q16 as f32 / (1u32 << 16) as f32));
    }
    let mut encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
    // openh264 prints warnings such as its quality-mode frame-skip note to stderr while it initialises
    let mut trace_level = openh264_sys2::WELS_LOG_ERROR as core::ffi::c_int;
    // SAFETY: the trace level option reads one int through the pointer and openh264 accepts it before initialisation.
    let status = unsafe {
        encoder.raw_api().set_option(
            openh264_sys2::ENCODER_OPTION_TRACE_LEVEL,
            (&raw mut trace_level).cast(),
        )
    };
    if status != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(encoder)
}

fn set_slice_count(encoder: &mut Encoder, settings: &Settings) -> Result<(), G2gError> {
    let mut params = openh264_sys2::SEncParamExt::default();
    // SAFETY: OpenH264 copies its current parameters into the struct the pointer names.
    let status = unsafe {
        encoder.raw_api().get_option(
            openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
            (&raw mut params).cast(),
        )
    };
    if status != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let slicing = &mut params.sSpatialLayers[0].sSliceArgument;
    slicing.uiSliceMode = openh264_sys2::SM_FIXEDSLCNUM_SLICE;
    slicing.uiSliceNum = settings.slices;
    // OpenH264 cut the thread count to one for the single slice it started with
    params.iMultipleThreadIdc = settings.threads;
    // SAFETY: OpenH264 reads the parameters through the pointer and re-initialises with them.
    let status = unsafe {
        encoder.raw_api().set_option(
            openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
            (&raw mut params).cast(),
        )
    };
    if status != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(())
}

pub struct OpenH264Enc {
    settings: Settings,
    width: u32,
    height: u32,
    framerate: Rate,
    // carried on the output caps only, the openh264 crate sets no VUI sample aspect
    pixel_aspect: PixelAspect,
    encoder: Option<Encoder>,
    slice_count_pending: bool,
    emitted: u64,
    caps_sent: bool,
}

impl core::fmt::Debug for OpenH264Enc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpenH264Enc")
            .field("settings", &self.settings)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl Default for OpenH264Enc {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenH264Enc {
    pub fn new() -> Self {
        Self {
            settings: Settings::default(),
            width: 0,
            height: 0,
            framerate: Rate::Any,
            pixel_aspect: PixelAspect::Any,
            encoder: None,
            slice_count_pending: false,
            emitted: 0,
            caps_sent: false,
        }
    }

    pub fn with_bitrate_bps(mut self, bitrate_bps: u32) -> Self {
        self.settings.bitrate_bps = bitrate_bps;
        self
    }

    fn input_template() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    fn output_caps(width: Dim, height: Dim, framerate: Rate, pixel_aspect: PixelAspect) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height,
            framerate,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect,
            alignment: BitstreamAlignment::AccessUnit,
        }
    }

    // settings fix at initialisation, so a change takes a new encoder that starts on an IDR
    fn reopen(&mut self) -> Result<(), G2gError> {
        if self.encoder.is_some() {
            self.open()?;
        }
        Ok(())
    }

    fn open(&mut self) -> Result<(), G2gError> {
        self.encoder = Some(open_encoder(&self.settings, &self.framerate)?);
        self.slice_count_pending = self.settings.slices > 1;
        Ok(())
    }

    fn set_setting(&mut self, change: impl FnOnce(&mut Settings)) -> Result<(), PropError> {
        let previous = self.settings;
        change(&mut self.settings);
        if self.settings.qp_min > self.settings.qp_max {
            self.settings = previous;
            return Err(PropError::Value);
        }
        if self.settings != previous {
            self.reopen().map_err(|_| PropError::Value)?;
        }
        Ok(())
    }

    fn encode(&mut self, i420: &[u8], pts_ns: u64) -> Result<Option<Vec<u8>>, G2gError> {
        let format = RawVideoFormat::I420;
        let plane = |index: usize| -> Result<&[u8], G2gError> {
            let offset = format
                .plane_offset(index, self.width, self.height)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or(G2gError::CapsMismatch)?;
            let stride = format
                .plane_stride(index, self.width)
                .ok_or(G2gError::CapsMismatch)?;
            let rows = format
                .plane_rows(index, self.height)
                .ok_or(G2gError::CapsMismatch)?;
            let length = usize::try_from(u64::from(stride) * u64::from(rows))
                .map_err(|_| G2gError::CapsMismatch)?;
            i420.get(offset..offset + length)
                .ok_or(G2gError::CapsMismatch)
        };
        let stride = |index: usize| -> Result<usize, G2gError> {
            format
                .plane_stride(index, self.width)
                .and_then(|stride| usize::try_from(stride).ok())
                .ok_or(G2gError::CapsMismatch)
        };
        let source = YUVSlices::new(
            (plane(0)?, plane(1)?, plane(2)?),
            (self.width as usize, self.height as usize),
            (stride(0)?, stride(1)?, stride(2)?),
        );
        let timestamp = if pts_ns == FrameTiming::PTS_NONE {
            Timestamp::ZERO
        } else {
            Timestamp::from_millis(pts_ns / 1_000_000)
        };
        let encoder = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        let access_unit = encoder
            .encode_at(&source, timestamp)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .to_vec();
        // openh264's Encoder initialises OpenH264 with one slice on its first encode
        if self.slice_count_pending {
            set_slice_count(encoder, &self.settings)?;
            self.slice_count_pending = false;
        }
        Ok((!access_unit.is_empty()).then_some(access_unit))
    }

    async fn emit(
        &mut self,
        access_unit: Vec<u8>,
        pts_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = Self::output_caps(
            Dim::Fixed(self.width),
            Dim::Fixed(self.height),
            self.framerate.clone(),
            self.pixel_aspect,
        );
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            Vec::from([(access_unit, pts_ns)]),
            &caps,
            out,
        )
        .await?;
        if let Some(bitrate_bps) = feedback.bitrate_bps.filter(|&bps| {
            crate::encoder_base::bitrate_change_is_significant(
                u64::from(self.settings.bitrate_bps),
                u64::from(bps),
            )
        }) {
            self.settings.bitrate_bps = bitrate_bps.max(1);
            self.reopen()?;
        } else if feedback.force_keyframe {
            if let Some(encoder) = self.encoder.as_mut() {
                encoder.force_intra_frame();
            }
        }
        Ok(())
    }
}

impl AsyncElement for OpenH264Enc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    fn handles_keyframe_requests(&self) -> bool {
        true
    }

    fn handles_bitrate_requests(&self) -> bool {
        true
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width,
                height,
                framerate,
                pixel_aspect,
                ..
            } => CapsSet::one(Self::output_caps(
                width.clone(),
                height.clone(),
                framerate.clone(),
                *pixel_aspect,
            )),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            pixel_aspect,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.width = *width;
        self.height = *height;
        self.framerate = framerate.clone();
        self.pixel_aspect = *pixel_aspect;
        self.caps_sent = false;
        self.open()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "OpenH264 video encoder",
            "Codec/Encoder/Video",
            "Encodes I420 to H.264 with OpenH264",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("bitrate", PropKind::Uint, "target bitrate, bits/second")
                .with_default("128000"),
            PropertySpec::new("gop-size", PropKind::Uint, "frames between intra frames")
                .with_default("90"),
            PropertySpec::new(
                "rate-control",
                PropKind::Str,
                "rate control: quality | bitrate | buffer | off",
            )
            .with_default("quality"),
            PropertySpec::new(
                "complexity",
                PropKind::Str,
                "complexity: low | medium | high",
            )
            .with_default("medium"),
            PropertySpec::new("usage-type", PropKind::Str, "content type: camera | screen")
                .with_default("camera"),
            PropertySpec::new("qp-min", PropKind::Uint, "minimum quantizer, 0..51")
                .with_default("0"),
            PropertySpec::new("qp-max", PropKind::Uint, "maximum quantizer, 0..51")
                .with_default("51"),
            PropertySpec::new("multi-thread", PropKind::Uint, "encoder threads, 0 = auto")
                .with_default("0"),
            PropertySpec::new(
                "num-slices",
                PropKind::Uint,
                "slices per frame, encoded on separate threads from the second frame on",
            )
            .with_default("1"),
            PropertySpec::new(
                "enable-frame-skip",
                PropKind::Bool,
                "skip frames to reach the target bitrate",
            )
            .with_default("false"),
            PropertySpec::new(
                "scene-change-detection",
                PropKind::Bool,
                "start an IDR on a scene change",
            )
            .with_default("true"),
            PropertySpec::new(
                "adaptive-quantization",
                PropKind::Bool,
                "adaptive quantization",
            )
            .with_default("true"),
            PropertySpec::new(
                "background-detection",
                PropKind::Bool,
                "background detection",
            )
            .with_default("true"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let uint = |value: &PropValue| value.as_uint().ok_or(PropError::Type);
        let flag = |value: &PropValue| value.as_bool().ok_or(PropError::Type);
        let qp = |value: &PropValue| {
            u8::try_from(uint(value)?)
                .ok()
                .filter(|&qp| qp <= MAX_QP)
                .ok_or(PropError::Value)
        };
        match name {
            "bitrate" => {
                let bitrate = u32::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
                self.set_setting(|settings| settings.bitrate_bps = bitrate)
            }
            "gop-size" => {
                let gop_size = u32::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
                self.set_setting(|settings| settings.gop_size = gop_size)
            }
            "rate-control" => {
                let mode = property_name(RATE_CONTROL_NAMES, value)?;
                self.set_setting(|settings| settings.rate_control = mode)
            }
            "complexity" => {
                let complexity = property_name(COMPLEXITY_NAMES, value)?;
                self.set_setting(|settings| settings.complexity = complexity)
            }
            "usage-type" => {
                let usage_type = property_name(USAGE_TYPE_NAMES, value)?;
                self.set_setting(|settings| settings.usage_type = usage_type)
            }
            "qp-min" => {
                let qp_min = qp(&value)?;
                self.set_setting(|settings| settings.qp_min = qp_min)
            }
            "qp-max" => {
                let qp_max = qp(&value)?;
                self.set_setting(|settings| settings.qp_max = qp_max)
            }
            "multi-thread" => {
                let threads = u16::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
                self.set_setting(|settings| settings.threads = threads)
            }
            "num-slices" => {
                let slices = u32::try_from(uint(&value)?)
                    .ok()
                    .filter(|&slices| slices > 0)
                    .ok_or(PropError::Value)?;
                self.set_setting(|settings| settings.slices = slices)
            }
            "enable-frame-skip" => {
                let enabled = flag(&value)?;
                self.set_setting(|settings| settings.frame_skip = enabled)
            }
            "scene-change-detection" => {
                let enabled = flag(&value)?;
                self.set_setting(|settings| settings.scene_change_detection = enabled)
            }
            "adaptive-quantization" => {
                let enabled = flag(&value)?;
                self.set_setting(|settings| settings.adaptive_quantization = enabled)
            }
            "background-detection" => {
                let enabled = flag(&value)?;
                self.set_setting(|settings| settings.background_detection = enabled)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let settings = &self.settings;
        Some(match name {
            "bitrate" => PropValue::Uint(u64::from(settings.bitrate_bps)),
            "gop-size" => PropValue::Uint(u64::from(settings.gop_size)),
            "rate-control" => PropValue::Str(settings.rate_control.into()),
            "complexity" => PropValue::Str(settings.complexity.into()),
            "usage-type" => PropValue::Str(settings.usage_type.into()),
            "qp-min" => PropValue::Uint(u64::from(settings.qp_min)),
            "qp-max" => PropValue::Uint(u64::from(settings.qp_max)),
            "multi-thread" => PropValue::Uint(u64::from(settings.threads)),
            "num-slices" => PropValue::Uint(u64::from(settings.slices)),
            "enable-frame-skip" => PropValue::Bool(settings.frame_skip),
            "scene-change-detection" => PropValue::Bool(settings.scene_change_detection),
            "adaptive-quantization" => PropValue::Bool(settings.adaptive_quantization),
            "background-detection" => PropValue::Bool(settings.background_detection),
            _ => return None,
        })
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.encoder.is_none() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let i420 = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let pts_ns = frame.timing.pts_ns;
                    if let Some(access_unit) = self.encode(i420, pts_ns)? {
                        self.emit(access_unit, pts_ns, out).await?;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for OpenH264Enc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(Self::output_caps(
                Dim::Any,
                Dim::Any,
                Rate::Any,
                PixelAspect::Any,
            ))),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openh264dec::OpenH264Dec;
    use crate::testutil::run_packets;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{PushOutcome, Reconfigure};

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 48;
    const FRAME_COUNT: u64 = 10;
    const FRAME_PERIOD_NS: u64 = 33_333_333;
    // the worst frame of this gradient measures a mean absolute luma error under 1 at the default settings
    const MAX_MEAN_LUMA_ERROR: f64 = 2.0;

    fn i420_caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    fn gradient(index: u64) -> Vec<u8> {
        let luma_size = (WIDTH * HEIGHT) as usize;
        let chroma_size = luma_size / 4;
        let mut pixels = Vec::with_capacity(luma_size + 2 * chroma_size);
        for row in 0..HEIGHT {
            for column in 0..WIDTH {
                pixels.push((column * 3 + row + index as u32 * 4) as u8);
            }
        }
        pixels.extend(core::iter::repeat_n(128u8, 2 * chroma_size));
        pixels
    }

    fn raw_frame(index: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(gradient(index).into_boxed_slice())),
            FrameTiming {
                pts_ns: index * FRAME_PERIOD_NS,
                ..FrameTiming::default()
            },
            index,
        ))
    }

    fn frame_payloads(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
        packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => {
                    frame.domain.as_system_slice().map(<[u8]>::to_vec)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn openh264dec_decodes_what_it_encodes() {
        let encoded = run_packets(
            &mut OpenH264Enc::new(),
            &i420_caps(),
            (0..FRAME_COUNT).map(raw_frame),
        );
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: BitstreamAlignment::AccessUnit,
        };
        let access_units: Vec<PipelinePacket> = encoded
            .packets
            .into_iter()
            .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
            .collect();
        assert_eq!(access_units.len() as u64, FRAME_COUNT);
        let decoded = run_packets(&mut OpenH264Dec::new(), &h264, access_units);
        let pictures = frame_payloads(&decoded.packets);
        assert_eq!(pictures.len() as u64, FRAME_COUNT);
        let luma_size = (WIDTH * HEIGHT) as usize;
        for (index, picture) in pictures.iter().enumerate() {
            let source = gradient(index as u64);
            let total_error: u64 = picture[..luma_size]
                .iter()
                .zip(&source[..luma_size])
                .map(|(decoded, source)| u64::from(decoded.abs_diff(*source)))
                .sum();
            let mean_error = total_error as f64 / luma_size as f64;
            assert!(
                mean_error <= MAX_MEAN_LUMA_ERROR,
                "frame {index}: mean luma error {mean_error}"
            );
        }
    }

    struct KeyframeRequestSink {
        access_units: Vec<Vec<u8>>,
        requested: bool,
    }

    impl OutputSink for KeyframeRequestSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            if let PipelinePacket::DataFrame(frame) = packet {
                self.access_units.push(
                    frame
                        .domain
                        .as_system_slice()
                        .expect("system memory")
                        .to_vec(),
                );
            }
            let outcome =
                if self.access_units.len() == 3 && !core::mem::replace(&mut self.requested, true) {
                    PushOutcome::Reconfigure(Reconfigure::ForceKeyframe)
                } else {
                    PushOutcome::Accepted
                };
            core::task::Poll::Ready(Ok(outcome))
        }
    }

    #[tokio::test]
    async fn a_keyframe_request_makes_the_next_access_unit_an_idr() {
        let mut encoder = OpenH264Enc::new();
        encoder
            .set_property("scene-change-detection", PropValue::Bool(false))
            .unwrap();
        encoder.configure_pipeline(&i420_caps()).unwrap();
        let mut sink = KeyframeRequestSink {
            access_units: Vec::new(),
            requested: false,
        };
        for index in 0..FRAME_COUNT {
            encoder.process(raw_frame(index), &mut sink).await.unwrap();
        }
        let keyframes: Vec<usize> = sink
            .access_units
            .iter()
            .enumerate()
            .filter(|(_, access_unit)| crate::annexb::au_is_keyframe(VideoCodec::H264, access_unit))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(keyframes, [0, 3]);
    }

    #[test]
    fn each_frame_after_the_first_has_the_set_number_of_slices() {
        const SLICES: usize = 4;
        // openh264 keeps one slice in a frame of 48 macroblocks or fewer
        const SLICED_WIDTH: u32 = 176;
        const SLICED_HEIGHT: u32 = 144;
        let caps = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(SLICED_WIDTH),
            height: Dim::Fixed(SLICED_HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        let frame_bytes = (SLICED_WIDTH * SLICED_HEIGHT * 3 / 2) as usize;
        let frames = (0..FRAME_COUNT).map(|index| {
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    alloc::vec![(index * 20) as u8; frame_bytes].into_boxed_slice(),
                )),
                FrameTiming {
                    pts_ns: index * FRAME_PERIOD_NS,
                    ..FrameTiming::default()
                },
                index,
            ))
        });
        let mut encoder = OpenH264Enc::new();
        encoder
            .set_property("num-slices", PropValue::Uint(SLICES as u64))
            .unwrap();
        let encoded = run_packets(&mut encoder, &caps, frames);
        let slice_counts: Vec<usize> = frame_payloads(&encoded.packets)
            .iter()
            .map(|access_unit| {
                crate::annexb::nal_units(access_unit)
                    .filter(|nal| matches!(crate::annexb::h264_nal_type(nal), Some(1 | 5)))
                    .count()
            })
            .collect();
        assert_eq!(slice_counts.len() as u64, FRAME_COUNT);
        assert!(
            slice_counts[1..].iter().all(|&count| count == SLICES),
            "slices per frame: {slice_counts:?}"
        );
        let mut params = openh264_sys2::SEncParamExt::default();
        let openh264 = encoder.encoder.as_mut().unwrap();
        // SAFETY: OpenH264 copies its current parameters into the struct the pointer names.
        let status = unsafe {
            openh264.raw_api().get_option(
                openh264_sys2::ENCODER_OPTION_SVC_ENCODE_PARAM_EXT,
                (&raw mut params).cast(),
            )
        };
        assert_eq!(status, 0);
        assert!(
            params.iMultipleThreadIdc > 1,
            "encoder threads: {}",
            params.iMultipleThreadIdc
        );
    }

    #[test]
    fn a_minimum_quantizer_above_the_maximum_is_refused() {
        let mut encoder = OpenH264Enc::new();
        encoder.set_property("qp-max", PropValue::Uint(30)).unwrap();
        assert_eq!(
            encoder.set_property("qp-min", PropValue::Uint(31)),
            Err(PropError::Value)
        );
        assert_eq!(encoder.get_property("qp-min"), Some(PropValue::Uint(0)));
    }
}
