use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;
use core::ptr::NonNull;

use alloc::boxed::Box;
use alloc::vec::Vec;

use objc2_audio_toolbox::{
    kAudioConverterEncodeBitRate, kAudioConverterPropertyMaximumOutputPacketSize,
    AudioConverterDispose, AudioConverterFillComplexBuffer, AudioConverterGetProperty,
    AudioConverterNew, AudioConverterRef, AudioConverterSetProperty,
};
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger,
    kAudioFormatLinearPCM, kAudioFormatMPEG4AAC, AudioBuffer, AudioBufferList,
    AudioStreamBasicDescription, AudioStreamPacketDescription,
};

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ChannelLayout, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::aacparse::adts_frame;

const DEFAULT_BITRATE_BPS: u32 = 128_000;
// AAC-LC codes 1024 samples per channel in each access unit
const SAMPLES_PER_ACCESS_UNIT: usize = 1024;
// ADTS `profile` for AAC-LC: the audio object type (2) minus one
const ADTS_PROFILE_AAC_LC: u8 = 1;
const MAX_CHANNELS: u8 = 2;
// returned from the input callback when it has handed over all buffered PCM, AudioConverter passes it back unchanged
const INPUT_EXHAUSTED: i32 = 1;

fn sample_rate_index(rate: u32) -> Option<u8> {
    crate::aacparse::SAMPLE_RATES
        .iter()
        .position(|&supported| supported == rate)
        .and_then(|index| u8::try_from(index).ok())
}

fn pcm_description(
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
) -> Result<AudioStreamBasicDescription, G2gError> {
    let (bits, flags) = match format {
        AudioFormat::PcmS16Le => (16u32, kAudioFormatFlagIsSignedInteger),
        AudioFormat::PcmF32Le => (32u32, kAudioFormatFlagIsFloat),
        _ => return Err(G2gError::CapsMismatch),
    };
    let bytes_per_frame = bits / 8 * u32::from(channels);
    Ok(AudioStreamBasicDescription {
        mSampleRate: f64::from(sample_rate),
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: flags | kAudioFormatFlagIsPacked,
        mBytesPerPacket: bytes_per_frame,
        mFramesPerPacket: 1,
        mBytesPerFrame: bytes_per_frame,
        mChannelsPerFrame: u32::from(channels),
        mBitsPerChannel: bits,
        mReserved: 0,
    })
}

fn check(status: i32) -> Result<(), G2gError> {
    if status == 0 {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Other))
    }
}

struct PendingInput {
    pcm: *const u8,
    frames: u32,
    bytes_per_frame: u32,
    channels: u32,
    end_of_stream: bool,
}

unsafe extern "C-unwind" fn supply_input(
    _converter: AudioConverterRef,
    packet_count: NonNull<u32>,
    buffers: NonNull<AudioBufferList>,
    _descriptions: *mut *mut AudioStreamPacketDescription,
    user_data: *mut c_void,
) -> i32 {
    // SAFETY: user_data is the PendingInput that encode_pending passes to AudioConverterFillComplexBuffer, alive for the call.
    let input = unsafe { &mut *user_data.cast::<PendingInput>() };
    let (packet_count, buffers) = (packet_count.as_ptr(), buffers.as_ptr());
    if input.frames == 0 {
        // SAFETY: AudioConverter passes a valid pointer to the packet count.
        unsafe { *packet_count = 0 };
        return if input.end_of_stream {
            0
        } else {
            INPUT_EXHAUSTED
        };
    }
    // SAFETY: AudioConverter passes valid pointers to the packet count and a one-buffer list, and the PCM stays in AtEnc::pending until FillComplexBuffer returns.
    unsafe {
        *packet_count = input.frames;
        (*buffers).mBuffers[0] = AudioBuffer {
            mNumberChannels: input.channels,
            mDataByteSize: input.frames * input.bytes_per_frame,
            mData: input.pcm.cast_mut().cast(),
        };
    }
    input.frames = 0;
    0
}

struct Converter(AudioConverterRef);

// SAFETY: an AudioConverter has no thread affinity, and the element that owns it is driven by one thread at a time.
unsafe impl Send for Converter {}

impl Drop for Converter {
    fn drop(&mut self) {
        // SAFETY: the converter came from AudioConverterNew and is disposed once.
        unsafe { AudioConverterDispose(self.0) };
    }
}

impl Converter {
    fn new(input: &AudioStreamBasicDescription, bitrate_bps: u32) -> Result<Self, G2gError> {
        let output = AudioStreamBasicDescription {
            mSampleRate: input.mSampleRate,
            mFormatID: kAudioFormatMPEG4AAC,
            mFormatFlags: 0,
            mBytesPerPacket: 0,
            mFramesPerPacket: SAMPLES_PER_ACCESS_UNIT as u32,
            mBytesPerFrame: 0,
            mChannelsPerFrame: input.mChannelsPerFrame,
            mBitsPerChannel: 0,
            mReserved: 0,
        };
        let mut converter: AudioConverterRef = core::ptr::null_mut();
        // SAFETY: both descriptions and the out pointer are valid for the call.
        check(unsafe {
            AudioConverterNew(
                NonNull::from(input),
                NonNull::from(&output),
                NonNull::from(&mut converter),
            )
        })?;
        let converter = Self(converter);
        let mut bitrate = bitrate_bps;
        // SAFETY: the bitrate property is one UInt32.
        check(unsafe {
            AudioConverterSetProperty(
                converter.0,
                kAudioConverterEncodeBitRate,
                core::mem::size_of::<u32>() as u32,
                NonNull::from(&mut bitrate).cast(),
            )
        })?;
        Ok(converter)
    }

    fn maximum_packet_size(&self) -> Result<usize, G2gError> {
        let mut size = 0u32;
        let mut property_size = core::mem::size_of::<u32>() as u32;
        // SAFETY: the maximum output packet size property is one UInt32.
        check(unsafe {
            AudioConverterGetProperty(
                self.0,
                kAudioConverterPropertyMaximumOutputPacketSize,
                NonNull::from(&mut property_size),
                NonNull::from(&mut size).cast(),
            )
        })?;
        usize::try_from(size).map_err(|_| G2gError::Hardware(HardwareError::Other))
    }

    // one access unit per call, None once AudioConverter needs more input than `input` holds
    fn next_access_unit(&self, input: &mut PendingInput) -> Result<Option<Vec<u8>>, G2gError> {
        let mut payload = alloc::vec![0u8; self.maximum_packet_size()?];
        let mut buffers = AudioBufferList {
            mNumberBuffers: 1,
            mBuffers: [AudioBuffer {
                mNumberChannels: input.channels,
                mDataByteSize: payload.len() as u32,
                mData: payload.as_mut_ptr().cast(),
            }],
        };
        let mut description = AudioStreamPacketDescription {
            mStartOffset: 0,
            mVariableFramesInPacket: 0,
            mDataByteSize: 0,
        };
        let mut packets = 1u32;
        // SAFETY: the output buffer, packet description and input state outlive the call.
        let status = unsafe {
            AudioConverterFillComplexBuffer(
                self.0,
                Some(supply_input),
                (input as *mut PendingInput).cast(),
                NonNull::from(&mut packets),
                NonNull::from(&mut buffers),
                &mut description,
            )
        };
        if status != 0 && status != INPUT_EXHAUSTED {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        if packets == 0 {
            return Ok(None);
        }
        payload.truncate(buffers.mBuffers[0].mDataByteSize as usize);
        Ok(Some(payload))
    }
}

pub struct AtEnc {
    bitrate_bps: u32,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    converter: Option<Converter>,
    pending: Vec<u8>,
    access_units_out: u64,
    emitted: u64,
    caps_sent: bool,
}

impl core::fmt::Debug for AtEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AtEnc")
            .field("bitrate_bps", &self.bitrate_bps)
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("access_units_out", &self.access_units_out)
            .finish_non_exhaustive()
    }
}

impl Default for AtEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl AtEnc {
    pub fn new() -> Self {
        Self {
            bitrate_bps: DEFAULT_BITRATE_BPS,
            format: AudioFormat::PcmS16Le,
            channels: 0,
            sample_rate: 0,
            converter: None,
            pending: Vec::new(),
            access_units_out: 0,
            emitted: 0,
            caps_sent: false,
        }
    }

    pub fn with_bitrate_bps(mut self, bitrate_bps: u32) -> Self {
        self.bitrate_bps = bitrate_bps;
        self
    }

    fn input_template() -> CapsSet {
        let pcm = |format| Caps::Audio {
            format,
            channels: 0,
            sample_rate: 0,
            channel_layout: ChannelLayout::UNSPECIFIED,
        };
        CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ]))
    }

    fn output_caps(channels: u8, sample_rate: u32) -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels,
            sample_rate,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    }

    fn bytes_per_frame(&self) -> usize {
        let sample_bytes = match self.format {
            AudioFormat::PcmF32Le => 4,
            _ => 2,
        };
        sample_bytes * usize::from(self.channels)
    }

    // hands AudioConverter the whole access units in `pending` and collects what it emits
    fn encode_pending(&mut self, end_of_stream: bool) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let bytes_per_frame = self.bytes_per_frame();
        let whole_frames = if end_of_stream {
            self.pending.len() / bytes_per_frame
        } else {
            self.pending.len() / bytes_per_frame / SAMPLES_PER_ACCESS_UNIT * SAMPLES_PER_ACCESS_UNIT
        };
        let converter = self.converter.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut input = PendingInput {
            pcm: self.pending.as_ptr(),
            frames: u32::try_from(whole_frames).map_err(|_| G2gError::CapsMismatch)?,
            bytes_per_frame: bytes_per_frame as u32,
            channels: u32::from(self.channels),
            end_of_stream,
        };
        let rate_index = sample_rate_index(self.sample_rate).ok_or(G2gError::CapsMismatch)?;
        let mut access_units = Vec::new();
        while let Some(payload) = converter.next_access_unit(&mut input)? {
            let access_unit = adts_frame(ADTS_PROFILE_AAC_LC, rate_index, self.channels, &payload)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let pts_ns = self.access_units_out * SAMPLES_PER_ACCESS_UNIT as u64 * 1_000_000_000
                / u64::from(self.sample_rate);
            self.access_units_out += 1;
            access_units.push((access_unit, pts_ns));
        }
        self.pending.drain(..whole_frames * bytes_per_frame);
        Ok(access_units)
    }

    async fn emit(
        &mut self,
        access_units: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = Self::output_caps(self.channels, self.sample_rate);
        crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            access_units,
            &caps,
            out,
        )
        .await?;
        Ok(())
    }

    fn open(&mut self) -> Result<(), G2gError> {
        let input = pcm_description(self.format, self.channels, self.sample_rate)?;
        self.converter = Some(Converter::new(&input, self.bitrate_bps)?);
        self.pending.clear();
        self.access_units_out = 0;
        Ok(())
    }
}

impl AsyncElement for AtEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::input_template()
            .alternatives()
            .iter()
            .find_map(|template| upstream_caps.intersect(template).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le | AudioFormat::PcmF32Le,
                channels,
                sample_rate,
                ..
            } => CapsSet::one(Self::output_caps(*channels, *sample_rate)),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::Audio {
            format: format @ (AudioFormat::PcmS16Le | AudioFormat::PcmF32Le),
            channels: channels @ 1..=MAX_CHANNELS,
            sample_rate,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        sample_rate_index(*sample_rate).ok_or(G2gError::CapsMismatch)?;
        self.format = *format;
        self.channels = *channels;
        self.sample_rate = *sample_rate;
        self.caps_sent = false;
        self.open()?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "AudioToolbox AAC encoder",
            "Codec/Encoder/Audio",
            "Encodes PCM to ADTS-framed AAC-LC with macOS AudioToolbox",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] =
            &[
                PropertySpec::new("bitrate", PropKind::Uint, "target bitrate, bits/second")
                    .with_default("128000"),
            ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                let bitrate = value.as_uint().ok_or(PropError::Type)?;
                self.bitrate_bps = u32::try_from(bitrate)
                    .ok()
                    .filter(|&bitrate| bitrate > 0)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(u64::from(self.bitrate_bps))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.converter.is_none() {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let pcm = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    if !pcm.len().is_multiple_of(self.bytes_per_frame()) {
                        return Err(G2gError::CapsMismatch);
                    }
                    self.pending.extend_from_slice(pcm);
                    let access_units = self.encode_pending(false)?;
                    self.emit(access_units, out).await?;
                }
                PipelinePacket::Eos => {
                    let access_units = self.encode_pending(true)?;
                    self.emit(access_units, out).await?;
                }
                PipelinePacket::Flush => {
                    self.open()?;
                    out.push(PipelinePacket::Flush).await?;
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

impl PadTemplates for AtEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(Self::input_template()),
            PadTemplate::source(CapsSet::one(crate::typefind::elementary_audio_caps(
                AudioFormat::Aac,
            ))),
        ])
    }
}
