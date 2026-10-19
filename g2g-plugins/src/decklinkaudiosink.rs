use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    g2g_error, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ChannelLayout, ClockSync,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PresentationPacer, PropError, PropKind, PropValue, PropertySpec,
};

use crate::decklink::{log_target, NANOS_PER_SECOND};
use crate::decklinkoutput::{find_output, hold_until_due, sleep_ns, SharedOutput};

const ELEMENT_NAME: &str = "decklinkaudiosink";

const SAMPLE_RATE: u32 = 48_000;
const FORMATS: [AudioFormat; 2] = [AudioFormat::PcmS16Le, AudioFormat::PcmS32Le];
const CHANNEL_COUNTS: [u8; 3] = [2, 8, 16];

// how long to wait before offering samples again to a board whose audio buffer is full
const FULL_BUFFER_RETRY_NS: u64 = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoundLayout {
    format: AudioFormat,
    channels: u8,
    bytes_per_sample_frame: usize,
}

#[derive(Debug)]
pub struct DeckLinkAudioSink {
    device_number: u32,
    output: Option<Arc<SharedOutput>>,
    layout: Option<SoundLayout>,
    pacer: PresentationPacer,
}

impl Default for DeckLinkAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

fn pcm_caps(format: AudioFormat, channels: u8) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate: SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

fn accepted_caps() -> CapsSet {
    CapsSet::from_alternatives(
        FORMATS
            .iter()
            .flat_map(|format| CHANNEL_COUNTS.map(|channels| pcm_caps(*format, channels)))
            .collect(),
    )
}

fn layout_for(caps: &Caps) -> Result<SoundLayout, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
        ..
    } = *caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let supported = FORMATS.contains(&format)
        && CHANNEL_COUNTS.contains(&channels)
        && sample_rate == SAMPLE_RATE;
    if !supported {
        return Err(G2gError::CapsMismatch);
    }
    let bytes_per_sample = format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?;
    Ok(SoundLayout {
        format,
        channels,
        bytes_per_sample_frame: bytes_per_sample * usize::from(channels),
    })
}

impl DeckLinkAudioSink {
    pub fn new() -> Self {
        Self {
            device_number: 0,
            output: None,
            layout: None,
            pacer: PresentationPacer::new(),
        }
    }

    async fn schedule(&mut self, frame: Frame) -> Result<(), G2gError> {
        let layout = self.layout.ok_or(G2gError::NotConfigured)?;
        let output = self.output.as_ref().ok_or(G2gError::NotConfigured)?;
        let sync: ClockSync = self
            .pacer
            .clock_sync()
            .cloned()
            .ok_or(G2gError::NotConfigured)?;
        let samples = frame.domain.require_system_slice(ELEMENT_NAME)?;
        if samples.len() % layout.bytes_per_sample_frame != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let sample_frames = samples.len() / layout.bytes_per_sample_frame;
        let Some(deadline_ns) = output.deadline_ns(&sync, &self.pacer, frame.timing.pts_ns)? else {
            return Ok(());
        };
        hold_until_due(&sync, deadline_ns).await;

        let mut scheduled = 0;
        while scheduled < sample_frames {
            let remaining = &samples[scheduled * layout.bytes_per_sample_frame..];
            let offset_ns = scheduled as u64 * NANOS_PER_SECOND / u64::from(SAMPLE_RATE);
            let count =
                u32::try_from(sample_frames - scheduled).map_err(|_| G2gError::CapsMismatch)?;
            let written =
                output.schedule_audio(&sync, deadline_ns + offset_ns, remaining, count)?;
            if written == 0 {
                sleep_ns(FULL_BUFFER_RETRY_NS).await;
            }
            scheduled += written as usize;
        }
        Ok(())
    }
}

impl AsyncElement for DeckLinkAudioSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        layout_for(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let layout = layout_for(absolute_caps).inspect_err(|_| {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} takes S16LE or S32LE at 48 kHz in 2, 8 or 16 channels, got {absolute_caps:?}"
            );
        })?;
        if let Some(configured) = self.layout {
            if configured == layout {
                return Ok(ConfigureOutcome::Accepted);
            }
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} cannot change format while the output runs"
            );
            return Err(G2gError::CapsMismatch);
        }
        // the board schedules audio against the video frames, so a video sink must own the device
        let output = find_output(self.device_number).ok_or_else(|| {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} needs a decklinkvideosink with device-number {}",
                self.device_number
            );
            G2gError::NotConfigured
        })?;
        output.enable_audio(layout.format, layout.channels)?;
        self.output = Some(output);
        self.layout = Some(layout);
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SDI audio sink",
            "Sink/Audio/Hardware",
            "Plays interleaved PCM out of an SDI output for Blackmagic Design DeckLink",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "device-number",
            PropKind::Int,
            "Output device index, the same as its decklinkvideosink",
        )
        .with_default("0")
        .with_range("0", "2147483647")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device-number" => {
                let device_number = value
                    .as_int()
                    .ok_or(PropError::Type)
                    .and_then(|n| u32::try_from(n).map_err(|_| PropError::Value))?;
                if self.layout.is_some() {
                    return Err(PropError::Value);
                }
                self.device_number = device_number;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device-number" => Some(PropValue::Int(i64::from(self.device_number))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.schedule(frame).await,
                PipelinePacket::CapsChanged(caps) => {
                    self.configure_pipeline(&caps)?;
                    Ok(())
                }
                PipelinePacket::Segment(segment) => {
                    self.pacer.set_segment(segment);
                    Ok(())
                }
                // the video sink stops the shared output on a flush and plays out on end of stream
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for DeckLinkAudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}
