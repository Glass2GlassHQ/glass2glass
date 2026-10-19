use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, Colorimetry, ConfigureOutcome, Dim, DriftClock, ElementMetadata, G2gError,
    Interlace, OutputSink, PadTemplate, PadTemplates, PipelineClock, PipelinePacket,
    PresentationPacer, PropError, PropKind, PropValue, PropertySpec, Rate,
};

use crate::decklink::{
    display_mode, log_target, mode_for_caps, sdk_opener, video_format_supported, DeckLinkOpener,
    DisplayMode, DISPLAY_MODES, PIXEL_FORMATS,
};
use crate::decklinkoutput::{claim_output, hold_until_due, sleep_ns, PictureLayout, SharedOutput};

const ELEMENT_NAME: &str = "decklinkvideosink";

const AUTO_MODE: &str = "auto";

// "auto" then every name in DISPLAY_MODES, in order
const MODE_VALUES: &str = "auto | ntsc | ntsc2398 | pal | ntsc-p | pal-p | 1080p2398 | 1080p24 | \
1080p25 | 1080p2997 | 1080p30 | 1080i50 | 1080i5994 | 1080i60 | 1080p50 | 1080p5994 | 1080p60 | \
720p50 | 720p5994 | 720p60 | 1556p2398 | 1556p24 | 1556p25 | 2kdcip2398 | 2kdcip24 | 2kdcip25 | \
2kdcip2997 | 2kdcip30 | 2kdcip50 | 2kdcip5994 | 2kdcip60 | 2160p2398 | 2160p24 | 2160p25 | \
2160p2997 | 2160p30 | 2160p50 | 2160p5994 | 2160p60 | 4kdcip2398 | 4kdcip24 | 4kdcip25 | \
4kdcip2997 | 4kdcip30 | 4kdcip50 | 4kdcip5994 | 4kdcip60 | 8kp2398 | 8kp24 | 8kp25 | 8kp2997 | \
8kp30 | 8kp50 | 8kp5994 | 8kp60 | 8kdcip2398 | 8kdcip24 | 8kdcip25 | 8kdcip2997 | 8kdcip30 | \
8kdcip50 | 8kdcip5994 | 8kdcip60 | 640x480p60 | 800x600p60 | 1440x900p50 | 1440x900p60 | \
1440x1080p50 | 1440x1080p60 | 1600x1200p50 | 1600x1200p60 | 1920x1200p50 | 1920x1200p60 | \
1920x1440p50 | 1920x1440p60 | 2560x1440p50 | 2560x1440p60 | 2560x1600p50 | 2560x1600p60";

pub struct DeckLinkVideoSink {
    device_number: u32,
    // None picks the mode from the negotiated caps
    mode: Option<&'static DisplayMode>,
    opener: DeckLinkOpener,
    output: Arc<SharedOutput>,
    layout: Option<PictureLayout>,
    pacer: PresentationPacer,
    last_frame_end_ns: Option<u64>,
}

impl core::fmt::Debug for DeckLinkVideoSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeckLinkVideoSink")
            .field("device_number", &self.device_number)
            .field("mode", &self.mode.map(|mode| mode.name))
            .field("layout", &self.layout)
            .finish()
    }
}

impl Default for DeckLinkVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl DeckLinkVideoSink {
    pub fn new() -> Self {
        Self::with_opener(0, sdk_opener())
    }

    pub fn with_opener(device_number: u32, opener: DeckLinkOpener) -> Self {
        Self {
            device_number,
            mode: None,
            output: claim_output(device_number, Arc::clone(&opener)),
            opener,
            layout: None,
            pacer: PresentationPacer::new(),
            last_frame_end_ns: None,
        }
    }

    pub fn clock(&self) -> Arc<DriftClock> {
        self.output.clock()
    }

    fn layout_for(&self, caps: &Caps) -> Result<PictureLayout, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Fixed(framerate_q16),
            interlace,
            ..
        } = *caps
        else {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} needs fixed raw video caps, got {caps:?}"
            );
            return Err(G2gError::CapsMismatch);
        };
        if !video_format_supported(format) {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} cannot output {}",
                format.gst_name()
            );
            return Err(G2gError::CapsMismatch);
        }
        let mode = match self.mode {
            Some(mode) if mode.matches(width, height, framerate_q16, interlace) => mode,
            Some(mode) => {
                g2g_error!(
                    log_target(),
                    "{ELEMENT_NAME} mode {} is {}x{} at {}/{} fps, the caps are {caps:?}",
                    mode.name,
                    mode.width,
                    mode.height,
                    mode.framerate_numerator,
                    mode.framerate_denominator
                );
                return Err(G2gError::CapsMismatch);
            }
            None => mode_for_caps(width, height, framerate_q16, interlace).ok_or_else(|| {
                g2g_error!(log_target(), "{ELEMENT_NAME} has no mode for {caps:?}");
                G2gError::CapsMismatch
            })?,
        };
        let row_bytes = format.row_stride(width).ok_or(G2gError::CapsMismatch)?;
        let frame_bytes = format
            .frame_bytes(u64::from(row_bytes), u64::from(height))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(G2gError::CapsMismatch)?;
        Ok(PictureLayout {
            mode,
            format,
            row_bytes,
            frame_bytes,
        })
    }

    async fn schedule(&mut self, frame: g2g_core::frame::Frame) -> Result<(), G2gError> {
        let layout = self.layout.ok_or(G2gError::NotConfigured)?;
        let sync: ClockSync = self
            .pacer
            .clock_sync()
            .cloned()
            .ok_or(G2gError::NotConfigured)?;
        let pixels = frame.domain.require_system_slice(ELEMENT_NAME)?;
        let pixels = pixels
            .get(..layout.frame_bytes)
            .ok_or(G2gError::CapsMismatch)?;
        let Some(deadline_ns) = self
            .output
            .deadline_ns(&sync, &self.pacer, frame.timing.pts_ns)?
        else {
            return Ok(());
        };
        hold_until_due(&sync, deadline_ns).await;
        let end_ns = self
            .output
            .schedule_video(&sync, deadline_ns, &layout, pixels)?;
        self.last_frame_end_ns = Some(end_ns);
        Ok(())
    }

    // lets the last scheduled picture play out before the stream ends
    async fn finish(&mut self) -> Result<(), G2gError> {
        let Some(end_ns) = self.last_frame_end_ns.take() else {
            return Ok(());
        };
        let board_now = self.output.stop_at(end_ns)?;
        if end_ns > board_now {
            sleep_ns(end_ns - board_now).await;
        }
        Ok(())
    }
}

impl AsyncElement for DeckLinkVideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::RawVideo { format, .. } if video_format_supported(*format) => {
                Ok(upstream_caps.clone())
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        let modes: Vec<&'static DisplayMode> = match self.mode {
            Some(mode) => Vec::from([mode]),
            None => DISPLAY_MODES.iter().collect(),
        };
        let caps = modes
            .iter()
            .flat_map(|mode| PIXEL_FORMATS.iter().map(|(format, _)| mode.caps(*format)))
            .collect();
        CapsConstraint::Accepts(CapsSet::from_alternatives(caps))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let layout = self.layout_for(absolute_caps)?;
        if let Some(configured) = self.layout {
            if configured == layout {
                return Ok(ConfigureOutcome::Accepted);
            }
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} cannot change mode or format while the output runs"
            );
            return Err(G2gError::CapsMismatch);
        }
        self.output.enable_video(layout.mode)?;
        self.layout = Some(layout);
        Ok(ConfigureOutcome::Accepted)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.output.clock();
        Some(ClockCandidate::new(ClockPriority::AudioProvider, clock))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SDI video sink",
            "Sink/Video/Hardware",
            "Plays raw video out of an SDI output for Blackmagic Design DeckLink",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "device-number",
                PropKind::Int,
                "Output device index, shared with the decklinkaudiosink of the same number",
            )
            .with_default("0")
            .with_range("0", "2147483647"),
            PropertySpec::new(
                "mode",
                PropKind::Str,
                "Video mode, auto picks it from the negotiated caps",
            )
            .with_default(AUTO_MODE)
            .with_enum_values(MODE_VALUES),
        ];
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
                self.output = claim_output(device_number, Arc::clone(&self.opener));
                self.device_number = device_number;
                Ok(())
            }
            "mode" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.mode = match name {
                    AUTO_MODE => None,
                    _ => Some(display_mode(name).ok_or(PropError::Value)?),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device-number" => Some(PropValue::Int(i64::from(self.device_number))),
            "mode" => Some(PropValue::Str(
                self.mode.map_or(AUTO_MODE, |mode| mode.name).into(),
            )),
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
                PipelinePacket::Flush => {
                    self.last_frame_end_ns = None;
                    self.output.flush()
                }
                PipelinePacket::Eos => self.finish().await,
                _ => Ok(()),
            }
        })
    }
}

impl PadTemplates for DeckLinkVideoSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let caps = PIXEL_FORMATS.map(|(format, _)| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        });
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(caps.to_vec()))])
    }
}
