use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{bounded, Receiver, SourceLoop};
use g2g_core::{
    g2g_error, Caps, CaptureAnchor, ClockCandidate, ClockPriority, ClockSync, ConfigureOutcome,
    DeviceClockOffset, DriftClock, ElementMetadata, FrameTiming, G2gError, HardwareError,
    LatencyReport, MemoryDomain, MonotonicClock, OutputSink, PadTemplate, PadTemplates,
    PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat,
};

use crate::decklink::{
    any_mode_caps, device_number_from_property, log_target, mode_from_property,
    mode_property_value, sdk_input_opener, video_connection, CapturedVideoFrame, DeckLinkInput,
    DeckLinkInputOpener, DisplayMode, InputEvent, VideoConnection, AUTO_MODE, DISPLAY_MODES,
    MODE_VALUES, PIXEL_FORMATS,
};
use crate::inter::deliver_dropping_oldest;
use crate::st291::AncillaryMeta;

const ELEMENT_NAME: &str = "decklinkvideosrc";

// GStreamer's name for the connection and video-format the board picks
const AUTO: &str = "auto";

// "auto" then every name in VIDEO_CONNECTIONS, in order
const CONNECTION_VALUES: &str = "auto | sdi | hdmi | optical-sdi | component | composite | svideo";

// "auto" then every name in PIXEL_FORMATS, in order
const VIDEO_FORMAT_VALUES: &str = "auto | 8bit-yuv | 8bit-bgra | 10bit-yuv | 10bit-rgb";

// GStreamer captures 8-bit YUV when video-format is auto, until the input says otherwise
const FORMAT_WHEN_AUTO: RawVideoFormat = RawVideoFormat::Uyvy;

// GStreamer's buffer-size default, in frames
const DEFAULT_BUFFER_SIZE: usize = 5;
const DEFAULT_BUFFER_SIZE_TEXT: &str = "5";

struct Capture {
    board: Box<dyn DeckLinkInput>,
    events: Receiver<InputEvent>,
    mode: &'static DisplayMode,
    format: RawVideoFormat,
    signal: bool,
    // the frame intercept_caps read to learn the input mode
    first_frame: Option<CapturedVideoFrame>,
}

impl Capture {
    fn caps(&self) -> Caps {
        self.mode.caps(self.format)
    }

    fn follow_input(
        &mut self,
        mode: Option<&'static DisplayMode>,
        format: Option<RawVideoFormat>,
        video_format: Option<RawVideoFormat>,
    ) -> Result<(), G2gError> {
        let Some(mode) = mode else {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} cannot capture the input's display mode"
            );
            return Err(G2gError::CapsMismatch);
        };
        let Some(format) = format else {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} cannot capture the input's signal format"
            );
            return Err(G2gError::CapsMismatch);
        };
        if video_format.is_some_and(|wanted| wanted != format) {
            g2g_error!(
                log_target(),
                "{ELEMENT_NAME} video-format is set, but the input is {}",
                format.gst_name()
            );
            return Err(G2gError::CapsMismatch);
        }
        self.board.restart_video(mode, format)?;
        self.mode = mode;
        self.format = format;
        Ok(())
    }

    async fn next_frame(
        &mut self,
        video_format: Option<RawVideoFormat>,
        drop_no_signal_frames: bool,
    ) -> Result<Option<CapturedVideoFrame>, G2gError> {
        if let Some(frame) = self.first_frame.take() {
            return Ok(Some(frame));
        }
        loop {
            let Some(event) = self.events.recv().await else {
                return Ok(None);
            };
            let frame = match event {
                InputEvent::Failed(error) => return Err(error),
                InputEvent::FormatChanged { mode, format } => {
                    self.follow_input(mode, format, video_format)?;
                    continue;
                }
                InputEvent::Frame(frame) => frame,
            };
            // frames queued before a restart are still in the old mode
            let current = frame.width == self.mode.width
                && frame.height == self.mode.height
                && frame.format == self.format;
            if !current {
                continue;
            }
            self.signal = !frame.no_signal;
            if frame.no_signal && drop_no_signal_frames {
                continue;
            }
            return Ok(Some(frame));
        }
    }
}

pub struct DeckLinkVideoSrc {
    device_number: u32,
    // None detects the mode on the input
    mode: Option<&'static DisplayMode>,
    // None keeps the board's connection
    connection: Option<&'static VideoConnection>,
    // None takes the format the input carries
    video_format: Option<RawVideoFormat>,
    buffer_size: usize,
    drop_no_signal_frames: bool,
    output_vanc: bool,
    num_buffers: u64,
    // whether the last frame the board captured had an input signal
    signal: bool,
    opener: DeckLinkInputOpener,
    clock: Arc<DriftClock>,
    clock_offset: DeviceClockOffset,
    clock_sync: Option<ClockSync>,
    capture: Option<Capture>,
    configured: bool,
}

impl core::fmt::Debug for DeckLinkVideoSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeckLinkVideoSrc")
            .field("device_number", &self.device_number)
            .field("mode", &self.mode.map(|mode| mode.name))
            .field("video_format", &self.video_format)
            .field("capturing", &self.capture.is_some())
            .finish()
    }
}

impl Default for DeckLinkVideoSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl DeckLinkVideoSrc {
    pub fn new() -> Self {
        Self::with_opener(0, sdk_input_opener())
    }

    pub fn with_opener(device_number: u32, opener: DeckLinkInputOpener) -> Self {
        Self {
            device_number,
            mode: None,
            connection: None,
            video_format: None,
            buffer_size: DEFAULT_BUFFER_SIZE,
            drop_no_signal_frames: false,
            output_vanc: false,
            num_buffers: u64::MAX,
            signal: false,
            opener,
            clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
            clock_offset: DeviceClockOffset::default(),
            clock_sync: None,
            capture: None,
            configured: false,
        }
    }

    pub fn clock(&self) -> Arc<DriftClock> {
        Arc::clone(&self.clock)
    }

    fn open_capture(&self) -> Result<Capture, G2gError> {
        let mut board = (self.opener)(self.device_number)?;
        if let Some(connection) = self.connection {
            board.set_connection(connection)?;
        }
        // GStreamer starts input format detection in its first mode, ntsc
        let mode = self.mode.unwrap_or(&DISPLAY_MODES[0]);
        let format = self.video_format.unwrap_or(FORMAT_WHEN_AUTO);
        board.enable_video(mode, format, self.mode.is_none())?;
        let (sender, events) = bounded(self.buffer_size);
        board.start(Box::new(move |event| {
            deliver_dropping_oldest(&sender, event, |queued| {
                matches!(queued, InputEvent::Frame(_))
            });
        }))?;
        Ok(Capture {
            board,
            events,
            mode,
            format,
            signal: false,
            first_frame: None,
        })
    }

    fn take_capture(&mut self) -> Result<Capture, G2gError> {
        match self.capture.take() {
            Some(capture) => Ok(capture),
            None => self.open_capture(),
        }
    }

    fn set_mode_caps(&self) -> Option<Caps> {
        let mode = self.mode?;
        Some(mode.caps(self.video_format.unwrap_or(FORMAT_WHEN_AUTO)))
    }

    async fn caps(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = self.set_mode_caps() {
            return Ok(caps);
        }
        let capture = self.take_capture()?;
        let capture = self.capture.insert(capture);
        if capture.first_frame.is_none() {
            let frame = capture
                .next_frame(self.video_format, self.drop_no_signal_frames)
                .await?
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            capture.first_frame = Some(frame);
        }
        let caps = capture.caps();
        self.signal = capture.signal;
        Ok(caps)
    }

    fn frame_period_ns(&self) -> u64 {
        let mode = self
            .capture
            .as_ref()
            .map(|capture| capture.mode)
            .or(self.mode);
        mode.map_or(0, DisplayMode::frame_duration_ns)
    }

    fn reject_once_open(&self) -> Result<(), PropError> {
        if self.capture.is_some() || self.configured {
            return Err(PropError::Value);
        }
        Ok(())
    }

    async fn stream(&mut self, out: &mut dyn OutputSink) -> Result<u64, G2gError> {
        let mut advertised = self.caps().await?;
        let mut capture = self.take_capture()?;
        let mut anchor = CaptureAnchor::new();
        let mut first_hardware_ns = None;
        let mut sequence = 0;
        while sequence < self.num_buffers {
            let Some(captured) = capture
                .next_frame(self.video_format, self.drop_no_signal_frames)
                .await?
            else {
                break;
            };
            self.signal = capture.signal;
            let caps = capture.caps();
            if caps != advertised {
                if out
                    .push(PipelinePacket::CapsChanged(caps.clone()))
                    .await
                    .is_err()
                {
                    return Ok(sequence);
                }
                advertised = caps;
            }
            let local_ns = self.clock.reference_now();
            let board_ns = capture.board.hardware_time_ns()?;
            let master_ns = self.clock_offset.master_ns(&self.clock, local_ns, board_ns);
            self.clock.observe(local_ns, master_ns);

            let period_ns = capture.mode.frame_duration_ns();
            let first_ns = *first_hardware_ns.get_or_insert(captured.hardware_time_ns);
            let elapsed_ns = captured
                .hardware_time_ns
                .checked_sub(first_ns)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            // a frame finished arriving one period before the board handed it over
            let pts_ns = match &self.clock_sync {
                Some(sync) => anchor.stamp(sync, elapsed_ns, period_ns),
                None => elapsed_ns,
            };
            let mut frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    captured.pixels.into_boxed_slice(),
                )),
                timing: FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    duration_ns: period_ns,
                    capture_ns: pts_ns,
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    keyframe: true,
                },
                sequence,
                meta: Default::default(),
            };
            crate::paddedrows::declare_padded_rows(
                &mut frame,
                captured.format,
                captured.width as usize,
                captured.height as usize,
                0,
                captured.row_bytes as usize,
            );
            if self.output_vanc && !captured.ancillary.is_empty() {
                frame.meta.attach(AncillaryMeta {
                    packets: captured.ancillary,
                });
            }
            if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                return Ok(sequence);
            }
            sequence += 1;
        }
        out.push(PipelinePacket::Eos).await?;
        Ok(sequence)
    }
}

impl SourceLoop for DeckLinkVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(self.caps())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(self.frame_period_ns(), None)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.clock.clone();
        Some(ClockCandidate::new(ClockPriority::LiveSource, clock))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SDI video source",
            "Source/Video/Hardware",
            "Captures raw video from an SDI input for Blackmagic Design DeckLink",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("device-number", PropKind::Int, "Input device index")
                .with_default("0")
                .with_range("0", "2147483647"),
            PropertySpec::new(
                "mode",
                PropKind::Str,
                "Video mode, auto detects it on the input",
            )
            .with_default(AUTO_MODE)
            .with_enum_values(MODE_VALUES),
            PropertySpec::new(
                "connection",
                PropKind::Str,
                "Video input connection, auto keeps the board's",
            )
            .with_default(AUTO)
            .with_enum_values(CONNECTION_VALUES),
            PropertySpec::new(
                "video-format",
                PropKind::Str,
                "Pixel format to capture, auto takes the input's",
            )
            .with_default(AUTO)
            .with_enum_values(VIDEO_FORMAT_VALUES),
            PropertySpec::new(
                "buffer-size",
                PropKind::Uint,
                "Captured frames held for downstream before the oldest is dropped",
            )
            .with_default(DEFAULT_BUFFER_SIZE_TEXT)
            .with_range("1", "2147483647"),
            PropertySpec::new(
                "drop-no-signal-frames",
                PropKind::Bool,
                "Drop the frames the board captures while it has no input signal",
            )
            .with_default("false"),
            PropertySpec::new(
                "output-vanc",
                PropKind::Bool,
                "Attach the VANC packets of each frame as AncillaryMeta",
            )
            .with_default("false"),
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "frames to capture then EOS (-1 = forever)",
            )
            .with_default("-1"),
            PropertySpec::new(
                "signal",
                PropKind::Bool,
                "True while the input has a signal",
            )
            .with_default("false")
            .read_only(),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device-number" => {
                let device_number = device_number_from_property(&value)?;
                self.reject_once_open()?;
                self.device_number = device_number;
            }
            "mode" => {
                let mode = mode_from_property(&value)?;
                self.reject_once_open()?;
                self.mode = mode;
            }
            "connection" => {
                let connection = match value.as_str().ok_or(PropError::Type)? {
                    AUTO => None,
                    name => Some(video_connection(name).ok_or(PropError::Value)?),
                };
                self.reject_once_open()?;
                self.connection = connection;
            }
            "video-format" => {
                let video_format = match value.as_str().ok_or(PropError::Type)? {
                    AUTO => None,
                    name => Some(
                        PIXEL_FORMATS
                            .iter()
                            .find(|pixel| pixel.name == name)
                            .ok_or(PropError::Value)?
                            .format,
                    ),
                };
                self.reject_once_open()?;
                self.video_format = video_format;
            }
            "buffer-size" => {
                let buffer_size = value.as_uint().ok_or(PropError::Type)?;
                let buffer_size = usize::try_from(buffer_size).map_err(|_| PropError::Value)?;
                if buffer_size == 0 {
                    return Err(PropError::Value);
                }
                self.reject_once_open()?;
                self.buffer_size = buffer_size;
            }
            "drop-no-signal-frames" => {
                self.drop_no_signal_frames = value.as_bool().ok_or(PropError::Type)?;
            }
            "output-vanc" => self.output_vanc = value.as_bool().ok_or(PropError::Type)?,
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.num_buffers, &value)?,
            "signal" => return Err(PropError::ReadOnly),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device-number" => Some(PropValue::Int(i64::from(self.device_number))),
            "mode" => Some(mode_property_value(self.mode)),
            "connection" => Some(PropValue::Str(
                self.connection
                    .map_or(AUTO, |connection| connection.name)
                    .into(),
            )),
            "video-format" => Some(PropValue::Str(
                PIXEL_FORMATS
                    .iter()
                    .find(|pixel| Some(pixel.format) == self.video_format)
                    .map_or(AUTO, |pixel| pixel.name)
                    .into(),
            )),
            "buffer-size" => Some(PropValue::Uint(self.buffer_size as u64)),
            "drop-no-signal-frames" => Some(PropValue::Bool(self.drop_no_signal_frames)),
            "output-vanc" => Some(PropValue::Bool(self.output_vanc)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.num_buffers)),
            "signal" => Some(PropValue::Bool(self.signal)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.num_buffers, out).await? {
                return Ok(0);
            }
            self.stream(out).await
        })
    }
}

impl PadTemplates for DeckLinkVideoSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(any_mode_caps())])
    }
}
