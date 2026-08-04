//! Wall-clock overlay (`clockoverlay`). Burns the current time of day into a
//! packed RGBA / BGRA frame, the g2g analog of GStreamer's `clockoverlay`. The
//! text is a strftime-style `time-format` (default `%H:%M:%S`), and the box
//! rendering, placement and styling come from [`timeoverlay`], its
//! buffer-timestamp sibling. std-gated: it needs a system clock, unlike
//! `timeoverlay` (buffer PTS), which is `no_std`.
//!
//! Two deliberate differences from GStreamer: the time is UTC, because the
//! baseline carries no timezone database, and it is read through an injectable
//! [`PipelineClock`] whose `now_ns` is nanoseconds since the UNIX epoch
//! ([`UnixEpochClock`](crate::clock::UnixEpochClock) by default), so a test can
//! pin the rendered text to a fixed instant. Only the strftime fields listed on
//! [`strftime_utc`] are substituted; anything else is drawn literally.
//!
//! [`timeoverlay`]: crate::timeoverlay

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::clock::UnixEpochClock;
use crate::timeoverlay::{
    OverlayCore, OverlayStyle, COLOR_PROP, HALIGN_PROP, SCALE_PROP, SHADED_PROP, VALIGN_PROP,
    XPAD_PROP, YPAD_PROP,
};
use crate::xmlutil::civil;

const DEFAULT_TIME_FORMAT: &str = "%H:%M:%S";

const DAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
/// Days elapsed before the 1st of each month in a non-leap year.
const MONTH_START: [u64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

/// Format `unix_ns` (nanoseconds since the UNIX epoch) as UTC civil time per a
/// strftime `fmt`. Supported fields: `%Y %y %C %m %d %e %j %H %I %M %S %p %P %a
/// %A %b %B %F %T %R %D %s %n %t %%`. An unsupported field is emitted verbatim
/// (`%Q` renders as `%Q`), so a typo shows up on the frame instead of vanishing.
pub fn strftime_utc(unix_ns: u64, fmt: &str) -> String {
    let secs = unix_ns / 1_000_000_000;
    let (year, month, day, hh, mm, ss) = civil(secs);
    // 1970-01-01 was a Thursday (index 4 with Sunday first).
    let wday = ((secs / 86_400 + 4) % 7) as usize;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let yday = MONTH_START[(month - 1) as usize] + day as u64 + u64::from(leap && month > 2);
    let hour12 = match hh % 12 {
        0 => 12,
        h => h,
    };

    let mut out = String::with_capacity(fmt.len() + 8);
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(spec) = chars.next() else {
            // Trailing '%': nothing to substitute, draw it.
            out.push('%');
            break;
        };
        match spec {
            'Y' => out.push_str(&format!("{year:04}")),
            'y' => out.push_str(&format!("{:02}", year.rem_euclid(100))),
            'C' => out.push_str(&format!("{:02}", year.div_euclid(100))),
            'm' => out.push_str(&format!("{month:02}")),
            'd' => out.push_str(&format!("{day:02}")),
            'e' => out.push_str(&format!("{day:2}")),
            'j' => out.push_str(&format!("{yday:03}")),
            'H' => out.push_str(&format!("{hh:02}")),
            'I' => out.push_str(&format!("{hour12:02}")),
            'M' => out.push_str(&format!("{mm:02}")),
            'S' => out.push_str(&format!("{ss:02}")),
            'p' => out.push_str(if hh < 12 { "AM" } else { "PM" }),
            'P' => out.push_str(if hh < 12 { "am" } else { "pm" }),
            'a' => out.push_str(&DAYS[wday][..3]),
            'A' => out.push_str(DAYS[wday]),
            'b' | 'h' => out.push_str(&MONTHS[(month - 1) as usize][..3]),
            'B' => out.push_str(MONTHS[(month - 1) as usize]),
            'F' => out.push_str(&format!("{year:04}-{month:02}-{day:02}")),
            'T' => out.push_str(&format!("{hh:02}:{mm:02}:{ss:02}")),
            'R' => out.push_str(&format!("{hh:02}:{mm:02}")),
            'D' => out.push_str(&format!("{month:02}/{day:02}/{:02}", year.rem_euclid(100))),
            's' => out.push_str(&format!("{secs}")),
            'n' => out.push(' '),
            't' => out.push(' '),
            '%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other);
            }
        }
    }
    out
}

pub struct ClockOverlay {
    core: OverlayCore,
    time_format: String,
    /// Time-of-day source: `now_ns` is nanoseconds since the UNIX epoch.
    clock: Arc<dyn PipelineClock + Send + Sync>,
}

impl core::fmt::Debug for ClockOverlay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClockOverlay")
            .field("time_format", &self.time_format)
            .field("now_ns", &self.clock.now_ns())
            .finish()
    }
}

impl Default for ClockOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl ClockOverlay {
    pub fn new() -> Self {
        Self {
            core: OverlayCore::default(),
            time_format: DEFAULT_TIME_FORMAT.to_string(),
            clock: Arc::new(UnixEpochClock),
        }
    }

    pub fn with_scale(mut self, scale: u32) -> Self {
        self.core.style_mut().scale = scale.max(1);
        self
    }

    /// Placement and styling of the text box.
    pub fn with_style(mut self, style: OverlayStyle) -> Self {
        *self.core.style_mut() = style;
        self
    }

    /// The strftime format drawn (the `time-format` property).
    pub fn with_time_format(mut self, format: &str) -> Self {
        self.time_format = format.to_string();
        self
    }

    /// Read the time of day from `clock` instead of the system clock. `now_ns`
    /// must be nanoseconds since the UNIX epoch.
    pub fn with_clock(mut self, clock: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        self.clock = clock;
        self
    }
}

impl AsyncElement for ClockOverlay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        OverlayCore::intercept_caps(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        OverlayCore::constraint()
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.core.configure(absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let text = strftime_utc(self.clock.now_ns(), &self.time_format);
                    self.core.render(frame, &text, out).await?;
                }
                other => self.core.control(other, out).await?,
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CLOCKOVERLAY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Clock overlay",
            "Filter/Editor/Video",
            "Overlays the wall-clock time on video",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "time-format" => {
                self.time_format = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => self.core.set_style_property(name, value),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "time-format" => Some(PropValue::Str(self.time_format.clone())),
            _ => self.core.get_style_property(name),
        }
    }
}

static CLOCKOVERLAY_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "time-format",
        PropKind::Str,
        "strftime format for the UTC time of day, e.g. \"%F %T\"",
    )
    .with_default(DEFAULT_TIME_FORMAT),
    SCALE_PROP,
    HALIGN_PROP,
    VALIGN_PROP,
    XPAD_PROP,
    YPAD_PROP,
    COLOR_PROP,
    SHADED_PROP,
];

impl PadTemplates for ClockOverlay {
    fn pad_templates() -> Vec<PadTemplate> {
        OverlayCore::pad_templates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Dim, FrameTiming, PushOutcome, Rate, RawVideoFormat};

    /// 2026-08-04T13:45:07Z, a Tuesday, day 216 of the year.
    const FIXED_NS: u64 = 1_785_851_107 * 1_000_000_000;

    #[derive(Debug)]
    struct FixedClock(u64);

    impl PipelineClock for FixedClock {
        fn now_ns(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn strftime_substitutes_the_supported_fields() {
        assert_eq!(strftime_utc(FIXED_NS, "%H:%M:%S"), "13:45:07");
        assert_eq!(strftime_utc(FIXED_NS, "%F %T"), "2026-08-04 13:45:07");
        assert_eq!(strftime_utc(FIXED_NS, "%Y-%m-%d"), "2026-08-04");
        assert_eq!(strftime_utc(FIXED_NS, "%I:%M %p"), "01:45 PM");
        assert_eq!(strftime_utc(FIXED_NS, "%a %b %e"), "Tue Aug  4");
        assert_eq!(strftime_utc(FIXED_NS, "%A %B"), "Tuesday August");
        assert_eq!(strftime_utc(FIXED_NS, "%D %R"), "08/04/26 13:45");
        assert_eq!(strftime_utc(FIXED_NS, "%y %C %j"), "26 20 216");
        assert_eq!(strftime_utc(FIXED_NS, "%s"), "1785851107");
        assert_eq!(strftime_utc(FIXED_NS, "100%% done"), "100% done");
        // Unsupported fields and a trailing percent are drawn as written.
        assert_eq!(strftime_utc(FIXED_NS, "%Q at %H%"), "%Q at 13%");
    }

    #[test]
    fn day_of_year_counts_the_leap_day() {
        // 2024-03-01 (leap year) is day 61; 2023-03-01 is day 60.
        assert_eq!(
            strftime_utc(1_709_251_200 * 1_000_000_000, "%F %j"),
            "2024-03-01 061"
        );
        assert_eq!(
            strftime_utc(1_677_628_800 * 1_000_000_000, "%F %j"),
            "2023-03-01 060"
        );
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[derive(Default)]
    struct PixelSink {
        last: Option<Vec<u8>>,
    }

    impl OutputSink for PixelSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.last = Some(slice.to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    /// A pinned clock makes the burnt-in text deterministic: the element's output
    /// must equal the same text drawn straight onto a white canvas.
    #[tokio::test]
    async fn burns_the_pinned_clock_time() {
        let (w, h) = (96u32, 24u32);
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        let mut ov = ClockOverlay::new()
            .with_style(style.clone())
            .with_time_format("%F %T")
            .with_clock(Arc::new(FixedClock(FIXED_NS)));
        ov.configure_pipeline(&rgba_caps(w, h)).unwrap();

        let buf = vec![255u8; (w * h * 4) as usize].into_boxed_slice();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(buf)),
            FrameTiming::default(),
            0,
        );
        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        let out = sink.last.take().expect("frame forwarded");

        let mut expected = vec![255u8; (w * h * 4) as usize];
        crate::timeoverlay::draw_text(
            &mut expected,
            w as usize,
            h as usize,
            "2026-08-04 13:45:07",
            &style,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn configure_rejects_non_video() {
        let mut c = ClockOverlay::new();
        let bad = Caps::Audio {
            format: g2g_core::AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(
            c.configure_pipeline(&bad).unwrap_err(),
            G2gError::CapsMismatch
        );
    }

    #[test]
    fn properties_round_trip() {
        let mut c = ClockOverlay::new();
        assert_eq!(
            c.get_property("time-format"),
            Some(PropValue::Str("%H:%M:%S".into()))
        );
        c.set_property("time-format", PropValue::Str("%F".into()))
            .unwrap();
        assert_eq!(
            c.get_property("time-format"),
            Some(PropValue::Str("%F".into()))
        );
    }
}
