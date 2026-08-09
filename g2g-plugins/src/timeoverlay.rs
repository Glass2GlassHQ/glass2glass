//! Buffer-time overlay (`timeoverlay`). Burns a timestamp derived from each
//! frame, formatted as `HH:MM:SS.mmm`, into a packed RGBA / BGRA frame with the
//! embedded 8x8 [`bitmapfont`], preserving format and geometry. CPU-only
//! `no_std`.
//!
//! The g2g analog of GStreamer's `timeoverlay`. Which timestamp is drawn is the
//! `time-mode` property ([`TimeMode`]): the raw PTS, its stream / running time
//! through the active segment, running time relative to the first frame, the
//! buffer count, or the frame number. A frame carrying the source's own SMPTE
//! timecode (`TimecodeMeta`, mined from the bitstream by the H.264 / H.265
//! parser) draws that instead of the PTS in the default mode, so gst's separate
//! `time-code` mode has no counterpart here. `reference-timestamp` is absent:
//! there is no reference-timestamp frame-meta to read.
//!
//! With `show-times-as-dates` the time is instead drawn as a civil date, that
//! time counted from `datetime-epoch` and formatted per `datetime-format`. The
//! epoch is a string (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`, UTC) rather than
//! GStreamer's `GDateTime`, and defaults the same way (1900-01-01).
//!
//! This module also owns the text-box renderer, the strftime formatter
//! ([`strftime_utc`]) and the placement / styling knobs (`halignment`,
//! `valignment`, `xpad`, `ypad`, `color`, `shaded-background`) that the
//! wall-clock sibling [`clockoverlay`](crate::clockoverlay) shares, since the
//! two elements differ only in the string they draw. Unlike GStreamer's
//! Pango-backed base class, the box is shaded by default
//! (`shaded-background=true`) and hugs the aligned corner (`xpad`/`ypad` default
//! `0`): the 8x8 font is far smaller than a Pango one, so gst's 25px inset would
//! push it off a small frame.
//!
//! [`bitmapfont`]: crate::bitmapfont

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat, Segment,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
use crate::paint::blend_px;
use crate::xmlutil::{civil_signed, days_from_civil};

const FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Bgra8];

const DEFAULT_DATETIME_FORMAT: &str = "%F %T";
/// GStreamer's "prime epoch" default for `datetime-epoch`, and the same date in
/// seconds from the UNIX epoch (negative: it predates 1970). `parse_epoch` of
/// the string must equal the number, which `dates_count_from_the_epoch` checks.
const DEFAULT_DATETIME_EPOCH: &str = "1900-01-01";
const DEFAULT_EPOCH_SECS: i64 = -2_208_988_800;

/// Format a SMPTE 12M timecode as `HH:MM:SS:FF`, with the `;` frame separator
/// that marks a drop-frame count.
#[cfg(feature = "metadata")]
fn format_timecode(tc: &g2g_core::meta::TimecodeMeta) -> String {
    let sep = if tc.drop_frame { ';' } else { ':' };
    format!(
        "{:02}:{:02}:{:02}{sep}{:02}",
        tc.hours, tc.minutes, tc.seconds, tc.frames
    )
}

/// Format a nanosecond timestamp as `HH:MM:SS.mmm`.
fn format_time(ns: u64) -> String {
    let total_ms = ns / 1_000_000;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60_000) % 60;
    let h = total_ms / 3_600_000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

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

/// Gregorian leap year, `rem_euclid` so a year before 1 CE still answers
/// correctly.
fn is_leap(year: i64) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

/// Format `unix_ns` (nanoseconds since the UNIX epoch) as UTC civil time per a
/// strftime `fmt`. Supported fields: `%Y %y %C %m %d %e %j %H %I %M %S %p %P %a
/// %A %b %B %F %T %R %D %s %n %t %%`. An unsupported field is emitted verbatim
/// (`%Q` renders as `%Q`), so a typo shows up on the frame instead of vanishing.
pub fn strftime_utc(unix_ns: u64, fmt: &str) -> String {
    strftime_utc_secs((unix_ns / 1_000_000_000) as i64, fmt)
}

/// [`strftime_utc`] for signed seconds since the UNIX epoch, so a date before
/// 1970 renders (`timeoverlay`'s date epoch defaults to 1900-01-01).
pub fn strftime_utc_secs(secs: i64, fmt: &str) -> String {
    let (year, month, day, hh, mm, ss) = civil_signed(secs);
    // 1970-01-01 was a Thursday (index 4 with Sunday first).
    let wday = (secs.div_euclid(86_400) + 4).rem_euclid(7) as usize;
    let leap = is_leap(year);
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

/// Days in `month` of `year`, for validating a `datetime-epoch`.
fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        2 => 28 + u32::from(is_leap(year)),
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Parse a `datetime-epoch`: an ISO-8601 UTC `YYYY-MM-DD` or
/// `YYYY-MM-DD HH:MM:SS` (a `T` separator is accepted too) to signed seconds
/// from the UNIX epoch. `None` for anything malformed or out of range, so a bad
/// value is rejected rather than silently read as 1900.
fn parse_epoch(s: &str) -> Option<i64> {
    let s = s.trim();
    let (date, time) = match s.split_once(['T', ' ']) {
        Some((d, t)) => (d, Some(t)),
        None => (s, None),
    };
    let mut ymd = date.split('-');
    let year: i64 = ymd.next()?.parse().ok()?;
    let month: u32 = ymd.next()?.parse().ok()?;
    let day: u32 = ymd.next()?.parse().ok()?;
    if ymd.next().is_some()
        || !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
    {
        return None;
    }
    let (mut hh, mut mi, mut ss) = (0u32, 0u32, 0u32);
    if let Some(t) = time {
        let mut hms = t.split(':');
        hh = hms.next()?.parse().ok()?;
        mi = hms.next()?.parse().ok()?;
        ss = hms.next()?.parse().ok()?;
        if hms.next().is_some() || hh > 23 || mi > 59 || ss > 59 {
            return None;
        }
    }
    Some(days_from_civil(year, month, day) * 86_400 + (hh * 3_600 + mi * 60 + ss) as i64)
}

/// Which time [`TimeOverlay`] draws (GStreamer's `time-mode`).
// Closed set: intentionally exhaustive; see STABILITY.md.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeMode {
    /// The frame's PTS, unmapped.
    #[default]
    BufferTime,
    /// The PTS mapped to stream time through the active segment.
    StreamTime,
    /// The PTS mapped to running time through the active segment.
    RunningTime,
    /// Running time relative to the first frame of the current segment, so
    /// playback always starts at `00:00:00.000`.
    ElapsedRunningTime,
    /// Count of frames drawn so far, as a plain integer.
    BufferCount,
    /// Frame number: running time scaled by the negotiated framerate.
    BufferOffset,
}

impl TimeMode {
    fn from_nick(nick: &str) -> Option<TimeMode> {
        Some(match nick {
            "buffer-time" => TimeMode::BufferTime,
            "stream-time" => TimeMode::StreamTime,
            "running-time" => TimeMode::RunningTime,
            "elapsed-running-time" => TimeMode::ElapsedRunningTime,
            "buffer-count" => TimeMode::BufferCount,
            "buffer-offset" => TimeMode::BufferOffset,
            _ => return None,
        })
    }

    fn nick(self) -> &'static str {
        match self {
            TimeMode::BufferTime => "buffer-time",
            TimeMode::StreamTime => "stream-time",
            TimeMode::RunningTime => "running-time",
            TimeMode::ElapsedRunningTime => "elapsed-running-time",
            TimeMode::BufferCount => "buffer-count",
            TimeMode::BufferOffset => "buffer-offset",
        }
    }
}

/// Horizontal placement of the text box (`halignment`).
// Closed set: intentionally exhaustive; see STABILITY.md.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Vertical placement of the text box (`valignment`).
// Closed set: intentionally exhaustive; see STABILITY.md.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VAlign {
    #[default]
    Top,
    Center,
    Bottom,
}

/// Placement and styling of the burnt-in text box, shared by `timeoverlay` and
/// `clockoverlay`.
#[derive(Clone, Debug)]
pub struct OverlayStyle {
    /// Integer font magnification (>= 1).
    pub scale: u32,
    pub halign: HAlign,
    pub valign: VAlign,
    /// Inset from the aligned frame edge, in output pixels.
    pub xpad: u32,
    pub ypad: u32,
    /// Glyph color as `[R, G, B, A]`.
    pub color: [u8; 4],
    /// Draw a translucent black box behind the glyphs.
    pub shaded: bool,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            scale: 2,
            halign: HAlign::Left,
            valign: VAlign::Top,
            xpad: 0,
            ypad: 0,
            color: [255, 255, 255, 255],
            shaded: true,
        }
    }
}

/// Caps handling, frame copying and text rendering shared by the two timestamp
/// overlays: they differ only in the string they draw, so the whole element body
/// apart from that lives here.
#[derive(Debug, Default)]
pub(crate) struct OverlayCore {
    style: OverlayStyle,
    input: Option<(RawVideoFormat, u32, u32, Rate)>,
    configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl OverlayCore {
    pub(crate) fn style_mut(&mut self) -> &mut OverlayStyle {
        &mut self.style
    }

    /// Negotiated framerate, in Q16 fps, once configured.
    fn rate(&self) -> Option<Rate> {
        self.input.as_ref().map(|(_, _, _, r)| r.clone())
    }

    /// Frames pushed downstream so far.
    pub(crate) fn emitted(&self) -> u64 {
        self.emitted
    }

    fn accept_input(caps: &Caps) -> Result<(RawVideoFormat, u32, u32, Rate), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate,
            interlace: _,
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !FORMATS.contains(format) || *w == 0 || *h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        Ok((*format, *w, *h, framerate.clone()))
    }

    pub(crate) fn intercept_caps(upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for format in FORMATS {
            let candidate = Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    pub(crate) fn constraint() -> CapsConstraint<'static> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo { format, .. } if FORMATS.contains(format) => {
                CapsSet::one(input.clone())
            }
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    pub(crate) fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
        };
        let set = CapsSet::from_alternatives(FORMATS.map(any_geometry).to_vec());
        Vec::from([PadTemplate::sink(set.clone()), PadTemplate::source(set)])
    }

    pub(crate) fn configure(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(Self::accept_input(absolute_caps)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Copy `frame`, burn `text` into the copy, and push it (emitting the output
    /// caps first when they changed).
    pub(crate) async fn render(
        &mut self,
        frame: Frame,
        text: &str,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let (format, w, h, rate) = match &self.input {
            Some((f, w, h, r)) => (*f, *w, *h, r.clone()),
            None => return Err(G2gError::NotConfigured),
        };
        let Some(src) = frame.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = (w as usize) * (h as usize) * 4;
        if src.len() < bytes {
            return Err(G2gError::CapsMismatch);
        }
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);
        draw_text(&mut dst, w as usize, h as usize, text, &self.style);

        let new_caps = Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: rate,
            interlace: g2g_core::Interlace::Any,
        };
        if self.last_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                .await?;
            self.last_caps = Some(new_caps);
        }
        let out_frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(dst)),
            timing: frame.timing,
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }

    /// Handle everything that is not a data frame: mid-stream caps refinement,
    /// flush, and plain forwarding. `Eos` is swallowed (the runner's transform
    /// arm emits it).
    pub(crate) async fn control(
        &mut self,
        packet: PipelinePacket,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        match packet {
            PipelinePacket::CapsChanged(c) => {
                self.input = Some(Self::accept_input(&c)?);
            }
            PipelinePacket::Flush => {
                self.last_caps = None;
                out.push(PipelinePacket::Flush).await?;
            }
            PipelinePacket::Eos => {}
            other => {
                out.push(other).await?;
            }
        }
        Ok(())
    }

    /// Apply one of the shared placement / styling properties. Unknown names
    /// return [`PropError::Unknown`], so an element matches its own properties
    /// first and delegates the rest here.
    pub(crate) fn set_style_property(
        &mut self,
        name: &str,
        value: PropValue,
    ) -> Result<(), PropError> {
        match name {
            "scale" => {
                let s = value.as_uint().ok_or(PropError::Type)?;
                if s == 0 {
                    return Err(PropError::Value);
                }
                self.style.scale = s as u32;
            }
            "halignment" => {
                self.style.halign = match value.as_str().ok_or(PropError::Type)? {
                    "left" => HAlign::Left,
                    "center" => HAlign::Center,
                    "right" => HAlign::Right,
                    _ => return Err(PropError::Value),
                };
            }
            "valignment" => {
                self.style.valign = match value.as_str().ok_or(PropError::Type)? {
                    "top" => VAlign::Top,
                    "center" => VAlign::Center,
                    "bottom" => VAlign::Bottom,
                    _ => return Err(PropError::Value),
                };
            }
            "xpad" => self.style.xpad = value.as_uint().ok_or(PropError::Type)? as u32,
            "ypad" => self.style.ypad = value.as_uint().ok_or(PropError::Type)? as u32,
            // 0xAARRGGBB packed color, the gst textoverlay convention. Stored
            // as [R, G, B, A].
            "color" => {
                let argb = value.as_uint().ok_or(PropError::Type)? as u32;
                self.style.color = [
                    (argb >> 16) as u8,
                    (argb >> 8) as u8,
                    argb as u8,
                    (argb >> 24) as u8,
                ];
            }
            "shaded-background" => self.style.shaded = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    /// Read back one of the shared placement / styling properties.
    pub(crate) fn get_style_property(&self, name: &str) -> Option<PropValue> {
        let s = &self.style;
        Some(match name {
            "scale" => PropValue::Uint(s.scale as u64),
            "halignment" => PropValue::Str(
                match s.halign {
                    HAlign::Left => "left",
                    HAlign::Center => "center",
                    HAlign::Right => "right",
                }
                .into(),
            ),
            "valignment" => PropValue::Str(
                match s.valign {
                    VAlign::Top => "top",
                    VAlign::Center => "center",
                    VAlign::Bottom => "bottom",
                }
                .into(),
            ),
            "xpad" => PropValue::Uint(s.xpad as u64),
            "ypad" => PropValue::Uint(s.ypad as u64),
            "color" => {
                let [r, g, b, a] = s.color;
                PropValue::Uint(
                    ((a as u64) << 24) | ((r as u64) << 16) | ((g as u64) << 8) | b as u64,
                )
            }
            "shaded-background" => PropValue::Bool(s.shaded),
            _ => return None,
        })
    }
}

// The placement / styling properties both timestamp overlays expose. `const`
// (not a static slice) so each element can list them in its own
// `&'static [PropertySpec]` table.
pub(crate) const SCALE_PROP: PropertySpec =
    PropertySpec::new("scale", PropKind::Uint, "integer font magnification (>= 1)")
        .with_default("2");
pub(crate) const HALIGN_PROP: PropertySpec = PropertySpec::new(
    "halignment",
    PropKind::Str,
    "horizontal placement of the text box",
)
.with_enum_values("left | center | right")
.with_default("left");
pub(crate) const VALIGN_PROP: PropertySpec = PropertySpec::new(
    "valignment",
    PropKind::Str,
    "vertical placement of the text box",
)
.with_enum_values("top | center | bottom")
.with_default("top");
pub(crate) const XPAD_PROP: PropertySpec = PropertySpec::new(
    "xpad",
    PropKind::Uint,
    "horizontal inset from the aligned frame edge, in pixels",
)
.with_default("0");
pub(crate) const YPAD_PROP: PropertySpec = PropertySpec::new(
    "ypad",
    PropKind::Uint,
    "vertical inset from the aligned frame edge, in pixels",
)
.with_default("0");
pub(crate) const COLOR_PROP: PropertySpec = PropertySpec::new(
    "color",
    PropKind::Uint,
    "glyph color as 0xAARRGGBB (e.g. 4294967295 = opaque white)",
)
.with_default("4294967295");
pub(crate) const SHADED_PROP: PropertySpec = PropertySpec::new(
    "shaded-background",
    PropKind::Bool,
    "draw a translucent black box behind the glyphs",
)
.with_default("true");

/// # Example
///
/// ```no_run
/// use g2g_plugins::timeoverlay::{TimeMode, TimeOverlay};
///
/// let overlay = TimeOverlay::new()
///     .with_time_mode(TimeMode::RunningTime)
///     .with_scale(2);
/// ```
#[derive(Debug)]
pub struct TimeOverlay {
    core: OverlayCore,
    mode: TimeMode,
    /// Active segment, used by the stream- / running-time modes. Defaults to a
    /// full open segment, so those modes read the PTS until one arrives.
    segment: Segment,
    /// Running time of the first frame since the last segment / flush, for
    /// [`TimeMode::ElapsedRunningTime`].
    elapsed_base: Option<u64>,
    /// Draw the time as a date counted from `epoch_secs` (`show-times-as-dates`).
    show_dates: bool,
    datetime_format: String,
    /// The `datetime-epoch` string as set, so the property reads back verbatim,
    /// alongside the seconds it parsed to (signed: the default predates 1970).
    datetime_epoch: String,
    epoch_secs: i64,
}

impl Default for TimeOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeOverlay {
    pub fn new() -> Self {
        Self {
            core: OverlayCore::default(),
            mode: TimeMode::default(),
            segment: Segment::new(),
            elapsed_base: None,
            show_dates: false,
            datetime_format: DEFAULT_DATETIME_FORMAT.to_string(),
            datetime_epoch: DEFAULT_DATETIME_EPOCH.to_string(),
            epoch_secs: DEFAULT_EPOCH_SECS,
        }
    }

    pub fn with_scale(mut self, scale: u32) -> Self {
        self.core.style_mut().scale = scale.max(1);
        self
    }

    /// Which time to draw (the `time-mode` property).
    pub fn with_time_mode(mut self, mode: TimeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Placement and styling of the text box.
    pub fn with_style(mut self, style: OverlayStyle) -> Self {
        *self.core.style_mut() = style;
        self
    }

    /// Draw the time as a date counted from the epoch (`show-times-as-dates`).
    pub fn with_show_times_as_dates(mut self, on: bool) -> Self {
        self.show_dates = on;
        self
    }

    /// The strftime format dates are drawn in (`datetime-format`).
    pub fn with_datetime_format(mut self, format: &str) -> Self {
        self.datetime_format = format.to_string();
        self
    }

    /// The UTC date times are counted from (`datetime-epoch`): `YYYY-MM-DD` or
    /// `YYYY-MM-DD HH:MM:SS`. Rejects anything else rather than falling back to
    /// the default epoch.
    pub fn with_datetime_epoch(mut self, epoch: &str) -> Result<Self, PropError> {
        self.epoch_secs = parse_epoch(epoch).ok_or(PropError::Value)?;
        self.datetime_epoch = epoch.to_string();
        Ok(self)
    }

    /// Running time of `pts` through the active segment. A frame outside the
    /// segment has none; fall back to the raw PTS rather than drawing a
    /// misleading zero.
    fn running_time(&self, pts: u64) -> u64 {
        self.segment.to_running_time(pts).unwrap_or(pts)
    }

    /// The text drawn for `frame`, per the active [`TimeMode`].
    ///
    /// A frame carrying the source's own SMPTE timecode draws that instead of the
    /// PTS in the default [`TimeMode::BufferTime`]: it is the same instant, told
    /// on the clock the source actually counted. The other modes ask for a
    /// specific pipeline clock, so they are left alone, as is `show-times-as-dates`
    /// (a timecode is not a date).
    fn text_for(&mut self, frame: &Frame) -> String {
        #[cfg(feature = "metadata")]
        if self.mode == TimeMode::BufferTime && !self.show_dates {
            if let Some(tc) = frame.meta.get::<g2g_core::meta::TimecodeMeta>() {
                return format_timecode(tc);
            }
        }
        let timing = &frame.timing;
        let pts = timing.pts_ns;
        let ns = match self.mode {
            TimeMode::BufferTime => pts,
            TimeMode::StreamTime => self.segment.to_stream_time(pts).unwrap_or(pts),
            TimeMode::RunningTime => self.running_time(pts),
            TimeMode::ElapsedRunningTime => {
                let rt = self.running_time(pts);
                let base = *self.elapsed_base.get_or_insert(rt);
                rt.saturating_sub(base)
            }
            // Counts are not times, so `show-times-as-dates` does not apply to
            // them, as in gst.
            TimeMode::BufferCount => return format!("{}", self.core.emitted()),
            TimeMode::BufferOffset => {
                return format!("{}", frame_number(self.running_time(pts), self.core.rate()))
            }
        };
        if self.show_dates {
            let secs = self.epoch_secs.saturating_add((ns / 1_000_000_000) as i64);
            strftime_utc_secs(secs, &self.datetime_format)
        } else {
            format_time(ns)
        }
    }

    /// Count of frames drawn so far.
    pub fn drawn_count(&self) -> u64 {
        self.core.emitted()
    }
}

/// Frame number of `running_ns` at the negotiated framerate: `running * fps`.
/// Zero when the rate is not fixed (no framerate to count against).
fn frame_number(running_ns: u64, rate: Option<Rate>) -> u64 {
    let Some(Rate::Fixed(q16)) = rate else {
        return 0;
    };
    // fps = q16 / 65536, so frames = running_ns * q16 / (65536 * 1e9). u128 so a
    // long running time at a high rate cannot overflow the multiply.
    let num = running_ns as u128 * q16 as u128;
    (num / (65_536u128 * 1_000_000_000)) as u64
}

impl AsyncElement for TimeOverlay {
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
                    let text = self.text_for(&frame);
                    self.core.render(frame, &text, out).await?;
                }
                // A new segment re-bases stream / running time, so the elapsed
                // origin restarts with it.
                PipelinePacket::Segment(seg) => {
                    self.segment = seg;
                    self.elapsed_base = None;
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Flush => {
                    self.elapsed_base = None;
                    self.core.control(PipelinePacket::Flush, out).await?;
                }
                other => self.core.control(other, out).await?,
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TIMEOVERLAY_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Time overlay",
            "Filter/Editor/Video",
            "Overlays the buffer time on video",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "time-mode" => {
                let nick = value.as_str().ok_or(PropError::Type)?;
                self.mode = TimeMode::from_nick(nick).ok_or(PropError::Value)?;
                Ok(())
            }
            "show-times-as-dates" => {
                self.show_dates = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "datetime-format" => {
                self.datetime_format = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "datetime-epoch" => {
                let epoch = value.as_str().ok_or(PropError::Type)?;
                self.epoch_secs = parse_epoch(epoch).ok_or(PropError::Value)?;
                self.datetime_epoch = epoch.to_string();
                Ok(())
            }
            _ => self.core.set_style_property(name, value),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "time-mode" => Some(PropValue::Str(self.mode.nick().into())),
            "show-times-as-dates" => Some(PropValue::Bool(self.show_dates)),
            "datetime-format" => Some(PropValue::Str(self.datetime_format.clone())),
            "datetime-epoch" => Some(PropValue::Str(self.datetime_epoch.clone())),
            _ => self.core.get_style_property(name),
        }
    }
}

static TIMEOVERLAY_PROPS: &[PropertySpec] = &[
    PropertySpec::new("time-mode", PropKind::Str, "which time to draw")
        .with_enum_values(
            "buffer-time | stream-time | running-time | elapsed-running-time | \
             buffer-count | buffer-offset",
        )
        .with_default("buffer-time"),
    PropertySpec::new(
        "show-times-as-dates",
        PropKind::Bool,
        "draw the time, counted from datetime-epoch, as a date",
    )
    .with_default("false"),
    PropertySpec::new(
        "datetime-format",
        PropKind::Str,
        "strftime format for the date drawn when show-times-as-dates is set",
    )
    .with_default(DEFAULT_DATETIME_FORMAT),
    PropertySpec::new(
        "datetime-epoch",
        PropKind::Str,
        "UTC date times are counted from, \"YYYY-MM-DD\" or \"YYYY-MM-DD HH:MM:SS\"",
    )
    .with_default(DEFAULT_DATETIME_EPOCH),
    SCALE_PROP,
    HALIGN_PROP,
    VALIGN_PROP,
    XPAD_PROP,
    YPAD_PROP,
    COLOR_PROP,
    SHADED_PROP,
];

impl PadTemplates for TimeOverlay {
    fn pad_templates() -> Vec<PadTemplate> {
        OverlayCore::pad_templates()
    }
}

/// Draw `text` into a packed RGBA/BGRA buffer per `style`: an optional
/// translucent black box, then the glyphs. The glyph bitmap is
/// channel-symmetric, so it renders the same in RGBA and BGRA. Shared with
/// `clockoverlay`.
pub(crate) fn draw_text(buf: &mut [u8], w: usize, h: usize, text: &str, style: &OverlayStyle) {
    let scale = style.scale.max(1) as i32;
    let inner = 2 * scale;
    let cell_w = GLYPH_ADVANCE as i32 * scale;
    let glyph_h = GLYPH_HEIGHT as i32 * scale;
    let box_w = inner * 2 + cell_w * text.chars().count() as i32;
    let box_h = inner * 2 + glyph_h;
    let (wi, hi) = (w as i32, h as i32);
    let (xpad, ypad) = (style.xpad as i32, style.ypad as i32);
    let box_x = match style.halign {
        HAlign::Left => xpad,
        HAlign::Center => (wi - box_w) / 2,
        HAlign::Right => wi - box_w - xpad,
    };
    let box_y = match style.valign {
        VAlign::Top => ypad,
        VAlign::Center => (hi - box_h) / 2,
        VAlign::Bottom => hi - box_h - ypad,
    };

    let dims = (w, h);
    if style.shaded {
        fill_rect(buf, dims, box_x, box_y, box_w, box_h, [0, 0, 0, 160]);
    }
    for (i, c) in text.chars().enumerate() {
        let gx = box_x + inner + i as i32 * cell_w;
        blit_glyph(buf, dims, gx, box_y + inner, scale, glyph(c), style.color);
    }
}

fn fill_rect(
    buf: &mut [u8],
    dims: (usize, usize),
    x: i32,
    y: i32,
    rw: i32,
    rh: i32,
    color: [u8; 4],
) {
    let (wi, hi) = (dims.0 as i32, dims.1 as i32);
    for py in y..y + rh {
        if py < 0 || py >= hi {
            continue;
        }
        for px in x..x + rw {
            if px < 0 || px >= wi {
                continue;
            }
            blend_px(buf, ((py * wi + px) * 4) as usize, color, 255);
        }
    }
}

fn blit_glyph(
    buf: &mut [u8],
    dims: (usize, usize),
    gx: i32,
    gy: i32,
    scale: i32,
    rows: [u8; 8],
    color: [u8; 4],
) {
    for (ry, bits) in rows.iter().enumerate() {
        if *bits == 0 {
            continue;
        }
        for col in 0..8i32 {
            if bits & (0x80 >> col) != 0 {
                fill_rect(
                    buf,
                    dims,
                    gx + col * scale,
                    gy + ry as i32 * scale,
                    scale,
                    scale,
                    color,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::FrameTiming;
    use g2g_core::PushOutcome;

    #[test]
    fn formats_time_as_hms() {
        assert_eq!(format_time(0), "00:00:00.000");
        assert_eq!(format_time(1_500_000_000), "00:00:01.500");
        assert_eq!(format_time(3_661_250_000_000), "01:01:01.250");
    }

    #[test]
    fn draws_something_onto_a_blank_frame() {
        // 128x16 white RGBA frame; after overlay some pixels must differ (the box
        // + glyphs), proving the overlay actually wrote to the buffer.
        let (w, h) = (128usize, 16usize);
        let mut buf = vec![255u8; w * h * 4];
        let before = buf.clone();
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        draw_text(&mut buf, w, h, "00:00:01.000", &style);
        assert_ne!(buf, before);
        // top-left pixel is inside the translucent black box, so it darkened.
        assert!(buf[0] < 255);
    }

    /// Bounding box (min_x, min_y, max_x, max_y) of the pixels `draw_text`
    /// touched on an all-white canvas.
    fn touched_bounds(buf: &[u8], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                if buf[i] != 255 || buf[i + 1] != 255 || buf[i + 2] != 255 {
                    bounds = Some(match bounds {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }
        bounds
    }

    #[test]
    fn alignment_and_padding_place_the_box() {
        let (w, h) = (200usize, 80usize);
        let text = "00:00:00.000";
        // scale 1: 2px inner margin each side, 6px per glyph cell, 8px tall.
        let box_w = 2 * 2 + 6 * text.chars().count();
        let box_h = 2 * 2 + 8;

        let draw = |style: &OverlayStyle| {
            let mut buf = vec![255u8; w * h * 4];
            draw_text(&mut buf, w, h, text, style);
            touched_bounds(&buf, w, h).expect("something was drawn")
        };

        let top_left = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        assert_eq!(draw(&top_left), (0, 0, box_w - 1, box_h - 1));

        let bottom_right = OverlayStyle {
            scale: 1,
            halign: HAlign::Right,
            valign: VAlign::Bottom,
            ..Default::default()
        };
        assert_eq!(
            draw(&bottom_right),
            (w - box_w, h - box_h, w - 1, h - 1),
            "flush to the bottom-right corner"
        );

        let padded = OverlayStyle {
            scale: 1,
            halign: HAlign::Right,
            valign: VAlign::Bottom,
            xpad: 10,
            ypad: 5,
            ..Default::default()
        };
        assert_eq!(
            draw(&padded),
            (w - box_w - 10, h - box_h - 5, w - 11, h - 6),
            "inset from the bottom-right corner by xpad / ypad"
        );

        let centered = OverlayStyle {
            scale: 1,
            halign: HAlign::Center,
            valign: VAlign::Center,
            ..Default::default()
        };
        let (x0, y0, x1, y1) = draw(&centered);
        assert_eq!((x0, y0), ((w - box_w) / 2, (h - box_h) / 2));
        assert_eq!((x1 - x0 + 1, y1 - y0 + 1), (box_w, box_h));
    }

    #[test]
    fn unshaded_draws_only_glyphs() {
        // Without the box, an all-white canvas painted with white glyphs is
        // untouched; a red glyph color then shows exactly the glyph pixels.
        let (w, h) = (128usize, 16usize);
        let mut buf = vec![255u8; w * h * 4];
        let white = OverlayStyle {
            scale: 1,
            shaded: false,
            ..Default::default()
        };
        draw_text(&mut buf, w, h, "00:00:01.000", &white);
        assert_eq!(touched_bounds(&buf, w, h), None, "white on white, no box");

        let red = OverlayStyle {
            color: [255, 0, 0, 255],
            ..white
        };
        draw_text(&mut buf, w, h, "1", &red);
        // The '1' glyph is 5 columns wide (its serif row) and 7 rows tall, drawn
        // at the box's 2px inner margin.
        assert_eq!(
            touched_bounds(&buf, w, h).expect("glyph painted"),
            (2, 2, 2 + 4, 2 + 6)
        );
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

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
        }
    }

    /// A 1x1 frame at PTS 0 with no metadata, for the text-selection tests.
    fn plain_frame() -> Frame {
        white_frame(1, 1, 0)
    }

    fn white_frame(w: u32, h: u32, pts_ns: u64) -> Frame {
        let buf = vec![255u8; (w * h * 4) as usize].into_boxed_slice();
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(buf)),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            0,
        )
    }

    /// Render `text` on its own, as the reference the element's output must match.
    fn reference(w: usize, h: usize, text: &str, style: &OverlayStyle) -> Vec<u8> {
        let mut buf = vec![255u8; w * h * 4];
        draw_text(&mut buf, w, h, text, style);
        buf
    }

    /// Drive one frame at `pts_ns` through the element and return the pixels.
    async fn drawn(ov: &mut TimeOverlay, pts_ns: u64) -> Vec<u8> {
        let mut sink = PixelSink::default();
        ov.process(
            PipelinePacket::DataFrame(white_frame(64, 24, pts_ns)),
            &mut sink,
        )
        .await
        .unwrap();
        sink.last.take().expect("frame forwarded")
    }

    #[tokio::test]
    async fn buffer_time_burns_the_pts() {
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        let mut ov = TimeOverlay::new().with_style(style.clone());
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        let out = drawn(&mut ov, 2_500_000_000).await;
        assert_eq!(
            out,
            reference(64, 24, "00:00:02.500", &style),
            "the frame's PTS, formatted HH:MM:SS.mmm"
        );
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn running_and_elapsed_modes_use_the_segment() {
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        // A segment starting 10s into the stream, playing from running time 1s.
        let segment = Segment {
            base: 1_000_000_000,
            start: 10_000_000_000,
            time: 5_000_000_000,
            ..Segment::new()
        };

        // running-time: base + (pts - start) = 1s + 2s.
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_time_mode(TimeMode::RunningTime);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::Segment(segment), &mut sink)
            .await
            .unwrap();
        let out = drawn(&mut ov, 12_000_000_000).await;
        assert_eq!(out, reference(64, 24, "00:00:03.000", &style));

        // stream-time: time + (pts - start) = 5s + 2s.
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_time_mode(TimeMode::StreamTime);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        ov.process(PipelinePacket::Segment(segment), &mut sink)
            .await
            .unwrap();
        let out = drawn(&mut ov, 12_000_000_000).await;
        assert_eq!(out, reference(64, 24, "00:00:07.000", &style));

        // elapsed-running-time: the first frame reads zero, the next its delta.
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_time_mode(TimeMode::ElapsedRunningTime);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        ov.process(PipelinePacket::Segment(segment), &mut sink)
            .await
            .unwrap();
        let first = drawn(&mut ov, 12_000_000_000).await;
        assert_eq!(first, reference(64, 24, "00:00:00.000", &style));
        let second = drawn(&mut ov, 12_500_000_000).await;
        assert_eq!(second, reference(64, 24, "00:00:00.500", &style));
    }

    #[tokio::test]
    async fn count_and_offset_modes_draw_integers() {
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_time_mode(TimeMode::BufferCount);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        assert_eq!(drawn(&mut ov, 0).await, reference(64, 24, "0", &style));
        assert_eq!(drawn(&mut ov, 1).await, reference(64, 24, "1", &style));

        // buffer-offset at 30 fps: 2s of running time is frame 60.
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_time_mode(TimeMode::BufferOffset);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        assert_eq!(
            drawn(&mut ov, 2_000_000_000).await,
            reference(64, 24, "60", &style)
        );
    }

    #[test]
    fn frame_number_scales_by_the_framerate() {
        assert_eq!(frame_number(1_000_000_000, Some(Rate::Fixed(25 << 16))), 25);
        // 29.97 fps in Q16: 1s is 29 whole frames.
        let q16 = ((30_000u64 << 16) / 1001) as u32;
        assert_eq!(frame_number(1_000_000_000, Some(Rate::Fixed(q16))), 29);
        assert_eq!(frame_number(1_000_000_000, Some(Rate::Any)), 0);
        assert_eq!(
            frame_number(u64::MAX, Some(Rate::Fixed(60 << 16))),
            1_106_804_644_422
        );
    }

    /// Epoch + elapsed, rendered through the date path.
    fn dated(epoch: &str, elapsed_secs: i64, fmt: &str) -> String {
        let secs = parse_epoch(epoch).expect("epoch parses") + elapsed_secs;
        strftime_utc_secs(secs, fmt)
    }

    #[test]
    fn dates_count_from_the_epoch() {
        // The 1900 default epoch is before 1970, so the seconds are negative.
        assert_eq!(
            parse_epoch(DEFAULT_DATETIME_EPOCH),
            Some(DEFAULT_EPOCH_SECS)
        );
        assert_eq!(dated("1900-01-01", 0, "%F %T"), "1900-01-01 00:00:00");
        assert_eq!(dated("1900-01-01", 3661, "%F %T"), "1900-01-01 01:01:01");
        // Rollover past midnight moves to the next day.
        assert_eq!(dated("1900-01-01", 86_399, "%F %T"), "1900-01-01 23:59:59");
        assert_eq!(dated("1900-01-01", 86_400, "%F %T"), "1900-01-02 00:00:00");
        // A time-of-day epoch is honoured, and a custom format renders date
        // codes.
        assert_eq!(
            dated("2026-08-04 13:45:07", 0, "%Y/%m/%d %H:%M:%S %a %b %j"),
            "2026/08/04 13:45:07 Tue Aug 216"
        );
    }

    #[test]
    fn leap_years_follow_the_gregorian_rule() {
        // 1900 is divisible by 100 but not 400: no Feb 29.
        assert_eq!(dated("1900-02-28", 86_400, "%F"), "1900-03-01");
        // 2000 is divisible by 400: Feb 29 exists.
        assert_eq!(dated("2000-02-28", 86_400, "%F"), "2000-02-29");
        assert_eq!(dated("2000-02-28", 2 * 86_400, "%F"), "2000-03-01");
        // A leap day mid-cycle, and the ordinal day it shifts.
        assert_eq!(dated("1904-02-28", 86_400, "%F %j"), "1904-02-29 060");
        assert_eq!(dated("1900-02-28", 86_400, "%F %j"), "1900-03-01 060");
    }

    #[test]
    fn epoch_parsing_rejects_garbage() {
        assert!(parse_epoch("not a date").is_none());
        assert!(parse_epoch("1900-01").is_none());
        assert!(parse_epoch("1900-13-01").is_none());
        assert!(
            parse_epoch("1900-02-29").is_none(),
            "1900 is not a leap year"
        );
        assert!(parse_epoch("2000-02-30").is_none());
        assert!(parse_epoch("1900-01-01 24:00:00").is_none());
        assert!(parse_epoch("1900-01-01 00:00").is_none());
        assert!(parse_epoch("").is_none());
        // ISO 'T' separator is accepted, same instant as the space form.
        assert_eq!(
            parse_epoch("1900-01-01T06:00:00"),
            parse_epoch("1900-01-01 06:00:00")
        );
    }

    #[tokio::test]
    async fn dates_are_burnt_in_when_enabled() {
        let style = OverlayStyle {
            scale: 1,
            ..Default::default()
        };
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_show_times_as_dates(true);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        // The default epoch plus the frame's PTS of 1h01m01s.
        let out = drawn(&mut ov, 3_661_000_000_000).await;
        assert_eq!(out, reference(64, 24, "1900-01-01 01:01:01", &style));

        // A count mode stays an integer: it is not a time to date.
        let mut ov = TimeOverlay::new()
            .with_style(style.clone())
            .with_show_times_as_dates(true)
            .with_time_mode(TimeMode::BufferCount);
        ov.configure_pipeline(&rgba_caps(64, 24)).unwrap();
        assert_eq!(drawn(&mut ov, 0).await, reference(64, 24, "0", &style));
    }

    /// Cross-check the formatter against GNU `date`, the reference for the same
    /// civil arithmetic. Skipped (with a reason) where `date` is absent or not
    /// GNU, since only GNU accepts the "<date> UTC + N seconds" form.
    #[test]
    fn matches_gnu_date() {
        extern crate std;
        use std::eprintln;
        use std::process::Command;

        let probe = Command::new("date").arg("--version").output();
        let gnu = match &probe {
            Ok(o) => String::from_utf8_lossy(&o.stdout).contains("GNU coreutils"),
            Err(_) => false,
        };
        if !gnu {
            eprintln!("skipping: no GNU `date` on this host");
            return;
        }

        for (epoch, offset) in [
            ("1900-01-01", 3661i64),
            ("1900-01-01", 86_400),
            ("1900-02-28", 86_400),
            ("2000-02-28", 86_400),
            ("1969-12-31 23:00:00", 7200),
            ("2026-08-04 13:45:07", 0),
        ] {
            let spec = format!("{epoch} UTC + {offset} seconds");
            let out = Command::new("date")
                .args(["-u", "-d", &spec, "+%F %T"])
                .output()
                .expect("date runs");
            let want = String::from_utf8_lossy(&out.stdout).trim().to_string();
            assert_eq!(dated(epoch, offset, "%F %T"), want, "for {spec}");
        }
    }

    #[test]
    fn properties_round_trip() {
        let mut ov = TimeOverlay::new();
        ov.set_property("time-mode", PropValue::Str("running-time".into()))
            .unwrap();
        assert_eq!(
            ov.get_property("time-mode"),
            Some(PropValue::Str("running-time".into()))
        );
        assert!(ov
            .set_property("time-mode", PropValue::Str("time-code".into()))
            .is_err());
    }

    #[test]
    fn date_properties_round_trip() {
        let mut ov = TimeOverlay::new();
        for name in ["show-times-as-dates", "datetime-format", "datetime-epoch"] {
            assert!(
                TIMEOVERLAY_PROPS.iter().any(|p| p.name == name),
                "{name} is declared"
            );
        }
        // gst defaults.
        assert_eq!(
            ov.get_property("show-times-as-dates"),
            Some(PropValue::Bool(false))
        );
        assert_eq!(
            ov.get_property("datetime-format"),
            Some(PropValue::Str("%F %T".into()))
        );
        assert_eq!(
            ov.get_property("datetime-epoch"),
            Some(PropValue::Str("1900-01-01".into()))
        );

        ov.set_property("show-times-as-dates", PropValue::Bool(true))
            .unwrap();
        ov.set_property("datetime-format", PropValue::Str("%H:%M".into()))
            .unwrap();
        ov.set_property(
            "datetime-epoch",
            PropValue::Str("2026-08-04 13:45:07".into()),
        )
        .unwrap();
        assert_eq!(
            ov.get_property("show-times-as-dates"),
            Some(PropValue::Bool(true))
        );
        assert_eq!(
            ov.get_property("datetime-format"),
            Some(PropValue::Str("%H:%M".into()))
        );
        assert_eq!(
            ov.get_property("datetime-epoch"),
            Some(PropValue::Str("2026-08-04 13:45:07".into()))
        );
        assert_eq!(ov.text_for(&plain_frame()), "13:45");

        // A bad epoch is refused and leaves the accepted one in place.
        assert!(ov
            .set_property("datetime-epoch", PropValue::Str("yesterday".into()))
            .is_err());
        assert_eq!(
            ov.get_property("datetime-epoch"),
            Some(PropValue::Str("2026-08-04 13:45:07".into()))
        );
        assert!(TimeOverlay::new().with_datetime_epoch("yesterday").is_err());
    }

    #[test]
    #[cfg(feature = "metadata")]
    fn a_carried_timecode_is_drawn_instead_of_the_pts() {
        use g2g_core::meta::TimecodeMeta;
        let mut ov = TimeOverlay::new();
        // No meta: the PTS, as before.
        assert_eq!(
            ov.text_for(&white_frame(1, 1, 2_500_000_000)),
            "00:00:02.500"
        );

        let mut f = white_frame(1, 1, 2_500_000_000);
        f.meta.attach(TimecodeMeta {
            hours: 10,
            minutes: 59,
            seconds: 58,
            frames: 29,
            drop_frame: true,
            framerate_q16: None,
        });
        // The source's own count wins, with the drop-frame `;` separator.
        assert_eq!(ov.text_for(&f), "10:59:58;29");

        // An explicit clock mode still asks for that clock, not the timecode.
        let mut ov = TimeOverlay::new().with_time_mode(TimeMode::BufferCount);
        assert_eq!(ov.text_for(&f), "0");
    }
}
