//! Element-granular logging facade (M179), the `GST_DEBUG` analog.
//!
//! A hand-rolled `no_std` logging layer: levels, per-category thresholds, and a
//! pluggable sink, so an element emits a record only when its category is enabled
//! and the record is routed wherever the host installed a sink (stderr on `std`,
//! a UART / RTT writer on an RTOS). It pulls no external logging crate, matching
//! the `no_std + alloc` baseline.
//!
//! **Categories and instances.** A log record carries a `category` (the element
//! *type*, e.g. `"opusenc"`, the GStreamer `GST_DEBUG_CATEGORY` analog) and an
//! optional `instance` name (the element *instance*, e.g. `"opusenc0"`, the
//! `<object>` in a GStreamer log line). Filtering is per category; the instance
//! is for disambiguation in the output. An element exposes both by implementing
//! [`LogSource`]; the runner logs about an element via a [`Target`].
//!
//! **Filtering.** [`configure`] parses a `GST_DEBUG`-style spec
//! (`"*:warning,opusenc:debug"`): `*:LEVEL` (or a bare `LEVEL`) sets the default
//! threshold, `name:LEVEL` overrides one category, and a `name` with `*` / `?`
//! wildcards overrides every matching category (`*sink*:5`; an exact override
//! wins over a glob). A message at `level` is emitted
//! when `level <= threshold`. The common no-override case is checked against an
//! atomic without locking, so a disabled `g2g_trace!` in a hot loop is cheap.
//!
//! **Macros.** [`g2g_error!`](crate::g2g_error) / [`g2g_warn!`](crate::g2g_warn) /
//! [`g2g_fixme!`](crate::g2g_fixme) / [`g2g_info!`](crate::g2g_info) /
//! [`g2g_debug!`](crate::g2g_debug) / [`g2g_log!`](crate::g2g_log) /
//! [`g2g_trace!`](crate::g2g_trace) take a [`LogSource`] then a
//! `format_args!` message; they check the threshold *before* formatting.
//! [`g2g_log_fields!`](crate::g2g_log_fields) adds structured [`LogField`]s a sink can render or ship
//! without re-parsing the message.
//!
//! **Timestamps.** Core has no clock, so a record's `timestamp_ns` is filled
//! from a host-installed [`set_time_source`] (on `std`, [`init_from_env`]
//! installs the UNIX-epoch one) and is `None` otherwise.
//!
//! **Sinks.** [`StderrSink`] (`std`), `TracingSink` (`tracing` feature), and
//! [`RingSink`], a bounded in-memory flight recorder for postmortem dumps.

#[cfg(feature = "std")]
extern crate std;

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use spin::Mutex;

/// Severity of a log record, ordered most-severe (`Error`) to least (`Trace`),
/// mirroring GStreamer's debug levels (minus `MEMDUMP`). `Off` disables a
/// category. The discriminants match GStreamer's numeric levels so a
/// `G2G_DEBUG=opusenc:5` numeric spec reads the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    /// No logging for this category.
    Off = 0,
    /// A fatal or recoverable error.
    Error = 1,
    /// A warning: something unexpected but handled.
    Warn = 2,
    /// A known-incomplete code path (GStreamer's `FIXME`).
    Fixme = 3,
    /// High-level informational lifecycle messages.
    Info = 4,
    /// Detailed debugging messages.
    Debug = 5,
    /// Very frequent messages (per-buffer scope).
    Log = 6,
    /// The most verbose (per-byte / per-iteration) tracing.
    Trace = 7,
}

impl LogLevel {
    /// The uppercase label used in a log line and accepted by [`parse`](Self::parse).
    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Off => "OFF",
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Fixme => "FIXME",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Log => "LOG",
            LogLevel::Trace => "TRACE",
        }
    }

    /// Parse a level from a name (case-insensitive, `WARNING` also accepted) or a
    /// `0..=7` number, as used in a `G2G_DEBUG` spec. `None` if unrecognized.
    pub fn parse(s: &str) -> Option<LogLevel> {
        let s = s.trim();
        if let Ok(n) = s.parse::<u8>() {
            return Self::from_u8(n);
        }
        Some(match () {
            _ if s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("none") => LogLevel::Off,
            _ if s.eq_ignore_ascii_case("error") => LogLevel::Error,
            _ if s.eq_ignore_ascii_case("warn") || s.eq_ignore_ascii_case("warning") => {
                LogLevel::Warn
            }
            _ if s.eq_ignore_ascii_case("fixme") => LogLevel::Fixme,
            _ if s.eq_ignore_ascii_case("info") => LogLevel::Info,
            _ if s.eq_ignore_ascii_case("debug") => LogLevel::Debug,
            _ if s.eq_ignore_ascii_case("log") => LogLevel::Log,
            _ if s.eq_ignore_ascii_case("trace") => LogLevel::Trace,
            _ => return None,
        })
    }

    /// The level for a numeric value `0..=7`, else `None`.
    pub fn from_u8(n: u8) -> Option<LogLevel> {
        Some(match n {
            0 => LogLevel::Off,
            1 => LogLevel::Error,
            2 => LogLevel::Warn,
            3 => LogLevel::Fixme,
            4 => LogLevel::Info,
            5 => LogLevel::Debug,
            6 => LogLevel::Log,
            7 => LogLevel::Trace,
            _ => return None,
        })
    }
}

/// The short type name of `T` (the last `::` segment of
/// [`core::any::type_name`]), used as the default log category for an element so
/// every element type gets a filtering key for free (e.g. `"OpusEnc"`). Still a
/// `&'static str` (a slice into the static type name).
pub fn short_type_name<T: ?Sized>() -> &'static str {
    let full = core::any::type_name::<T>();
    // Strip generic parameters first (`Foo<Bar>` -> `Foo`); otherwise the last
    // `::` segment is the parameter's path tail (e.g. `SystemClock>`), not the
    // element type's own name.
    let base = full.split_once('<').map_or(full, |(head, _)| head);
    match base.rsplit("::").next() {
        Some(s) if !s.is_empty() => s,
        _ => base,
    }
}

/// A thing that can be logged about: its [`category`](Self::log_category) (type)
/// and optional [`instance`](Self::log_instance) name. Elements implement this so
/// the logging macros pick up both from `self`; the runner uses [`Target`].
pub trait LogSource {
    /// The element type's category, e.g. `"opusenc"`, the filtering key.
    fn log_category(&self) -> &'static str;
    /// The element instance name, e.g. `"opusenc0"`, for the log line. Default
    /// none (filtering is by category regardless).
    fn log_instance(&self) -> Option<&str> {
        None
    }
    /// A per-instance category override (M845), replacing the type category for
    /// *both* filtering and output, so `G2G_DEBUG=my-cat:debug` (and globs) key
    /// off the override. Default none: the type name is the category. An
    /// element stores one in a [`LogName`] and returns it here.
    fn log_category_override(&self) -> Option<&str> {
        None
    }
}

/// The per-instance log identity an element stores when it logs about itself:
/// the runner-assigned instance name plus an optional category override.
/// Elements hold one, feed it from `set_instance_name` / `set_log_category`, and
/// return its two accessors from their [`LogSource`] impl.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LogName {
    instance: Option<String>,
    category: Option<String>,
}

impl LogName {
    pub const fn new() -> Self {
        Self {
            instance: None,
            category: None,
        }
    }

    /// Store the runner-assigned instance name.
    pub fn set_instance(&mut self, name: String) {
        self.instance = Some(name);
    }

    /// Override the log category for this instance.
    pub fn set_category(&mut self, category: String) {
        self.category = Some(category);
    }

    pub fn instance(&self) -> Option<&str> {
        self.instance.as_deref()
    }

    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }
}

/// A standalone [`LogSource`] for logging about a named element from outside it
/// (the runner naming `<category>N`), or for an ad-hoc log site.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub category: &'static str,
    pub instance: Option<&'a str>,
}

impl<'a> Target<'a> {
    /// A target with a category and an instance name.
    pub fn named(category: &'static str, instance: &'a str) -> Self {
        Self {
            category,
            instance: Some(instance),
        }
    }

    /// A target with only a category (no instance name).
    pub fn category(category: &'static str) -> Self {
        Self {
            category,
            instance: None,
        }
    }
}

impl LogSource for Target<'_> {
    fn log_category(&self) -> &'static str {
        self.category
    }
    fn log_instance(&self) -> Option<&str> {
        self.instance
    }
}

/// Per-category instance counter, shared by every runner so an element is named
/// and logged the same way whichever one drives it (M842). Hands out
/// `<category>N` names (the GStreamer `videotestsrc0` convention) and emits the
/// "added to pipeline" lifecycle line.
#[derive(Debug, Default)]
pub struct InstanceNamer {
    counts: Vec<(&'static str, u32)>,
    reserved: Vec<String>,
}

impl InstanceNamer {
    pub fn new() -> Self {
        Self::default()
    }

    /// A namer that counts past `reserved`, the names the graph already carries
    /// from its launch line's `name=`. Without this an auto-named instance can
    /// land on a name a user chose for another one, and two instances answering
    /// to the same name make either unaddressable.
    pub fn with_reserved(reserved: impl IntoIterator<Item = String>) -> Self {
        Self {
            counts: Vec::new(),
            reserved: reserved.into_iter().collect(),
        }
    }

    /// Name an element instance and log its addition, returning the name.
    /// `explicit` is a launch line's `name=`: it is taken verbatim and, as in
    /// gst-launch, does not consume a number, so auto-named siblings of the same
    /// category keep counting from 0.
    pub fn add(&mut self, category: &'static str, explicit: Option<&str>) -> String {
        let name = match explicit {
            Some(n) => String::from(n),
            None => loop {
                let n = match self.counts.iter_mut().find(|(c, _)| *c == category) {
                    Some(e) => {
                        let v = e.1;
                        e.1 += 1;
                        v
                    }
                    None => {
                        self.counts.push((category, 1));
                        0
                    }
                };
                let candidate = alloc::format!("{category}{n}");
                if !self.reserved.contains(&candidate) {
                    break candidate;
                }
            },
        };
        crate::g2g_info!(Target::named(category, &name), "added to pipeline");
        name
    }
}

// Forward through references so the logging macros accept `self` (a `&Self` or
// `&mut Self` inside a method) and `&target` uniformly: the macro passes `&$src`
// and type inference picks the right blanket.
impl<T: LogSource + ?Sized> LogSource for &T {
    fn log_category(&self) -> &'static str {
        (**self).log_category()
    }
    fn log_instance(&self) -> Option<&str> {
        (**self).log_instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        (**self).log_category_override()
    }
}

impl<T: LogSource + ?Sized> LogSource for &mut T {
    fn log_category(&self) -> &'static str {
        (**self).log_category()
    }
    fn log_instance(&self) -> Option<&str> {
        (**self).log_instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        (**self).log_category_override()
    }
}

/// A structured value on a log record (M845): the scalar kinds a sink can
/// render or ship (JSON, a tracing field) without knowing the log site. `Str`
/// borrows at the site, so a record a sink drops allocates nothing; the owned
/// form comes from [`into_owned`](Self::into_owned).
#[derive(Debug, Clone, PartialEq)]
pub enum LogValue<'a> {
    Str(Cow<'a, str>),
    Int(i64),
    Uint(u64),
    Float(f64),
    Bool(bool),
}

impl LogValue<'_> {
    /// Detach from the log site so the value can outlive it (what [`RingSink`]
    /// stores).
    pub fn into_owned(self) -> LogValue<'static> {
        match self {
            LogValue::Str(s) => LogValue::Str(Cow::Owned(s.into_owned())),
            LogValue::Int(v) => LogValue::Int(v),
            LogValue::Uint(v) => LogValue::Uint(v),
            LogValue::Float(v) => LogValue::Float(v),
            LogValue::Bool(v) => LogValue::Bool(v),
        }
    }
}

impl core::fmt::Display for LogValue<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LogValue::Str(s) => f.write_str(s),
            LogValue::Int(v) => write!(f, "{v}"),
            LogValue::Uint(v) => write!(f, "{v}"),
            LogValue::Float(v) => write!(f, "{v}"),
            LogValue::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl<'a> From<&'a str> for LogValue<'a> {
    fn from(v: &'a str) -> Self {
        LogValue::Str(Cow::Borrowed(v))
    }
}

impl From<String> for LogValue<'static> {
    fn from(v: String) -> Self {
        LogValue::Str(Cow::Owned(v))
    }
}

impl From<bool> for LogValue<'_> {
    fn from(v: bool) -> Self {
        LogValue::Bool(v)
    }
}

macro_rules! log_value_from {
    ($variant:ident, $($ty:ty),+) => {$(
        impl From<$ty> for LogValue<'_> {
            fn from(v: $ty) -> Self {
                LogValue::$variant(v.into())
            }
        }
    )+};
}
log_value_from!(Int, i8, i16, i32, i64);
log_value_from!(Uint, u8, u16, u32, u64);
log_value_from!(Float, f32, f64);

impl From<usize> for LogValue<'_> {
    fn from(v: usize) -> Self {
        LogValue::Uint(v as u64)
    }
}

/// One structured key/value on a log record.
#[derive(Debug, Clone, PartialEq)]
pub struct LogField<'a> {
    pub key: Cow<'a, str>,
    pub value: LogValue<'a>,
}

impl<'a> LogField<'a> {
    pub fn new(key: impl Into<Cow<'a, str>>, value: impl Into<LogValue<'a>>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }

    /// Detach from the log site (see [`LogValue::into_owned`]).
    pub fn into_owned(self) -> LogField<'static> {
        LogField {
            key: Cow::Owned(self.key.into_owned()),
            value: self.value.into_owned(),
        }
    }
}

/// One log record handed to a [`LogSink`]. The message is `format_args!` so a
/// sink that drops the record (or buffers selectively) pays no formatting cost;
/// `fields` carries the same information structured, so a sink renders or ships
/// it without re-parsing the message.
#[derive(Debug)]
pub struct LogRecord<'a> {
    pub level: LogLevel,
    pub category: &'a str,
    pub instance: Option<&'a str>,
    /// Nanoseconds from the installed [`set_time_source`], `None` when the host
    /// installed none (the `no_std` default: core reads no clock itself).
    pub timestamp_ns: Option<u64>,
    pub fields: &'a [LogField<'a>],
    pub message: core::fmt::Arguments<'a>,
}

impl LogRecord<'_> {
    /// Copy the record (message formatted, fields and names owned) so it can be
    /// buffered past the log site, as [`RingSink`] does.
    pub fn to_owned_record(&self) -> OwnedLogRecord {
        OwnedLogRecord {
            level: self.level,
            category: self.category.to_string(),
            instance: self.instance.map(|s| s.to_string()),
            timestamp_ns: self.timestamp_ns,
            fields: self
                .fields
                .iter()
                .cloned()
                .map(LogField::into_owned)
                .collect(),
            message: alloc::format!("{}", self.message),
        }
    }
}

/// An owned [`LogRecord`], what a buffering sink stores and hands back.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedLogRecord {
    pub level: LogLevel,
    pub category: String,
    pub instance: Option<String>,
    pub timestamp_ns: Option<u64>,
    pub fields: Vec<LogField<'static>>,
    pub message: String,
}

impl OwnedLogRecord {
    /// The value of one structured field by key, `None` if absent.
    pub fn field(&self, key: &str) -> Option<&LogValue<'static>> {
        self.fields.iter().find(|f| f.key == key).map(|f| &f.value)
    }

    /// Hand a buffered record to `sink`, the reverse of
    /// [`to_owned_record`](LogRecord::to_owned_record). For replaying what a
    /// buffering sink captured once a real destination exists: the TUI diverts
    /// logging into a ring for its log pane, and the pane goes away with the
    /// alternate screen, so anything explaining a failure has to be played back
    /// onto stderr afterwards or it is lost.
    pub fn emit_to(&self, sink: &dyn LogSink) {
        sink.emit(&LogRecord {
            level: self.level,
            category: &self.category,
            instance: self.instance.as_deref(),
            timestamp_ns: self.timestamp_ns,
            fields: &self.fields,
            message: format_args!("{}", self.message),
        });
    }
}

/// A destination for log records. The host installs one via [`set_sink`]; without
/// one, records are dropped. `Send + Sync` so it lives in a global behind a lock.
pub trait LogSink: Send + Sync {
    fn emit(&self, record: &LogRecord<'_>);
}

/// The mutable filter configuration: a default threshold plus per-category
/// overrides. Pure (no globals), so it is unit-testable in isolation; the process
/// global is a thin wrapper over one of these.
#[derive(Debug, Clone)]
pub struct LogConfig {
    default: LogLevel,
    overrides: Vec<(String, LogLevel)>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl LogConfig {
    /// A config defaulting every category to `Error` (errors always surface; the
    /// host raises the level to see more).
    pub const fn new() -> Self {
        Self {
            default: LogLevel::Error,
            overrides: Vec::new(),
        }
    }

    /// The effective threshold for `category`: an exact override, else the first
    /// matching glob override (`*` / `?` wildcards, e.g. `G2G_DEBUG=*sink*:5`),
    /// else the default. Exact wins over glob regardless of spec order.
    pub fn level_for(&self, category: &str) -> LogLevel {
        for (k, v) in &self.overrides {
            if k == category {
                return *v;
            }
        }
        for (k, v) in &self.overrides {
            if k.contains(['*', '?']) && glob_match(k, category) {
                return *v;
            }
        }
        self.default
    }

    /// Whether a `level` message in `category` should be emitted.
    pub fn enabled(&self, category: &str, level: LogLevel) -> bool {
        level != LogLevel::Off && (level as u8) <= (self.level_for(category) as u8)
    }

    /// Set the default threshold (the `*:LEVEL` of a spec).
    pub fn set_default(&mut self, level: LogLevel) {
        self.default = level;
    }

    /// Override (or add) one category's threshold.
    pub fn set_category(&mut self, category: &str, level: LogLevel) {
        if let Some(e) = self.overrides.iter_mut().find(|(k, _)| k == category) {
            e.1 = level;
        } else {
            self.overrides.push((category.to_string(), level));
        }
    }

    /// Apply a `GST_DEBUG`-style spec: comma-separated `name:LEVEL` entries, with
    /// `*:LEVEL` or a bare `LEVEL` setting the default. Unparseable entries are
    /// skipped.
    pub fn parse_spec(&mut self, spec: &str) {
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match part.split_once(':') {
                Some((name, lvl)) => {
                    if let Some(level) = LogLevel::parse(lvl) {
                        if name.trim() == "*" {
                            self.set_default(level);
                        } else {
                            self.set_category(name.trim(), level);
                        }
                    }
                }
                None => {
                    if let Some(level) = LogLevel::parse(part) {
                        self.set_default(level);
                    }
                }
            }
        }
    }

    fn has_overrides(&self) -> bool {
        !self.overrides.is_empty()
    }
}

/// Minimal glob over ASCII category names: `*` matches any run (including
/// empty), `?` exactly one byte. Iterative with single-star backtracking, so a
/// pathological pattern cannot recurse.
fn glob_match(pattern: &str, s: &str) -> bool {
    let (p, t) = (pattern.as_bytes(), s.as_bytes());
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    while ti < t.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some((pi, ti));
            pi += 1;
        } else if let Some((sp, sm)) = star {
            // Backtrack: let the last `*` consume one more byte.
            star = Some((sp, sm + 1));
            pi = sp + 1;
            ti = sm + 1;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

// Process-global filter + sink. `DEFAULT_LEVEL` / `HAS_OVERRIDES` mirror `CONFIG`
// so the common no-override `enabled` check reads an atomic without locking.
static DEFAULT_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Error as u8);
static HAS_OVERRIDES: AtomicBool = AtomicBool::new(false);
static CONFIG: Mutex<LogConfig> = Mutex::new(LogConfig::new());
#[allow(clippy::type_complexity)]
static SINK: Mutex<Option<Box<dyn LogSink>>> = Mutex::new(None);

fn sync_caches(cfg: &LogConfig) {
    DEFAULT_LEVEL.store(cfg.default as u8, Ordering::Relaxed);
    HAS_OVERRIDES.store(cfg.has_overrides(), Ordering::Relaxed);
}

/// Whether a `level` message in `category` is enabled by the global config. The
/// macros call this before formatting; a hot disabled site costs one atomic load.
pub fn enabled(category: &str, level: LogLevel) -> bool {
    if matches!(level, LogLevel::Off) {
        return false;
    }
    let lvl = level as u8;
    if HAS_OVERRIDES.load(Ordering::Relaxed) {
        lvl <= CONFIG.lock().level_for(category) as u8
    } else {
        lvl <= DEFAULT_LEVEL.load(Ordering::Relaxed)
    }
}

/// Emit a record to the installed sink (no-op if none). Called by the macros
/// after the [`enabled`] check; a direct caller should gate on [`enabled`] too.
pub fn emit(
    category: &str,
    instance: Option<&str>,
    level: LogLevel,
    message: core::fmt::Arguments<'_>,
) {
    emit_fields(category, instance, level, &[], message);
}

/// [`emit`] with structured fields attached (M845).
pub fn emit_fields(
    category: &str,
    instance: Option<&str>,
    level: LogLevel,
    fields: &[LogField<'_>],
    message: core::fmt::Arguments<'_>,
) {
    if let Some(sink) = SINK.lock().as_deref() {
        sink.emit(&LogRecord {
            level,
            category,
            instance,
            timestamp_ns: timestamp_now(),
            fields,
            message,
        });
    }
}

/// A nanosecond timestamp source the host installs (see [`set_time_source`]).
/// Any epoch, as long as it is consistent: a sink reports it verbatim.
pub type TimeSource = fn() -> u64;

static TIME_SOURCE: Mutex<Option<TimeSource>> = Mutex::new(None);
// Mirrors `TIME_SOURCE.is_some()` so the unset case skips the lock.
static HAS_TIME_SOURCE: AtomicBool = AtomicBool::new(false);

/// Install the clock that stamps records. Core reads no clock of its own (the
/// `no_std` baseline has none), so without this every record's `timestamp_ns`
/// is `None`. On `std`, [`init_from_env`] installs [`unix_time_source`].
pub fn set_time_source(source: TimeSource) {
    *TIME_SOURCE.lock() = Some(source);
    HAS_TIME_SOURCE.store(true, Ordering::Relaxed);
}

/// The current timestamp from the installed source, `None` if none.
pub fn timestamp_now() -> Option<u64> {
    if !HAS_TIME_SOURCE.load(Ordering::Relaxed) {
        return None;
    }
    let source = *TIME_SOURCE.lock();
    source.map(|f| f())
}

/// Nanoseconds since the UNIX epoch, the `std` [`TimeSource`].
#[cfg(feature = "std")]
pub fn unix_time_source() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// Install (replace) the global log sink. Without one, records are dropped.
pub fn set_sink(sink: Box<dyn LogSink>) {
    *SINK.lock() = Some(sink);
}

/// Set the global default threshold (applies to categories with no override).
pub fn set_default_level(level: LogLevel) {
    let mut cfg = CONFIG.lock();
    cfg.set_default(level);
    sync_caches(&cfg);
}

/// Override one category's global threshold.
pub fn set_category_level(category: &str, level: LogLevel) {
    let mut cfg = CONFIG.lock();
    cfg.set_category(category, level);
    sync_caches(&cfg);
}

/// Apply a `GST_DEBUG`-style spec to the global config (see
/// [`LogConfig::parse_spec`]).
pub fn configure(spec: &str) {
    let mut cfg = CONFIG.lock();
    cfg.parse_spec(spec);
    sync_caches(&cfg);
}

/// Reset the global config to defaults and remove the sink and time source
/// (for tests).
pub fn reset() {
    let mut cfg = CONFIG.lock();
    *cfg = LogConfig::new();
    sync_caches(&cfg);
    *SINK.lock() = None;
    *TIME_SOURCE.lock() = None;
    HAS_TIME_SOURCE.store(false, Ordering::Relaxed);
}

/// A bounded in-memory [`LogSink`] (M845), the flight recorder: it keeps the
/// most recent `capacity` records and overwrites the oldest, so a postmortem
/// dump on a target with no live log stream (an RTOS board, a crashed field
/// unit) still shows what led up to the fault. `no_std + alloc`: records are
/// stored owned (see [`OwnedLogRecord`]), so the buffer's memory is bounded by
/// capacity times record size, not by run length.
///
/// Cloning shares one buffer: install a clone as the sink and keep the original
/// to [`snapshot`](Self::snapshot) or [`drain`](Self::drain) it.
///
/// ```
/// # use g2g_core::log::{self, LogLevel, RingSink, Target};
/// # use g2g_core::g2g_error;
/// let ring = RingSink::new(64);
/// log::set_sink(Box::new(ring.clone()));
/// log::set_default_level(LogLevel::Warn);
/// g2g_error!(Target::category("demo"), "boom");
/// assert_eq!(ring.drain().len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct RingSink {
    inner: Arc<Mutex<Ring>>,
}

#[derive(Debug)]
struct Ring {
    capacity: usize,
    records: VecDeque<OwnedLogRecord>,
    dropped: u64,
}

impl RingSink {
    /// A recorder holding at most `capacity` records (clamped to at least one).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(Mutex::new(Ring {
                capacity,
                records: VecDeque::with_capacity(capacity),
                dropped: 0,
            })),
        }
    }

    /// Copy the buffered records, oldest first, leaving them in place.
    pub fn snapshot(&self) -> Vec<OwnedLogRecord> {
        self.inner.lock().records.iter().cloned().collect()
    }

    /// Take the buffered records, oldest first, emptying the buffer.
    pub fn drain(&self) -> Vec<OwnedLogRecord> {
        self.inner.lock().records.drain(..).collect()
    }

    /// Records currently buffered.
    pub fn len(&self) -> usize {
        self.inner.lock().records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The buffer's fixed capacity.
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity
    }

    /// How many records the recorder has overwritten since it was created: a
    /// non-zero count means the dump is a tail, not the whole run.
    pub fn overwritten(&self) -> u64 {
        self.inner.lock().dropped
    }
}

impl LogSink for RingSink {
    fn emit(&self, record: &LogRecord<'_>) {
        let mut ring = self.inner.lock();
        if ring.records.len() == ring.capacity {
            ring.records.pop_front();
            ring.dropped += 1;
        }
        ring.records.push_back(record.to_owned_record());
    }
}

/// The reserved log category the caps-negotiation explainer emits under
/// (DESIGN.md 4.20a). Not an element type: it names the solver's narration, so
/// `G2G_DEBUG=caps:debug` (or the `G2G_CAPS_TRACE` shortcut) turns it on
/// independent of element logging.
pub const CAPS_CATEGORY: &str = "caps";

/// The reserved log category the runners emit under. Not an element type: it
/// names the runner's own narration, notably which element instance raised the
/// error that ended a run.
pub const RUNTIME_CATEGORY: &str = "runtime";

/// Name the element instance whose arm returned `err`, at error level (so it
/// prints without `G2G_DEBUG`). The runners call this at the one point they pick
/// a run's reported error, since [`G2gError`](crate::G2gError) itself carries no
/// element identity. An unnamed arm (a plain broadcast tee, the coordinator)
/// stays quiet: the caller already prints the error.
pub fn report_element_failure(name: Option<&str>, err: &crate::G2gError) {
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        return;
    };
    crate::g2g_error!(
        Target::category(RUNTIME_CATEGORY),
        "pipeline error in {name}: {err:?}"
    );
}

/// Map a filesystem / device I/O failure onto [`G2gError`](crate::G2gError),
/// which carries an errno rather than the `io::Error` itself. The single mapping
/// every path-opening element and the flight-recorder dump share; prefer
/// [`path_io_err`], which also says which path failed.
#[cfg(feature = "std")]
pub fn io_err(e: std::io::Error) -> crate::G2gError {
    crate::G2gError::Hardware(crate::error::HardwareError::Io(
        e.raw_os_error().unwrap_or(0),
    ))
}

/// [`io_err`] plus an error log naming the file and the OS message. The errno in
/// `Hardware(Io)` alone does not say which path failed or what went wrong, so
/// every element that opens a path reports through here.
#[cfg(feature = "std")]
pub fn path_io_err<P: AsRef<std::path::Path>>(
    category: &'static str,
    verb: &str,
    path: P,
    e: std::io::Error,
) -> crate::G2gError {
    crate::g2g_error!(
        Target::category(category),
        "cannot {verb} {}: {e}",
        path.as_ref().display()
    );
    io_err(e)
}

/// Install the stderr sink and apply logging from the environment. The sink is
/// always installed, so ERROR-level diagnostics print by default; the
/// `G2G_DEBUG` environment variable (a `GST_DEBUG`-style spec) tunes thresholds
/// up from the default Error level. Also honors `G2G_CAPS_TRACE`
/// as a shortcut for the caps explainer: a boolean-ish value (`1` / `true` / `on`
/// / `yes`) raises the [`CAPS_CATEGORY`] to `Debug`, or a level name / number
/// (`debug`, `trace`, `7`) sets that verbosity, installing the stderr sink if
/// `G2G_DEBUG` did not. Call once at startup; the `g2g-launch` / `g2g-inspect`
/// binaries and apps invoke it.
#[cfg(feature = "std")]
pub fn init_from_env() {
    // Always install the stderr sink so ERROR-level diagnostics (notably the
    // caps-negotiation narration, which already runs on every failed solve) are
    // visible by default without opting in. The default threshold is Error
    // (LogConfig::new), so a normal run stays quiet; G2G_DEBUG only tunes it up.
    set_sink(Box::new(StderrSink));
    set_time_source(unix_time_source);
    if let Ok(spec) = std::env::var("G2G_DEBUG") {
        configure(&spec);
    }
    if let Ok(v) = std::env::var("G2G_CAPS_TRACE") {
        let v = v.trim();
        let enable = !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false");
        if enable {
            // A bare on-switch means Debug; a level name / number tunes it.
            let level = match v.to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => LogLevel::Debug,
                other => LogLevel::parse(other)
                    .filter(|l| *l != LogLevel::Off)
                    .unwrap_or(LogLevel::Debug),
            };
            set_category_level(CAPS_CATEGORY, level);
        }
    }
}

/// A [`LogSink`] that writes one line per record to stderr, in the shape
/// `LEVEL category <instance> message [k=v ...]` (the `<instance>` omitted when
/// unnamed, the `k=v` tail only when the record carries structured fields).
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct StderrSink;

#[cfg(feature = "std")]
impl LogSink for StderrSink {
    fn emit(&self, r: &LogRecord<'_>) {
        use core::fmt::Write;
        let mut tail = String::new();
        for f in r.fields {
            let _ = write!(tail, " {}={}", f.key, f.value);
        }
        match r.instance {
            Some(i) => {
                std::eprintln!(
                    "{:<5} {:<16} <{}> {}{}",
                    r.level.as_str(),
                    r.category,
                    i,
                    r.message,
                    tail
                )
            }
            None => std::eprintln!(
                "{:<5} {:<16} {}{}",
                r.level.as_str(),
                r.category,
                r.message,
                tail
            ),
        }
    }
}

/// A [`LogSink`] that forwards each record to the [`tracing`] crate, so a host
/// running a `tracing` subscriber (fmt, journald, OTLP / Jaeger, tokio-console)
/// receives g2g's logs in its existing observability pipeline. The g2g element
/// *category* and *instance* are emitted as `tracing` fields under a fixed
/// `g2g` target, and the message is forwarded lazily (it is only formatted if
/// the subscriber enables the event).
///
/// **Level mapping.** `tracing` has five levels to g2g's seven, so two pairs
/// collapse: `Fixme` maps to `WARN` and `Log` maps to `TRACE`. The original g2g
/// level is preserved verbatim in the `g2g_level` field, so nothing is lost,
/// the subscriber can still distinguish `FIXME` from `WARN`.
///
/// **Filtering.** With this sink installed, let the `tracing` subscriber own
/// filtering (e.g. `RUST_LOG=g2g=debug`) rather than g2g's per-category
/// thresholds, by raising g2g's default to pass everything through.
/// [`init_tracing`] does exactly that.
#[cfg(feature = "tracing")]
#[derive(Debug, Default)]
pub struct TracingSink;

#[cfg(feature = "tracing")]
impl LogSink for TracingSink {
    fn emit(&self, r: &LogRecord<'_>) {
        let category = r.category;
        let instance = r.instance.unwrap_or("");
        let g2g_level = r.level.as_str();
        let message = r.message;
        // `tracing::event!` needs a const level and target, so dispatch per
        // level. The message is passed as `format_args!`, so tracing formats it
        // only when the event is enabled by the subscriber.
        match r.level {
            LogLevel::Error => tracing::event!(
                target: "g2g", tracing::Level::ERROR,
                category, instance, g2g_level, "{message}"
            ),
            LogLevel::Warn | LogLevel::Fixme => tracing::event!(
                target: "g2g", tracing::Level::WARN,
                category, instance, g2g_level, "{message}"
            ),
            LogLevel::Info => tracing::event!(
                target: "g2g", tracing::Level::INFO,
                category, instance, g2g_level, "{message}"
            ),
            LogLevel::Debug => tracing::event!(
                target: "g2g", tracing::Level::DEBUG,
                category, instance, g2g_level, "{message}"
            ),
            LogLevel::Log | LogLevel::Trace => tracing::event!(
                target: "g2g", tracing::Level::TRACE,
                category, instance, g2g_level, "{message}"
            ),
            // `emit` is only reached for an enabled (non-`Off`) record.
            LogLevel::Off => {}
        }
    }
}

/// Route g2g's logging into the `tracing` ecosystem: install [`TracingSink`] and
/// raise the g2g default threshold to `Trace` so g2g stops filtering and the
/// installed `tracing` subscriber owns verbosity (e.g. `RUST_LOG=g2g=debug`).
/// Call once at startup, after setting up your subscriber. Records flow to
/// `tracing` under the `g2g` target with `category` / `instance` / `g2g_level`
/// fields.
#[cfg(feature = "tracing")]
pub fn init_tracing() {
    set_sink(Box::new(TracingSink));
    set_default_level(LogLevel::Trace);
}

/// Implementation hook for the logging macros: check the category threshold and,
/// when enabled, format and emit. Generic over `&S` so the macro can pass `&$src`
/// whether `$src` is `self` (a `&`/`&mut Self`) or a [`Target`] value (the
/// reference forwarding impls cover the extra indirection). Not called directly.
#[doc(hidden)]
pub fn __log<S: LogSource + ?Sized>(src: &S, level: LogLevel, args: core::fmt::Arguments<'_>) {
    __log_fields(src, level, &[], args)
}

/// [`__log`] with structured fields. Not called directly.
#[doc(hidden)]
pub fn __log_fields<S: LogSource + ?Sized>(
    src: &S,
    level: LogLevel,
    fields: &[LogField<'_>],
    args: core::fmt::Arguments<'_>,
) {
    // A per-instance override replaces the type category for filtering too, so
    // a G2G_DEBUG entry (or glob) written against the override matches.
    let category = match src.log_category_override() {
        Some(c) => c,
        None => src.log_category(),
    };
    if enabled(category, level) {
        emit_fields(category, src.log_instance(), level, fields, args);
    }
}

/// Log at `level` about a [`LogSource`], checking the category threshold before
/// formatting the message. Prefer the level-specific macros.
#[macro_export]
macro_rules! g2g_log_at {
    ($level:expr, $src:expr, $($arg:tt)+) => {
        $crate::log::__log(&$src, $level, ::core::format_args!($($arg)+))
    };
}

/// Log at `level` with structured fields plus the formatted message, so a sink
/// can render or ship the values without re-parsing the line:
/// `g2g_log_fields!(LogLevel::Info, self, ["width" => w, "height" => h],
/// "configured {w}x{h}")`. Field values are anything convertible into a
/// [`LogValue`] (strings, integers, floats, bools).
#[macro_export]
macro_rules! g2g_log_fields {
    ($level:expr, $src:expr, [$($k:expr => $v:expr),* $(,)?], $($arg:tt)+) => {
        $crate::log::__log_fields(
            &$src,
            $level,
            &[$($crate::log::LogField::new($k, $v)),*],
            ::core::format_args!($($arg)+),
        )
    };
}

/// `ERROR`-level log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_error {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Error, $src, $($arg)+) };
}
/// `WARN`-level log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_warn {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Warn, $src, $($arg)+) };
}
/// `FIXME`-level log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_fixme {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Fixme, $src, $($arg)+) };
}
/// `INFO`-level log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_info {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Info, $src, $($arg)+) };
}
/// `DEBUG`-level log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_debug {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Debug, $src, $($arg)+) };
}
/// `LOG`-level (per-buffer) log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_log {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Log, $src, $($arg)+) };
}
/// `TRACE`-level (most verbose) log about a [`LogSource`].
#[macro_export]
macro_rules! g2g_trace {
    ($src:expr, $($arg:tt)+) => { $crate::g2g_log_at!($crate::log::LogLevel::Trace, $src, $($arg)+) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::sync::Arc;

    /// What the TUI leans on at teardown: records buffered while logging was
    /// diverted into a ring can be handed to a real sink afterwards, or the only
    /// account of a failure dies with the screen that displayed it.
    #[test]
    fn a_buffered_record_replays_into_another_sink() {
        let ring = RingSink::new(4);
        ring.emit(&LogRecord {
            level: LogLevel::Error,
            category: "FileSink",
            instance: Some("FileSink0"),
            timestamp_ns: None,
            fields: &[],
            message: format_args!("reads host memory but got a Cuda frame"),
        });

        let replayed = RingSink::new(4);
        for record in ring.snapshot() {
            record.emit_to(&replayed);
        }

        let out = replayed.snapshot();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].level, LogLevel::Error);
        assert_eq!(out[0].category, "FileSink");
        assert_eq!(out[0].instance.as_deref(), Some("FileSink0"));
        assert_eq!(out[0].message, "reads host memory but got a Cuda frame");
    }

    #[test]
    fn short_type_name_strips_generics_and_path() {
        struct Inner;
        struct Outer<T>(core::marker::PhantomData<T>);
        assert_eq!(short_type_name::<Inner>(), "Inner");
        // A generic element keys on its own name, not the parameter's path tail.
        assert_eq!(short_type_name::<Outer<Inner>>(), "Outer");
    }

    #[test]
    fn level_parse_accepts_names_and_numbers() {
        assert_eq!(LogLevel::parse("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("WARNING"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("5"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("off"), Some(LogLevel::Off));
        assert_eq!(LogLevel::parse("nope"), None);
        assert_eq!(LogLevel::parse("9"), None);
    }

    #[test]
    fn config_filters_by_category_and_default() {
        let mut cfg = LogConfig::new(); // default Error
        assert!(cfg.enabled("opusenc", LogLevel::Error));
        assert!(
            !cfg.enabled("opusenc", LogLevel::Debug),
            "default Error hides Debug"
        );

        cfg.set_default(LogLevel::Warn);
        cfg.set_category("opusenc", LogLevel::Trace);
        // The override lets opusenc through at Trace; others stay at Warn.
        assert!(cfg.enabled("opusenc", LogLevel::Trace));
        assert!(cfg.enabled("opusenc", LogLevel::Debug));
        assert!(
            !cfg.enabled("videoscale", LogLevel::Info),
            "non-overridden uses default Warn"
        );
        assert!(cfg.enabled("videoscale", LogLevel::Warn));
        // Off is never enabled.
        cfg.set_category("muted", LogLevel::Off);
        assert!(!cfg.enabled("muted", LogLevel::Error));
    }

    #[test]
    fn parse_spec_sets_default_and_overrides() {
        let mut cfg = LogConfig::new();
        cfg.parse_spec("*:warning,opusenc:debug, videoscale:5");
        assert_eq!(cfg.level_for("opusenc"), LogLevel::Debug);
        assert_eq!(cfg.level_for("videoscale"), LogLevel::Debug);
        assert_eq!(cfg.level_for("anything-else"), LogLevel::Warn);
        // A bare level sets the default.
        let mut c2 = LogConfig::new();
        c2.parse_spec("info");
        assert_eq!(c2.level_for("x"), LogLevel::Info);
    }

    #[test]
    fn glob_overrides_match_categories() {
        let mut cfg = LogConfig::new();
        cfg.parse_spec("*:warning,*sink*:5,opus?nc:debug,waylandsink:error");
        // Glob hits every matching category...
        assert_eq!(cfg.level_for("filesink"), LogLevel::Debug);
        assert_eq!(cfg.level_for("sinkpad"), LogLevel::Debug);
        // ...`?` matches exactly one byte...
        assert_eq!(cfg.level_for("opusenc"), LogLevel::Debug);
        assert_eq!(cfg.level_for("opusnc"), LogLevel::Warn);
        // ...an exact override wins over a matching glob, in either spec order...
        assert_eq!(cfg.level_for("waylandsink"), LogLevel::Error);
        // ...and non-matches keep the default.
        assert_eq!(cfg.level_for("videoscale"), LogLevel::Warn);
    }

    #[test]
    fn glob_match_handles_edges() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", ""));
        assert!(glob_match("a*b*c", "a-long-b-run-c"));
        assert!(!glob_match("a*b*c", "a-long-b-run"));
        assert!(glob_match("??", "ab"));
        assert!(!glob_match("??", "a"));
        assert!(!glob_match("abc", "abd"));
    }

    /// One captured log record (level, category, instance, formatted message).
    type CapturedRecord = (LogLevel, String, Option<String>, String);
    /// A capturing sink for the global-path test.
    struct CaptureSink(Arc<Mutex<Vec<CapturedRecord>>>);
    impl LogSink for CaptureSink {
        fn emit(&self, r: &LogRecord<'_>) {
            self.0.lock().push((
                r.level,
                r.category.to_string(),
                r.instance.map(|s| s.to_string()),
                format!("{}", r.message),
            ));
        }
    }

    // Serializes the few tests that touch the process-global config / sink.
    static GLOBAL_GUARD: Mutex<()> = Mutex::new(());

    #[test]
    fn macros_respect_global_filtering_and_route_to_sink() {
        let _g = GLOBAL_GUARD.lock();
        reset();
        let captured = Arc::new(Mutex::new(Vec::new()));
        set_sink(Box::new(CaptureSink(captured.clone())));
        configure("*:warning,opusenc:debug");

        let enc = Target::named("opusenc", "opusenc0");
        let scale = Target::named("videoscale", "videoscale0");

        // opusenc is at DEBUG: a debug line is captured with the instance name.
        g2g_debug!(enc, "encoded {} bytes", 42);
        // videoscale is at the WARNING default: a debug line is filtered out.
        g2g_debug!(scale, "scaled a frame");
        // A warning on videoscale passes.
        g2g_warn!(scale, "odd dimension");

        let recs = captured.lock();
        assert_eq!(recs.len(), 2, "got: {recs:?}");
        assert_eq!(recs[0].0, LogLevel::Debug);
        assert_eq!(recs[0].1, "opusenc");
        assert_eq!(recs[0].2.as_deref(), Some("opusenc0"));
        assert_eq!(recs[0].3, "encoded 42 bytes");
        assert_eq!(recs[1].0, LogLevel::Warn);
        assert_eq!(recs[1].1, "videoscale");
        drop(recs);
        reset();
    }

    /// An element-shaped source: a fixed type category plus a settable
    /// per-instance name and category override (what [`LogName`] gives an
    /// element).
    struct FakeElement {
        name: LogName,
    }
    impl LogSource for FakeElement {
        fn log_category(&self) -> &'static str {
            "VideoFlip"
        }
        fn log_instance(&self) -> Option<&str> {
            self.name.instance()
        }
        fn log_category_override(&self) -> Option<&str> {
            self.name.category()
        }
    }

    #[test]
    fn category_override_replaces_the_type_category_for_filtering() {
        let _g = GLOBAL_GUARD.lock();
        reset();
        let captured = Arc::new(Mutex::new(Vec::new()));
        set_sink(Box::new(CaptureSink(captured.clone())));
        // The type category is off; only the override (and a glob covering it)
        // is enabled.
        configure("*:off,flip-a:debug,*-glob:info");

        let mut plain = FakeElement {
            name: LogName::new(),
        };
        plain.name.set_instance(String::from("VideoFlip0"));
        let mut renamed = FakeElement {
            name: LogName::new(),
        };
        renamed.name.set_instance(String::from("VideoFlip1"));
        renamed.name.set_category(String::from("flip-a"));
        let mut globbed = FakeElement {
            name: LogName::new(),
        };
        globbed.name.set_category(String::from("via-glob"));

        g2g_debug!(plain, "type category is off");
        g2g_debug!(renamed, "override is at debug");
        g2g_info!(globbed, "override matches the glob");

        let recs = captured.lock();
        assert_eq!(recs.len(), 2, "got: {recs:?}");
        // The override is the category the sink sees, not just the filter key.
        assert_eq!(recs[0].1, "flip-a");
        assert_eq!(recs[0].2.as_deref(), Some("VideoFlip1"));
        assert_eq!(recs[1].1, "via-glob");
        drop(recs);
        reset();
    }

    #[test]
    fn structured_fields_and_timestamp_reach_the_sink() {
        let _g = GLOBAL_GUARD.lock();
        reset();
        let owned: Arc<Mutex<Vec<OwnedLogRecord>>> = Arc::new(Mutex::new(Vec::new()));
        struct OwningSink(Arc<Mutex<Vec<OwnedLogRecord>>>);
        impl LogSink for OwningSink {
            fn emit(&self, r: &LogRecord<'_>) {
                self.0.lock().push(r.to_owned_record());
            }
        }
        set_sink(Box::new(OwningSink(owned.clone())));
        set_time_source(|| 42);
        configure("*:debug");

        let width = 1920u32;
        g2g_log_fields!(
            LogLevel::Info,
            Target::named("videoscale", "videoscale0"),
            ["width" => width, "height" => 1080u32, "format" => "NV12", "scaled" => true, "ratio" => 1.5f64],
            "configured {width}"
        );

        let recs = owned.lock();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        // The fields survive as typed values, so a sink renders or ships them
        // without re-parsing the message.
        assert_eq!(r.field("width"), Some(&LogValue::Uint(1920)));
        assert_eq!(r.field("height"), Some(&LogValue::Uint(1080)));
        assert_eq!(
            r.field("format"),
            Some(&LogValue::Str(Cow::Borrowed("NV12")))
        );
        assert_eq!(r.field("scaled"), Some(&LogValue::Bool(true)));
        assert_eq!(r.field("ratio"), Some(&LogValue::Float(1.5)));
        assert_eq!(r.field("missing"), None);
        assert_eq!(r.timestamp_ns, Some(42));
        assert_eq!(r.message, "configured 1920");
        assert_eq!(r.instance.as_deref(), Some("videoscale0"));
        drop(recs);
        reset();
    }

    #[test]
    fn ring_sink_keeps_the_newest_records_and_drains() {
        let _g = GLOBAL_GUARD.lock();
        reset();
        let ring = RingSink::new(3);
        set_sink(Box::new(ring.clone()));
        configure("*:debug");

        for i in 0..5 {
            g2g_info!(Target::category("demo"), "record {i}");
        }

        assert_eq!(ring.len(), 3, "bounded at capacity");
        assert_eq!(ring.capacity(), 3);
        assert_eq!(ring.overwritten(), 2, "two oldest were overwritten");
        let snap = ring.snapshot();
        let messages: Vec<&str> = snap.iter().map(|r| r.message.as_str()).collect();
        assert_eq!(messages, ["record 2", "record 3", "record 4"]);
        // A snapshot leaves the buffer intact; a drain empties it.
        assert_eq!(ring.len(), 3);
        let drained = ring.drain();
        assert_eq!(drained.len(), 3);
        assert!(ring.is_empty());
        // Records carry no timestamp when the host installed no time source.
        assert_eq!(drained[0].timestamp_ns, None);

        // The recorder keeps working after a drain.
        g2g_info!(Target::category("demo"), "after drain");
        assert_eq!(ring.snapshot()[0].message, "after drain");
        reset();
    }

    #[test]
    fn no_sink_drops_records_without_panic() {
        let _g = GLOBAL_GUARD.lock();
        reset();
        configure("*:trace");
        // No sink installed: emitting must be a harmless no-op.
        g2g_error!(Target::category("x"), "no sink, {}", "dropped");
        reset();
    }

    /// The `tracing` bridge forwards g2g records to the active `tracing`
    /// subscriber, carrying category / instance / original level as fields, with
    /// `Fixme` collapsing to `WARN` but preserved verbatim in `g2g_level`.
    #[cfg(feature = "tracing")]
    #[test]
    fn tracing_sink_forwards_records_to_subscriber() {
        use core::fmt::Write;
        use tracing::field::{Field, Visit};

        // A subscriber that captures each event as a flat "level target k=v ..." line.
        #[derive(Default)]
        struct Capture {
            events: Mutex<Vec<String>>,
        }
        struct Recorder<'a>(&'a mut String);
        impl Visit for Recorder<'_> {
            fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
                let _ = write!(self.0, "{}={:?} ", field.name(), value);
            }
            fn record_str(&mut self, field: &Field, value: &str) {
                let _ = write!(self.0, "{}={} ", field.name(), value);
            }
        }
        impl tracing::Subscriber for Capture {
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                let meta = event.metadata();
                let mut line = String::new();
                let _ = write!(line, "{} {} ", meta.level(), meta.target());
                event.record(&mut Recorder(&mut line));
                self.events.lock().push(line);
            }
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
        }

        let _g = GLOBAL_GUARD.lock();
        reset();
        init_tracing();

        let capture = Arc::new(Capture::default());
        tracing::subscriber::with_default(capture.clone(), || {
            let enc = Target::named("opusenc", "opusenc0");
            g2g_info!(enc, "encoded {} bytes", 42);
            g2g_fixme!(Target::category("videoscale"), "todo: odd dims");
        });

        let events = capture.events.lock();
        assert_eq!(events.len(), 2, "got: {events:?}");
        // INFO event carries category, instance, and the forwarded message.
        assert!(events[0].contains("INFO"), "{}", events[0]);
        assert!(events[0].contains("category=opusenc"), "{}", events[0]);
        assert!(events[0].contains("instance=opusenc0"), "{}", events[0]);
        assert!(events[0].contains("encoded 42 bytes"), "{}", events[0]);
        // FIXME collapses to WARN at the tracing level but is kept in g2g_level.
        assert!(events[1].contains("WARN"), "{}", events[1]);
        assert!(events[1].contains("g2g_level=FIXME"), "{}", events[1]);
        drop(events);
        reset();
    }
}
