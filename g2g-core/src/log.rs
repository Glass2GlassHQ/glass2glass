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
//! threshold, `name:LEVEL` overrides one category. A message at `level` is emitted
//! when `level <= threshold`. The common no-override case is checked against an
//! atomic without locking, so a disabled `g2g_trace!` in a hot loop is cheap.
//!
//! **Macros.** [`g2g_error!`] / [`g2g_warn!`] / [`g2g_fixme!`] / [`g2g_info!`] /
//! [`g2g_debug!`] / [`g2g_log!`] / [`g2g_trace!`] take a [`LogSource`] then a
//! `format_args!` message; they check the threshold *before* formatting.

#[cfg(feature = "std")]
extern crate std;

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
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
}

impl<T: LogSource + ?Sized> LogSource for &mut T {
    fn log_category(&self) -> &'static str {
        (**self).log_category()
    }
    fn log_instance(&self) -> Option<&str> {
        (**self).log_instance()
    }
}

/// One log record handed to a [`LogSink`]. The message is `format_args!` so a
/// sink that drops the record (or buffers selectively) pays no formatting cost.
#[derive(Debug)]
pub struct LogRecord<'a> {
    pub level: LogLevel,
    pub category: &'a str,
    pub instance: Option<&'a str>,
    pub message: core::fmt::Arguments<'a>,
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

    /// The effective threshold for `category`: its override, else the default.
    pub fn level_for(&self, category: &str) -> LogLevel {
        for (k, v) in &self.overrides {
            if k == category {
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
    if let Some(sink) = SINK.lock().as_deref() {
        sink.emit(&LogRecord {
            level,
            category,
            instance,
            message,
        });
    }
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

/// Reset the global config to defaults and remove the sink (for tests).
pub fn reset() {
    let mut cfg = CONFIG.lock();
    *cfg = LogConfig::new();
    sync_caches(&cfg);
    *SINK.lock() = None;
}

/// The reserved log category the caps-negotiation explainer emits under
/// (DESIGN.md 4.20a). Not an element type: it names the solver's narration, so
/// `G2G_DEBUG=caps:debug` (or the `G2G_CAPS_TRACE` shortcut) turns it on
/// independent of element logging.
pub const CAPS_CATEGORY: &str = "caps";

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
/// `LEVEL category <instance> message` (the `<instance>` omitted when unnamed).
#[cfg(feature = "std")]
#[derive(Debug, Default)]
pub struct StderrSink;

#[cfg(feature = "std")]
impl LogSink for StderrSink {
    fn emit(&self, r: &LogRecord<'_>) {
        match r.instance {
            Some(i) => {
                std::eprintln!(
                    "{:<5} {:<16} <{}> {}",
                    r.level.as_str(),
                    r.category,
                    i,
                    r.message
                )
            }
            None => std::eprintln!("{:<5} {:<16} {}", r.level.as_str(), r.category, r.message),
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
    let category = src.log_category();
    if enabled(category, level) {
        emit(category, src.log_instance(), level, args);
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
