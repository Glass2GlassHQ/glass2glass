//! Runtime element properties (M104): a name/value bag layered over the
//! compile-time `with_*` builders, the GObject-property analog.
//!
//! The builders (`VideoTestSrc::new().with_pattern(..)`) stay the zero-cost,
//! type-checked construction path and the only one the `no_std` / RTOS baseline
//! needs. This module adds the *runtime* face GStreamer tooling expects: set a
//! property by string name and value, read it back, and enumerate an element's
//! properties without instantiating tooling-specific code. That runtime face is
//! what a `gst-launch` text pipeline parser and a `gst-inspect` introspection
//! dump build on (M105 / M106).
//!
//! It costs the baseline nothing: the [`properties`](crate::AsyncElement::properties)
//! / [`set_property`](crate::AsyncElement::set_property) /
//! [`get_property`](crate::AsyncElement::get_property) trait methods default to
//! "no properties", exactly like [`latency`](crate::AsyncElement::latency), so an
//! element opts in only by overriding them and an RTOS build that never calls
//! them pays nothing.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The type of a property value, used in a [`PropertySpec`] (so tooling knows how
/// to parse a string for it) and to validate a [`PropValue`] on assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropKind {
    Bool,
    /// Signed integer (`i64`).
    Int,
    /// Unsigned integer (`u64`).
    Uint,
    /// Floating point (`f64`).
    Double,
    /// A `num/den` fraction (e.g. a framerate `30/1`).
    Fraction,
    /// UTF-8 string.
    Str,
}

/// A runtime property value. The variants mirror [`PropKind`].
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Double(f64),
    /// `(numerator, denominator)`.
    Fraction(i32, i32),
    Str(String),
}

impl PropValue {
    /// The [`PropKind`] this value holds.
    pub fn kind(&self) -> PropKind {
        match self {
            PropValue::Bool(_) => PropKind::Bool,
            PropValue::Int(_) => PropKind::Int,
            PropValue::Uint(_) => PropKind::Uint,
            PropValue::Double(_) => PropKind::Double,
            PropValue::Fraction(_, _) => PropKind::Fraction,
            PropValue::Str(_) => PropKind::Str,
        }
    }

    /// Parse a textual value (as it appears in a `gst-launch` pipeline) for the
    /// given [`PropKind`]. `true`/`false` for bools; `n/d` for fractions; a bare
    /// integer is also accepted as a fraction `n/1`. The string kind takes the
    /// text verbatim.
    pub fn parse(kind: PropKind, text: &str) -> Result<PropValue, PropError> {
        let t = text.trim();
        match kind {
            PropKind::Bool => match t {
                "true" | "1" | "yes" => Ok(PropValue::Bool(true)),
                "false" | "0" | "no" => Ok(PropValue::Bool(false)),
                _ => Err(PropError::Value),
            },
            PropKind::Int => t.parse::<i64>().map(PropValue::Int).map_err(|_| PropError::Value),
            PropKind::Uint => t.parse::<u64>().map(PropValue::Uint).map_err(|_| PropError::Value),
            PropKind::Double => {
                t.parse::<f64>().map(PropValue::Double).map_err(|_| PropError::Value)
            }
            PropKind::Fraction => match t.split_once('/') {
                Some((n, d)) => {
                    let n = n.trim().parse::<i32>().map_err(|_| PropError::Value)?;
                    let d = d.trim().parse::<i32>().map_err(|_| PropError::Value)?;
                    if d == 0 {
                        return Err(PropError::Value);
                    }
                    Ok(PropValue::Fraction(n, d))
                }
                None => {
                    let n = t.parse::<i32>().map_err(|_| PropError::Value)?;
                    Ok(PropValue::Fraction(n, 1))
                }
            },
            PropKind::Str => Ok(PropValue::Str(t.to_string())),
        }
    }

    /// Borrow the value as `bool`, if it is one.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow the value as `i64`, if it is an [`Int`](PropValue::Int).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Borrow the value as `u64`, if it is a [`Uint`](PropValue::Uint).
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            PropValue::Uint(v) => Some(*v),
            _ => None,
        }
    }

    /// Borrow the value as `f64`, if it is a [`Double`](PropValue::Double).
    pub fn as_double(&self) -> Option<f64> {
        match self {
            PropValue::Double(v) => Some(*v),
            _ => None,
        }
    }

    /// Borrow the value as a `(num, den)` fraction, if it is one.
    pub fn as_fraction(&self) -> Option<(i32, i32)> {
        match self {
            PropValue::Fraction(n, d) => Some((*n, *d)),
            _ => None,
        }
    }

    /// Borrow the value as `&str`, if it is a [`Str`](PropValue::Str).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Read/write access flags for a property, the GObject `G_PARAM_READABLE` /
/// `G_PARAM_WRITABLE` analog shown in a `gst-inspect` dump. Default is
/// read+write; a derived/computed property is read-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropFlags {
    pub readable: bool,
    pub writable: bool,
}

impl PropFlags {
    /// Readable and writable (the default).
    pub const READWRITE: Self = Self { readable: true, writable: true };
    /// Readable only (a computed / status property).
    pub const READ_ONLY: Self = Self { readable: true, writable: false };
}

impl Default for PropFlags {
    fn default() -> Self {
        Self::READWRITE
    }
}

/// Static metadata for one settable property: its name, type, a one-line
/// description, and (optionally) its default, accepted range, and access flags.
/// The element type declares these (via
/// [`properties`](crate::AsyncElement::properties)) so tooling can enumerate and
/// document them without a live instance carrying the strings. All textual fields
/// are `&'static str` so the struct stays `Copy` / `const`-declarable.
///
/// Build with [`new`](Self::new) (name + kind + blurb) and refine with the
/// `const` builders ([`with_default`](Self::with_default),
/// [`with_range`](Self::with_range), [`read_only`](Self::read_only),
/// [`with_enum_values`](Self::with_enum_values)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropertySpec {
    /// Property name, as used in a `gst-launch` pipeline (`key=value`).
    pub name: &'static str,
    /// The value type, so a textual value can be parsed for it.
    pub kind: PropKind,
    /// One-line human description, for a `gst-inspect`-style dump.
    pub blurb: &'static str,
    /// Default value as text (parseable via [`PropValue::parse`]), or `None` if
    /// the property has no meaningful default.
    pub default: Option<&'static str>,
    /// Accepted `(min, max)` range as text, for a numeric property.
    pub range: Option<(&'static str, &'static str)>,
    /// The named choices of an enum-like string property
    /// (e.g. `"horizontal-mirror | vertical-mirror | rotate-180"`).
    pub enum_values: Option<&'static str>,
    /// Read/write access.
    pub flags: PropFlags,
}

impl PropertySpec {
    /// A new spec (a `const fn` so a static `&[PropertySpec]` table is cheap).
    /// Defaults to no default value, no range, and read+write.
    pub const fn new(name: &'static str, kind: PropKind, blurb: &'static str) -> Self {
        Self {
            name,
            kind,
            blurb,
            default: None,
            range: None,
            enum_values: None,
            flags: PropFlags::READWRITE,
        }
    }

    /// Set the textual default value shown by `gst-inspect`.
    pub const fn with_default(mut self, default: &'static str) -> Self {
        self.default = Some(default);
        self
    }

    /// Set the accepted `(min, max)` numeric range.
    pub const fn with_range(mut self, min: &'static str, max: &'static str) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set the named choices of an enum-like string property.
    pub const fn with_enum_values(mut self, values: &'static str) -> Self {
        self.enum_values = Some(values);
        self
    }

    /// Mark the property read-only (a computed / status value).
    pub const fn read_only(mut self) -> Self {
        self.flags = PropFlags::READ_ONLY;
        self
    }
}

/// Static, type-level description of an element for `gst-inspect`-style
/// introspection (M178), the GStreamer element-class-metadata analog
/// (`gst_element_class_set_static_metadata`). All `&'static str` so it is
/// `const`-declarable next to the element and costs a live instance nothing.
/// An element opts in by overriding `metadata()` (default: empty), exactly like
/// [`properties`](crate::AsyncElement::properties).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ElementMetadata {
    /// Human-readable name, e.g. `"Opus audio encoder"`.
    pub long_name: &'static str,
    /// Classification (GStreamer's `klass`), e.g. `"Codec/Encoder/Audio"`.
    pub klass: &'static str,
    /// One-paragraph description of what the element does.
    pub description: &'static str,
    /// Author / origin, e.g. `"g2g"`.
    pub author: &'static str,
}

impl ElementMetadata {
    /// A new metadata block (a `const fn` for a `const` declaration on the type).
    pub const fn new(
        long_name: &'static str,
        klass: &'static str,
        description: &'static str,
        author: &'static str,
    ) -> Self {
        Self { long_name, klass, description, author }
    }

    /// Whether any field is set (an element that overrode `metadata()`).
    pub fn is_set(&self) -> bool {
        !(self.long_name.is_empty()
            && self.klass.is_empty()
            && self.description.is_empty()
            && self.author.is_empty())
    }
}

/// Why a [`set_property`](crate::AsyncElement::set_property) (or a value parse)
/// failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropError {
    /// No property of that name on this element.
    Unknown,
    /// The value's [`PropKind`] does not match the property's.
    Type,
    /// The value is the right kind but out of the accepted range / not parseable.
    Value,
    /// The property exists but is read-only.
    ReadOnly,
}

impl core::fmt::Display for PropError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            PropError::Unknown => "unknown property",
            PropError::Type => "property type mismatch",
            PropError::Value => "invalid property value",
            PropError::ReadOnly => "read-only property",
        };
        f.write_str(s)
    }
}

/// The human label for a [`PropKind`], as `gst-inspect` names the type.
fn kind_label(kind: PropKind) -> &'static str {
    match kind {
        PropKind::Bool => "Boolean",
        PropKind::Int => "Integer",
        PropKind::Uint => "Unsigned Integer",
        PropKind::Double => "Double",
        PropKind::Fraction => "Fraction",
        PropKind::Str => "String",
    }
}

/// Format a property spec table the way `gst-inspect` details it: a header line
/// per property (name + blurb), then indented `flags`, type, range/enum, and
/// default lines. Used by the registry's introspection dump (M105, enriched
/// M178).
pub fn format_specs(specs: &[PropertySpec]) -> String {
    use core::fmt::Write;
    let mut out = String::new();
    for s in specs {
        let _ = writeln!(out, "  {}: {}", s.name, s.blurb);
        let flags = match (s.flags.readable, s.flags.writable) {
            (true, true) => "readable, writable",
            (true, false) => "readable",
            (false, true) => "writable",
            (false, false) => "",
        };
        let _ = writeln!(out, "    flags: {flags}");
        let _ = write!(out, "    {}", kind_label(s.kind));
        if let Some((min, max)) = s.range {
            let _ = write!(out, ". Range: {min} - {max}");
        }
        if let Some(values) = s.enum_values {
            let _ = write!(out, ". Values: {values}");
        }
        if let Some(default) = s.default {
            let _ = write!(out, ". Default: {default}");
        }
        out.push('\n');
    }
    out
}

/// Format an [`ElementMetadata`] block the way `gst-inspect` opens with its
/// "Factory Details" section. `name` is the registry/factory name (the element's
/// `gst-launch` identifier). Empty metadata fields are omitted.
pub fn format_metadata(name: &str, meta: &ElementMetadata) -> String {
    use core::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "Factory Details:");
    let _ = writeln!(out, "  Name        {name}");
    if !meta.long_name.is_empty() {
        let _ = writeln!(out, "  Long-name   {}", meta.long_name);
    }
    if !meta.klass.is_empty() {
        let _ = writeln!(out, "  Klass       {}", meta.klass);
    }
    if !meta.description.is_empty() {
        let _ = writeln!(out, "  Description {}", meta.description);
    }
    if !meta.author.is_empty() {
        let _ = writeln!(out, "  Author      {}", meta.author);
    }
    out
}

/// Collect the names of a spec table (helper for tests / tooling).
pub fn spec_names(specs: &[PropertySpec]) -> Vec<&'static str> {
    specs.iter().map(|s| s.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_matches_kind() {
        assert_eq!(PropValue::parse(PropKind::Bool, "true").unwrap(), PropValue::Bool(true));
        assert_eq!(PropValue::parse(PropKind::Bool, "0").unwrap(), PropValue::Bool(false));
        assert_eq!(PropValue::parse(PropKind::Int, "-7").unwrap(), PropValue::Int(-7));
        assert_eq!(PropValue::parse(PropKind::Uint, "42").unwrap(), PropValue::Uint(42));
        assert_eq!(PropValue::parse(PropKind::Fraction, "30/1").unwrap(), PropValue::Fraction(30, 1));
        // A bare integer parses as n/1 for a fraction property.
        assert_eq!(PropValue::parse(PropKind::Fraction, "25").unwrap(), PropValue::Fraction(25, 1));
        assert_eq!(
            PropValue::parse(PropKind::Str, "file.mp4").unwrap(),
            PropValue::Str("file.mp4".into())
        );
    }

    #[test]
    fn parse_rejects_bad_values() {
        assert_eq!(PropValue::parse(PropKind::Int, "x"), Err(PropError::Value));
        assert_eq!(PropValue::parse(PropKind::Uint, "-1"), Err(PropError::Value));
        assert_eq!(PropValue::parse(PropKind::Fraction, "1/0"), Err(PropError::Value));
        assert_eq!(PropValue::parse(PropKind::Bool, "maybe"), Err(PropError::Value));
    }

    #[test]
    fn kind_round_trips_value() {
        assert_eq!(PropValue::Int(3).kind(), PropKind::Int);
        assert_eq!(PropValue::Fraction(30, 1).kind(), PropKind::Fraction);
        assert_eq!(PropValue::Str("x".into()).kind(), PropKind::Str);
    }

    #[test]
    fn format_specs_details_each_property() {
        let specs = [
            PropertySpec::new("pattern", PropKind::Str, "test pattern")
                .with_enum_values("smpte | snow | ball")
                .with_default("smpte"),
            PropertySpec::new("num-buffers", PropKind::Int, "frames then EOS (-1 = forever)")
                .with_range("-1", "9223372036854775807")
                .with_default("-1"),
        ];
        let dump = format_specs(&specs);
        // Header line: name + blurb.
        assert!(dump.contains("pattern: test pattern"), "got:\n{dump}");
        // Detail lines: flags, type, enum values, default.
        assert!(dump.contains("flags: readable, writable"));
        assert!(dump.contains("String. Values: smpte | snow | ball. Default: smpte"));
        assert!(dump.contains("Integer. Range: -1 - 9223372036854775807. Default: -1"));
        assert_eq!(spec_names(&specs), ["pattern", "num-buffers"]);
    }

    #[test]
    fn read_only_flag_renders() {
        let specs = [PropertySpec::new("dropped", PropKind::Uint, "frames dropped").read_only()];
        assert!(format_specs(&specs).contains("flags: readable\n"));
    }

    #[test]
    fn metadata_block_omits_empty_fields() {
        let meta = ElementMetadata::new("Opus encoder", "Codec/Encoder/Audio", "", "g2g");
        let dump = format_metadata("opusenc", &meta);
        assert!(dump.contains("Name        opusenc"));
        assert!(dump.contains("Long-name   Opus encoder"));
        assert!(dump.contains("Klass       Codec/Encoder/Audio"));
        assert!(dump.contains("Author      g2g"));
        assert!(!dump.contains("Description"), "empty description omitted");
        assert!(!ElementMetadata::default().is_set());
        assert!(meta.is_set());
    }
}
