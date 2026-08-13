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
#[non_exhaustive]
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
    /// A set of named flags, written `a+b+c` (gst's flags-property syntax, e.g.
    /// `flags=video+audio`). The accepted nicks are the property's
    /// [`enum_values`](PropertySpec::enum_values); the element receives them as a
    /// [`PropValue::Flags`] list, so it never splits the text itself.
    Flags,
}

impl PropKind {
    /// The human label for this kind, as `gst-inspect` names the type
    /// (`"Boolean"`, `"Unsigned Integer"`, ...). Shared by the text dump and the
    /// structured [`PropertyDoc`](crate::runtime::PropertyDoc).
    pub fn label(self) -> &'static str {
        match self {
            PropKind::Bool => "Boolean",
            PropKind::Int => "Integer",
            PropKind::Uint => "Unsigned Integer",
            PropKind::Double => "Double",
            PropKind::Fraction => "Fraction",
            PropKind::Str => "String",
            PropKind::Flags => "Flags",
        }
    }
}

/// A runtime property value. The variants mirror [`PropKind`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PropValue {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Double(f64),
    /// `(numerator, denominator)`.
    Fraction(i32, i32),
    Str(String),
    /// The nicks of a set-valued property, in the order they were written.
    Flags(Vec<String>),
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
            PropValue::Flags(_) => PropKind::Flags,
        }
    }

    /// Parse a textual value (as it appears in a `gst-launch` pipeline) for the
    /// given [`PropKind`]. `true`/`false` for bools; `n/d` for fractions; a bare
    /// integer is also accepted as a fraction `n/1`. The string kind takes the
    /// text verbatim.
    pub fn parse(kind: PropKind, text: &str) -> Result<PropValue, PropError> {
        let t = text.trim();
        match kind {
            // Case-insensitive, so a pipeline pasted from gst-launch (which
            // takes `True` as readily as `true`) parses here too.
            PropKind::Bool => match t.to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" => Ok(PropValue::Bool(true)),
                "false" | "0" | "no" => Ok(PropValue::Bool(false)),
                _ => Err(PropError::Value),
            },
            PropKind::Int => t
                .parse::<i64>()
                .map(PropValue::Int)
                .map_err(|_| PropError::Value),
            PropKind::Uint => t
                .parse::<u64>()
                .map(PropValue::Uint)
                .map_err(|_| PropError::Value),
            PropKind::Double => t
                .parse::<f64>()
                .map(PropValue::Double)
                .map_err(|_| PropError::Value),
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
            PropKind::Flags => {
                let mut set = Vec::new();
                for nick in t.split('+') {
                    let nick = nick.trim();
                    if nick.is_empty() {
                        return Err(PropError::Value);
                    }
                    set.push(nick.to_string());
                }
                Ok(PropValue::Flags(set))
            }
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

    /// Borrow the value as the nicks of a [`Flags`](PropValue::Flags) set. The
    /// parser has already split the `a+b` text and (when the property declares
    /// `enum_values`) checked every nick, so an element only matches them.
    pub fn as_flags(&self) -> Option<&[String]> {
        match self {
            PropValue::Flags(set) => Some(set),
            _ => None,
        }
    }

    /// Whether a [`Flags`](PropValue::Flags) set contains `nick`. `false` for any
    /// other kind.
    pub fn has_flag(&self, nick: &str) -> bool {
        self.as_flags()
            .is_some_and(|set| set.iter().any(|n| n == nick))
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
    pub const READWRITE: Self = Self {
        readable: true,
        writable: true,
    };
    /// Readable only (a computed / status property).
    pub const READ_ONLY: Self = Self {
        readable: true,
        writable: false,
    };
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
    /// (e.g. `"horizontal-mirror | vertical-mirror | rotate-180"`), `|`
    /// separated. For a [`Str`](PropKind::Str) or [`Flags`](PropKind::Flags)
    /// property this is the *closed* set [`parse_value`](Self::parse_value)
    /// validates against, so every nick the element accepts (aliases included)
    /// must be listed. On a numeric property it stays a documentation list (the
    /// entries may carry a note, `"2 (2.5 ms) | 5"`) and is not enforced.
    pub enum_values: Option<&'static str>,
    /// Read/write access.
    pub flags: PropFlags,
}

/// [`PropertySpec::name`] of the entry an element adds to say it takes
/// properties beyond the ones it declares.
///
/// No pipeline can spell this as a key, so it cannot collide with a real one.
pub const UNDECLARED_PROPERTIES: &str = "*";

/// Whether these specs let a name none of them declares through.
pub fn takes_undeclared_properties(specs: &[PropertySpec]) -> bool {
    specs.iter().any(|s| s.name == UNDECLARED_PROPERTIES)
}

impl PropertySpec {
    /// The entry that lets a key none of the other specs names through, as text,
    /// for whatever does know it to interpret.
    ///
    /// For an element whose real property set is not known until something loads
    /// at run time: a `pyelement` takes whatever the hosted Python class
    /// declares, so the list cannot be written down here. `blurb` says where the
    /// rest come from, since a `gst-inspect` dump shows this entry in their place.
    pub const fn undeclared(blurb: &'static str) -> Self {
        Self::new(UNDECLARED_PROPERTIES, PropKind::Str, blurb)
    }

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

    /// The declared nicks of an enum / flag property: [`enum_values`](Self::enum_values)
    /// split on `|`, each entry's leading word (an entry may carry a trailing
    /// note, `"2 (2.5 ms)"`). Empty when the property declares none.
    pub fn enum_nicks(&self) -> impl Iterator<Item = &'static str> {
        self.enum_values
            .unwrap_or("")
            .split('|')
            .filter_map(|entry| entry.split_whitespace().next())
    }

    /// Parse a textual value (a `gst-launch` `key=value`) for this property:
    /// [`PropValue::parse`] for the kind, plus nick validation against
    /// [`enum_values`](Self::enum_values) for a string / flag set. Validating here
    /// means a launch parser can name the valid choices in its error instead of
    /// surfacing a bare [`PropError::Value`] from the element.
    pub fn parse_value(&self, text: &str) -> Result<PropValue, ValueError> {
        let value = PropValue::parse(self.kind, text).map_err(|e| {
            // A flag set parses unless an entry is empty (`a++b`, a trailing
            // `+`); report the whole text so the error can list the nicks.
            if self.kind == PropKind::Flags {
                ValueError::Nick(text.trim().to_string())
            } else {
                ValueError::Kind(e)
            }
        })?;
        if self.enum_values.is_none() {
            return Ok(value);
        }
        let nicks: &[String] = match &value {
            PropValue::Str(s) => core::slice::from_ref(s),
            PropValue::Flags(set) => set,
            // A numeric property's enum_values is documentation, not a closed set.
            _ => return Ok(value),
        };
        for nick in nicks {
            if !self.enum_nicks().any(|d| d == nick) {
                return Err(ValueError::Nick(nick.clone()));
            }
        }
        Ok(value)
    }
}

/// Why [`PropertySpec::parse_value`] rejected a textual property value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    /// The text did not parse for the property's [`PropKind`].
    Kind(PropError),
    /// A name that is not one of the property's declared choices. The string is
    /// the offending nick (one entry of a `+`-joined flag set), or the whole
    /// value when a flag set was malformed.
    Nick(String),
}

impl core::fmt::Display for ValueError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ValueError::Kind(e) => write!(f, "{e}"),
            ValueError::Nick(n) => write!(f, "unknown value '{n}'"),
        }
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
        Self {
            long_name,
            klass,
            description,
            author,
        }
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
    kind.label()
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
        assert_eq!(
            PropValue::parse(PropKind::Bool, "true").unwrap(),
            PropValue::Bool(true)
        );
        assert_eq!(
            PropValue::parse(PropKind::Bool, "0").unwrap(),
            PropValue::Bool(false)
        );
        assert_eq!(
            PropValue::parse(PropKind::Bool, "True").unwrap(),
            PropValue::Bool(true)
        );
        assert_eq!(
            PropValue::parse(PropKind::Bool, "FALSE").unwrap(),
            PropValue::Bool(false)
        );
        assert_eq!(
            PropValue::parse(PropKind::Int, "-7").unwrap(),
            PropValue::Int(-7)
        );
        assert_eq!(
            PropValue::parse(PropKind::Uint, "42").unwrap(),
            PropValue::Uint(42)
        );
        assert_eq!(
            PropValue::parse(PropKind::Fraction, "30/1").unwrap(),
            PropValue::Fraction(30, 1)
        );
        // A bare integer parses as n/1 for a fraction property.
        assert_eq!(
            PropValue::parse(PropKind::Fraction, "25").unwrap(),
            PropValue::Fraction(25, 1)
        );
        assert_eq!(
            PropValue::parse(PropKind::Str, "file.mp4").unwrap(),
            PropValue::Str("file.mp4".into())
        );
    }

    #[test]
    fn parse_rejects_bad_values() {
        assert_eq!(PropValue::parse(PropKind::Int, "x"), Err(PropError::Value));
        assert_eq!(
            PropValue::parse(PropKind::Uint, "-1"),
            Err(PropError::Value)
        );
        assert_eq!(
            PropValue::parse(PropKind::Fraction, "1/0"),
            Err(PropError::Value)
        );
        assert_eq!(
            PropValue::parse(PropKind::Bool, "maybe"),
            Err(PropError::Value)
        );
    }

    #[test]
    fn flag_set_parses_into_nicks() {
        assert_eq!(
            PropValue::parse(PropKind::Flags, "video+audio").unwrap(),
            PropValue::Flags(alloc::vec!["video".into(), "audio".into()])
        );
        // Whitespace around a nick is trimmed (a quoted `"video + audio"`).
        assert_eq!(
            PropValue::parse(PropKind::Flags, "video + audio").unwrap(),
            PropValue::Flags(alloc::vec!["video".into(), "audio".into()])
        );
        let v = PropValue::parse(PropKind::Flags, "video+audio").unwrap();
        assert!(v.has_flag("audio") && !v.has_flag("text"));
        assert_eq!(v.as_flags().unwrap().len(), 2);
    }

    #[test]
    fn malformed_flag_set_is_rejected() {
        for bad in ["video+", "+video", "video++audio", ""] {
            assert_eq!(
                PropValue::parse(PropKind::Flags, bad),
                Err(PropError::Value),
                "{bad} must not parse"
            );
        }
    }

    #[test]
    fn spec_validates_enum_nicks() {
        let spec = PropertySpec::new("backend", PropKind::Str, "encoder")
            .with_enum_values("nvenc | software");
        assert_eq!(
            spec.parse_value("software").unwrap(),
            PropValue::Str("software".into())
        );
        assert_eq!(
            spec.parse_value("nvidia"),
            Err(ValueError::Nick("nvidia".into()))
        );
        // No declared values: any string goes through.
        let free = PropertySpec::new("location", PropKind::Str, "path");
        assert!(free.parse_value("anything").is_ok());
    }

    #[test]
    fn spec_validates_each_flag_nick() {
        let spec = PropertySpec::new("protocols", PropKind::Flags, "transports")
            .with_enum_values("udp | tcp");
        assert_eq!(
            spec.parse_value("udp+tcp").unwrap(),
            PropValue::Flags(alloc::vec!["udp".into(), "tcp".into()])
        );
        // The offending nick is named, not the whole value.
        assert_eq!(
            spec.parse_value("udp+quic"),
            Err(ValueError::Nick("quic".into()))
        );
        // A malformed set reports the whole text (there is no one bad nick).
        assert_eq!(
            spec.parse_value("udp+"),
            Err(ValueError::Nick("udp+".into()))
        );
        assert_eq!(spec.enum_nicks().collect::<Vec<_>>(), ["udp", "tcp"]);
    }

    #[test]
    fn numeric_enum_values_stay_documentation() {
        // `opusenc frame-size` lists annotated numbers; the list documents the
        // choices but the kind (not the list) decides what parses.
        let spec = PropertySpec::new("frame-size", PropKind::Uint, "ms")
            .with_enum_values("2 (2.5 ms) | 5 | 10");
        assert_eq!(spec.parse_value("20").unwrap(), PropValue::Uint(20));
        assert_eq!(
            spec.parse_value("x"),
            Err(ValueError::Kind(PropError::Value))
        );
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
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "frames then EOS (-1 = forever)",
            )
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
