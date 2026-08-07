//! M840: the launch-line value grammar. Covers the two things a pasted
//! `gst-launch` line needs beyond bare `key=value`: values carrying spaces
//! (quoted or escaped) and named choices (an enum nick, or a `+`-joined flag
//! set), with an error that names the element, the property, and the valid
//! choices.
//!
//! The fixture element pair is registered into a real [`Registry`] and driven
//! through the real [`parse_launch`], so the grammar under test is the shipped
//! one. What each line does in GStreamer 1.26 is noted per test; verified on this
//! host with `gst-launch-1.0 fakesrc num-buffers=0 ! filesink location=<value>`
//! (the created file name is the parsed value).
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, OnceLock};

use g2g_core::runtime::{
    parse_launch, LaunchFactory, ParseError, Registry, SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink, PadTemplate,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(2),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// What the last constructed `probesrc` was given, so a test can assert the
/// element received the parsed value (the factory takes a bare `fn()`, so the
/// fixture cannot carry a handle).
fn applied() -> &'static Mutex<Vec<(String, PropValue)>> {
    static APPLIED: OnceLock<Mutex<Vec<(String, PropValue)>>> = OnceLock::new();
    APPLIED.get_or_init(|| Mutex::new(Vec::new()))
}

static PROBE_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "free-form path"),
    PropertySpec::new("pattern", PropKind::Str, "drawn pattern").with_enum_values("solid | noise"),
    PropertySpec::new("protocols", PropKind::Flags, "transports, in order")
        .with_enum_values("udp | tcp")
        .with_default("tcp"),
];

/// Source whose only job is to record the properties the parser applies.
struct ProbeSrc;

impl SourceLoop for ProbeSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn properties(&self) -> &'static [PropertySpec] {
        PROBE_PROPS
    }
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if !PROBE_PROPS.iter().any(|s| s.name == name) {
            return Err(PropError::Unknown);
        }
        applied().lock().unwrap().push((name.to_string(), value));
        Ok(())
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

struct ProbeSink;

impl AsyncElement for ProbeSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("probesrc", caps(), || {
        Box::new(ProbeSrc)
    }));
    reg.register_launch(LaunchFactory::new(
        "probesink",
        Vec::from([PadTemplate::sink(CapsSet::one(caps()))]),
        || Box::new(ProbeSink),
    ));
    reg
}

/// Parse `probesrc <props> ! probesink` and return the properties the source was
/// given. The record is process-wide, so parses serialize against each other.
fn apply(props: &str) -> Result<Vec<(String, PropValue)>, ParseError> {
    static SERIAL: Mutex<()> = Mutex::new(());
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    applied().lock().unwrap().clear();
    parse_launch(&registry(), &format!("probesrc {props} ! probesink"))?;
    let got = applied().lock().unwrap().clone();
    Ok(got)
}

fn one_str(props: &str) -> String {
    let got = apply(props).expect("line parses");
    assert_eq!(got.len(), 1, "one property applied: {got:?}");
    got[0].1.as_str().expect("string value").to_string()
}

#[test]
fn quoting_matrix_yields_the_literal_value() {
    // gst-launch 1.26 accepts the double-quoted and backslash-escaped forms and
    // strips / resolves them exactly like this. It does NOT treat `'` as a quote
    // (the file it creates is literally named `'a b.txt'`) and it rejects a quote
    // opened mid-fragment ("erroneous pipeline: no element ..."), so the single
    // quote and mid-fragment rows below are g2g accepting a superset: every line
    // gst accepts parses here to the same value.
    assert_eq!(one_str(r#"location="/my file.ts""#), "/my file.ts");
    assert_eq!(one_str("location='/my file.ts'"), "/my file.ts");
    assert_eq!(one_str(r"location=/my\ file.ts"), "/my file.ts");
    assert_eq!(one_str(r#"location="a\"b.ts""#), "a\"b.ts");
    assert_eq!(one_str(r#"location='a\'b.ts'"#), "a'b.ts");
    // A quoted region in the middle of a fragment, and two of them.
    assert_eq!(
        one_str(r#"location=/tmp/"my dir"/a.ts"#),
        "/tmp/my dir/a.ts"
    );
    assert_eq!(
        one_str(r#"location="my dir"/"my file".ts"#),
        "my dir/my file.ts"
    );
    // An unquoted `\` before an ordinary character stays literal, so a Windows
    // path needs no escaping (gst-launch would eat these backslashes).
    assert_eq!(one_str(r"location=C:\videos\a.ts"), r"C:\videos\a.ts");
    // A `!` or a `#` survives inside quotes / behind an escape.
    assert_eq!(one_str(r#"location="a ! b.ts""#), "a ! b.ts");
    assert_eq!(one_str(r"location=a\#b.ts"), "a#b.ts");
}

#[test]
fn enum_nick_is_validated_against_the_declared_values() {
    assert_eq!(one_str("pattern=noise"), "noise");
    let err = apply("pattern=sparkle").unwrap_err();
    assert_eq!(
        err,
        ParseError::BadEnumValue {
            element: "probesrc".into(),
            key: "pattern".into(),
            value: "sparkle".into(),
            values: "solid | noise",
        }
    );
    // The message names element, property, and the valid choices.
    let msg = err.to_string();
    assert!(
        msg.contains("probesrc") && msg.contains("'pattern'") && msg.contains("solid | noise"),
        "{msg}"
    );
}

#[test]
fn flag_set_reaches_the_element_as_nicks() {
    // gst spells a flags property the same way (`playbin flags=video+audio`).
    let got = apply("protocols=udp+tcp").expect("line parses");
    assert_eq!(
        got,
        [(
            "protocols".to_string(),
            PropValue::Flags(Vec::from(["udp".to_string(), "tcp".to_string()]))
        )]
    );
    // A quoted set may be spaced out; the element still receives bare nicks.
    let got = apply(r#"protocols="tcp + udp""#).expect("line parses");
    assert!(got[0].1.has_flag("tcp") && got[0].1.has_flag("udp"));
}

#[test]
fn bad_flag_nick_names_the_offender_and_the_choices() {
    let err = apply("protocols=udp+quic").unwrap_err();
    assert_eq!(
        err,
        ParseError::BadEnumValue {
            element: "probesrc".into(),
            key: "protocols".into(),
            value: "quic".into(),
            values: "udp | tcp",
        }
    );
    assert!(err.to_string().contains("udp | tcp"), "{err}");
}

#[test]
fn malformed_flag_set_reports_the_whole_value_with_a_syntax_hint() {
    for bad in ["protocols=udp+", "protocols=+udp", "protocols=udp++tcp"] {
        let err = apply(bad).unwrap_err();
        let ParseError::BadEnumValue { value, .. } = &err else {
            panic!("{bad}: expected an enum-value error, got {err:?}");
        };
        assert!(value.contains('+'), "{bad}: reports the whole set: {value}");
        assert!(
            err.to_string().contains("flag set"),
            "{bad}: hints the syntax: {err}"
        );
    }
}

#[test]
fn unknown_property_names_the_element_and_key() {
    let err = apply("bogus=1").unwrap_err();
    assert_eq!(
        err,
        ParseError::UnknownProperty {
            element: "probesrc".into(),
            key: "bogus".into(),
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("probesrc") && msg.contains("'bogus'"), "{msg}");
}
