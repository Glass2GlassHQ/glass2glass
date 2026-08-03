//! M840: enum nicks and flag sets on real elements, through the real launch
//! parser. An element's declared `enum_values` is the closed set the parser
//! validates a `key=nick` against, so a typo names the element, the property, and
//! the valid choices instead of surfacing a bare "invalid property value".

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, ParseError};
use g2g_plugins::registry::default_registry;

fn parse(line: &str) -> Result<(), ParseError> {
    parse_launch(&default_registry(), line).map(|_| ())
}

#[test]
fn declared_nicks_and_their_aliases_all_parse() {
    // Every spelling `videoflip` / `videotestsrc` accept is declared, so none of
    // these is rejected by the parser before the element sees it.
    for method in [
        "none",
        "identity",
        "clockwise",
        "rotate-90cw",
        "rotate-180",
        "counterclockwise",
        "rotate-90ccw",
        "horizontal-flip",
        "horizontal-mirror",
        "vertical-flip",
        "vertical-mirror",
    ] {
        parse(&format!(
            "videotestsrc num-buffers=1 ! videoflip method={method} ! fakesink"
        ))
        .unwrap_or_else(|e| panic!("method={method}: {e}"));
    }
    for pattern in [
        "smpte",
        "snow",
        "bar",
        "moving-bar",
        "checkers-8",
        "checker",
    ] {
        parse(&format!(
            "videotestsrc num-buffers=1 pattern={pattern} ! fakesink"
        ))
        .unwrap_or_else(|e| panic!("pattern={pattern}: {e}"));
    }
}

#[test]
fn unknown_enum_nick_names_the_element_property_and_choices() {
    let err =
        parse("videotestsrc num-buffers=1 ! videoflip method=sideways ! fakesink").unwrap_err();
    let ParseError::BadEnumValue {
        element,
        key,
        value,
        values,
    } = &err
    else {
        panic!("expected an enum-value error, got {err:?}");
    };
    assert_eq!(
        (element.as_str(), key.as_str(), value.as_str()),
        ("videoflip", "method", "sideways")
    );
    assert!(values.contains("clockwise"), "declared choices: {values}");
    let msg = err.to_string();
    assert!(
        msg.contains("videoflip") && msg.contains("'method'") && msg.contains("valid:"),
        "{msg}"
    );
}

#[test]
fn unknown_property_is_still_reported_by_name() {
    let err = parse("videotestsrc num-buffers=1 sparkle=3 ! fakesink").unwrap_err();
    assert_eq!(
        err,
        ParseError::UnknownProperty {
            element: "videotestsrc".into(),
            key: "sparkle".into(),
        }
    );
}

/// `rtspsrc protocols=` is g2g's flag-set property: a `+`-joined transport list
/// applied in the order written (gst's `rtspsrc protocols=udp+tcp`). No network:
/// the assertion is on the property surface, which is what the launch line sets.
#[cfg(feature = "rtsp")]
mod rtsp_protocols {
    use g2g_core::runtime::SourceLoop;
    use g2g_core::{PropValue, PropertySpec, ValueError};
    use g2g_plugins::rtspsrc::RtspSrc;

    fn spec(src: &RtspSrc) -> PropertySpec {
        *src.properties()
            .iter()
            .find(|s| s.name == "protocols")
            .expect("rtspsrc declares protocols")
    }

    #[test]
    fn flag_set_sets_the_transport_order() {
        let mut src = RtspSrc::new("rtsp://example/stream");
        // The default is TCP (interleaved), retina's default too.
        assert_eq!(
            src.get_property("protocols"),
            Some(PropValue::Flags(Vec::from(["tcp".to_string()])))
        );
        let value = spec(&src).parse_value("udp+tcp").expect("parses");
        src.set_property("protocols", value).unwrap();
        assert_eq!(
            src.get_property("protocols"),
            Some(PropValue::Flags(Vec::from([
                "udp".to_string(),
                "tcp".to_string()
            ])))
        );
    }

    #[test]
    fn undeclared_transport_is_rejected_by_the_spec() {
        let src = RtspSrc::new("rtsp://example/stream");
        assert_eq!(
            spec(&src).parse_value("udp+quic"),
            Err(ValueError::Nick("quic".into()))
        );
        // gst also spells multicast `udp-mcast`; retina has no such transport, so
        // it is not declared and the parser says so rather than silently ignoring.
        assert_eq!(
            spec(&src).parse_value("udp-mcast"),
            Err(ValueError::Nick("udp-mcast".into()))
        );
    }

    #[test]
    fn launch_line_applies_the_flag_set() {
        use g2g_core::runtime::parse_launch;
        use g2g_plugins::registry::default_registry;
        // Parse only: no DESCRIBE happens until the graph runs.
        parse_launch(
            &default_registry(),
            "rtspsrc location=rtsp://example/stream protocols=udp+tcp width=320 height=240 ! fakesink",
        )
        .expect("flag set parses on a real element");
    }
}
