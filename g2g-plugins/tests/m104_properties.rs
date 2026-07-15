//! M104 runtime property system: set/get an element's config by string name and
//! value, the GObject-property analog the `gst-launch` parser + `gst-inspect`
//! dump build on. Exercises a source (`VideoTestSrc`, on `SourceLoop`) and two
//! transforms (`VideoFlip` / `VideoRate`, on `AsyncElement`), plus the dyn-erased
//! path that `Box<dyn DynSourceLoop>` / `Box<dyn DynAsyncElement>` take.

use g2g_core::runtime::SourceLoop;
use g2g_core::{AsyncElement, PropError, PropKind, PropValue};
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videorate::VideoRate;
use g2g_plugins::videotestsrc::VideoTestSrc;

#[test]
fn videotestsrc_props_round_trip() {
    let mut src = VideoTestSrc::new(320, 240, 30, 100);

    // String enum, int, uint, and fraction properties all set + read back.
    src.set_property("pattern", PropValue::Str("snow".into())).unwrap();
    src.set_property("num-buffers", PropValue::Int(5)).unwrap();
    src.set_property("width", PropValue::Uint(640)).unwrap();
    src.set_property("height", PropValue::Uint(480)).unwrap();
    src.set_property("framerate", PropValue::Fraction(25, 1)).unwrap();

    assert_eq!(src.get_property("pattern"), Some(PropValue::Str("snow".into())));
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(5)));
    assert_eq!(src.get_property("width"), Some(PropValue::Uint(640)));
    assert_eq!(src.get_property("height"), Some(PropValue::Uint(480)));
    assert_eq!(src.get_property("framerate"), Some(PropValue::Fraction(25, 1)));

    // -1 num-buffers means "forever".
    src.set_property("num-buffers", PropValue::Int(-1)).unwrap();
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
}

#[test]
fn property_errors_are_typed() {
    let mut src = VideoTestSrc::new(320, 240, 30, 100);
    // Unknown name.
    assert_eq!(src.set_property("nope", PropValue::Int(1)), Err(PropError::Unknown));
    // Right name, wrong value kind (width wants a uint).
    assert_eq!(src.set_property("width", PropValue::Str("x".into())), Err(PropError::Type));
    // Right name + kind, invalid value (unknown pattern string).
    assert_eq!(
        src.set_property("pattern", PropValue::Str("zigzag".into())),
        Err(PropError::Value)
    );
    // A bad framerate denominator is a value error.
    assert_eq!(src.set_property("framerate", PropValue::Fraction(30, 0)), Err(PropError::Value));
}

#[test]
fn spec_tables_describe_each_property() {
    let src = VideoTestSrc::new(2, 2, 30, 1);
    let names: Vec<_> = SourceLoop::properties(&src).iter().map(|s| s.name).collect();
    assert!(names.contains(&"pattern") && names.contains(&"framerate"));
    // The framerate property advertises the fraction kind so a parser knows how
    // to turn "30/1" into a value.
    let fr = SourceLoop::properties(&src).iter().find(|s| s.name == "framerate").unwrap();
    assert_eq!(fr.kind, PropKind::Fraction);

    let flip = VideoFlip::new(FlipMethod::Rotate180);
    let m = AsyncElement::properties(&flip).iter().find(|s| s.name == "method").unwrap();
    assert_eq!(m.kind, PropKind::Str);
}

#[test]
fn videoflip_method_enum_property() {
    let mut flip = VideoFlip::new(FlipMethod::HorizontalMirror);
    // Canonical GStreamer nickname round-trips unchanged (M182).
    flip.set_property("method", PropValue::Str("clockwise".into())).unwrap();
    assert_eq!(flip.method(), FlipMethod::Rotate90Cw);
    assert_eq!(flip.get_property("method"), Some(PropValue::Str("clockwise".into())));
    // The historical g2g spelling is still accepted as an alias, normalized to
    // the gst name on read.
    flip.set_property("method", PropValue::Str("rotate-90ccw".into())).unwrap();
    assert_eq!(flip.method(), FlipMethod::Rotate90Ccw);
    assert_eq!(flip.get_property("method"), Some(PropValue::Str("counterclockwise".into())));
}

#[test]
fn videorate_fraction_property() {
    let mut rate = VideoRate::new(30.0);
    rate.set_property("framerate", PropValue::Fraction(10, 1)).unwrap();
    assert_eq!(rate.get_property("framerate"), Some(PropValue::Fraction(10, 1)));
}

#[test]
fn set_property_through_dyn_erasure() {
    use g2g_core::element::DynAsyncElement;
    use g2g_core::runtime::DynSourceLoop;
    // The whole point of the dyn mirrors: a registry holds Box<dyn ...> and still
    // sets properties by name. This is the path the gst-launch parser uses.
    let mut src: Box<dyn DynSourceLoop> = Box::new(VideoTestSrc::new(16, 16, 30, 1));
    // Old g2g spelling accepted as an alias; normalized to the gst name on read.
    src.set_property("pattern", PropValue::Str("moving-bar".into())).unwrap();
    assert_eq!(src.get_property("pattern"), Some(PropValue::Str("bar".into())));
    assert!(src.properties().iter().any(|s| s.name == "pattern"));

    let mut flip: Box<dyn DynAsyncElement> = Box::new(VideoFlip::new(FlipMethod::Rotate180));
    flip.set_property("method", PropValue::Str("vertical-mirror".into())).unwrap();
    assert_eq!(flip.get_property("method"), Some(PropValue::Str("vertical-flip".into())));
}
