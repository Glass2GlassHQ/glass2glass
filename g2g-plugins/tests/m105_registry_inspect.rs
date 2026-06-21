//! M105 registry build-by-name + `gst-inspect` dump. Registers a source and two
//! transforms/sinks under names, then: lists them, inspects their property /
//! pad-template tables, and constructs them by name with properties applied (the
//! path the M106 `gst-launch` parser drives).

use g2g_core::runtime::{LaunchFactory, Registry, SourceFactory};
use g2g_core::{Caps, Dim, PropValue, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("videotestsrc", rgba_any(), || {
        Box::new(VideoTestSrc::new(320, 240, 30, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(FlipMethod::Rotate180))
    }));
    reg.register_launch(LaunchFactory::new("fakesink", Vec::new(), || Box::new(FakeSink::new())));
    reg
}

#[test]
fn lists_registered_element_names() {
    let reg = registry();
    let names = reg.element_names();
    assert!(names.contains(&"videotestsrc"));
    assert!(names.contains(&"videoflip"));
    assert!(names.contains(&"fakesink"));
}

#[test]
fn inspect_dumps_properties_and_templates() {
    let reg = registry();

    let src = reg.inspect("videotestsrc").expect("source registered");
    // M178: gst-inspect-shaped dump with a Factory Details header + role.
    assert!(src.contains("Factory Details:"), "has the metadata header:\n{src}");
    assert!(src.contains("Long-name   Video test source"), "shows the long name:\n{src}");
    assert!(src.contains("Role        source"));
    assert!(src.contains("pattern"), "lists the pattern property:\n{src}");
    assert!(src.contains("framerate"), "lists the framerate property:\n{src}");
    // Enriched property detail: the pattern default and its flags line.
    assert!(src.contains("Default: smpte"), "shows the pattern default:\n{src}");
    assert!(src.contains("flags: readable, writable"), "shows property flags:\n{src}");

    let flip = reg.inspect("videoflip").expect("element registered");
    assert!(flip.contains("Role        element"));
    assert!(flip.contains("Klass       Filter/Effect/Video"), "shows the classification:\n{flip}");
    assert!(flip.contains("method"), "lists the method property:\n{flip}");
    // VideoFlip declares pad templates, so the dump shows SINK / SRC lines.
    assert!(flip.contains("SINK") && flip.contains("SRC"), "lists pad templates:\n{flip}");

    assert!(reg.inspect("nonesuch").is_none(), "unknown name -> None");
}

#[test]
fn make_by_name_then_set_property() {
    let reg = registry();

    let mut src = reg.make_source("videotestsrc").expect("source built");
    src.set_property("num-buffers", PropValue::Int(3)).unwrap();
    assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(3)));

    let mut flip = reg.make_element("videoflip").expect("element built");
    // gst nickname canonical; old g2g `rotate-90cw` still accepted as an alias.
    flip.set_property("method", PropValue::Str("clockwise".into())).unwrap();
    assert_eq!(flip.get_property("method"), Some(PropValue::Str("clockwise".into())));

    assert!(reg.make_source("videoflip").is_none(), "videoflip is not a source");
    assert!(reg.make_element("videotestsrc").is_none(), "videotestsrc is not a launch element");
}
