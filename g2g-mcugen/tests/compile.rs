//! Compiler behavior: the flagship graph compiles to the expected structure
//! and ring budget, the emitted source is deterministic (a golden snapshot),
//! and mis-wired graphs are rejected with the right diagnostic before any Rust
//! is emitted. The end-to-end correctness proof (generated wire == reference)
//! lives in `examples/mcugen-audio`; these tests pin the compiler's contract.

use g2g_mcugen::{compile_str, CompileError};

const FLAGSHIP: &str = include_str!("../examples/flagship.yaml");
const DISPLAY: &str = include_str!("../examples/display.yaml");

#[test]
fn flagship_ring_budget_matches_the_hand_written_reference() {
    let c = compile_str(FLAGSHIP).expect("compile");
    // The hand-written noalloc-pipeline::audio uses exactly these sizes:
    // A=160, B=1920, conv=960, resample=160, mix=160, enc=80.
    assert_eq!(c.ring_bytes_total, 160 + 1920 + 960 + 160 + 160 + 80);
    assert_eq!(c.rings.len(), 6, "one ring per lending element (the sink lends none)");
    let cap_b = c.rings.iter().find(|(n, _)| n == "ring_cap_b").expect("cap_b ring");
    assert_eq!(cap_b.1, 1920, "48 kHz x 10 ms x 4 bytes = 1920");
    assert_eq!(c.entry, "run_flagship_with");
    assert_eq!(c.grabber_params, ["grab_cap_a", "grab_cap_b"]);
}

#[test]
fn emitted_source_is_deterministic() {
    let a = compile_str(FLAGSHIP).expect("compile");
    let b = compile_str(FLAGSHIP).expect("compile");
    assert_eq!(a.source, b.source, "codegen must be a pure function of the document");
    // Structural anchors (the full snapshot is the checked-in mcugen-audio/src/graph.rs).
    assert!(a.source.contains("pub async fn run_flagship_with<G0, G1, S>"));
    assert!(a.source.contains("run_sources_fanin_sink(cap_a, SourceChain(cap_b, Chain(conv, rs)), mix, SinkChain(enc, &mut rtp))"));
    assert!(a.source.contains("StaticLendRing<1, 1920>"));
    assert!(a.source.contains("RtpSink::new(sender, MediaClock::audio(8000), 0, 0x67676767, 0)"));
}

#[test]
fn the_display_graph_compiles_to_a_video_sink_pipeline() {
    // The video catalog: a raster camera -> SPI display, no fan-in. Proves the
    // compiler is not audio-specific (the end-to-end wire proof is in
    // examples/mcugen-graphs).
    let c = compile_str(DISPLAY).expect("display compile");
    // One 4x4 RGBA frame = 64 bytes; the display sink lends no ring.
    assert_eq!(c.ring_bytes_total, 4 * 4 * 4);
    assert_eq!(c.rings.len(), 1, "only the camera lends a ring");
    assert_eq!(c.entry, "run_display_with");
    assert_eq!(c.grabber_params, ["grab_cam"]);
    // The sink seams are the SPI bus, D/C pin, and delay, in entry order.
    assert_eq!(c.sink_params, ["spi", "dc", "delay"]);
    // A display sink is a plain linear source->sink (no transform, no fan-in),
    // returns unit (no `-> ()` emitted), and drives its panel over embedded-hal.
    assert!(c.source.contains("pub async fn run_display_with<G0, SPI, DC, DLY>"));
    assert!(c.source.contains("SpiDisplaySink::st7789(spi, dc, 4, 4)"));
    assert!(c.source.contains("mut delay: DLY"), "the delay seam must bind `mut` (it is borrowed)");
    assert!(c.source.contains("run_source_sink(cam, &mut disp)"));
    assert!(!c.source.contains("run_sources_fanin_sink"), "no fan-in in the display graph");
    assert!(!c.source.contains("-> ()"), "a unit-returning entry omits the return arrow");
}

#[test]
fn a_display_fed_the_wrong_pixel_format_is_rejected() {
    // An RGB565 (bpp 2) camera into an RGBA-expecting panel: bad geometry, so
    // the mis-wire fails the compile rather than emitting a broken pipeline.
    let doc = r#"
name: badpix
frame_ns: 33333333
frames: 8
nodes:
  - { id: cam,  element: grabbersrc, props: { width-px: 4, height-px: 4, format: rgb565 } }
  - { id: disp, element: spidisplaysink, props: { driver: st7789, width-px: 4, height-px: 4 } }
edges:
  - { from: cam, to: disp }
"#;
    assert!(matches!(compile_str(doc), Err(CompileError::BadGeometry { .. })));
}

#[test]
fn a_display_geometry_mismatch_is_rejected() {
    // A 4x4 camera into an 8x8 panel: the input dimensions must match the panel.
    let doc = r#"
name: baddim
frame_ns: 33333333
frames: 8
nodes:
  - { id: cam,  element: grabbersrc, props: { width-px: 4, height-px: 4 } }
  - { id: disp, element: spidisplaysink, props: { driver: st7789, width-px: 8, height-px: 8 } }
edges:
  - { from: cam, to: disp }
"#;
    assert!(matches!(compile_str(doc), Err(CompileError::BadGeometry { .. })));
}

#[test]
fn a_linear_graph_compiles_without_a_fan_in() {
    // source -> resample -> encode -> rtp (no mixer): exercises the linear path.
    let doc = r#"
name: line
frame_ns: 10000000
frames: 10
nodes:
  - { id: cap, element: grabbersrc, props: { sample-rate: 16000, width: 2 } }
  - { id: rs,  element: resample, props: { from: 16000, to: 8000 } }
  - { id: enc, element: g711enc, props: { law: alaw } }
  - { id: rtp, element: rtpsink, props: { ssrc: 1 } }
edges:
  - { from: cap, to: rs }
  - { from: rs,  to: enc }
  - { from: enc, to: rtp }
"#;
    let c = compile_str(doc).expect("linear compile");
    assert!(c.source.contains("run_source_transform_sink"));
    assert!(!c.source.contains("run_sources_fanin_sink"), "no fan-in in a linear graph");
    assert!(c.source.contains("Law::Alaw"));
    // 16 kHz S16 capture = 320 B; resample-to-8k = 160 B; a-law encode = 80 B.
    assert_eq!(c.ring_bytes_total, 320 + 160 + 80);
}

#[test]
fn mixer_input_mismatch_is_rejected() {
    // Two 8 kHz and 16 kHz branches into the mixer: unequal geometry.
    let doc = r#"
name: bad
frame_ns: 10000000
frames: 10
nodes:
  - { id: a,   element: grabbersrc, props: { sample-rate: 8000, width: 2 } }
  - { id: b,   element: grabbersrc, props: { sample-rate: 16000, width: 2 } }
  - { id: mix, element: mixer }
  - { id: rtp, element: rtpsink, props: { ssrc: 1 } }
edges:
  - { from: a, to: mix }
  - { from: b, to: mix }
  - { from: mix, to: rtp }
"#;
    assert!(matches!(compile_str(doc), Err(CompileError::MixerInputMismatch { .. })));
}

#[test]
fn encoder_fed_wrong_width_is_rejected() {
    // g711enc directly after a 32-bit-slot capture (no pcmconvert): bad width.
    let doc = r#"
name: bad2
frame_ns: 10000000
frames: 10
nodes:
  - { id: cap, element: grabbersrc, props: { sample-rate: 8000, width: 4 } }
  - { id: enc, element: g711enc }
  - { id: rtp, element: rtpsink, props: { ssrc: 1 } }
edges:
  - { from: cap, to: enc }
  - { from: enc, to: rtp }
"#;
    assert!(matches!(compile_str(doc), Err(CompileError::BadGeometry { .. })));
}

#[test]
fn fractional_frame_is_rejected() {
    // 44100 Hz is not in the catalog, but even a catalog rate with a bad
    // period must be caught: 8000 Hz over 3 ms = 24 samples exactly, but
    // over 100000 ns (0.1 ms) = 0.8 samples.
    let doc = r#"
name: frac
frame_ns: 100000
frames: 10
nodes:
  - { id: cap, element: grabbersrc, props: { sample-rate: 8000, width: 2 } }
  - { id: rtp, element: rtpsink, props: { ssrc: 1 } }
edges:
  - { from: cap, to: rtp }
"#;
    assert!(matches!(compile_str(doc), Err(CompileError::FractionalFrame { .. })));
}

#[test]
fn unknown_element_and_bad_name_are_rejected() {
    let bad_el = r#"
name: x
frame_ns: 10000000
frames: 1
nodes: [ { id: a, element: nosuchthing } ]
edges: []
"#;
    assert!(matches!(compile_str(bad_el), Err(CompileError::UnknownElement(_))));

    let bad_name = r#"
name: "1bad"
frame_ns: 10000000
frames: 1
nodes: [ { id: a, element: grabbersrc, props: { sample-rate: 8000 } } ]
edges: []
"#;
    assert!(matches!(compile_str(bad_name), Err(CompileError::BadName(_))));
}
