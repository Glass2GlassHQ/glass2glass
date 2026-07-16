//! The M646/M648 conformance oracle: each `g2g-mcugen`-generated flagship graph
//! must reproduce its hand-written reference's wire output exactly. Both sides
//! are driven with identical mock peripherals; the generated pipeline's wire
//! checksum must equal the reference constant. This is what makes the compiler
//! trustworthy across both catalogs: not that it emits Rust that compiles, but
//! that the Rust it emits produces the same bytes on the wire as the graphs a
//! human wrote and verified (audio against ffmpeg and a float DSP reference;
//! video against the ST7789 wire protocol).

use noalloc_pipeline::audio::AUDIO_EXPECTED_CHECKSUM;
use noalloc_pipeline::EXPECTED_CHECKSUM;

#[test]
fn generated_audio_graph_matches_the_reference_wire() {
    assert_eq!(
        mcugen_graphs::run_audio_generated(),
        AUDIO_EXPECTED_CHECKSUM,
        "the generated audio graph must produce the same RTP wire as the hand-written reference"
    );
}

#[test]
fn generated_video_graph_matches_the_reference_wire() {
    // The reference's `Touch` transform is a byte no-op, so the generated
    // camera->display graph (no transform) puts the identical bytes on the
    // panel bus and must reproduce the same checksum.
    assert_eq!(
        mcugen_graphs::run_video_generated(),
        EXPECTED_CHECKSUM,
        "the generated display graph must drive the same panel wire as the hand-written reference"
    );
}

#[test]
fn generated_graphs_are_deterministic() {
    assert_eq!(mcugen_graphs::run_audio_generated(), mcugen_graphs::run_audio_generated());
    assert_eq!(mcugen_graphs::run_video_generated(), mcugen_graphs::run_video_generated());
}
