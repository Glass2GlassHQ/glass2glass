use noalloc_pipeline::keyword::{run_keyword, EXPECTED_RUN, INPUT_SCALE, INPUT_ZERO_POINT};

const QUANTIZATION: &str = include_str!("../../../g2g-mcu/tests/fixtures/keyword/quantization.txt");

fn quantization(name: &str) -> f64 {
    QUANTIZATION
        .lines()
        .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        .unwrap_or_else(|| panic!("{name} missing from quantization.txt"))
}

#[test]
fn keyword_graph_reproduces_the_pinned_features_and_scores() {
    let run = run_keyword().expect("keyword graph run");
    assert_eq!(
        run.features_digest, EXPECTED_RUN.features_digest,
        "features differ from features.i8"
    );
    assert_eq!(
        run.scores, EXPECTED_RUN.scores,
        "scores differ from scores.f32le"
    );
}

#[test]
fn quantizer_constants_are_the_model_input_quantization() {
    assert_eq!(INPUT_SCALE, quantization("input_scale") as f32);
    assert_eq!(INPUT_ZERO_POINT as f64, quantization("input_zero_point"));
}
