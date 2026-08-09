use burn_onnx::{LoadStrategy, ModelGen};

const FIXTURES: [&str; 2] = ["model/tiny_classifier.onnx", "model/tiny_attention.onnx"];

fn main() {
    let mut generator = ModelGen::new();
    for fixture in FIXTURES {
        println!("cargo:rerun-if-changed={fixture}");
        generator.input(fixture);
    }
    generator
        .out_dir("model/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
