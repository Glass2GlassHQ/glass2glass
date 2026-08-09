use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    println!("cargo:rerun-if-changed=model/tiny_classifier.onnx");
    ModelGen::new()
        .input("model/tiny_classifier.onnx")
        .out_dir("model/")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}
