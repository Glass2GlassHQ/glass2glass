# g2g-onnx-import: an ONNX topology running in `BurnInference`

The Burn backend's graph-topology import. `safetensors` (`WgpuInference`) carries
trained weights into a topology that stays compiled Rust; this brings the
topology itself in from an ONNX file.

`build.rs` runs `burn-onnx`'s `ModelGen` over
[`model/tiny_classifier.onnx`](model/tiny_classifier.onnx), which generates a
burn `Module` plus a burnpack weight blob embedded in the binary. `burn-onnx` is
build-time codegen, not a runtime loader, so the generated `Model<Wgpu>` reaches
the pipeline through `g2g_ml::burninfer::BurnModule`: [`TinyClassifier`](src/lib.rs)
implements that trait and `BurnInference::module` drives it per frame, exactly
like the element's built-in linear layer.

The imported graph is `Conv2d -> BatchNorm -> ReLU -> global average pool ->
linear`, 4x4 RGB in, 2 logits out. Attention topologies are not validated through
this path yet.

## Why it is workspace-excluded

`burn-onnx` needs rustc 1.92 (burn 0.21's own MSRV) and the publishable workspace
is pinned at 1.86, so this crate is in the root `Cargo.toml` `exclude` list with
its own `Cargo.lock` and path-deps back to `g2g-ml`. Build, test, `fmt` and
`clippy` it from this directory, not from the workspace root.

## Run

```sh
cargo test
```

Needs a wgpu adapter; the test skips itself when none is found. It runs the
imported model through the real element and asserts the logits match the ONNX
Runtime reference for the same frame (tolerance 1e-3, f32 GPU conv/BN drift).

## Regenerate the fixture

```sh
uv run --with onnx --with onnxruntime --with numpy ../../tools/onnx-fixture.py \
    model/tiny_classifier.onnx
```

Deterministic (fixed seed): it rewrites the `.onnx` and prints the `RGBA_FRAME`
and `EXPECTED_LOGITS` constants to paste into
[`tests/onnx_import.rs`](tests/onnx_import.rs).
