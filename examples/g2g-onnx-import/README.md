# g2g-onnx-import: ONNX topologies running in `BurnInference`

The Burn backend's graph-topology import. `safetensors` (`WgpuInference`) carries
trained weights into a topology that stays compiled Rust; this brings the
topology itself in from an ONNX file.

`build.rs` runs `burn-onnx`'s `ModelGen` over each fixture in [`model/`](model),
which generates a burn `Module` plus a burnpack weight blob embedded in the
binary. `burn-onnx` is build-time codegen, not a runtime loader, so the generated
`Model<Wgpu>` reaches the pipeline through `g2g_ml::burninfer::BurnModule`: the
wrappers in [`src/lib.rs`](src/lib.rs) implement that trait and
`BurnInference::module` drives them per frame, exactly like the element's built-in
linear layer.

Two topologies, both 4x4 RGB in and 2 logits out:

| Fixture | Graph |
| :--- | :--- |
| `tiny_classifier.onnx` | `Conv2d -> BatchNorm -> ReLU -> global average pool -> linear` |
| `tiny_attention.onnx` | the 16 pixels as a token sequence -> multi-head self-attention -> mean pool -> linear |

The attention model is one standard-domain ONNX `Attention` node (opset 23) for
the whole multi-head block, which `burn-onnx` lowers onto
`burn::tensor::module::attention`, so the GPU runs burn's own attention kernel
rather than a hand-unrolled matmul/softmax chain.

## Why it is workspace-excluded

`burn-onnx` drags burn's whole codegen tree (and burn itself) into whatever
lockfile resolves it, so this crate is in the root `Cargo.toml` `exclude` list
with its own `Cargo.lock` and path-deps back to `g2g-ml`. Build, test, `fmt` and
`clippy` it from this directory, not from the workspace root.

## Run

```sh
cargo test
```

Needs a wgpu adapter; the tests skip themselves when none is found. Each runs its
imported model through the real element and asserts the logits match the ONNX
Runtime reference for the same frame (tolerance 1e-3, f32 GPU drift).

## Regenerate a fixture

```sh
uv run --with onnx --with onnxruntime --with numpy ../../tools/onnx-fixture.py \
    classifier model/tiny_classifier.onnx
uv run --with onnx --with onnxruntime --with numpy ../../tools/onnx-fixture.py \
    attention model/tiny_attention.onnx
```

Deterministic (fixed seed): it rewrites the `.onnx` and prints the `RGBA_FRAME`
and logits constants to paste into [`tests/onnx_import.rs`](tests/onnx_import.rs).
For the attention model it also folds the attention formula in numpy and asserts
ONNX Runtime agrees, so the reference logits do not rest on ORT alone.
