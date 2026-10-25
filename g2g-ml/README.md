# g2g-ml

ML inference elements for
[glass2glass](https://github.com/Glass2GlassHQ/glass2glass): Burn (pure Rust, wgpu),
ONNX Runtime and ExecuTorch backends, GPU tensor preprocessing, detection
post-processing into the frame analytics metadata, and a multi-stream tensor
batcher.

## Building with `executorch`

The `executorch` feature links the ExecuTorch 1.5.1 C++ static libraries, which
cargo does not build. Build them once, export the directory the script prints,
then build with the feature:

```sh
tools/executorch-build.sh ~/executorch
export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=~/executorch/cmake-out
cargo test -p g2g-ml --features executorch --test executorch_inference
```
