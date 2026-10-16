#!/usr/bin/env bash
# usage: tools/executorch-build.sh <prefix>, builds the ExecuTorch static libraries `executorchinfer` links into <prefix>/cmake-out
set -euo pipefail

EXECUTORCH_VERSION=1.5.1
TORCH_VERSION=2.14.0
TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
EXECUTORCH_REPOSITORY=https://github.com/pytorch/executorch.git
PYTHON_VERSION=3.12

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi

prefix=$(realpath -m "$1")
# the ExecuTorch CMakeLists refuses a checkout not named executorch
source_dir="$prefix/executorch"
venv_dir="$prefix/venv"
build_dir="$prefix/cmake-out"

if [ -f "$build_dir/libexecutorch.a" ] && [ -f "$build_dir/kernels/portable/libportable_ops_lib.a" ]; then
    echo "export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$build_dir"
    exit 0
fi

if [ ! -d "$source_dir/.git" ]; then
    git clone --depth 1 --branch "v$EXECUTORCH_VERSION" "$EXECUTORCH_REPOSITORY" "$source_dir"
fi
git -C "$source_dir" submodule update --init --recursive --depth 1

# the kernel codegen imports torchgen (from the torch wheel) and yaml
uv venv --allow-existing -p "$PYTHON_VERSION" "$venv_dir"
# the CPU index comes first, so torch resolves there, pyyaml is only on PyPI
uv pip install -p "$venv_dir/bin/python" "torch==$TORCH_VERSION" pyyaml --index "$TORCH_CPU_INDEX"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DPYTHON_EXECUTABLE="$venv_dir/bin/python" \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON
ninja -C "$build_dir"

echo "export EXECUTORCH_RS_EXECUTORCH_LIB_DIR=$build_dir"
