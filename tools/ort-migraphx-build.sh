#!/usr/bin/env bash
# usage: tools/ort-migraphx-build.sh <prefix> <migraphx install dir>, builds ONNX Runtime with the MIGraphX provider into <prefix>/onnxruntime/build/Release
set -euo pipefail

ONNXRUNTIME_VERSION=1.24.4
ONNXRUNTIME_REPOSITORY=https://github.com/microsoft/onnxruntime.git
BUILD_JOBS=8

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <prefix> <migraphx install dir>" >&2
    exit 2
fi

prefix=$(realpath -m "$1")
migraphx_home=$(realpath "$2")
patch_file="$(dirname "$(realpath "$0")")/onnxruntime-$ONNXRUNTIME_VERSION-migraphx-6.4.patch"
source_dir="$prefix/onnxruntime"
library_dir="$source_dir/build/Release"

if [ -f "$library_dir/libonnxruntime_providers_migraphx.so" ]; then
    echo "export ORT_LIB_LOCATION=$library_dir ORT_PREFER_DYNAMIC_LINK=1"
    exit 0
fi

if [ ! -d "$source_dir/.git" ]; then
    git clone --depth 1 --branch "v$ONNXRUNTIME_VERSION" "$ONNXRUNTIME_REPOSITORY" "$source_dir"
    # the onnxruntime 1.24 provider calls fp4 and bf16 functions migraphx 6.4 lacks, and turns an empty model cache dir into the path ""
    git -C "$source_dir" apply "$patch_file"
fi

(cd "$source_dir" && ./build.sh --config Release --build_dir build --build_shared_lib \
    --parallel "$BUILD_JOBS" --skip_tests --use_migraphx --migraphx_home "$migraphx_home" \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_PREFIX_PATH=/usr/lib64/rocm)

echo "export ORT_LIB_LOCATION=$library_dir ORT_PREFER_DYNAMIC_LINK=1"
