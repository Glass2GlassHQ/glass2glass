#!/usr/bin/env bash
# usage: tools/ffmpeg-build.sh <prefix>, builds the LGPL ffmpeg shared libraries the release binaries ship into <prefix>
#
# On Windows run it from an MSYS2 shell that inherits a Visual Studio developer
# environment (cl.exe on PATH) and has make and nasm: it builds with MSVC, puts
# the DLLs in <prefix>/bin and the import libraries in <prefix>/lib, and prints
# the FFMPEG_DIR that ffmpeg-sys-next reads instead of a PKG_CONFIG_PATH.
set -euo pipefail

FFMPEG_VERSION=8.1.3
FFMPEG_SHA256=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi

windows=false
case "$(uname -s)" in
MINGW* | MSYS*) windows=true ;;
esac

prefix=$(realpath -m "$1")
tarball="$prefix/ffmpeg-$FFMPEG_VERSION.tar.xz"
source_dir="$prefix/ffmpeg-$FFMPEG_VERSION"

print_env() {
    if $windows; then
        echo "export FFMPEG_DIR=$(cygpath -w "$prefix")"
    else
        echo "export PKG_CONFIG_PATH=$prefix/lib/pkgconfig"
    fi
}

if [ -f "$prefix/lib/pkgconfig/libavcodec.pc" ]; then
    print_env
    exit 0
fi

mkdir -p "$prefix"
if [ ! -f "$tarball" ]; then
    curl -sfL -o "$tarball" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
fi
echo "$FFMPEG_SHA256  $tarball" | sha256sum -c - >&2
tar -xJf "$tarball" -C "$prefix"

platform=()
if $windows; then
    platform=(--toolchain=msvc --target-os=win64 --arch=x86_64)
    # MSYS2's coreutils `link` would shadow the MSVC linker that sits beside cl.exe
    cl_path=$(command -v cl) || { echo "cl.exe is not on PATH: run from a Visual Studio developer environment" >&2; exit 1; }
    PATH="$(dirname "$cl_path"):$PATH"
fi

cd "$source_dir"
# --disable-autodetect keeps out system libraries such as libva, zlib and libxcb that a bare host lacks
./configure \
    --prefix="$prefix" \
    "${platform[@]}" \
    --enable-shared --disable-static \
    --disable-autodetect \
    --disable-programs --disable-doc --disable-debug \
    --disable-avdevice --disable-avfilter --disable-swresample \
    >&2
make -j"$(nproc)" >&2
make install >&2
if $windows; then
    # the MSVC install puts the import libraries beside the DLLs, ffmpeg-sys-next looks in lib/
    mv "$prefix"/bin/*.lib "$prefix/lib/"
fi

print_env
