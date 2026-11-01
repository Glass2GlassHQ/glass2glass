#!/usr/bin/env bash
# usage: tools/ffmpeg-build.sh <prefix>, builds the LGPL ffmpeg shared libraries the release binaries ship into <prefix>
set -euo pipefail

FFMPEG_VERSION=8.1.3
FFMPEG_SHA256=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi

prefix=$(realpath -m "$1")
tarball="$prefix/ffmpeg-$FFMPEG_VERSION.tar.xz"
source_dir="$prefix/ffmpeg-$FFMPEG_VERSION"

if [ -f "$prefix/lib/pkgconfig/libavcodec.pc" ]; then
    echo "export PKG_CONFIG_PATH=$prefix/lib/pkgconfig"
    exit 0
fi

mkdir -p "$prefix"
if [ ! -f "$tarball" ]; then
    curl -sfL -o "$tarball" "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz"
fi
echo "$FFMPEG_SHA256  $tarball" | sha256sum -c - >&2
tar -xJf "$tarball" -C "$prefix"

cd "$source_dir"
# --disable-autodetect keeps out system libraries such as libva, zlib and libxcb that a bare host lacks
./configure \
    --prefix="$prefix" \
    --enable-shared --disable-static \
    --disable-autodetect \
    --disable-programs --disable-doc --disable-debug \
    --disable-avdevice --disable-avfilter --disable-swresample \
    >&2
make -j"$(nproc)" >&2
make install >&2

echo "export PKG_CONFIG_PATH=$prefix/lib/pkgconfig"
