#!/usr/bin/env bash
# usage: tools/libvpx-build.sh <prefix>, builds the static libvpx that the release binaries link into <prefix>
set -euo pipefail

LIBVPX_VERSION=1.17.0
LIBVPX_SHA256=1020f184046187baa2985dbde38e0691f49c44088bca7a1842b0236c6081dc0a

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi

prefix=$(realpath -m "$1")
tarball="$prefix/libvpx-$LIBVPX_VERSION.tar.gz"
source_dir="$prefix/libvpx-$LIBVPX_VERSION"

if [ ! -f "$prefix/lib/pkgconfig/vpx.pc" ]; then
    mkdir -p "$prefix"
    if [ ! -f "$tarball" ]; then
        curl -sfL -o "$tarball" "https://github.com/webmproject/libvpx/archive/refs/tags/v$LIBVPX_VERSION.tar.gz"
    fi
    echo "$LIBVPX_SHA256  $tarball" | sha256sum -c - >&2
    tar -xzf "$tarball" -C "$prefix"
    cd "$source_dir"
    ./configure \
        --prefix="$prefix" \
        --enable-static --disable-shared --enable-pic \
        --enable-vp8 --enable-vp9 \
        --disable-examples --disable-tools --disable-docs --disable-unit-tests \
        >&2
    make -j"$(nproc)" >&2
    make install >&2
fi

echo "export PKG_CONFIG_PATH=$prefix/lib/pkgconfig"
# pkg-config-rs links the static archive and its private libraries when this is set
echo "export VPX_STATIC=1"
