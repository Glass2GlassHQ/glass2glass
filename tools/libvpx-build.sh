#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# usage: tools/libvpx-build.sh <prefix>, builds the static libvpx that the release binaries link into <prefix>
set -euo pipefail

LIBVPX_VERSION=1.17.0
LIBVPX_SHA256=1020f184046187baa2985dbde38e0691f49c44088bca7a1842b0236c6081dc0a

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <prefix>" >&2
    exit 2
fi

mkdir -p "$1"
prefix=$(cd "$1" && pwd)

case "$(uname -s)" in
MINGW* | MSYS*)
    # vcpkg's libvpx port drives libvpx's own MSVC project generator, the -windows-static triplets link the static C runtime
    # vcvarsall sets VSCMD_ARG_TGT_ARCH to x64 or arm64, the same names vcpkg's triplets use
    triplet="$VSCMD_ARG_TGT_ARCH-windows-static"
    vcpkg install --triplet "$triplet" libvpx pkgconf >&2
    vcpkg_root=$(cygpath -m "$VCPKG_INSTALLATION_ROOT")
    installed="$vcpkg_root/installed/$triplet"
    cat "$installed/share/libvpx/copyright" "$vcpkg_root"/buildtrees/libvpx/src/*/PATENTS > "$prefix/LICENSE-LIBVPX.txt"
    echo "export PKG_CONFIG=$installed/tools/pkgconf/pkgconf.exe"
    echo "export PKG_CONFIG_PATH=$installed/lib/pkgconfig"
    echo "export VPX_STATIC=1"
    exit 0
    ;;
esac

tarball="$prefix/libvpx-$LIBVPX_VERSION.tar.gz"
source_dir="$prefix/libvpx-$LIBVPX_VERSION"

if [ ! -f "$prefix/lib/pkgconfig/vpx.pc" ]; then
    if [ ! -f "$tarball" ]; then
        curl -sfL -o "$tarball" "https://github.com/webmproject/libvpx/archive/refs/tags/v$LIBVPX_VERSION.tar.gz"
    fi
    # macOS has shasum but no sha256sum
    if command -v sha256sum > /dev/null; then
        echo "$LIBVPX_SHA256  $tarball" | sha256sum -c - >&2
    else
        echo "$LIBVPX_SHA256  $tarball" | shasum -a 256 -c - >&2
    fi
    tar -xzf "$tarball" -C "$prefix"
    cd "$source_dir"
    ./configure \
        --prefix="$prefix" \
        --enable-static --disable-shared --enable-pic \
        --enable-vp8 --enable-vp9 \
        --disable-examples --disable-tools --disable-docs --disable-unit-tests \
        >&2
    make -j"$(getconf _NPROCESSORS_ONLN)" >&2
    make install >&2
fi

# x86inc is ISC licensed and assembled into libvpx on x86 and x64
cat "$source_dir/LICENSE" "$source_dir/PATENTS" "$source_dir/third_party/x86inc/LICENSE" > "$prefix/LICENSE-LIBVPX.txt"

echo "export PKG_CONFIG_PATH=$prefix/lib/pkgconfig"
# pkg-config-rs links the static archive and its private libraries when this is set
echo "export VPX_STATIC=1"
