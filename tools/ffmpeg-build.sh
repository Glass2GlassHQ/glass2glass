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
ZLIB_VERSION=1.3.2
ZLIB_SHA256=d7a0654783a4da529d1bb793b7ad9c3318020af77667bcae35f95d0e42a792f3

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
zlib_tarball="$prefix/zlib-$ZLIB_VERSION.tar.xz"
zlib_dir="$prefix/zlib-$ZLIB_VERSION"
# zlib is built static into its own prefix, so nothing but the ffmpeg libraries ships
zlib_prefix="$prefix/zlib"

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
if [ ! -f "$zlib_tarball" ]; then
    curl -sfL -o "$zlib_tarball" "https://github.com/madler/zlib/releases/download/v$ZLIB_VERSION/zlib-$ZLIB_VERSION.tar.xz"
fi
echo "$ZLIB_SHA256  $zlib_tarball" | sha256sum -c - >&2
tar -xJf "$zlib_tarball" -C "$prefix"

platform=()
if $windows; then
    platform=(--toolchain=msvc --target-os=win64 --arch=x86_64)
    # MSYS2's coreutils `link` would shadow the MSVC linker that sits beside cl.exe
    cl_path=$(command -v cl) || { echo "cl.exe is not on PATH: run from a Visual Studio developer environment" >&2; exit 1; }
    PATH="$(dirname "$cl_path"):$PATH"
fi

# the zlib codecs (ZMBV, PNG, Screenpresso, ...) need it; static, and position
# independent so it links into the shared libavcodec / libavformat
mkdir -p "$zlib_prefix/include" "$zlib_prefix/lib"
if $windows; then
    # Makefile.msc builds against the DLL C runtime (-MD), as ffmpeg's MSVC build does
    (cd "$zlib_dir" && nmake -nologo -f win32/Makefile.msc zlib.lib >&2)
    cp "$zlib_dir/zlib.h" "$zlib_dir/zconf.h" "$zlib_prefix/include/"
    # the MSVC toolchain turns `-lz` into zlib.lib, and hands -L to link.exe as is, which ignores it
    cp "$zlib_dir/zlib.lib" "$zlib_prefix/lib/"
    zlib_include=$(cygpath -m "$zlib_prefix/include")
    zlib_ldflags="-libpath:$(cygpath -m "$zlib_prefix/lib")"
else
    (cd "$zlib_dir" && CFLAGS="-O2 -fPIC" ./configure --static --prefix="$zlib_prefix" >&2 && make -j"$(nproc)" libz.a >&2)
    cp "$zlib_dir/zlib.h" "$zlib_dir/zconf.h" "$zlib_prefix/include/"
    cp "$zlib_dir/libz.a" "$zlib_prefix/lib/"
    zlib_include=$zlib_prefix/include
    zlib_ldflags="-L$zlib_prefix/lib"
fi

cd "$source_dir"
# --disable-autodetect keeps out system libraries such as libva and libxcb that a bare host lacks;
# zlib comes from the static build above, its -L ahead of any system libz
./configure \
    --prefix="$prefix" \
    "${platform[@]}" \
    --enable-shared --disable-static \
    --disable-autodetect \
    --enable-zlib --extra-cflags="-I$zlib_include" --extra-ldflags="$zlib_ldflags" \
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
